#!/usr/bin/env python3
"""Session 15 post-deploy smoke test — frontier, TLS, and estimate shape.

POST-DEPLOY ONLY. Do NOT run from CI: the estimate check calls the real LLM.

Default mode probes the public surface (``--base-url``). Pass ``--ai-url``
(or set ``AI_SERVICE_URL``) for in-network checks (readiness, 401, estimate).

    docker compose -f docker-compose.yml exec web \\
      python scripts/smoke_test_s15.py --base-url http://localhost:8501 --ai-url http://ai-service:8000
"""

from __future__ import annotations

import argparse
import os
import socket
import ssl
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx

TOKEN = (
    os.getenv("AI_SERVICE_TOKEN", "").strip()
    or os.getenv("ESTIMATE_API_KEY", "").strip()
)
IDEMPOTENCY_KEY = "s15-smoke-happy-path"
VALID_CONFIDENCE = frozenset({"high", "medium", "low"})
PRIVATE_PORTS = (8000, 8501, 5432, 6379)
REDIRECT_CODES = frozenset({301, 302, 307, 308})

_CANDIDATE_TRANSCRIPTS = (
    Path(__file__).resolve().parent / "data" / "s15_smoke_transcript.txt",
    Path(__file__).resolve().parents[1]
    / "exercises"
    / "session-14"
    / "sample_transcript_happy_path.txt",
    Path("/app/scripts/data/s15_smoke_transcript.txt"),
    Path("/app/exercises/session-14/sample_transcript_happy_path.txt"),
)


@dataclass
class Counter:
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    failures: list[str] = field(default_factory=list)

    def ok(self, name: str, detail: str = "") -> None:
        self.passed += 1
        suffix = f" {detail}" if detail else ""
        print(f"PASS {name}{suffix}")

    def fail(self, name: str, detail: str) -> None:
        self.failed += 1
        self.failures.append(f"{name}: {detail}")
        print(f"FAIL {name} — {detail}", file=sys.stderr)

    def skip(self, name: str, detail: str) -> None:
        self.skipped += 1
        print(f"SKIP {name} — {detail}")


def _load_transcript() -> str:
    for path in _CANDIDATE_TRANSCRIPTS:
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if len(text) >= 100:
                print(f"transcript {path} ({len(text)} chars)")
                return text
    print(
        "FAIL: sample transcript not found "
        "(expected exercises/session-14/sample_transcript_happy_path.txt)",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("SMOKE_BASE_URL", "http://localhost:8501"),
        help="Public surface (Streamlit / Caddy). Default: %(default)s",
    )
    parser.add_argument(
        "--ai-url",
        default=os.getenv("AI_SERVICE_URL", ""),
        help="In-network AI service URL. Enables readiness / 401 / estimate checks.",
    )
    parser.add_argument(
        "--allow-insufficient",
        action="store_true",
        help="Accept confidence=insufficient (empty corpus). Not valid as check-4 evidence.",
    )
    parser.add_argument(
        "--skip-estimation",
        action="store_true",
        help="Skip the token-spending estimate call.",
    )
    return parser.parse_args()


def check_public_surface(base_url: str, counter: Counter) -> None:
    root = base_url.rstrip("/")
    with httpx.Client(timeout=15.0, follow_redirects=True) as client:
        try:
            home = client.get(root)
        except httpx.HTTPError as exc:
            counter.fail("public_surface", f"UI not reachable: {exc}")
            return
        if home.status_code != 200:
            counter.fail("public_surface", f"UI status {home.status_code}")
        else:
            counter.ok("public_surface", str(home.status_code))

        try:
            health = client.get(f"{root}/_stcore/health")
        except httpx.HTTPError as exc:
            counter.fail("stcore_health", str(exc))
            return
        if health.status_code != 200:
            counter.fail("stcore_health", f"status {health.status_code}")
        else:
            counter.ok("stcore_health", str(health.status_code))


def check_model_badge(base_url: str, counter: Counter) -> None:
    """Home prints the effective primary model only after GET /api/v1/config/models."""
    with httpx.Client(timeout=15.0, follow_redirects=True) as client:
        try:
            home = client.get(base_url.rstrip("/"))
        except httpx.HTTPError as exc:
            counter.fail("model_badge", str(exc))
            return
        body = home.text
        if "Modelo primario" in body or "PRIMARY_MODEL" in body:
            counter.ok("model_badge", "primary model visible (web → ai-service)")
            return
        lowered = body.lower()
        if any(token in lowered for token in ("gpt-", "claude", "gemini", "openai")):
            counter.ok("model_badge", "model identifier present in UI payload")
            return
        counter.fail(
            "model_badge",
            "home did not show the primary model — Streamlit could not reach "
            "GET /api/v1/config/models",
        )


def check_private_ports(host: str, public_port: int | None, counter: Counter) -> None:
    try:
        infos = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as exc:
        counter.fail(
            "private_ports",
            f"host {host!r} did not resolve ({exc}); refusing to treat closed ports as evidence",
        )
        return

    addresses = []
    for info in infos:
        addr = info[4][0]
        if addr not in addresses:
            addresses.append(addr)
    target = addresses[0]
    print(f"private_ports resolved {host} -> {target}")

    local_names = {"localhost", "127.0.0.1", "::1"}
    for port in PRIVATE_PORTS:
        if public_port is not None and port == public_port:
            counter.skip(f"port {port}", "public entry of --base-url")
            continue
        if host in local_names and port in {5432, 6379}:
            counter.skip(f"port {port}", "localhost may run developer Postgres/Redis")
            continue
        sock = socket.socket(socket.AF_INET if ":" not in target else socket.AF_INET6)
        sock.settimeout(2.0)
        try:
            result = sock.connect_ex((target, port))
        except OSError as exc:
            counter.ok(f"port {port}", f"not reachable ({exc})")
            continue
        finally:
            sock.close()
        if result == 0:
            counter.fail(
                f"port {port}",
                f"{host}:{port} accepted a connection — frontier leak",
            )
        else:
            counter.ok(f"port {port}", "refused")


def check_tls(base_url: str, counter: Counter) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme != "https":
        counter.skip("tls", "base-url is not https")
        return
    host = parsed.hostname or ""
    http_url = f"http://{host}"
    with httpx.Client(timeout=10.0, follow_redirects=False) as client:
        try:
            response = client.get(http_url)
        except httpx.HTTPError as exc:
            counter.fail("tls_redirect", str(exc))
        else:
            if response.status_code in REDIRECT_CODES:
                counter.ok("tls_redirect", str(response.status_code))
            else:
                counter.fail(
                    "tls_redirect",
                    f"HTTP {response.status_code}, expected {sorted(REDIRECT_CODES)}",
                )
    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, parsed.port or 443), timeout=10.0) as sock:
            with context.wrap_socket(sock, server_hostname=host) as wrapped:
                counter.ok("tls_handshake", wrapped.version() or "ok")
    except ssl.SSLError as exc:
        counter.fail("tls_handshake", str(exc))
    except OSError as exc:
        counter.fail("tls_handshake", str(exc))


def check_readiness(ai_url: str, counter: Counter) -> None:
    with httpx.Client(timeout=15.0) as client:
        try:
            ready = client.get(f"{ai_url.rstrip('/')}/health/ready")
        except httpx.HTTPError as exc:
            counter.fail("readiness", str(exc))
            return
        print("ready", ready.status_code, ready.text[:300])
        if ready.status_code == 200:
            counter.ok("readiness")
            return
        if ready.status_code == 503:
            counter.fail("readiness", f"dependency down: {ready.text[:400]}")
            return
        counter.fail("readiness", f"status {ready.status_code}")


def check_unauthenticated_rejected(ai_url: str, counter: Counter) -> None:
    with httpx.Client(timeout=15.0) as client:
        try:
            response = client.post(
                f"{ai_url.rstrip('/')}/api/v1/estimate",
                json={
                    "description": "Smoke test description long enough.",
                    "project_type": "web_saas",
                    "detail_level": "medium",
                    "output_format": "line_items",
                },
            )
        except httpx.HTTPError as exc:
            counter.fail("unauthenticated", str(exc))
            return
        if response.status_code == 401:
            counter.ok("unauthenticated", "401")
        else:
            counter.fail("unauthenticated", f"expected 401, got {response.status_code}")


def check_estimation(
    ai_url: str,
    *,
    allow_insufficient: bool,
    counter: Counter,
) -> None:
    if not TOKEN:
        counter.fail("estimate", "set ESTIMATE_API_KEY or AI_SERVICE_TOKEN")
        return
    headers = {"X-API-Key": TOKEN}
    transcript = _load_transcript()
    with httpx.Client(timeout=180.0) as client:
        health = client.get(f"{ai_url.rstrip('/')}/health")
        print(
            "health",
            health.status_code,
            health.json() if health.status_code == 200 else health.text[:200],
        )
        if health.status_code != 200:
            counter.fail("estimate_health", f"status {health.status_code}")
            return
        counter.ok("estimate_health")

        config = client.get(
            f"{ai_url.rstrip('/')}/api/v1/config/models", headers=headers
        )
        print("config/models", config.status_code)
        if config.status_code != 200:
            counter.fail("estimate_config", config.text[:400])
            return
        counter.ok("estimate_config")

        estimate = client.post(
            f"{ai_url.rstrip('/')}/v1/estimate/from-transcript",
            headers=headers,
            json={"transcript": transcript, "idempotency_key": IDEMPOTENCY_KEY},
        )
        print("POST /v1/estimate/from-transcript", estimate.status_code)
        if estimate.status_code != 200:
            counter.fail("estimate", estimate.text[:800])
            return

        body = estimate.json()
        confidence = body.get("confidence")
        sources = body.get("sources") or []
        modules = body.get("modules") or []
        days = body.get("total_engineer_days")

        if confidence == "insufficient":
            msg = (
                "confidence=insufficient — corpus likely empty "
                "(run Fase A / build_task_corpus.py --ingest)"
            )
            if allow_insufficient:
                print(f"WARN {msg}", file=sys.stderr)
                counter.skip("estimate_shape", "insufficient allowed")
                return
            counter.fail("estimate_shape", msg)
            return

        if confidence not in VALID_CONFIDENCE:
            counter.fail("estimate_shape", f"confidence={confidence!r}")
            return
        if not sources:
            counter.fail("estimate_shape", "sources is empty")
            return
        for i, src in enumerate(sources):
            sid = src.get("source_id") if isinstance(src, dict) else None
            if not isinstance(sid, int):
                counter.fail("estimate_shape", f"sources[{i}].source_id={sid!r}")
                return
        if days is None or not isinstance(days, int) or days <= 0:
            counter.fail("estimate_shape", f"total_engineer_days={days!r}")
            return
        if not modules:
            counter.fail("estimate_shape", "modules is empty")
            return
        counter.ok(
            "estimate_shape",
            f"confidence={confidence} sources={len(sources)} days={days} modules={len(modules)}",
        )


def main() -> int:
    args = _parse_args()
    counter = Counter()
    parsed = urlparse(args.base_url)
    host = parsed.hostname or ""
    if not host:
        print("FAIL: --base-url has no hostname", file=sys.stderr)
        return 2

    check_public_surface(args.base_url, counter)
    check_model_badge(args.base_url, counter)
    check_private_ports(host, parsed.port, counter)
    check_tls(args.base_url, counter)

    ai_url = (args.ai_url or "").rstrip("/")
    if ai_url:
        check_readiness(ai_url, counter)
        check_unauthenticated_rejected(ai_url, counter)
        if args.skip_estimation:
            counter.skip("estimate", "--skip-estimation")
        else:
            check_estimation(
                ai_url,
                allow_insufficient=args.allow_insufficient,
                counter=counter,
            )
    else:
        counter.skip("in_network", "pass --ai-url for readiness / 401 / estimate")

    print(
        f"{counter.passed} passed, {counter.failed} failed, {counter.skipped} skipped"
    )
    if counter.failed:
        for item in counter.failures:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("OK — smoke_test_s15 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
