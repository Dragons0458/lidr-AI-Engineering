#!/usr/bin/env python3
"""Verify that Streamlit-consumed routes exist in the generated OpenAPI spec.

No server, no network, no LLM: imports ``app.main:app`` and calls ``app.openapi()``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "docs" / "contract" / "web-consumed-routes.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_contract(path: Path) -> dict:
    if not path.is_file():
        print(f"Contract file not found: {path}", file=sys.stderr)
        raise SystemExit(2)
    return json.loads(path.read_text(encoding="utf-8"))


def _check_routes(spec_paths: dict, routes: list[dict]) -> list[str]:
    failures: list[str] = []
    for route in routes:
        path = route["path"]
        method = route["method"].lower()
        client = route.get("client", "?")
        if path not in spec_paths:
            failures.append(
                f"{client} {method.upper()} {path} — path missing from OpenAPI"
            )
            continue
        if method not in spec_paths[path]:
            available = ", ".join(sorted(spec_paths[path]))
            failures.append(
                f"{client} {method.upper()} {path} — method missing (have: {available})"
            )
    return failures


def _check_probes(spec_paths: dict, probes: list[dict]) -> list[str]:
    from app.api.security import is_public_path

    failures: list[str] = []
    for probe in probes:
        path = probe["path"]
        method = probe["method"].lower()
        consumer = probe.get("consumer", "?")
        if path not in spec_paths:
            failures.append(f"probe {consumer} {method.upper()} {path} — path missing")
            continue
        if method not in spec_paths[path]:
            failures.append(
                f"probe {consumer} {method.upper()} {path} — method missing"
            )
        if probe.get("unauthenticated") and not is_public_path(path):
            failures.append(
                f"probe {consumer} {method.upper()} {path} — "
                "unauthenticated=true but is_public_path() is False"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT,
        help="JSON artefact listing consumed routes (default: %(default)s)",
    )
    args = parser.parse_args()
    contract = _load_contract(args.contract)

    from app.main import app

    spec_paths = app.openapi().get("paths") or {}
    routes = contract.get("routes") or []
    probes = contract.get("probes") or []
    failures = _check_routes(spec_paths, routes) + _check_probes(spec_paths, probes)
    checks = len(routes) + len(probes)
    if failures:
        print(
            f"Contract FAILED — {len(failures)} issue(s), {checks} checks:",
            file=sys.stderr,
        )
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1

    print(f"Contract OK -- {checks} checks passed ({len(routes)} consumed routes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
