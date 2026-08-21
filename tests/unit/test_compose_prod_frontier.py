"""Production Compose frontier: only Caddy publishes host ports."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docker-compose.yml"
PROD = ROOT / "docker-compose.prod.yml"


class _Reset:
    """Compose ``!reset`` tag: replace the base value instead of merging."""

    def __init__(self, value):
        self.value = value


def _reset_constructor(loader: yaml.SafeLoader, node: yaml.Node):
    if isinstance(node, yaml.SequenceNode):
        return _Reset(loader.construct_sequence(node))
    if isinstance(node, yaml.MappingNode):
        return _Reset(loader.construct_mapping(node))
    if isinstance(node, yaml.ScalarNode):
        raw = loader.construct_scalar(node)
        if raw in {"", "~", "null", "Null", "NULL"}:
            return _Reset(None)
        return _Reset(raw)
    return _Reset(None)


class _ComposeLoader(yaml.SafeLoader):
    pass


_ComposeLoader.add_constructor("!reset", _reset_constructor)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=_ComposeLoader)


def _merged_ports(base: dict, override: dict) -> dict[str, list]:
    ports: dict[str, list] = {}
    for name, service in (base.get("services") or {}).items():
        ports[name] = list(service.get("ports") or [])
    for name, service in (override.get("services") or {}).items():
        if "ports" not in (service or {}):
            ports.setdefault(name, [])
            continue
        value = service["ports"]
        if isinstance(value, _Reset):
            ports[name] = list(value.value or [])
        else:
            ports[name] = list(ports.get(name) or []) + list(value or [])
    return ports


def test_production_frontier_publishes_only_caddy_80_and_443() -> None:
    ports = _merged_ports(_load(BASE), _load(PROD))
    published = {name: entries for name, entries in ports.items() if entries}
    assert set(published) == {"caddy"}
    rendered = " ".join(str(item) for item in published["caddy"])
    assert "80" in rendered
    assert "443" in rendered
    assert "8501" not in rendered
    assert "8000" not in rendered
    assert "5432" not in rendered
    assert "6379" not in rendered
