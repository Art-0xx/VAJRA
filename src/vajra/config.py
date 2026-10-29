# Load YAML configs into typed dataclasses.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str | Path, _is_recursive: bool = False) -> Config:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    if "inherits" in raw:
        parent = load_config(path.parent / raw.pop("inherits"), _is_recursive=True)
        raw = {**parent.raw, **raw}
    if not _is_recursive:
        local = path.parent / "local.yaml"
        if local.exists():
            with local.open() as f:
                override = yaml.safe_load(f) or {}
            raw = _deep_merge(raw, override)
    return Config(raw=raw, path=path)