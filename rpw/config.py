"""Configuration: base YAML <- optional overlay YAML <- --set key.path=value overrides."""
from __future__ import annotations

import copy
import pathlib
from typing import Any

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "config" / "v2" / "base.yaml"


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_path(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"unknown config section in override: {dotted}")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"unknown config key in override: {dotted}")
    node[keys[-1]] = value


def load_config(overlay: str | None = None, sets: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    if overlay:
        p = pathlib.Path(overlay)
        if not p.exists():
            p = ROOT / overlay
        cfg = deep_merge(cfg, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    for s in sets or []:
        if "=" not in s:
            raise ValueError(f"--set expects key.path=value, got {s!r}")
        k, v = s.split("=", 1)
        set_path(cfg, k.strip(), yaml.safe_load(v))
    return cfg
