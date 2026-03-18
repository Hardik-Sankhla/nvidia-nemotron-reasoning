"""YAML config loading utilities used by train/eval/full-run scripts."""

from pathlib import Path
import yaml


def load_yaml(path):
    p = Path(path)
    with p.open("r", encoding="utf8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base, override):
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out
