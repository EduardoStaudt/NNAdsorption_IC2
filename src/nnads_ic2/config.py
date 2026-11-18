"""YAML configuration files.

A config may name a parent file with the key `inherit: other.yaml` (path relative to the
child). The child's values override the parent's, key by key (deep merge), so a small
override file only lists what changes.
"""

import copy
import os

import yaml


def deep_merge(base, override):
    """Return a copy of `base` with `override` merged into it (nested dicts merged key by key)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path):
    """Read a YAML config, resolving `inherit` recursively."""
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent is not None:
        parent_path = os.path.join(os.path.dirname(path), parent)
        cfg = deep_merge(load_config(parent_path), cfg)
    return cfg


def apply_overrides(cfg, items):
    """Command-line overrides 'a.b.c=value' (value parsed as YAML, e.g. 10, 1.0e-3, true, [1, 2])."""
    for item in items or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError("override must look like key.sub=value, got %r" % item)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def save_config(cfg, path):
    """Write the fully resolved config (saved next to every run for reproducibility)."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def resolve_path(cfg_path_value, base_dir):
    """Paths inside configs are relative to the repository root (`base_dir`)."""
    if cfg_path_value is None:
        return None
    if os.path.isabs(cfg_path_value):
        return cfg_path_value
    return os.path.normpath(os.path.join(base_dir, cfg_path_value))
