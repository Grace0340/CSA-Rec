import os
import random
import argparse
import numpy as np
import yaml


class Cfg(dict):
    """Dict with attribute access and nested get/set via dotted keys."""

    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v

    @staticmethod
    def _wrap(obj):
        if isinstance(obj, dict):
            return Cfg({k: Cfg._wrap(v) for k, v in obj.items()})
        return obj

    def dget(self, dotted, default=None):
        cur = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def dset(self, dotted, value):
        parts = dotted.split(".")
        cur = self
        for part in parts[:-1]:
            cur = cur.setdefault(part, Cfg())
        cur[parts[-1]] = value


def _coerce(s):
    """Turn a CLI string into bool/int/float/list where possible."""
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if "," in s:
        return [_coerce(x.strip()) for x in s.split(",")]
    return s


def load_config():
    """Load YAML config and apply --key.path value overrides."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args, extra = parser.parse_known_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = Cfg._wrap(yaml.safe_load(f))

    # extra is a flat list: --a.b val --c val ...
    i = 0
    while i < len(extra):
        key = extra[i]
        assert key.startswith("--"), f"bad override token: {key}"
        val = extra[i + 1]
        cfg.dset(key[2:], _coerce(val))
        i += 2
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def pick_device(requested):
    try:
        import torch
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
