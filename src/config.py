from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "run" / "config.yaml"


def load_config(path=None):
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if cfg["task"] not in ("segment", "detect"):
        raise SystemExit(f"[ERROR] task must be 'segment' or 'detect', got {cfg['task']!r}")
    return cfg


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
