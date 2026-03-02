"""Configuration loading and utilities: YAML config, path expansion, nested access, seeding."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config file and return a dict.

    Path keys under the top-level 'paths' key are expanded to absolute paths
    (relative to the current working directory).

    Args:
        path: Path to the YAML file.

    Returns:
        Config dict with path values expanded where applicable.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    base = Path.cwd()
    if "paths" in cfg and isinstance(cfg["paths"], dict):
        for key, value in cfg["paths"].items():
            if isinstance(value, str) and not Path(value).is_absolute():
                cfg["paths"][key] = str((base / value).resolve())

    return cfg


def get_path(cfg: dict[str, Any], *keys: str) -> Any:
    """Safely access a nested key in a config dict.

    Args:
        cfg: Config dictionary.
        *keys: One or more keys to traverse (e.g. "paths", "raw_dir").

    Returns:
        The value at the given path, or None if any key is missing.

    Example:
        >>> get_path(cfg, "paths", "raw_dir")
        "/abs/path/to/data/raw/msmarco_passage"
        >>> get_path(cfg, "limits", "max_passages")
        200000
    """
    current: Any = cfg
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Seeds the standard library random module, NumPy, and PyTorch (if installed)
    with the given seed.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
