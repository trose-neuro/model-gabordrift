"""General utilities for configuration, reproducibility, and outputs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def project_root() -> Path:
    """Return the repository root inferred from this file location."""
    return Path(__file__).resolve().parents[1]


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return an empty dict for an empty document."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``updates`` into ``base`` and return ``base``."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def load_config(path: str | Path = "configs/default.yaml", *, debug: bool = False) -> dict[str, Any]:
    """Load a config, merging non-default configs over ``configs/default.yaml``.

    Parameters
    ----------
    path:
        Config path. If it is not the default file, it is treated as an
        override on top of the default config.
    debug:
        Apply the ``debug`` override block from the merged config.
    """
    root = project_root()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    default_path = root / "configs" / "default.yaml"

    cfg = read_yaml(default_path)
    if config_path.resolve() != default_path.resolve():
        cfg = deep_update(cfg, read_yaml(config_path))

    if debug:
        cfg = apply_debug_overrides(cfg)
    return cfg


def apply_debug_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Return a config copy with its ``debug`` section merged into the top level."""
    cfg = deepcopy(config)
    debug_block = deepcopy(cfg.get("debug", {}))
    deep_update(cfg, debug_block)
    cfg.setdefault("runtime", {})["debug"] = True
    return cfg


def rng_from_config(config: dict[str, Any], *, offset: int = 0) -> np.random.Generator:
    """Create a reproducible NumPy random generator from the config seed."""
    seed = int(config.get("project", {}).get("random_seed", 0)) + int(offset)
    return np.random.default_rng(seed)


def ensure_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    """Create and return standard output directories."""
    root = project_root()
    output_dir = Path(config.get("runtime", {}).get("output_dir", "results"))
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    paths = {
        "root": output_dir,
        "figures": output_dir / "figures",
        "arrays": output_dir / "arrays",
        "tables": output_dir / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def degree_grid(spec: dict[str, float]) -> np.ndarray:
    """Create an inclusive floating-point degree grid from min/max/step values."""
    start = float(spec["min"])
    stop = float(spec["max"])
    step = float(spec["step"])
    count = int(np.floor((stop - start) / step + 0.5)) + 1
    return start + step * np.arange(count)


def write_text(path: str | Path, text: str) -> None:
    """Write UTF-8 text, creating parent directories when needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def config_label(path: str | Path) -> str:
    """Return a compact label for a config path."""
    return Path(path).stem.replace("_", " ")
