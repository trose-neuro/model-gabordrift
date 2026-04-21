#!/usr/bin/env python
"""Run the main gaze-induced tuning simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.simulation import run_main
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML path.")
    parser.add_argument("--debug", action="store_true", help="Use fast debug overrides from the config.")
    parser.add_argument("--skip-decomposition", action="store_true", help="Skip the four-way decomposition.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, debug=args.debug)
    result = run_main(config, run_decomp=not args.skip_decomposition)
    paths = result["paths"]
    print(f"Saved tables to {Path(paths['tables']).resolve()}")
    print(f"Saved figures to {Path(paths['figures']).resolve()}")


if __name__ == "__main__":
    main()
