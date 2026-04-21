#!/usr/bin/env python
"""Run compact sensitivity sweeps across RF, jitter, SNR, and mapping configs."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from src.plotting import plot_summary_bars, savefig, set_plot_style
from src.simulation import run_sweep
from src.utils import ensure_output_dirs, load_config


DEFAULT_SWEEP_CONFIGS = [
    "configs/default.yaml",
    "configs/broad_rf.yaml",
    "configs/narrow_rf.yaml",
    "configs/low_jitter.yaml",
    "configs/high_jitter.yaml",
    "configs/low_snr.yaml",
    "configs/high_snr.yaml",
    "configs/mapping_error_none.yaml",
    "configs/mapping_error_small.yaml",
    "configs/mapping_error_medium.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", default=DEFAULT_SWEEP_CONFIGS, help="Config paths to sweep.")
    parser.add_argument("--full", action="store_true", help="Disable debug overrides for sweeps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config("configs/default.yaml", debug=not args.full)
    paths = ensure_output_dirs(base_config)
    set_plot_style()
    table = run_sweep(base_config, args.configs, debug=not args.full)
    table.to_csv(paths["tables"] / "sweep_summary.csv", index=False)
    plot_summary_bars(table, paths["figures"] / "secondary_parameter_sweep_summary_bars.png")

    plt.figure(figsize=(7.2, 4.5))
    sns.scatterplot(
        data=table,
        x="median_abs_delta_po_deg",
        y="median_relative_response_change",
        hue="condition",
        s=80,
    )
    plt.xlabel("Median |ΔPO| at far offset (deg)")
    plt.ylabel("Median natural-image relative response change")
    plt.title("Parameter sweep bridge summary")
    plt.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(paths["figures"] / "secondary_parameter_sweep_bridge_scatter.png")
    print(f"Saved sweep table to {paths['tables'] / 'sweep_summary.csv'}")


if __name__ == "__main__":
    main()
