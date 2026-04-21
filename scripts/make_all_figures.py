#!/usr/bin/env python
"""Regenerate final figures from saved result tables when possible."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.plotting import (
    heatmap_from_table,
    plot_grating_gaze_rf_scheme,
    plot_moving_grating_gaze_rf_scheme,
    plot_summary_bars,
    set_plot_style,
)
from src.simulation import make_primary_figures
from src.utils import ensure_output_dirs, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="Config used to locate output directory.")
    parser.add_argument("--debug", action="store_true", help="Use debug output directory settings if configured.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, debug=args.debug)
    paths = ensure_output_dirs(config)
    set_plot_style()
    plot_grating_gaze_rf_scheme(paths["figures"] / "scheme_grating_gaze_rf_delta_po.png")
    plot_moving_grating_gaze_rf_scheme(paths["figures"] / "scheme_moving_grating_gaze_rf_delta_po.png")

    grating_path = paths["tables"] / "grating_shift_summary.csv"
    natural_path = paths["tables"] / "natural_image_shift_summary.csv"
    if not grating_path.exists() or not natural_path.exists():
        raise FileNotFoundError("Run scripts/run_main_simulation.py before regenerating figures.")
    grating = pd.read_csv(grating_path)
    natural = pd.read_csv(natural_path)
    make_primary_figures(grating, natural, paths)

    decomposition_path = paths["tables"] / "decomposition_summary.csv"
    if decomposition_path.exists():
        decomposition = pd.read_csv(decomposition_path)
        plot_summary_bars(
            decomposition.sort_values("summary_radius").groupby("condition").tail(1),
            paths["figures"] / "primary_10_noiseless_noisy_mapping_comparison.png",
        )

    model_path = paths["tables"] / "simple_vs_energy_summary.csv"
    if model_path.exists():
        plot_summary_bars(pd.read_csv(model_path), paths["figures"] / "primary_11_simple_vs_energy_summary.png")

    phase_sanity_path = paths["tables"] / "grating_phase_sanity_summary.csv"
    if phase_sanity_path.exists():
        import matplotlib.pyplot as plt
        import seaborn as sns

        phase_sanity = pd.read_csv(phase_sanity_path)
        plt.figure(figsize=(7.5, 4.8))
        sns.lineplot(data=phase_sanity, x="drift_deg", y="median_abs_delta_po_deg", hue="condition", marker="o")
        plt.xlabel("Diagonal gaze/FOV drift (deg)")
        plt.ylabel("Median |ΔPO| (deg)")
        plt.title("Full-field grating phase sanity check")
        plt.legend(frameon=False, fontsize=8)
        from src.plotting import savefig

        savefig(paths["figures"] / "validation_grating_phase_sanity.png")

    moving_sanity_path = paths["tables"] / "moving_grating_temporal_sanity_summary.csv"
    if moving_sanity_path.exists():
        import matplotlib.pyplot as plt
        import seaborn as sns
        from src.plotting import savefig

        moving_sanity = pd.read_csv(moving_sanity_path)
        plt.figure(figsize=(7.5, 4.8))
        sns.lineplot(data=moving_sanity, x="drift_deg", y="median_abs_delta_po_deg", hue="condition", marker="o")
        plt.xlabel("Diagonal gaze/FOV drift (deg)")
        plt.ylabel("Median |ΔPO| (deg)")
        plt.title("Moving-grating temporal averaging sanity check")
        plt.legend(frameon=False, fontsize=8)
        savefig(paths["figures"] / "validation_moving_grating_temporal_sanity.png")

    bridge_path = paths["tables"] / "grating_natural_bridge_summary.csv"
    if bridge_path.exists():
        bridge = pd.read_csv(bridge_path)
        bridge["condition"] = "main_model"
        plot_summary_bars(
            bridge.sort_values("gaze_radius").tail(1),
            paths["figures"] / "primary_09_gratings_vs_natural_images_summary.png",
        )

    # A convenience alias with a descriptive name for browsing the output folder.
    heatmap_from_table(
        natural,
        "median_relative_response_change",
        Path(paths["figures"]) / "natural_image_relative_change_heatmap.png",
        title="Natural-image relative response change",
        cmap="mako",
    )
    print(f"Regenerated figures in {paths['figures']}")


if __name__ == "__main__":
    main()
