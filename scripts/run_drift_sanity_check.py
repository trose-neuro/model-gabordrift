#!/usr/bin/env python
"""Focused 1-10 degree FOV/gaze drift sanity check."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.natural_images import load_natural_images
from src.plotting import savefig, set_plot_style
from src.sampling import sample_population
from src.simulation import perfect_mapping_config, run_grating_shift_grid, run_natural_shift_grid
from src.utils import ensure_output_dirs, load_config, rng_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--debug", action="store_true", help="Use debug model sizes but keep the requested 1-10 degree drift range.")
    parser.add_argument("--max-degree", type=float, default=10.0)
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument("--n-neurons", type=int, default=None, help="Override neuron count for a focused sanity run.")
    parser.add_argument("--include-natural", action="store_true", help="Also run natural-image metrics for the same shifts.")
    return parser.parse_args()


def make_focused_shifts(max_degree: float, step: float) -> pd.DataFrame:
    degrees = [round(x * step, 10) for x in range(int(max_degree / step) + 1)]
    rows = []
    for degree in degrees:
        rows.append({"drift_path": "azimuth", "drift_deg": degree, "gaze_az_deg": degree, "gaze_el_deg": 0.0})
        rows.append({"drift_path": "elevation", "drift_deg": degree, "gaze_az_deg": 0.0, "gaze_el_deg": degree})
        rows.append({"drift_path": "diagonal", "drift_deg": degree, "gaze_az_deg": degree, "gaze_el_deg": degree})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config, debug=args.debug)
    if args.n_neurons is not None:
        config["cortical_field"]["n_neurons"] = int(args.n_neurons)
    paths = ensure_output_dirs(config)
    set_plot_style()
    rng = rng_from_config(config, offset=900)
    population = sample_population(config, rng)
    shifts = make_focused_shifts(args.max_degree, args.step)

    grating = run_grating_shift_grid(
        population,
        config,
        rng,
        mapping_config=perfect_mapping_config(),
        noise_config={"model": "none", "repeats": 1, "clip_nonnegative": True},
        response_model=config["rf"].get("response_model", "simple"),
        shifts=shifts[["gaze_az_deg", "gaze_el_deg"]],
    )["summary"]
    table = pd.concat([shifts.reset_index(drop=True), grating.drop(columns=["gaze_az_deg", "gaze_el_deg"])], axis=1)

    if args.include_natural:
        images = load_natural_images(config, rng)
        natural = run_natural_shift_grid(
            population,
            images,
            config,
            rng,
            mapping_config=perfect_mapping_config(),
            noise_config={"model": "none", "repeats": 1, "clip_nonnegative": True},
            response_model=config["rf"].get("response_model", "simple"),
            shifts=shifts[["gaze_az_deg", "gaze_el_deg"]],
        )["summary"]
        table = pd.concat(
            [table, natural.drop(columns=["gaze_az_deg", "gaze_el_deg"]).reset_index(drop=True)],
            axis=1,
        )

    out_path = paths["tables"] / "drift_sanity_summary.csv"
    table.to_csv(out_path, index=False)

    plt.figure(figsize=(7.2, 4.5))
    sns.lineplot(data=table, x="drift_deg", y="median_abs_delta_po_deg", hue="drift_path", marker="o")
    plt.axhline(config["analysis"].get("po_large_shift_threshold_deg", 10.0), color="0.35", lw=1, ls="--")
    plt.xlabel("FOV/gaze drift magnitude (deg)")
    plt.ylabel("Median |ΔPO| (deg)")
    plt.title("Focused 1-10 degree drift sanity check")
    savefig(paths["figures"] / "drift_sanity_median_abs_delta_po.png")

    if args.include_natural:
        plt.figure(figsize=(7.2, 4.5))
        sns.lineplot(data=table, x="drift_deg", y="population_response_correlation", hue="drift_path", marker="o")
        plt.xlabel("FOV/gaze drift magnitude (deg)")
        plt.ylabel("Natural-image population correlation")
        plt.title("Natural-image sensitivity over 1-10 degree drift")
        savefig(paths["figures"] / "drift_sanity_natural_population_correlation.png")

    print(f"Saved focused drift sanity table to {out_path}")


if __name__ == "__main__":
    main()
