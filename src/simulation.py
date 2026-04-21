"""Config-driven simulation pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .interpretation import interpretation_text, methods_summary_text
from .metrics import grating_shift_metrics, natural_image_metrics, neuronwise_grating_changes, safe_corrcoef
from .natural_images import load_natural_images
from .plotting import (
    heatmap_from_table,
    plot_mapping_validation,
    plot_parameter_distributions,
    plot_response_matrix,
    plot_retinotopy,
    plot_rf_examples,
    plot_summary_bars,
    plot_tuning_examples,
    savefig,
    set_plot_style,
)
from .responses import grating_response_mean, natural_response_trials
from .rf_models import gabor_kernel_bank
from .sampling import sample_population
from .stimuli import make_gaze_grid, make_grating_stimuli
from .tuning import estimate_orientation_tuning
from .utils import deep_update, ensure_output_dirs, rng_from_config, write_text


SMALL_MAPPING_ERROR = {
    "mode": "approximate",
    "error": {
        "name": "slight",
        "origin_offset_deg": [0.15, -0.10],
        "scale": [1.015, 0.990],
        "rotation_deg": 1.0,
        "anisotropy": [1.02, 0.98],
        "nonlinear": 0.0008,
    },
}


def perfect_mapping_config() -> dict:
    """Return an exact no-error mapping config."""
    return {
        "mode": "exact",
        "error": {
            "name": "perfect",
            "origin_offset_deg": [0.0, 0.0],
            "scale": [1.0, 1.0],
            "rotation_deg": 0.0,
            "anisotropy": [1.0, 1.0],
            "nonlinear": 0.0,
        },
    }


def default_noise_for_decomposition(config: dict) -> dict:
    """Return a nonzero noise config for decomposition when the main config is noiseless."""
    noise_cfg = deepcopy(config.get("noise", {}))
    if str(noise_cfg.get("model", "none")).lower() in {"none", "deterministic", "noiseless"}:
        noise_cfg.update({"model": "gaussian", "repeats": 2, "gaussian_sd": 0.10, "clip_nonnegative": True})
    return noise_cfg


def run_grating_shift_grid(
    population: pd.DataFrame,
    config: dict,
    rng: np.random.Generator,
    *,
    mapping_config: dict | None = None,
    noise_config: dict | None = None,
    response_model: str | None = None,
    shifts: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Run grating tuning estimation for all gaze shifts."""
    stimuli = make_grating_stimuli(config)
    shifts = make_gaze_grid(config) if shifts is None else shifts.copy()
    threshold = float(config["analysis"]["op_large_shift_threshold_deg"])

    baseline_mean, baseline_trials = grating_response_mean(
        population,
        stimuli,
        config,
        rng,
        gaze_shift=(0.0, 0.0),
        mapping_config=mapping_config,
        noise_config=noise_config,
        response_model=response_model,
    )
    baseline_tuning = estimate_orientation_tuning(
        baseline_mean,
        stimuli["orientations_deg"],
        stimuli["spatial_frequencies_cpd"],
    )

    rows = []
    far_state = None
    far_radius = -np.inf
    for shift in shifts.itertuples(index=False):
        gaze = (float(shift.gaze_az_deg), float(shift.gaze_el_deg))
        mean_response, trials = grating_response_mean(
            population,
            stimuli,
            config,
            rng,
            gaze_shift=gaze,
            mapping_config=mapping_config,
            noise_config=noise_config,
            response_model=response_model,
        )
        tuning = estimate_orientation_tuning(mean_response, stimuli["orientations_deg"], stimuli["spatial_frequencies_cpd"])
        row = {
            "gaze_az_deg": gaze[0],
            "gaze_el_deg": gaze[1],
            **grating_shift_metrics(baseline_tuning, tuning, threshold),
        }
        rows.append(row)
        radius = abs(gaze[0]) + abs(gaze[1])
        if radius > far_radius:
            far_radius = radius
            far_state = {
                "gaze": gaze,
                "response": mean_response,
                "trials": trials,
                "tuning": tuning,
                "neuron_changes": neuronwise_grating_changes(baseline_tuning, tuning),
            }

    return {
        "summary": pd.DataFrame(rows),
        "stimuli": stimuli,
        "baseline_response": baseline_mean,
        "baseline_trials": baseline_trials,
        "baseline_tuning": baseline_tuning,
        "far_state": far_state,
    }


def run_natural_shift_grid(
    population: pd.DataFrame,
    images: np.ndarray,
    config: dict,
    rng: np.random.Generator,
    *,
    mapping_config: dict | None = None,
    noise_config: dict | None = None,
    response_model: str | None = None,
    shifts: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Run natural-image response metrics for all gaze shifts."""
    shifts = make_gaze_grid(config) if shifts is None else shifts.copy()
    img_cfg = config["stimuli"]["natural_images"]
    kernels = gabor_kernel_bank(
        population,
        image_size_px=int(img_cfg["image_size_px"]),
        extent_deg=float(img_cfg["extent_deg"]),
    )
    threshold = float(config["analysis"]["natural_large_change_threshold"])

    baseline_trials = natural_response_trials(
        population,
        images,
        config,
        rng,
        gaze_shift=(0.0, 0.0),
        kernels=kernels,
        mapping_config=mapping_config,
        noise_config=noise_config,
        response_model=response_model,
    )
    baseline = baseline_trials.mean(axis=0)

    rows = []
    far_state = None
    far_radius = -np.inf
    for shift in shifts.itertuples(index=False):
        gaze = (float(shift.gaze_az_deg), float(shift.gaze_el_deg))
        trials = natural_response_trials(
            population,
            images,
            config,
            rng,
            gaze_shift=gaze,
            kernels=kernels,
            mapping_config=mapping_config,
            noise_config=noise_config,
            response_model=response_model,
        )
        response = trials.mean(axis=0)
        row = {
            "gaze_az_deg": gaze[0],
            "gaze_el_deg": gaze[1],
            **natural_image_metrics(baseline, response, large_change_threshold=threshold),
        }
        rows.append(row)
        radius = abs(gaze[0]) + abs(gaze[1])
        if radius > far_radius:
            far_radius = radius
            far_state = {"gaze": gaze, "response": response, "trials": trials}

    return {
        "summary": pd.DataFrame(rows),
        "images": images,
        "baseline_response": baseline,
        "baseline_trials": baseline_trials,
        "far_state": far_state,
        "kernels": kernels,
    }


def _condition_summary(
    condition: str,
    population: pd.DataFrame,
    images: np.ndarray,
    config: dict,
    rng: np.random.Generator,
    *,
    mapping_config: dict,
    noise_config: dict,
    response_model: str,
    shifts: pd.DataFrame,
) -> pd.DataFrame:
    grating = run_grating_shift_grid(
        population,
        config,
        rng,
        mapping_config=mapping_config,
        noise_config=noise_config,
        response_model=response_model,
        shifts=shifts,
    )["summary"]
    natural = run_natural_shift_grid(
        population,
        images,
        config,
        rng,
        mapping_config=mapping_config,
        noise_config=noise_config,
        response_model=response_model,
        shifts=shifts,
    )["summary"]
    merged = pd.merge(grating, natural, on=["gaze_az_deg", "gaze_el_deg"], how="inner")
    merged["condition"] = condition
    merged["summary_radius"] = merged["gaze_az_deg"].abs() + merged["gaze_el_deg"].abs()
    return merged


def run_decomposition(
    population: pd.DataFrame,
    images: np.ndarray,
    config: dict,
    rng: np.random.Generator,
    *,
    shifts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the four-way decomposition of gaze, mapping error, and noise."""
    shifts = make_gaze_grid(config) if shifts is None else shifts.copy()
    noiseless = deepcopy(config.get("noise", {}))
    noiseless.update({"model": "none", "repeats": 1})
    noisy = default_noise_for_decomposition(config)
    conditions = [
        ("gaze_only_perfect_noiseless", perfect_mapping_config(), noiseless),
        ("gaze_plus_mapping_error_noiseless", SMALL_MAPPING_ERROR, noiseless),
        ("gaze_plus_noise_perfect", perfect_mapping_config(), noisy),
        ("gaze_plus_mapping_error_plus_noise", SMALL_MAPPING_ERROR, noisy),
    ]
    tables = []
    for idx, (name, mapping, noise) in enumerate(conditions):
        tables.append(
            _condition_summary(
                name,
                population,
                images,
                config,
                np.random.default_rng(rng.integers(0, 2**32 - 1) + idx),
                mapping_config=mapping,
                noise_config=noise,
                response_model=config["rf"].get("response_model", "simple"),
                shifts=shifts,
            )
        )
    return pd.concat(tables, ignore_index=True)


def make_primary_figures(
    grating_summary: pd.DataFrame,
    natural_summary: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    """Generate the required primary heat maps."""
    fig_dir = paths["figures"]
    heatmap_from_table(
        grating_summary,
        "median_abs_op_shift_deg",
        fig_dir / "primary_01_median_abs_op_shift.png",
        title="Median absolute OP shift",
    )
    heatmap_from_table(
        grating_summary,
        "p90_abs_op_shift_deg",
        fig_dir / "primary_02_p90_abs_op_shift.png",
        title="90th percentile absolute OP shift",
    )
    heatmap_from_table(
        grating_summary,
        "fraction_abs_op_shift_gt_threshold",
        fig_dir / "primary_03_fraction_op_shift_gt_10deg.png",
        title="Fraction of neurons with |dOP| > threshold",
    )
    heatmap_from_table(
        grating_summary,
        "median_circular_variance_change",
        fig_dir / "primary_04_median_circular_variance_change.png",
        title="Median circular-variance change",
        cmap="vlag",
        center=0.0,
    )
    heatmap_from_table(
        grating_summary,
        "median_tuning_strength_change",
        fig_dir / "primary_05_median_tuning_strength_change.png",
        title="Median tuning-strength change",
        cmap="vlag",
        center=0.0,
    )
    heatmap_from_table(
        natural_summary,
        "population_response_correlation",
        fig_dir / "primary_06_natural_population_correlation.png",
        title="Natural-image population correlation to baseline",
        cmap="rocket_r",
    )
    heatmap_from_table(
        natural_summary,
        "rdm_similarity_to_baseline",
        fig_dir / "primary_07_natural_rdm_similarity.png",
        title="Natural-image RDM similarity to baseline",
        cmap="rocket_r",
    )
    heatmap_from_table(
        natural_summary,
        "median_relative_response_change",
        fig_dir / "primary_08_natural_response_change.png",
        title="Natural-image response-change magnitude",
        cmap="mako",
    )


def make_secondary_figures(
    population: pd.DataFrame,
    grating_result: dict[str, object],
    natural_result: dict[str, object],
    paths: dict[str, Path],
) -> None:
    """Generate a compact set of secondary validation and bridge figures."""
    fig_dir = paths["figures"]
    far = grating_result["far_state"]
    changes = far["neuron_changes"]

    plt.figure(figsize=(5.6, 3.8))
    sns.histplot(changes["abs_op_shift_deg"], bins=24, color="#4C78A8")
    plt.xlabel("|dOP| (deg)")
    plt.ylabel("Neurons")
    plt.title(f"OP shifts at gaze {far['gaze'][0]:.1f}, {far['gaze'][1]:.1f} deg")
    savefig(fig_dir / "secondary_hist_abs_op_shift_far_offset.png")

    plt.figure(figsize=(5.6, 3.8))
    sns.histplot(changes["circular_variance_change"], bins=24, color="#F58518")
    plt.xlabel("Circular-variance change")
    plt.ylabel("Neurons")
    plt.title("Circular-variance change at far offset")
    savefig(fig_dir / "secondary_hist_circular_variance_change_far_offset.png")

    mean_rf_size = 0.5 * (population["sigma_x_deg"].to_numpy() + population["sigma_y_deg"].to_numpy())
    plt.figure(figsize=(5.7, 4.1))
    plt.scatter(mean_rf_size, changes["abs_op_shift_deg"], s=13, alpha=0.65)
    plt.xlabel("Mean RF sigma (deg)")
    plt.ylabel("|dOP| (deg)")
    plt.title("OP shift versus RF size")
    savefig(fig_dir / "secondary_scatter_op_shift_vs_rf_size.png")

    plt.figure(figsize=(5.7, 4.1))
    plt.scatter(population["f0_cpd"], changes["abs_op_shift_deg"], s=13, alpha=0.65)
    plt.xlabel("Preferred SF (cpd)")
    plt.ylabel("|dOP| (deg)")
    plt.title("OP shift versus preferred SF")
    savefig(fig_dir / "secondary_scatter_op_shift_vs_sf.png")

    plt.figure(figsize=(5.7, 4.1))
    plt.scatter(mean_rf_size, changes["tuning_strength_change"], s=13, alpha=0.65)
    plt.axhline(0, color="0.3", lw=1)
    plt.xlabel("Mean RF sigma (deg)")
    plt.ylabel("Tuning-strength change")
    plt.title("Tuning-strength change versus RF size")
    savefig(fig_dir / "secondary_scatter_tuning_change_vs_rf_size.png")

    baseline_nat = natural_result["baseline_response"]
    far_nat = natural_result["far_state"]["response"]
    natural_change = np.sqrt(np.mean((far_nat - baseline_nat) ** 2, axis=0)) / (
        np.sqrt(np.mean(baseline_nat**2, axis=0)) + 1e-9
    )
    plt.figure(figsize=(5.7, 4.1))
    plt.scatter(changes["abs_op_shift_deg"], natural_change, s=13, alpha=0.65)
    plt.xlabel("|dOP| at far offset (deg)")
    plt.ylabel("Natural-image relative change")
    plt.title(f"Bridge sensitivity r={safe_corrcoef(changes['abs_op_shift_deg'], natural_change):.2f}")
    savefig(fig_dir / "secondary_scatter_natural_vs_grating_sensitivity.png")

    baseline_tuning = grating_result["baseline_tuning"]["tuning_curve"]
    shifted_tuning = far["tuning"]["tuning_curve"]
    selected = np.argsort(changes["abs_op_shift_deg"])
    neuron_ids = [int(selected[0]), int(selected[len(selected) // 2]), int(selected[-1])]
    plot_tuning_examples(
        baseline_tuning,
        shifted_tuning,
        grating_result["stimuli"]["orientations_deg"],
        neuron_ids,
        fig_dir / "secondary_example_tuning_curves.png",
    )
    plot_response_matrix(
        baseline_nat,
        fig_dir / "secondary_natural_response_matrix_baseline.png",
        title="Natural-image response matrix, baseline",
    )
    plot_response_matrix(
        far_nat,
        fig_dir / "secondary_natural_response_matrix_far_offset.png",
        title="Natural-image response matrix, far gaze offset",
    )


def compare_simple_energy(
    population: pd.DataFrame,
    images: np.ndarray,
    config: dict,
    rng: np.random.Generator,
    paths: dict[str, Path],
) -> pd.DataFrame:
    """Compare simple-cell and energy-model stability at the largest gaze offset."""
    shifts = make_gaze_grid(config)
    far = shifts.assign(radius=shifts["gaze_az_deg"].abs() + shifts["gaze_el_deg"].abs()).sort_values("radius").tail(1)
    rows = []
    for model in ["simple", "energy"]:
        table = _condition_summary(
            f"{model}_model",
            population,
            images,
            config,
            np.random.default_rng(rng.integers(0, 2**32 - 1)),
            mapping_config=config.get("mapping", perfect_mapping_config()),
            noise_config=config.get("noise", {"model": "none", "repeats": 1}),
            response_model=model,
            shifts=far[["gaze_az_deg", "gaze_el_deg"]],
        )
        rows.append(table)
    out = pd.concat(rows, ignore_index=True)
    plot_summary_bars(out, paths["figures"] / "primary_11_simple_vs_energy_summary.png")
    return out


def run_main(config: dict, *, run_decomp: bool = True) -> dict[str, object]:
    """Run the main simulation and save tables, arrays, figures, and text."""
    set_plot_style()
    paths = ensure_output_dirs(config)
    rng = rng_from_config(config)
    population = sample_population(config, rng)
    population.to_csv(paths["tables"] / "sampled_population.csv", index=False)

    plot_parameter_distributions(population, paths["figures"] / "validation_parameter_distributions.png")
    plot_retinotopy(population, paths["figures"] / "validation_retinotopy_maps.png")
    plot_rf_examples(population, config, paths["figures"] / "validation_receptive_field_examples.png")
    plot_mapping_validation(config, paths["figures"] / "validation_mapping_error_field.png")

    images = load_natural_images(config, np.random.default_rng(int(config["stimuli"]["natural_images"]["procedural_seed"])))
    grating_result = run_grating_shift_grid(
        population,
        config,
        rng,
        mapping_config=config.get("mapping", perfect_mapping_config()),
        noise_config=config.get("noise", {"model": "none", "repeats": 1}),
        response_model=config["rf"].get("response_model", "simple"),
    )
    natural_result = run_natural_shift_grid(
        population,
        images,
        config,
        rng,
        mapping_config=config.get("mapping", perfect_mapping_config()),
        noise_config=config.get("noise", {"model": "none", "repeats": 1}),
        response_model=config["rf"].get("response_model", "simple"),
    )

    grating_summary = grating_result["summary"]
    natural_summary = natural_result["summary"]
    grating_summary.to_csv(paths["tables"] / "grating_shift_summary.csv", index=False)
    natural_summary.to_csv(paths["tables"] / "natural_image_shift_summary.csv", index=False)
    np.savez_compressed(
        paths["arrays"] / "main_responses_and_images.npz",
        natural_images=images,
        grating_baseline_response=grating_result["baseline_response"],
        natural_baseline_response=natural_result["baseline_response"],
        natural_far_response=natural_result["far_state"]["response"],
    )

    make_primary_figures(grating_summary, natural_summary, paths)
    make_secondary_figures(population, grating_result, natural_result, paths)

    decomposition = pd.DataFrame()
    if run_decomp:
        decomposition = run_decomposition(population, images, config, rng)
        decomposition.to_csv(paths["tables"] / "decomposition_summary.csv", index=False)
        plot_summary_bars(
            decomposition.sort_values("summary_radius").groupby("condition").tail(1),
            paths["figures"] / "primary_10_noiseless_noisy_mapping_comparison.png",
        )

    model_comparison = compare_simple_energy(population, images, config, rng, paths)
    model_comparison.to_csv(paths["tables"] / "simple_vs_energy_summary.csv", index=False)

    far_grating = grating_summary.sort_values(
        by=["gaze_az_deg", "gaze_el_deg"], key=lambda col: col.abs()
    ).tail(1)
    bridge_summary = pd.merge(grating_summary, natural_summary, on=["gaze_az_deg", "gaze_el_deg"])
    bridge_summary["gaze_radius"] = bridge_summary["gaze_az_deg"].abs() + bridge_summary["gaze_el_deg"].abs()
    bridge_summary.to_csv(paths["tables"] / "grating_natural_bridge_summary.csv", index=False)
    plot_summary_bars(
        bridge_summary.sort_values("gaze_radius").tail(1).assign(condition="main_model"),
        paths["figures"] / "primary_09_gratings_vs_natural_images_summary.png",
    )
    del far_grating

    write_text(paths["tables"] / "methods_summary.md", methods_summary_text(config))
    write_text(paths["tables"] / "interpretation_summary.md", interpretation_text(grating_summary, natural_summary, decomposition))

    return {
        "population": population,
        "grating": grating_result,
        "natural": natural_result,
        "decomposition": decomposition,
        "model_comparison": model_comparison,
        "paths": paths,
    }


def run_sweep(config: dict, config_paths: list[str | Path], *, debug: bool = True) -> pd.DataFrame:
    """Run compact sensitivity sweeps across config overrides."""
    from .utils import load_config

    del config
    rows = []
    for idx, path in enumerate(config_paths):
        cfg = load_config(path, debug=debug)
        rng = rng_from_config(cfg, offset=idx * 100)
        population = sample_population(cfg, rng)
        images = load_natural_images(cfg, np.random.default_rng(int(cfg["stimuli"]["natural_images"]["procedural_seed"]) + idx))
        shifts = make_gaze_grid(cfg)
        far = shifts.assign(radius=shifts["gaze_az_deg"].abs() + shifts["gaze_el_deg"].abs()).sort_values("radius").tail(1)
        summary = _condition_summary(
            Path(path).stem,
            population,
            images,
            cfg,
            rng,
            mapping_config=cfg.get("mapping", perfect_mapping_config()),
            noise_config=cfg.get("noise", {"model": "none", "repeats": 1}),
            response_model=cfg["rf"].get("response_model", "simple"),
            shifts=far[["gaze_az_deg", "gaze_el_deg"]],
        )
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)
