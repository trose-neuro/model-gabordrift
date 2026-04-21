"""Plotting helpers for validation, heat maps, and summary figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .coordinate_mapping import mapping_error_field
from .rf_models import gabor_kernel


def set_plot_style() -> None:
    """Apply a quiet publication-oriented plotting style."""
    sns.set_theme(context="notebook", style="whitegrid", font_scale=1.0)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def savefig(path: str | Path) -> None:
    """Save and close the current Matplotlib figure."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(target, bbox_inches="tight")
    plt.close()


def heatmap_from_table(
    table: pd.DataFrame,
    value: str,
    path: str | Path,
    *,
    title: str,
    cmap: str = "viridis",
    center: float | None = None,
) -> None:
    """Plot a 2D gaze-shift heat map from a long-form table."""
    pivot = table.pivot(index="gaze_el_deg", columns="gaze_az_deg", values=value).sort_index(ascending=False)
    plt.figure(figsize=(6.2, 5.1))
    sns.heatmap(pivot, cmap=cmap, center=center, cbar_kws={"label": value.replace("_", " ")})
    plt.xlabel("Azimuth shift (deg)")
    plt.ylabel("Elevation shift (deg)")
    plt.title(title)
    savefig(path)


def plot_parameter_distributions(population: pd.DataFrame, path: str | Path) -> None:
    """Plot sampled RF and retinotopy parameter distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    fields = [
        ("theta0_deg", "PO (deg)"),
        ("f0_cpd", "Preferred SF (cpd)"),
        ("sigma_x_deg", "RF sigma x (deg)"),
        ("sigma_y_deg", "RF sigma y (deg)"),
        ("retino_jitter_az_deg", "Azimuth jitter (deg)"),
        ("retino_jitter_el_deg", "Elevation jitter (deg)"),
    ]
    for ax, (field, label) in zip(axes.ravel(), fields):
        sns.histplot(population[field], bins=24, ax=ax, color="#4C78A8")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
    savefig(path)


def plot_retinotopy(population: pd.DataFrame, path: str | Path) -> None:
    """Plot cortical field colored by RF azimuth/elevation and visual scatter."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    sc0 = axes[0].scatter(population["x_um"], population["y_um"], c=population["azimuth_deg"], s=10, cmap="coolwarm")
    axes[0].set_title("Cortex colored by azimuth")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    plt.colorbar(sc0, ax=axes[0], label="deg")
    sc1 = axes[1].scatter(population["x_um"], population["y_um"], c=population["elevation_deg"], s=10, cmap="viridis")
    axes[1].set_title("Cortex colored by elevation")
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("y (um)")
    plt.colorbar(sc1, ax=axes[1], label="deg")
    axes[2].scatter(population["azimuth_deg"], population["elevation_deg"], s=10, color="#F58518", alpha=0.75)
    axes[2].set_title("RF centers in visual space")
    axes[2].set_xlabel("Azimuth (deg)")
    axes[2].set_ylabel("Elevation (deg)")
    axes[2].axis("equal")
    savefig(path)


def plot_rf_examples(population: pd.DataFrame, config: dict, path: str | Path, *, n_examples: int = 6) -> None:
    """Render example Gabor RFs from sampled neurons."""
    extent = float(config["stimuli"]["natural_images"]["extent_deg"])
    size = 96
    axis = np.linspace(-0.5 * extent, 0.5 * extent, size)
    xx, yy = np.meshgrid(axis, axis)
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.8))
    idx = np.linspace(0, len(population) - 1, n_examples, dtype=int)
    for ax, row_idx in zip(axes.ravel(), idx):
        row = population.iloc[row_idx]
        kernel = gabor_kernel(
            xx,
            yy,
            center=(row["azimuth_deg"], row["elevation_deg"]),
            theta_deg=row["theta0_deg"],
            sf_cpd=row["f0_cpd"],
            sigma_x_deg=row["sigma_x_deg"],
            sigma_y_deg=row["sigma_y_deg"],
            phase_rad=row["phase_rad"],
            normalize=False,
        )
        ax.imshow(kernel, cmap="coolwarm", extent=[axis.min(), axis.max(), axis.min(), axis.max()], origin="lower")
        ax.set_title(f"Neuron {int(row['neuron_id'])}")
        ax.set_xlabel("Azimuth (deg)")
        ax.set_ylabel("Elevation (deg)")
    savefig(path)


def plot_mapping_validation(config: dict, path: str | Path) -> None:
    """Visualize distortion from an approximate mapping relative to exact projection."""
    grid = np.linspace(-15.0, 15.0, 61)
    aa, ee = np.meshgrid(grid, grid)
    _, _, mag = mapping_error_field(aa, ee, config["mapping"])
    plt.figure(figsize=(5.7, 4.8))
    plt.imshow(mag, extent=[grid.min(), grid.max(), grid.min(), grid.max()], origin="lower", cmap="magma")
    plt.colorbar(label="Mapping displacement (deg)")
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Elevation (deg)")
    plt.title("Configured mapping error field")
    savefig(path)


def plot_tuning_examples(
    baseline_tuning: np.ndarray,
    shifted_tuning: np.ndarray,
    orientations_deg: np.ndarray,
    neuron_ids: list[int],
    path: str | Path,
) -> None:
    """Plot example single-neuron orientation tuning curves before/after gaze."""
    n = len(neuron_ids)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.2), squeeze=False)
    for ax, neuron_id in zip(axes.ravel(), neuron_ids):
        ax.plot(orientations_deg, baseline_tuning[neuron_id], marker="o", label="0 deg")
        ax.plot(orientations_deg, shifted_tuning[neuron_id], marker="s", label="shift")
        ax.set_title(f"Neuron {neuron_id}")
        ax.set_xlabel("Orientation (deg)")
        ax.set_ylabel("Response")
        ax.legend(frameon=False)
    savefig(path)


def plot_summary_bars(summary: pd.DataFrame, path: str | Path) -> None:
    """Plot condition-level summary bars for decomposition or model comparison."""
    metrics = [
        "median_abs_delta_po_deg",
        "population_response_correlation",
        "rdm_similarity_to_baseline",
        "median_tuning_strength_change",
    ]
    available = [metric for metric in metrics if metric in summary.columns]
    long = summary.melt(id_vars=["condition"], value_vars=available, var_name="metric", value_name="value")
    plt.figure(figsize=(10.5, 4.8))
    sns.barplot(data=long, x="metric", y="value", hue="condition")
    plt.xticks(rotation=20, ha="right")
    plt.xlabel("")
    plt.ylabel("Summary value")
    plt.title("Condition comparison")
    plt.legend(frameon=False, fontsize=8)
    savefig(path)


def plot_response_matrix(matrix: np.ndarray, path: str | Path, *, title: str = "Natural-image responses") -> None:
    """Plot an image-by-neuron response matrix."""
    plt.figure(figsize=(7, 4.5))
    sns.heatmap(matrix, cmap="rocket", cbar_kws={"label": "Response"})
    plt.xlabel("Neuron")
    plt.ylabel("Image")
    plt.title(title)
    savefig(path)
