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


def plot_longitudinal_metric_panels(table: pd.DataFrame, path: str | Path) -> None:
    """Plot weekly drift metrics across modalities and candidate mechanisms."""
    metrics = [
        ("static_grating_median_abs_delta_po_deg", "Static grating median |ΔPO| (deg)"),
        ("moving_grating_median_abs_delta_po_deg", "Moving grating median |ΔPO| (deg)"),
        ("natural_population_response_correlation", "Natural-image population correlation"),
        ("natural_rdm_similarity_to_baseline", "Natural-image RDM similarity"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), sharex=True)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        sns.lineplot(
            data=table,
            x="weeks_from_reference",
            y=metric,
            hue="condition",
            style="condition",
            marker="o",
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Weeks from reference")
        ax.set_ylabel(title)
        if metric.endswith("correlation") or metric.endswith("similarity"):
            ax.set_ylim(-0.05, 1.05)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncols=2, frameon=False)
    fig.suptitle("Weekly drift decomposition across stimulus classes", y=1.02)
    savefig(path)


def plot_longitudinal_hypothesis_summary(table: pd.DataFrame, path: str | Path) -> None:
    """Plot measured gaze against similarity loss for each hypothesis."""
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), sharex=False)
    specs = [
        ("static_grating_median_abs_delta_po_deg", "Static grating |ΔPO|"),
        ("moving_grating_median_abs_delta_po_deg", "Moving grating |ΔPO|"),
        ("natural_population_response_correlation", "Natural-image correlation"),
    ]
    for ax, (metric, title) in zip(axes, specs):
        sns.scatterplot(
            data=table,
            x="measured_gaze_deg",
            y=metric,
            hue="condition",
            style="condition",
            s=80,
            ax=ax,
        )
        sns.lineplot(
            data=table,
            x="measured_gaze_deg",
            y=metric,
            hue="condition",
            estimator=None,
            legend=False,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Measured cumulative gaze drift (deg)")
        ax.set_ylabel(title)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncols=4, frameon=False)
    fig.suptitle("Does similarity track measured gaze under each mechanism?", y=1.02)
    savefig(path)


def plot_grating_gaze_rf_scheme(path: str | Path) -> None:
    """Draw a schematic of how gaze-shifted gratings can create apparent ΔPO.

    The scheme intentionally separates the physical stimulus operation from the
    estimator consequence: a gaze shift translates the grating over the RF,
    changing phase, and sparse phase-sensitive responses can reshape the tuning
    curve enough to move the inferred PO.
    """
    axis = np.linspace(-5.0, 5.0, 240)
    xx, yy = np.meshgrid(axis, axis)
    rf_theta = 30.0
    rf_sf = 0.05
    rf = gabor_kernel(
        xx,
        yy,
        theta_deg=rf_theta,
        sf_cpd=rf_sf,
        sigma_x_deg=1.7,
        sigma_y_deg=0.8,
        phase_rad=0.0,
        normalize=False,
    )
    envelope = np.exp(-0.5 * (((xx * np.cos(np.deg2rad(rf_theta)) + yy * np.sin(np.deg2rad(rf_theta))) / 1.7) ** 2))
    grating_theta = 30.0
    gaze = np.array([5.0, 5.0])
    theta_rad = np.deg2rad(grating_theta)
    normal = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    phase_advance = 2.0 * np.pi * rf_sf * float(np.dot(gaze, normal))
    baseline_grating = np.cos(2.0 * np.pi * rf_sf * (xx * normal[0] + yy * normal[1]) - 0.52)
    shifted_grating = np.cos(2.0 * np.pi * rf_sf * (xx * normal[0] + yy * normal[1]) - 0.52 + phase_advance)

    orientations = np.arange(0.0, 180.0, 10.0)
    diff = ((orientations - rf_theta + 90.0) % 180.0) - 90.0
    amplitude = np.exp(-0.5 * (diff / 28.0) ** 2)
    phase0 = -0.52
    tuning_base = np.maximum(amplitude * np.cos(phase0), 0.0) ** 1.25
    phase_by_orientation = 2.0 * np.pi * rf_sf * (
        gaze[0] * np.cos(np.deg2rad(orientations)) + gaze[1] * np.sin(np.deg2rad(orientations))
    )
    tuning_shift = np.maximum(amplitude * np.cos(phase0 + phase_by_orientation), 0.0) ** 1.25

    def estimate_po(curve: np.ndarray) -> float:
        weights = np.maximum(curve - np.min(curve), 0.0)
        vector = np.sum(weights * np.exp(2j * np.deg2rad(orientations)))
        return float((0.5 * np.rad2deg(np.angle(vector))) % 180.0)

    po_base = estimate_po(tuning_base)
    po_shift = estimate_po(tuning_shift)
    delta_po = ((po_shift - po_base + 90.0) % 180.0) - 90.0

    fig = plt.figure(figsize=(12.5, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], width_ratios=[1, 1, 1.05])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2]),
    ]

    for ax, grating, title in [
        (axes[0], baseline_grating, "Baseline: grating over RF"),
        (axes[1], shifted_grating, "After gaze shift: same orientation, new phase"),
    ]:
        ax.imshow(grating, extent=[axis.min(), axis.max(), axis.min(), axis.max()], origin="lower", cmap="gray", alpha=0.75)
        ax.contour(xx, yy, rf, levels=[-0.55, -0.25, 0.25, 0.55], colors=["#2B6CB0", "#63B3ED", "#F6AD55", "#C05621"], linewidths=1.0)
        ax.contour(xx, yy, envelope, levels=[0.25, 0.55, 0.85], colors="black", linewidths=0.7, alpha=0.55)
        ax.set_title(title)
        ax.set_xlabel("Azimuth (deg)")
        ax.set_ylabel("Elevation (deg)")
        ax.set_aspect("equal")
    axes[1].annotate("", xy=(3.4, 3.4), xytext=(1.0, 1.0), arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#D62728"})
    axes[1].text(3.5, 3.3, "gaze/FOV\ntranslation", color="#D62728", ha="left", va="center")

    orient_fine = np.linspace(0.0, 170.0, 18)
    for sf in [0.02, 0.05, 0.10, 0.20]:
        phase_cycles = sf * (gaze[0] * np.cos(np.deg2rad(orient_fine)) + gaze[1] * np.sin(np.deg2rad(orient_fine)))
        axes[2].plot(orient_fine, phase_cycles, marker="o", ms=3, label=f"{sf:.2f} cpd")
    axes[2].axhline(0, color="0.35", lw=1)
    axes[2].axvline(grating_theta, color="#D62728", lw=1.3, ls="--")
    axes[2].set_title("Phase advance depends on orientation")
    axes[2].set_xlabel("Grating orientation θ (deg)")
    axes[2].set_ylabel("Δφ / 2π (cycles)")
    axes[2].legend(frameon=False, fontsize=8)

    axes[3].plot(orientations, tuning_base, "o-", lw=2, label=f"baseline PO={po_base:.1f}°")
    axes[3].plot(orientations, tuning_shift, "s-", lw=2, label=f"shifted PO={po_shift:.1f}°")
    axes[3].axvline(po_base, color="#1F77B4", lw=1.5, ls="--")
    axes[3].axvline(po_shift, color="#FF7F0E", lw=1.5, ls="--")
    axes[3].set_title(f"Sparse simple-cell phase sampling can move inferred PO: ΔPO={delta_po:.1f}°")
    axes[3].set_xlabel("Stimulus orientation (deg)")
    axes[3].set_ylabel("Rectified response")
    axes[3].legend(frameon=False)

    axes[4].axis("off")
    scheme_text = (
        "Mechanism\n\n"
        "1. Gaze does not rotate a full-field grating.\n\n"
        "2. It translates the grating over each RF.\n\n"
        "3. Translation changes phase by:\n"
        "   Δφ = 2πf(Δa cosθ + Δe sinθ)\n\n"
        "4. With sparse phases + rectification,\n"
        "   different orientations are modulated\n"
        "   unequally.\n\n"
        "5. The fitted tuning curve can shift,\n"
        "   producing apparent ΔPO."
    )
    axes[4].text(0.0, 0.98, scheme_text, va="top", ha="left", fontsize=10)

    fig.suptitle("How gaze-shifted full-field gratings can create apparent ΔPO", y=1.02, fontsize=15)
    savefig(path)


def plot_moving_grating_gaze_rf_scheme(path: str | Path) -> None:
    """Draw the moving-grating analogue of the gaze/RF ΔPO mechanism.

    Motion sweeps grating phase over time. A gaze shift still adds a fixed
    spatial phase offset, but dense temporal averaging over a drift cycle should
    largely cancel that offset in the simple-cell model; the energy model is
    phase-invariant by construction.
    """
    axis = np.linspace(-5.0, 5.0, 240)
    xx, yy = np.meshgrid(axis, axis)
    rf_theta = 35.0
    sf_cpd = 0.06
    tf_hz = 2.0
    theta_rad = np.deg2rad(rf_theta)
    normal = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    gaze = np.array([5.0, 5.0])
    phase0 = -0.45
    phase_gaze = 2.0 * np.pi * sf_cpd * float(np.dot(gaze, normal))
    frame_t0 = np.cos(2.0 * np.pi * sf_cpd * (xx * normal[0] + yy * normal[1]) + phase0)
    shifted_t0 = np.cos(2.0 * np.pi * sf_cpd * (xx * normal[0] + yy * normal[1]) + phase0 + phase_gaze)
    t_later = 0.125
    shifted_t1 = np.cos(
        2.0 * np.pi * sf_cpd * (xx * normal[0] + yy * normal[1])
        + phase0
        + phase_gaze
        - 2.0 * np.pi * tf_hz * t_later
    )
    rf = gabor_kernel(
        xx,
        yy,
        theta_deg=rf_theta,
        sf_cpd=sf_cpd,
        sigma_x_deg=1.6,
        sigma_y_deg=0.8,
        phase_rad=0.0,
        normalize=False,
    )

    orientations = np.arange(0.0, 180.0, 10.0)
    diff = ((orientations - rf_theta + 90.0) % 180.0) - 90.0
    amplitude = np.exp(-0.5 * (diff / 28.0) ** 2)
    phase_by_orientation = 2.0 * np.pi * sf_cpd * (
        gaze[0] * np.cos(np.deg2rad(orientations)) + gaze[1] * np.sin(np.deg2rad(orientations))
    )

    def temporal_tuning(n_samples: int, shifted: bool) -> np.ndarray:
        t = np.linspace(0.0, 1.0 / tf_hz, n_samples, endpoint=False)
        temporal_phase = -2.0 * np.pi * tf_hz * t
        gaze_phase = phase_by_orientation if shifted else np.zeros_like(phase_by_orientation)
        drive = amplitude[:, None] * np.cos(phase0 + gaze_phase[:, None] + temporal_phase[None, :])
        return np.mean(np.maximum(drive, 0.0) ** 1.25, axis=1)

    def estimate_po(curve: np.ndarray) -> float:
        weights = np.maximum(curve - np.min(curve), 0.0)
        vector = np.sum(weights * np.exp(2j * np.deg2rad(orientations)))
        return float((0.5 * np.rad2deg(np.angle(vector))) % 180.0)

    sparse_base = temporal_tuning(1, shifted=False)
    sparse_shift = temporal_tuning(1, shifted=True)
    dense_base = temporal_tuning(24, shifted=False)
    dense_shift = temporal_tuning(24, shifted=True)
    po_sparse_base = estimate_po(sparse_base)
    po_sparse_shift = estimate_po(sparse_shift)
    po_dense_base = estimate_po(dense_base)
    po_dense_shift = estimate_po(dense_shift)
    delta_sparse = ((po_sparse_shift - po_sparse_base + 90.0) % 180.0) - 90.0
    delta_dense = ((po_dense_shift - po_dense_base + 90.0) % 180.0) - 90.0

    fig = plt.figure(figsize=(13.0, 8.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.95, 1.05])
    frame_axes = [fig.add_subplot(gs[0, idx]) for idx in range(3)]
    ax_phase = fig.add_subplot(gs[1, 0])
    ax_tuning = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[1, 2])

    for ax, frame, title in [
        (frame_axes[0], frame_t0, "t = 0, no gaze shift"),
        (frame_axes[1], shifted_t0, "t = 0, gaze-shifted phase"),
        (frame_axes[2], shifted_t1, "later: motion sweeps phase"),
    ]:
        ax.imshow(frame, extent=[axis.min(), axis.max(), axis.min(), axis.max()], origin="lower", cmap="gray", alpha=0.75)
        ax.contour(xx, yy, rf, levels=[-0.55, -0.25, 0.25, 0.55], colors=["#2B6CB0", "#63B3ED", "#F6AD55", "#C05621"], linewidths=1.0)
        ax.set_title(title)
        ax.set_xlabel("Azimuth (deg)")
        ax.set_ylabel("Elevation (deg)")
        ax.set_aspect("equal")
    frame_axes[1].annotate("", xy=(3.4, 3.4), xytext=(1.0, 1.0), arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#D62728"})
    frame_axes[1].text(3.5, 3.3, "gaze/FOV\ntranslation", color="#D62728", ha="left", va="center")

    time = np.linspace(0.0, 1.0 / tf_hz, 200)
    phase_cycles = (-tf_hz * time) % 1.0
    gaze_cycles = (phase_gaze / (2.0 * np.pi)) % 1.0
    ax_phase.plot(time, phase_cycles, lw=2, label="motion phase")
    ax_phase.axhline(gaze_cycles, color="#D62728", ls="--", lw=1.5, label="fixed gaze phase offset")
    sample_time = np.linspace(0.0, 1.0 / tf_hz, 4, endpoint=False)
    ax_phase.scatter(sample_time, (-tf_hz * sample_time) % 1.0, color="black", s=26, zorder=3, label="4 samples")
    ax_phase.set_title("Moving gratings convert time into phase samples")
    ax_phase.set_xlabel("Time (s)")
    ax_phase.set_ylabel("Phase (cycles)")
    ax_phase.legend(frameon=False, fontsize=8)

    ax_tuning.plot(orientations, sparse_base, "o-", lw=1.8, label=f"1 time, baseline PO={po_sparse_base:.1f}°")
    ax_tuning.plot(orientations, sparse_shift, "s-", lw=1.8, label=f"1 time, shifted PO={po_sparse_shift:.1f}°")
    ax_tuning.plot(orientations, dense_base, "o--", lw=1.6, color="#4C78A8", alpha=0.8, label=f"24 times, baseline PO={po_dense_base:.1f}°")
    ax_tuning.plot(orientations, dense_shift, "s--", lw=1.6, color="#F58518", alpha=0.8, label=f"24 times, shifted PO={po_dense_shift:.1f}°")
    ax_tuning.set_title(
        f"Temporal averaging suppresses apparent |ΔPO| ({abs(delta_sparse):.1f}° to {abs(delta_dense):.1f}°)"
    )
    ax_tuning.set_xlabel("Stimulus orientation (deg)")
    ax_tuning.set_ylabel("Mean rectified response")
    ax_tuning.legend(frameon=False, fontsize=7.5)

    ax_text.axis("off")
    scheme_text = (
        "Moving-grating sanity check\n\n"
        "1. Gaze adds a fixed spatial phase:\n"
        "   Δφgaze = 2πf(Δa cosθ + Δe sinθ)\n\n"
        "2. Motion adds a temporal phase:\n"
        "   φtime(t) = -2πTFt\n\n"
        "3. Sparse time samples can still make\n"
        "   phase-sensitive simple-cell responses\n"
        "   look like PO changes.\n\n"
        "4. Dense samples over a drift cycle, or an\n"
        "   energy model, should remove deterministic\n"
        "   gaze-induced ΔPO for full-field gratings."
    )
    ax_text.text(0.0, 0.98, scheme_text, va="top", ha="left", fontsize=10)

    fig.suptitle("How gaze and moving gratings interact over RFs", y=1.02, fontsize=15)
    savefig(path)
