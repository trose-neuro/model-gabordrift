"""Automatic methods and interpretation text generation."""

from __future__ import annotations

import pandas as pd


def methods_summary_text(config: dict) -> str:
    """Create a concise methods summary for the current config."""
    env_name = config["project"]["environment_name"]
    n = config["cortical_field"]["n_neurons"]
    size = config["cortical_field"]["size_um"]
    gaze = config["gaze"]
    return f"""# Methods Summary

This first-pass null model simulates {n} L2/3-like units sampled uniformly in a {size:.0f} x {size:.0f} um V1 imaging field. Cortical positions are mapped to visual RF centers by a configurable local linear retinotopic transform with Gaussian cell-to-cell scatter. Each unit has a single Gabor-like receptive field with sampled PO, spatial-frequency preference, RF envelope widths, phase, gain, baseline, and static nonlinearity.

Full-field gratings are evaluated with a Gabor-inspired analytic response model that separates orientation/SF matching from gaze-dependent phase. Natural-image responses are evaluated by applying sampled Gabor kernels to preprocessed grayscale images. If no image folder is provided, the pipeline uses procedural naturalistic images so the project remains runnable.

Gaze is modeled as a global translation of the retinal image relative to the RFs. The configured grid spans azimuth {gaze['azimuth_deg']['min']} to {gaze['azimuth_deg']['max']} deg and elevation {gaze['elevation_deg']['min']} to {gaze['elevation_deg']['max']} deg in {gaze['azimuth_deg']['step']} deg azimuth steps and {gaze['elevation_deg']['step']} deg elevation steps. Exact spherical projection and parameterized approximate mappings can be compared. Output noise can be disabled or simulated as Gaussian, Poisson-like, multiplicative gain noise, or combined Gaussian plus multiplicative noise.

All scripts, notebooks, and tests are intended to run after activating the dedicated conda environment `{env_name}`.
"""


def interpretation_text(
    grating_table: pd.DataFrame,
    natural_table: pd.DataFrame,
    decomposition_table: pd.DataFrame | None = None,
    phase_sanity_table: pd.DataFrame | None = None,
) -> str:
    """Generate a compact interpretation from saved summary metrics."""
    far_g = grating_table.assign(radius=grating_table["gaze_az_deg"].abs() + grating_table["gaze_el_deg"].abs())
    far_g = far_g.sort_values("radius").iloc[-1]
    far_n = natural_table.assign(radius=natural_table["gaze_az_deg"].abs() + natural_table["gaze_el_deg"].abs())
    far_n = far_n.sort_values("radius").iloc[-1]

    lines = [
        "# Interpretation Summary",
        "",
        f"At the largest simulated offset in this run ({far_g['gaze_az_deg']:.2f}, {far_g['gaze_el_deg']:.2f}) deg, the median absolute inferred ΔPO was {far_g['median_abs_delta_po_deg']:.2f} deg and the 90th percentile was {far_g['p90_abs_delta_po_deg']:.2f} deg.",
        f"The median tuning-strength change was {far_g['median_tuning_strength_change']:.3f}; this distinguishes PO rotation from tuning flattening or sharpening.",
        f"For natural images at the same edge of the grid, population response correlation to baseline was {far_n['population_response_correlation']:.3f}, RDM similarity was {far_n['rdm_similarity_to_baseline']:.3f}, and the median relative per-neuron response change was {far_n['median_relative_response_change']:.3f}.",
        "",
        "Why can a gaze shift affect PO for full-field gratings at all? Translating an infinite grating does not rotate it; it advances stimulus phase at each RF by Δφ = 2π f (Δa cosθ + Δe sinθ). Therefore a perfectly phase-invariant cell, or a simple-cell response averaged over dense stimulus phases, should show little to no deterministic ΔPO under perfect mapping and no noise. Apparent ΔPO appears when finite phase sampling and rectification let that phase advance modulate different orientations unevenly, or when mapping error and noise perturb the estimated tuning curve.",
        "",
        "Under this model, gaze matters little when RFs are broad, phase is averaged or an energy model is used, mapping is accurate, and SNR is high. Gaze matters more when RFs are narrow, high-SF units are common, phase-sensitive simple-cell responses are estimated from limited phases, or natural images contain local structure that translates across RF subfields.",
        "",
        "Mapping errors matter when approximate projection, wrong scale, origin offsets, rotation mismatch, or nonlinear distortion produce systematic phase and position errors comparable to the RF subfield scale. Low SNR dominates when repeated-trial averaging and bootstrap intervals show large PO uncertainty even at zero or small gaze offsets.",
    ]
    if decomposition_table is not None and not decomposition_table.empty:
        edge = decomposition_table.sort_values("summary_radius").groupby("condition").tail(1)
        lines.extend(["", "Decomposition at the largest analyzed offsets:"])
        for row in edge.itertuples(index=False):
            lines.append(
                f"- {row.condition}: median |ΔPO| {row.median_abs_delta_po_deg:.2f} deg, natural-image correlation {row.population_response_correlation:.3f}, RDM similarity {row.rdm_similarity_to_baseline:.3f}."
            )
    if phase_sanity_table is not None and not phase_sanity_table.empty:
        edge = phase_sanity_table.sort_values("drift_deg").groupby("condition").tail(1)
        lines.extend(["", "Full-field grating phase sanity check at the largest tested drift:"])
        for row in edge.sort_values("condition").itertuples(index=False):
            lines.append(f"- {row.condition}: median |ΔPO| {row.median_abs_delta_po_deg:.2f} deg.")
    lines.append("")
    lines.append(
        "Natural images are considered more gaze-sensitive than gratings here when their population correlation or RDM similarity drops substantially while grating ΔPO remains small. That pattern is expected because translating a structured image can change RF drive without necessarily changing the PO that best fits grating responses."
    )
    return "\n".join(lines) + "\n"
