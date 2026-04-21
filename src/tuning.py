"""Orientation tuning estimation and circular statistics."""

from __future__ import annotations

import numpy as np


def circular_difference_deg(a_deg: np.ndarray | float, b_deg: np.ndarray | float, period: float = 180.0) -> np.ndarray:
    """Return signed circular difference ``a - b`` in degrees."""
    return (np.asarray(a_deg) - np.asarray(b_deg) + 0.5 * period) % period - 0.5 * period


def circular_mean_deg(values_deg: np.ndarray, weights: np.ndarray | None = None, period: float = 180.0) -> float:
    """Return a weighted circular mean for periodic degree values."""
    values = np.asarray(values_deg, dtype=float)
    weights = np.ones_like(values) if weights is None else np.asarray(weights, dtype=float)
    angles = 2.0 * np.pi * values / float(period)
    vector = np.sum(weights * np.exp(1j * angles))
    if np.abs(vector) < 1e-12:
        return float("nan")
    return float((np.angle(vector) * period / (2.0 * np.pi)) % period)


def estimate_orientation_tuning(
    responses: np.ndarray,
    orientations_deg: np.ndarray,
    spatial_frequencies_cpd: np.ndarray,
) -> dict[str, np.ndarray]:
    """Estimate OP, best SF, circular variance, and vector strength.

    Parameters
    ----------
    responses:
        Response array with shape ``(n_neurons, n_orientations, n_sfs)``.
    orientations_deg:
        Orientation grid in degrees.
    spatial_frequencies_cpd:
        Spatial-frequency grid in cycles per degree.
    """
    resp = np.asarray(responses, dtype=float)
    if resp.ndim != 3:
        raise ValueError("responses must have shape (n_neurons, n_orientations, n_sfs)")
    n = resp.shape[0]
    orientations = np.asarray(orientations_deg, dtype=float)
    sfs = np.asarray(spatial_frequencies_cpd, dtype=float)

    best_sf_idx = np.argmax(np.max(resp, axis=1), axis=1)
    tuning = resp[np.arange(n), :, best_sf_idx]
    weights = np.maximum(tuning - np.min(tuning, axis=1, keepdims=True), 0.0)
    denom = np.sum(weights, axis=1)
    angles = np.exp(2j * np.deg2rad(orientations))[None, :]
    vector = np.sum(weights * angles, axis=1)
    vector_strength = np.divide(np.abs(vector), denom, out=np.zeros_like(denom), where=denom > 1e-12)
    pref = (0.5 * np.rad2deg(np.angle(vector))) % 180.0
    pref = np.where(denom > 1e-12, pref, np.nan)
    circ_var = 1.0 - vector_strength
    best_ori_idx = np.nanargmax(tuning, axis=1)

    return {
        "preferred_orientation_deg": pref,
        "best_spatial_frequency_cpd": sfs[best_sf_idx],
        "best_orientation_grid_deg": orientations[best_ori_idx],
        "orientation_selectivity": vector_strength,
        "circular_variance": circ_var,
        "tuning_curve": tuning,
        "best_sf_index": best_sf_idx,
    }


def bootstrap_orientation_ci(
    trial_responses: np.ndarray,
    orientations_deg: np.ndarray,
    spatial_frequencies_cpd: np.ndarray,
    *,
    n_bootstrap: int = 200,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Bootstrap preferred-orientation and circular-variance uncertainty.

    ``trial_responses`` must have shape ``(repeats, n_neurons, n_orientations,
    n_sfs)``. The returned CI widths are intentionally compact diagnostics
    rather than a full posterior representation.
    """
    if rng is None:
        rng = np.random.default_rng()
    trials = np.asarray(trial_responses, dtype=float)
    if trials.ndim != 4:
        raise ValueError("trial_responses must have shape (repeats, neurons, orientations, sfs)")
    repeats, n = trials.shape[:2]
    pref_samples = np.empty((int(n_bootstrap), n), dtype=float)
    cv_samples = np.empty((int(n_bootstrap), n), dtype=float)
    for idx in range(int(n_bootstrap)):
        sample_idx = rng.integers(0, repeats, size=repeats)
        estimate = estimate_orientation_tuning(np.mean(trials[sample_idx], axis=0), orientations_deg, spatial_frequencies_cpd)
        pref_samples[idx] = estimate["preferred_orientation_deg"]
        cv_samples[idx] = estimate["circular_variance"]

    center = circular_mean_deg(pref_samples, period=180.0)
    del center  # kept only to emphasize circular treatment in the implementation.
    pref_low = np.nanpercentile(pref_samples, 2.5, axis=0)
    pref_high = np.nanpercentile(pref_samples, 97.5, axis=0)
    pref_width = np.abs(circular_difference_deg(pref_high, pref_low, period=180.0))
    return {
        "preferred_orientation_ci_width_deg": pref_width,
        "circular_variance_ci_width": np.nanpercentile(cv_samples, 97.5, axis=0)
        - np.nanpercentile(cv_samples, 2.5, axis=0),
    }
