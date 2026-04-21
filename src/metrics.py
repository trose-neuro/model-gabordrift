"""Summary metrics for grating tuning and natural-image representations."""

from __future__ import annotations

import numpy as np

from .tuning import circular_difference_deg


def grating_shift_metrics(baseline: dict[str, np.ndarray], shifted: dict[str, np.ndarray], threshold_deg: float) -> dict[str, float]:
    """Summarize PO and tuning-strength changes for one gaze shift."""
    delta_po = circular_difference_deg(
        shifted["po_deg"],
        baseline["po_deg"],
        period=180.0,
    )
    abs_delta_po = np.abs(delta_po)
    cv_change = shifted["circular_variance"] - baseline["circular_variance"]
    strength_change = shifted["orientation_selectivity"] - baseline["orientation_selectivity"]
    return {
        "median_abs_delta_po_deg": float(np.nanmedian(abs_delta_po)),
        "p90_abs_delta_po_deg": float(np.nanpercentile(abs_delta_po, 90)),
        "fraction_abs_delta_po_gt_threshold": float(np.nanmean(abs_delta_po > float(threshold_deg))),
        "median_signed_delta_po_deg": float(np.nanmedian(delta_po)),
        "median_circular_variance_change": float(np.nanmedian(cv_change)),
        "median_tuning_strength_change": float(np.nanmedian(strength_change)),
        "median_abs_tuning_strength_change": float(np.nanmedian(np.abs(strength_change))),
    }


def neuronwise_grating_changes(baseline: dict[str, np.ndarray], shifted: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return per-neuron grating sensitivity values."""
    delta_po = circular_difference_deg(
        shifted["po_deg"],
        baseline["po_deg"],
        period=180.0,
    )
    return {
        "signed_delta_po_deg": delta_po,
        "abs_delta_po_deg": np.abs(delta_po),
        "circular_variance_change": shifted["circular_variance"] - baseline["circular_variance"],
        "tuning_strength_change": shifted["orientation_selectivity"] - baseline["orientation_selectivity"],
    }


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    """Return a finite Pearson correlation for flattened arrays."""
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    mask = np.isfinite(aa) & np.isfinite(bb)
    if np.sum(mask) < 3:
        return float("nan")
    aa = aa[mask] - np.mean(aa[mask])
    bb = bb[mask] - np.mean(bb[mask])
    denom = np.sqrt(np.sum(aa**2) * np.sum(bb**2))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(aa * bb) / denom)


def representational_dissimilarity(responses: np.ndarray) -> np.ndarray:
    """Compute an image-by-image RDM as one minus response correlation."""
    matrix = np.asarray(responses, dtype=float)
    n_images = matrix.shape[0]
    rdm = np.zeros((n_images, n_images), dtype=float)
    for i in range(n_images):
        for j in range(i + 1, n_images):
            value = 1.0 - safe_corrcoef(matrix[i], matrix[j])
            rdm[i, j] = value
            rdm[j, i] = value
    return rdm


def rdm_similarity(baseline_rdm: np.ndarray, shifted_rdm: np.ndarray) -> float:
    """Correlate upper triangles of two RDMs."""
    mask = np.triu(np.ones_like(baseline_rdm, dtype=bool), k=1)
    return safe_corrcoef(baseline_rdm[mask], shifted_rdm[mask])


def natural_image_metrics(
    baseline_responses: np.ndarray,
    shifted_responses: np.ndarray,
    *,
    large_change_threshold: float = 0.5,
) -> dict[str, float]:
    """Summarize natural-image response stability for one gaze shift."""
    diff = shifted_responses - baseline_responses
    per_neuron_change = np.sqrt(np.mean(diff**2, axis=0))
    baseline_scale = np.sqrt(np.mean(baseline_responses**2, axis=0)) + 1e-9
    relative_change = per_neuron_change / baseline_scale
    baseline_rdm = representational_dissimilarity(baseline_responses)
    shifted_rdm = representational_dissimilarity(shifted_responses)
    return {
        "population_response_correlation": safe_corrcoef(baseline_responses, shifted_responses),
        "median_per_neuron_response_change": float(np.nanmedian(per_neuron_change)),
        "median_relative_response_change": float(np.nanmedian(relative_change)),
        "fraction_large_response_change": float(np.nanmean(relative_change > large_change_threshold)),
        "rdm_similarity_to_baseline": rdm_similarity(baseline_rdm, shifted_rdm),
        "image_discriminability": float(np.nanmean(baseline_rdm[np.triu_indices_from(baseline_rdm, k=1)])),
        "shifted_image_discriminability": float(np.nanmean(shifted_rdm[np.triu_indices_from(shifted_rdm, k=1)])),
    }
