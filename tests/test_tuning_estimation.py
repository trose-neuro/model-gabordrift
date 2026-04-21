import numpy as np

from src.tuning import estimate_orientation_tuning


def test_orientation_estimation_recovers_synthetic_peak():
    orientations = np.arange(0, 180, 10)
    sfs = np.array([0.02, 0.08])
    diff = ((orientations - 40 + 90) % 180) - 90
    tuning = np.exp(-0.5 * (diff / 20) ** 2)
    responses = np.zeros((1, len(orientations), len(sfs)))
    responses[0, :, 1] = tuning
    estimate = estimate_orientation_tuning(responses, orientations, sfs)
    assert abs(((estimate["preferred_orientation_deg"][0] - 40 + 90) % 180) - 90) < 2
    assert estimate["best_spatial_frequency_cpd"][0] == 0.08
    assert estimate["orientation_selectivity"][0] > 0.5
