import numpy as np
import pandas as pd

from src.longitudinal import apply_circuit_drift, make_session_schedule, sample_circuit_drift_state
from src.natural_images import load_natural_images
from src.sampling import sample_population
from src.simulation import run_longitudinal_decomposition
from src.utils import load_config


def _population() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "theta0_deg": [10.0, 170.0],
            "f0_cpd": [0.04, 0.08],
            "gain": [1.0, 1.2],
            "baseline": [0.05, 0.10],
            "phase_rad": [0.1, 6.1],
            "azimuth_deg": [1.0, -1.0],
            "elevation_deg": [2.0, -2.0],
        }
    )


def test_make_session_schedule_matches_cumulative_gaze_rate():
    config = {
        "longitudinal": {
            "n_sessions": 4,
            "session_interval_weeks": 1.0,
            "cumulative_gaze_drift_deg_per_week": 1.0,
            "drift_direction_deg": 0.0,
        }
    }
    schedule = make_session_schedule(config)
    np.testing.assert_allclose(schedule["weeks_from_reference"], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(schedule["measured_gaze_deg"], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(schedule["gaze_az_deg"], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(schedule["gaze_el_deg"], 0.0)


def test_zero_circuit_drift_leaves_population_unchanged():
    population = _population()
    state = sample_circuit_drift_state(population, np.random.default_rng(0))
    drifted = apply_circuit_drift(population, 0.0, state)
    pd.testing.assert_frame_equal(drifted, population)


def test_circuit_drift_wraps_angles_and_preserves_positive_parameters():
    population = _population()
    state = {
        "theta": np.array([3.0, -3.0]),
        "log_sf": np.array([1.0, -1.0]),
        "log_gain": np.array([1.0, -1.0]),
        "baseline": np.array([-10.0, 10.0]),
        "phase": np.array([5.0, -5.0]),
        "azimuth": np.array([1.0, -1.0]),
        "elevation": np.array([1.0, -1.0]),
    }
    drifted = apply_circuit_drift(population, 2.0, state)
    assert np.all((drifted["theta0_deg"] >= 0.0) & (drifted["theta0_deg"] < 180.0))
    assert np.all(drifted["f0_cpd"] > 0.0)
    assert np.all(drifted["gain"] > 0.0)
    assert np.all(drifted["baseline"] >= 0.0)
    assert np.all((drifted["phase_rad"] >= 0.0) & (drifted["phase_rad"] < 2.0 * np.pi))
    assert not np.allclose(drifted["azimuth_deg"], population["azimuth_deg"])
    assert not np.allclose(drifted["elevation_deg"], population["elevation_deg"])


def test_longitudinal_decomposition_separates_hypotheses_but_shares_baseline():
    cfg = load_config("configs/default.yaml", debug=True)
    cfg["cortical_field"]["n_neurons"] = 20
    cfg["stimuli"]["natural_images"]["n_images"] = 4
    cfg["stimuli"]["natural_images"]["image_size_px"] = 24
    cfg["stimuli"]["gratings"]["phases_deg"] = [0.0, 90.0]
    cfg["stimuli"]["moving_gratings"]["time_samples"] = 4
    cfg["noise"] = {"model": "none", "repeats": 1, "clip_nonnegative": True}
    cfg["longitudinal"]["n_sessions"] = 3
    rng = np.random.default_rng(2)
    population = sample_population(cfg, rng)
    images = load_natural_images(cfg, np.random.default_rng(3))

    table = run_longitudinal_decomposition(population, images, cfg, rng)

    assert set(table["condition"]) == {
        "gaze_only",
        "gaze_plus_mapping_error",
        "circuit_drift_only",
        "gaze_plus_circuit_drift",
    }
    assert table["session_index"].nunique() == 3

    baseline = table[table["session_index"] == 0].sort_values(by="condition")
    np.testing.assert_allclose(baseline["measured_gaze_deg"], 0.0)
    np.testing.assert_allclose(baseline["applied_gaze_az_deg"], 0.0)
    np.testing.assert_allclose(baseline["applied_gaze_el_deg"], 0.0)
    np.testing.assert_allclose(baseline["circuit_drift_amplitude_deg"], 0.0)
    np.testing.assert_allclose(baseline["static_grating_median_abs_delta_po_deg"], baseline.iloc[0]["static_grating_median_abs_delta_po_deg"])
    np.testing.assert_allclose(baseline["moving_grating_median_abs_delta_po_deg"], baseline.iloc[0]["moving_grating_median_abs_delta_po_deg"])
    np.testing.assert_allclose(baseline["natural_population_response_correlation"], baseline.iloc[0]["natural_population_response_correlation"])

    circuit_only = table[table["condition"] == "circuit_drift_only"]
    np.testing.assert_allclose(circuit_only["applied_gaze_az_deg"], 0.0)
    np.testing.assert_allclose(circuit_only["applied_gaze_el_deg"], 0.0)