from copy import deepcopy

import numpy as np
import pandas as pd

from src.responses import moving_grating_response_mean
from src.simulation import perfect_mapping_config
from src.stimuli import make_moving_grating_stimuli
from src.tuning import circular_difference_deg, estimate_orientation_tuning
from src.utils import load_config


def _single_neuron_population() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "theta0_deg": [45.0],
            "f0_cpd": [0.06],
            "gain": [1.0],
            "baseline": [0.0],
            "gamma": [1.25],
            "phase_rad": [0.3],
            "azimuth_deg": [1.0],
            "elevation_deg": [-1.0],
        }
    )


def test_energy_moving_grating_response_is_gaze_invariant():
    cfg = load_config("configs/default.yaml", debug=True)
    stimuli = make_moving_grating_stimuli(cfg)
    pop = _single_neuron_population()
    rng = np.random.default_rng(0)
    baseline, _ = moving_grating_response_mean(
        pop,
        stimuli,
        cfg,
        rng,
        gaze_shift=(0.0, 0.0),
        mapping_config=perfect_mapping_config(),
        noise_config={"model": "none", "repeats": 1},
        response_model="energy",
    )
    shifted, _ = moving_grating_response_mean(
        pop,
        stimuli,
        cfg,
        rng,
        gaze_shift=(10.0, 10.0),
        mapping_config=perfect_mapping_config(),
        noise_config={"model": "none", "repeats": 1},
        response_model="energy",
    )
    np.testing.assert_allclose(baseline, shifted)


def test_dense_temporal_average_suppresses_simple_cell_moving_grating_delta_po():
    cfg = deepcopy(load_config("configs/default.yaml", debug=True))
    tf_hz = float(cfg["stimuli"]["moving_gratings"]["temporal_frequency_hz"])
    cfg["stimuli"]["moving_gratings"]["time_samples"] = 72
    cfg["stimuli"]["moving_gratings"]["duration_s"] = 1.0 / tf_hz
    cfg["stimuli"]["moving_gratings"]["phases_deg"] = [0.0]
    stimuli = make_moving_grating_stimuli(cfg)
    pop = _single_neuron_population()
    rng = np.random.default_rng(1)
    baseline, _ = moving_grating_response_mean(
        pop,
        stimuli,
        cfg,
        rng,
        gaze_shift=(0.0, 0.0),
        mapping_config=perfect_mapping_config(),
        noise_config={"model": "none", "repeats": 1},
        response_model="simple",
    )
    shifted, _ = moving_grating_response_mean(
        pop,
        stimuli,
        cfg,
        rng,
        gaze_shift=(10.0, 10.0),
        mapping_config=perfect_mapping_config(),
        noise_config={"model": "none", "repeats": 1},
        response_model="simple",
    )
    baseline_tuning = estimate_orientation_tuning(baseline, stimuli["orientations_deg"], stimuli["spatial_frequencies_cpd"])
    shifted_tuning = estimate_orientation_tuning(shifted, stimuli["orientations_deg"], stimuli["spatial_frequencies_cpd"])
    delta_po = circular_difference_deg(shifted_tuning["po_deg"][0], baseline_tuning["po_deg"][0])
    assert abs(delta_po) < 0.05
