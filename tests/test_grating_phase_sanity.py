from copy import deepcopy

import numpy as np
import pandas as pd

from src.responses import grating_response_mean
from src.simulation import perfect_mapping_config
from src.stimuli import make_grating_stimuli
from src.tuning import estimate_orientation_tuning
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


def test_energy_grating_po_is_gaze_invariant():
    cfg = load_config("configs/default.yaml", debug=True)
    stimuli = make_grating_stimuli(cfg)
    pop = _single_neuron_population()
    rng = np.random.default_rng(0)
    baseline, _ = grating_response_mean(
        pop,
        stimuli,
        cfg,
        rng,
        gaze_shift=(0.0, 0.0),
        mapping_config=perfect_mapping_config(),
        noise_config={"model": "none", "repeats": 1},
        response_model="energy",
    )
    shifted, _ = grating_response_mean(
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


def test_dense_phase_average_suppresses_simple_cell_gaze_delta_po():
    cfg = load_config("configs/default.yaml", debug=True)
    cfg = deepcopy(cfg)
    cfg["stimuli"]["gratings"]["phases_deg"] = list(np.linspace(0.0, 360.0, 72, endpoint=False))
    stimuli = make_grating_stimuli(cfg)
    pop = _single_neuron_population()
    rng = np.random.default_rng(1)
    baseline, _ = grating_response_mean(
        pop,
        stimuli,
        cfg,
        rng,
        gaze_shift=(0.0, 0.0),
        mapping_config=perfect_mapping_config(),
        noise_config={"model": "none", "repeats": 1},
        response_model="simple",
    )
    shifted, _ = grating_response_mean(
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
    assert abs(baseline_tuning["po_deg"][0] - shifted_tuning["po_deg"][0]) < 0.05
