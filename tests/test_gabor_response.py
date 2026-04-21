import numpy as np
import pandas as pd

from src.rf_models import analytic_grating_responses, gabor_kernel
from src.stimuli import make_grating_stimuli
from src.utils import load_config


def test_gabor_kernel_has_expected_peak_near_center():
    axis = np.linspace(-4, 4, 81)
    xx, yy = np.meshgrid(axis, axis)
    kernel = gabor_kernel(xx, yy, theta_deg=0.0, sf_cpd=0.05, sigma_x_deg=1.0, sigma_y_deg=1.0, phase_rad=0.0, normalize=False)
    center = kernel[40, 40]
    assert center == np.max(kernel)


def test_analytic_grating_response_prefers_matched_orientation_and_sf():
    cfg = load_config("configs/default.yaml", debug=True)
    cfg["stimuli"]["gratings"]["orientations_deg"] = {"start": 0, "stop": 180, "step": 30}
    cfg["stimuli"]["gratings"]["spatial_frequencies_cpd"] = {"min": 0.05, "max": 0.10, "count": 2, "spacing": "linear"}
    cfg["stimuli"]["gratings"]["phases_deg"] = [0.0]
    pop = pd.DataFrame(
        {
            "theta0_deg": [60.0],
            "f0_cpd": [0.05],
            "gain": [1.0],
            "baseline": [0.0],
            "gamma": [1.0],
            "phase_rad": [0.0],
            "azimuth_deg": [0.0],
            "elevation_deg": [0.0],
        }
    )
    stimuli = make_grating_stimuli(cfg)
    response = analytic_grating_responses(pop, stimuli, cfg, response_model="energy").mean(axis=-1)
    best_ori = stimuli["orientations_deg"][np.argmax(response[0].max(axis=1))]
    assert best_ori == 60
