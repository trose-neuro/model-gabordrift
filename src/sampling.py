"""Sampling of cortical locations, retinotopy, and RF parameters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .retinotopy import apply_retinotopy


def sample_cortical_positions(n_neurons: int, size_um: float, rng: np.random.Generator) -> pd.DataFrame:
    """Sample neuron positions uniformly in a square cortical field."""
    xy = rng.uniform(0.0, float(size_um), size=(int(n_neurons), 2))
    return pd.DataFrame({"neuron_id": np.arange(int(n_neurons)), "x_um": xy[:, 0], "y_um": xy[:, 1]})


def _sample_lognormal_clipped(
    rng: np.random.Generator,
    median: float,
    sigma_log: float,
    low: float,
    high: float,
    size: int,
) -> np.ndarray:
    values = rng.lognormal(mean=np.log(float(median)), sigma=float(sigma_log), size=size)
    return np.clip(values, float(low), float(high))


def sample_rf_parameters(n_neurons: int, config: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Sample simple-cell-like Gabor RF parameters for each neuron."""
    n = int(n_neurons)
    rf_cfg = config["rf"]
    sf_cfg = rf_cfg["spatial_frequency_cpd"]
    sigma_cfg = rf_cfg["sigma_deg"]
    gain_cfg = rf_cfg["gain"]

    theta0 = rng.uniform(0.0, 180.0, size=n)
    f0 = _sample_lognormal_clipped(
        rng,
        sf_cfg["median"],
        sf_cfg["sigma_log"],
        sf_cfg["min"],
        sf_cfg["max"],
        n,
    )
    sigma_x = _sample_lognormal_clipped(
        rng,
        sigma_cfg["median_x"],
        sigma_cfg["sigma_log"],
        sigma_cfg["min"],
        sigma_cfg["max"],
        n,
    )
    sigma_y = _sample_lognormal_clipped(
        rng,
        sigma_cfg["median_y"],
        sigma_cfg["sigma_log"],
        sigma_cfg["min"],
        sigma_cfg["max"],
        n,
    )
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n)
    gain = rng.lognormal(mean=np.log(float(gain_cfg["median"])), sigma=float(gain_cfg["sigma_log"]), size=n)
    baseline = np.full(n, float(rf_cfg.get("baseline", 0.0)))
    gamma = np.full(n, float(rf_cfg.get("gamma", 1.0)))

    return pd.DataFrame(
        {
            "theta0_deg": theta0,
            "f0_cpd": f0,
            "sigma_x_deg": sigma_x,
            "sigma_y_deg": sigma_y,
            "phase_rad": phase,
            "gain": gain,
            "baseline": baseline,
            "gamma": gamma,
        }
    )


def sample_population(config: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Sample a complete neuron table with cortical, retinotopic, and RF fields."""
    n = int(config["cortical_field"]["n_neurons"])
    size_um = float(config["cortical_field"]["size_um"])
    positions = sample_cortical_positions(n, size_um, rng)
    az, el, jitter = apply_retinotopy(positions[["x_um", "y_um"]].to_numpy(), config, rng, jitter=True)
    rf_params = sample_rf_parameters(n, config, rng)
    population = pd.concat([positions, rf_params], axis=1)
    population["azimuth_deg"] = az
    population["elevation_deg"] = el
    population["retino_jitter_az_deg"] = jitter[:, 0]
    population["retino_jitter_el_deg"] = jitter[:, 1]
    return population
