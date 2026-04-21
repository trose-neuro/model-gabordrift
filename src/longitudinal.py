"""Longitudinal session-drift helpers for disentangling gaze and circuit drift."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_session_schedule(config: dict) -> pd.DataFrame:
    """Return a cumulative session schedule aligned to measured gaze drift."""
    long_cfg = config.get("longitudinal", {})
    n_sessions = int(long_cfg.get("n_sessions", 6))
    interval_weeks = float(long_cfg.get("session_interval_weeks", 1.0))
    drift_rate = float(long_cfg.get("cumulative_gaze_drift_deg_per_week", 1.0))
    direction_deg = float(long_cfg.get("drift_direction_deg", 45.0))
    direction_rad = np.deg2rad(direction_deg)
    weeks = interval_weeks * np.arange(n_sessions, dtype=float)
    measured_gaze_deg = drift_rate * weeks
    return pd.DataFrame(
        {
            "session_index": np.arange(n_sessions, dtype=int),
            "weeks_from_reference": weeks,
            "measured_gaze_deg": measured_gaze_deg,
            "gaze_az_deg": measured_gaze_deg * np.cos(direction_rad),
            "gaze_el_deg": measured_gaze_deg * np.sin(direction_rad),
        }
    )


def sample_circuit_drift_state(
    population: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Sample neuron-specific latent directions for circuit drift."""
    n_neurons = len(population)
    return {
        "theta": rng.normal(size=n_neurons),
        "log_sf": rng.normal(size=n_neurons),
        "log_gain": rng.normal(size=n_neurons),
        "baseline": rng.normal(size=n_neurons),
        "phase": rng.normal(size=n_neurons),
        "azimuth": rng.normal(size=n_neurons),
        "elevation": rng.normal(size=n_neurons),
    }


def apply_circuit_drift(
    population: pd.DataFrame,
    drift_amplitude_deg: float,
    drift_state: dict[str, np.ndarray],
    drift_config: dict | None = None,
) -> pd.DataFrame:
    """Return a population with RF parameters drifted by a session-scale latent.

    The drift amplitude is expressed in the same degree-like units as measured
    gaze drift so longitudinal gaze and circuit hypotheses can be compared on
    the same session axis.
    """
    drift_config = drift_config or {}
    amplitude = float(drift_amplitude_deg)
    if amplitude == 0.0:
        return population.copy()

    drifted = population.copy()
    theta_scale = float(drift_config.get("theta_sd_deg_per_deg", 0.75))
    log_sf_scale = float(drift_config.get("log_sf_sd_per_deg", 0.03))
    log_gain_scale = float(drift_config.get("log_gain_sd_per_deg", 0.04))
    baseline_scale = float(drift_config.get("baseline_sd_per_deg", 0.005))
    phase_scale = float(drift_config.get("phase_sd_rad_per_deg", 0.10))
    azimuth_scale = float(drift_config.get("azimuth_sd_deg_per_deg", 0.20))
    elevation_scale = float(drift_config.get("elevation_sd_deg_per_deg", 0.20))

    if "azimuth_deg" in drifted.columns:
        drifted["azimuth_deg"] = drifted["azimuth_deg"].to_numpy(float) + amplitude * azimuth_scale * drift_state["azimuth"]
    if "elevation_deg" in drifted.columns:
        drifted["elevation_deg"] = drifted["elevation_deg"].to_numpy(float) + amplitude * elevation_scale * drift_state["elevation"]

    drifted["theta0_deg"] = np.mod(
        drifted["theta0_deg"].to_numpy(float) + amplitude * theta_scale * drift_state["theta"],
        180.0,
    )
    drifted["f0_cpd"] = np.clip(
        drifted["f0_cpd"].to_numpy(float) * np.exp(amplitude * log_sf_scale * drift_state["log_sf"]),
        1e-4,
        None,
    )
    drifted["gain"] = np.clip(
        drifted["gain"].to_numpy(float) * np.exp(amplitude * log_gain_scale * drift_state["log_gain"]),
        1e-4,
        None,
    )
    drifted["baseline"] = np.clip(
        drifted["baseline"].to_numpy(float) + amplitude * baseline_scale * drift_state["baseline"],
        0.0,
        None,
    )
    drifted["phase_rad"] = np.mod(
        drifted["phase_rad"].to_numpy(float) + amplitude * phase_scale * drift_state["phase"],
        2.0 * np.pi,
    )
    return drifted