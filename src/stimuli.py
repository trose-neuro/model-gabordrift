"""Stimulus grids for static gratings, moving gratings, and gaze shifts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import degree_grid


def _orientation_grid(stimulus_config: dict) -> np.ndarray:
    """Create an orientation grid from a stimulus config block."""
    ori_cfg = stimulus_config["orientations_deg"]
    orientations = np.arange(float(ori_cfg["start"]), float(ori_cfg["stop"]), float(ori_cfg["step"]))
    return orientations


def _spatial_frequency_grid(stimulus_config: dict) -> np.ndarray:
    """Create a spatial-frequency grid from a stimulus config block."""
    sf_cfg = stimulus_config["spatial_frequencies_cpd"]
    if sf_cfg.get("spacing", "log") == "log":
        return np.geomspace(float(sf_cfg["min"]), float(sf_cfg["max"]), int(sf_cfg["count"]))
    return np.linspace(float(sf_cfg["min"]), float(sf_cfg["max"]), int(sf_cfg["count"]))


def make_grating_stimuli(config: dict) -> dict[str, np.ndarray]:
    """Create orientation, spatial-frequency, and phase grids for static gratings."""
    grating_cfg = config["stimuli"]["gratings"]
    orientations = _orientation_grid(grating_cfg)
    sfs = _spatial_frequency_grid(grating_cfg)
    phases = np.deg2rad(np.asarray(grating_cfg.get("phases_deg", [0.0]), dtype=float))
    return {"orientations_deg": orientations, "spatial_frequencies_cpd": sfs, "phases_rad": phases}


def make_moving_grating_stimuli(config: dict) -> dict[str, np.ndarray | float]:
    """Create a moving-grating stimulus grid.

    Moving gratings use the same orientation and spatial-frequency axes as
    static gratings, but replace the phase axis with sampled temporal phases:

    ``I(x, y, t) = cos(2π f (x cosθ + y sinθ) - 2π TF t + ψ)``.

    Averaging across a dense set of ``time_s`` samples approximates averaging
    over a temporal drift cycle. This makes the moving-grating sanity checks
    directly comparable to the static grating phase checks.
    """
    fallback = config["stimuli"]["gratings"]
    moving_cfg = config["stimuli"].get("moving_gratings", fallback)
    orientations = _orientation_grid(moving_cfg)
    sfs = _spatial_frequency_grid(moving_cfg)

    temporal_frequency_hz = float(moving_cfg.get("temporal_frequency_hz", 2.0))
    duration_s = float(moving_cfg.get("duration_s", 1.0))
    time_samples = int(moving_cfg.get("time_samples", 16))
    if time_samples <= 0:
        raise ValueError("moving_gratings.time_samples must be positive")
    time_s = np.linspace(0.0, duration_s, time_samples, endpoint=False)
    initial_phases = np.deg2rad(np.asarray(moving_cfg.get("phases_deg", [0.0]), dtype=float))
    temporal_phase = -2.0 * np.pi * temporal_frequency_hz * time_s
    phases = (initial_phases[:, None] + temporal_phase[None, :]).reshape(-1)

    return {
        "orientations_deg": orientations,
        "spatial_frequencies_cpd": sfs,
        "phases_rad": phases,
        "time_s": time_s,
        "initial_phases_rad": initial_phases,
        "temporal_frequency_hz": temporal_frequency_hz,
        "duration_s": duration_s,
    }


def make_gaze_grid(config: dict) -> pd.DataFrame:
    """Create a Cartesian grid of gaze shifts in azimuth/elevation degrees."""
    gaze_cfg = config["gaze"]
    az = degree_grid(gaze_cfg["azimuth_deg"])
    el = degree_grid(gaze_cfg["elevation_deg"])
    if bool(gaze_cfg.get("symmetric", False)):
        az = np.unique(np.concatenate([-az[::-1], az]))
        el = np.unique(np.concatenate([-el[::-1], el]))
    aa, ee = np.meshgrid(az, el, indexing="xy")
    return pd.DataFrame({"gaze_az_deg": aa.ravel(), "gaze_el_deg": ee.ravel()})


def grating_image(
    orientation_deg: float,
    sf_cpd: float,
    phase_rad: float,
    *,
    extent_deg: float = 40.0,
    size_px: int = 128,
) -> np.ndarray:
    """Render a static sinusoidal grating image for plotting and sanity checks."""
    axis = np.linspace(-0.5 * extent_deg, 0.5 * extent_deg, int(size_px))
    xx, yy = np.meshgrid(axis, axis)
    theta = np.deg2rad(orientation_deg)
    phase = 2.0 * np.pi * sf_cpd * (xx * np.cos(theta) + yy * np.sin(theta)) + phase_rad
    return np.cos(phase)


def moving_grating_frame(
    orientation_deg: float,
    sf_cpd: float,
    temporal_frequency_hz: float,
    time_s: float,
    *,
    phase_rad: float = 0.0,
    extent_deg: float = 40.0,
    size_px: int = 128,
) -> np.ndarray:
    """Render one frame of a moving sinusoidal grating."""
    effective_phase = float(phase_rad) - 2.0 * np.pi * float(temporal_frequency_hz) * float(time_s)
    return grating_image(
        orientation_deg,
        sf_cpd,
        effective_phase,
        extent_deg=extent_deg,
        size_px=size_px,
    )
