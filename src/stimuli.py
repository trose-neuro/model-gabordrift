"""Stimulus grids for gratings and gaze shifts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import degree_grid


def make_grating_stimuli(config: dict) -> dict[str, np.ndarray]:
    """Create orientation, spatial-frequency, and phase grids for gratings."""
    grating_cfg = config["stimuli"]["gratings"]
    ori_cfg = grating_cfg["orientations_deg"]
    orientations = np.arange(float(ori_cfg["start"]), float(ori_cfg["stop"]), float(ori_cfg["step"]))
    sf_cfg = grating_cfg["spatial_frequencies_cpd"]
    if sf_cfg.get("spacing", "log") == "log":
        sfs = np.geomspace(float(sf_cfg["min"]), float(sf_cfg["max"]), int(sf_cfg["count"]))
    else:
        sfs = np.linspace(float(sf_cfg["min"]), float(sf_cfg["max"]), int(sf_cfg["count"]))
    phases = np.deg2rad(np.asarray(grating_cfg.get("phases_deg", [0.0]), dtype=float))
    return {"orientations_deg": orientations, "spatial_frequencies_cpd": sfs, "phases_rad": phases}


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
