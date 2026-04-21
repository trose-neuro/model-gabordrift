"""Gabor receptive fields and analytic response approximations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .coordinate_mapping import map_visual_degrees


def rotate_coordinates(x: np.ndarray, y: np.ndarray, theta_deg: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate coordinates into a RF preferred-orientation frame."""
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    xp = x * c + y * s
    yp = -x * s + y * c
    return xp, yp


def gabor_kernel(
    x_deg: np.ndarray,
    y_deg: np.ndarray,
    *,
    center: tuple[float, float] = (0.0, 0.0),
    theta_deg: float = 0.0,
    sf_cpd: float = 0.05,
    sigma_x_deg: float = 2.0,
    sigma_y_deg: float = 1.0,
    phase_rad: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """Evaluate a 2D Gabor RF on a visual-coordinate grid."""
    x = np.asarray(x_deg, dtype=float) - float(center[0])
    y = np.asarray(y_deg, dtype=float) - float(center[1])
    xp, yp = rotate_coordinates(x, y, theta_deg)
    envelope = np.exp(-0.5 * ((xp / sigma_x_deg) ** 2 + (yp / sigma_y_deg) ** 2))
    kernel = envelope * np.cos(2.0 * np.pi * sf_cpd * xp + phase_rad)
    if normalize:
        kernel = kernel - np.mean(kernel)
        norm = np.sqrt(np.sum(kernel**2))
        if norm > 0:
            kernel = kernel / norm
    return kernel


def gabor_kernel_bank(
    population: pd.DataFrame,
    *,
    image_size_px: int,
    extent_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build cosine and sine Gabor kernel banks for natural-image responses."""
    axis = np.linspace(-0.5 * extent_deg, 0.5 * extent_deg, int(image_size_px))
    xx, yy = np.meshgrid(axis, axis)
    cos_kernels = []
    sin_kernels = []
    for row in population.itertuples(index=False):
        center = (float(row.azimuth_deg), float(row.elevation_deg))
        cos_kernels.append(
            gabor_kernel(
                xx,
                yy,
                center=center,
                theta_deg=float(row.theta0_deg),
                sf_cpd=float(row.f0_cpd),
                sigma_x_deg=float(row.sigma_x_deg),
                sigma_y_deg=float(row.sigma_y_deg),
                phase_rad=float(row.phase_rad),
                normalize=True,
            ).ravel()
        )
        sin_kernels.append(
            gabor_kernel(
                xx,
                yy,
                center=center,
                theta_deg=float(row.theta0_deg),
                sf_cpd=float(row.f0_cpd),
                sigma_x_deg=float(row.sigma_x_deg),
                sigma_y_deg=float(row.sigma_y_deg),
                phase_rad=float(row.phase_rad) + 0.5 * np.pi,
                normalize=True,
            ).ravel()
        )
    return np.asarray(cos_kernels), np.asarray(sin_kernels)


def _orientation_tuning(stim_orientation: np.ndarray, pref_orientation: np.ndarray, bandwidth_deg: float) -> np.ndarray:
    diff = ((stim_orientation[None, :] - pref_orientation[:, None] + 90.0) % 180.0) - 90.0
    return np.exp(-0.5 * (diff / float(bandwidth_deg)) ** 2)


def _sf_tuning(stim_sf: np.ndarray, pref_sf: np.ndarray, bandwidth_octaves: float) -> np.ndarray:
    ratio = np.log2(stim_sf[None, :] / np.maximum(pref_sf[:, None], 1e-12))
    return np.exp(-0.5 * (ratio / float(bandwidth_octaves)) ** 2)


def analytic_grating_responses(
    population: pd.DataFrame,
    stimuli: dict[str, np.ndarray],
    config: dict,
    *,
    gaze_shift: tuple[float, float] = (0.0, 0.0),
    mapping_config: dict | None = None,
    response_model: str | None = None,
) -> np.ndarray:
    """Compute noiseless grating responses with a Gabor-inspired approximation.

    The amplitude envelope captures orientation and spatial-frequency matching,
    while the phase term captures how a global gaze shift changes a simple-cell
    RF's phase relative to a full-field grating. The energy model removes this
    phase dependence by using quadrature-pair energy.
    """
    response_model = response_model or config["rf"].get("response_model", "simple")
    mapping_config = mapping_config or config.get("mapping", {"mode": "exact", "error": {}})
    orientations = np.asarray(stimuli["orientations_deg"], dtype=float)
    sfs = np.asarray(stimuli["spatial_frequencies_cpd"], dtype=float)
    phases = np.asarray(stimuli["phases_rad"], dtype=float)

    theta0 = population["theta0_deg"].to_numpy(float)
    f0 = population["f0_cpd"].to_numpy(float)
    gain = population["gain"].to_numpy(float)
    baseline = population["baseline"].to_numpy(float)
    gamma = population["gamma"].to_numpy(float)
    pref_phase = population["phase_rad"].to_numpy(float)
    az = population["azimuth_deg"].to_numpy(float)
    el = population["elevation_deg"].to_numpy(float)

    mapped_x, mapped_y = map_visual_degrees(az, el, gaze_shift, mapping_config)
    ori_amp = _orientation_tuning(
        orientations,
        theta0,
        float(config["rf"].get("orientation_bandwidth_deg", 32.0)),
    )
    sf_amp = _sf_tuning(
        sfs,
        f0,
        float(config["rf"].get("sf_bandwidth_octaves", 1.1)),
    )
    amplitude = gain[:, None, None, None] * ori_amp[:, :, None, None] * sf_amp[:, None, :, None]

    theta = np.deg2rad(orientations)
    projection = mapped_x[:, None] * np.cos(theta)[None, :] + mapped_y[:, None] * np.sin(theta)[None, :]
    phase = 2.0 * np.pi * projection[:, :, None, None] * sfs[None, None, :, None]
    phase = phase + phases[None, None, None, :] - pref_phase[:, None, None, None]

    if response_model == "energy":
        response = baseline[:, None, None, None] + np.power(np.maximum(amplitude, 0.0), gamma[:, None, None, None])
    elif response_model == "simple":
        drive = amplitude * np.cos(phase)
        response = baseline[:, None, None, None] + np.power(np.maximum(drive, 0.0), gamma[:, None, None, None])
    else:
        raise ValueError(f"Unknown response model: {response_model}")
    return np.asarray(response, dtype=float)
