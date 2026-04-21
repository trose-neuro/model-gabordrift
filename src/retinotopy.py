"""Local linear retinotopic mapping for a two-photon field of view."""

from __future__ import annotations

import numpy as np


def rotation_matrix(degrees: float) -> np.ndarray:
    """Return a 2D rotation matrix."""
    radians = np.deg2rad(degrees)
    c, s = np.cos(radians), np.sin(radians)
    return np.array([[c, -s], [s, c]], dtype=float)


def retinotopy_matrix(config: dict) -> np.ndarray:
    """Build the configurable 2x2 cortical-to-visual retinotopic transform.

    The base matrix is expressed in degrees per micron. Rotation and anisotropy
    are explicit because local retinotopy in a 500 um field is usually smooth
    but not axis-aligned with the imaging frame.
    """
    ret_cfg = config["retinotopy"]
    base = np.asarray(ret_cfg["matrix_deg_per_um"], dtype=float)
    rot = rotation_matrix(float(ret_cfg.get("rotation_deg", 0.0)))
    anis = ret_cfg.get("anisotropy", {})
    scale = np.diag(
        [
            float(anis.get("azimuth_scale", 1.0)),
            float(anis.get("elevation_scale", 1.0)),
        ]
    )
    return scale @ rot @ base


def apply_retinotopy(
    xy_um: np.ndarray,
    config: dict,
    rng: np.random.Generator | None = None,
    *,
    jitter: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map cortical coordinates in microns to visual RF centers in degrees.

    Parameters
    ----------
    xy_um:
        Array with shape ``(n_neurons, 2)`` in imaging-field coordinates.
    config:
        Full simulation config.
    rng:
        Random generator used for retinotopic scatter.
    jitter:
        If false, return the deterministic local linear map.

    Returns
    -------
    azimuth, elevation, jitter_values:
        Each is in degrees. ``jitter_values`` has shape ``(n_neurons, 2)``.
    """
    xy_um = np.asarray(xy_um, dtype=float)
    size_um = float(config["cortical_field"]["size_um"])
    centered = xy_um - 0.5 * size_um
    center = np.array(
        [
            float(config["retinotopy"]["center_azimuth_deg"]),
            float(config["retinotopy"]["center_elevation_deg"]),
        ]
    )
    visual = center + centered @ retinotopy_matrix(config).T

    noise = np.zeros_like(visual)
    if jitter:
        if rng is None:
            rng = np.random.default_rng()
        jitter_cfg = config["retinotopy"]["jitter_sd_deg"]
        sd = np.array([float(jitter_cfg["azimuth"]), float(jitter_cfg["elevation"])])
        noise = rng.normal(0.0, sd, size=visual.shape)
        visual = visual + noise

    return visual[:, 0], visual[:, 1], noise
