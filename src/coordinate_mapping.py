"""Spherical-to-plane coordinate mappings and controlled mapping errors."""

from __future__ import annotations

import numpy as np

from .retinotopy import rotation_matrix


def spherical_to_plane(
    azimuth_deg: np.ndarray | float,
    elevation_deg: np.ndarray | float,
    *,
    method: str = "exact",
) -> tuple[np.ndarray, np.ndarray]:
    """Project gaze-centered spherical coordinates to stimulus-plane degrees.

    ``exact`` uses a gnomonic projection from azimuth/elevation to a tangent
    plane. ``planar`` is the small-angle approximation. Returned coordinates
    are scaled back to degree-like units so they match the grating and image
    coordinate conventions near the origin.
    """
    az = np.asarray(azimuth_deg, dtype=float)
    el = np.asarray(elevation_deg, dtype=float)
    if method == "exact":
        az_rad = np.deg2rad(az)
        el_rad = np.deg2rad(el)
        x = np.rad2deg(np.tan(az_rad))
        y = np.rad2deg(np.tan(el_rad) / np.maximum(np.cos(az_rad), 1e-9))
    elif method in {"planar", "approximate"}:
        x, y = az, el
    else:
        raise ValueError(f"Unknown projection method: {method}")
    return x, y


def apply_mapping_error(
    x_deg: np.ndarray,
    y_deg: np.ndarray,
    error: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a controlled affine-plus-radial coordinate mapping error."""
    if not error:
        return x_deg, y_deg

    x = np.asarray(x_deg, dtype=float).copy()
    y = np.asarray(y_deg, dtype=float).copy()

    origin = np.asarray(error.get("origin_offset_deg", [0.0, 0.0]), dtype=float)
    x = x - origin[0]
    y = y - origin[1]

    scale = np.asarray(error.get("scale", [1.0, 1.0]), dtype=float)
    anis = np.asarray(error.get("anisotropy", [1.0, 1.0]), dtype=float)
    x = x * scale[0] * anis[0]
    y = y * scale[1] * anis[1]

    rot = rotation_matrix(float(error.get("rotation_deg", 0.0)))
    stacked = np.vstack([np.ravel(x), np.ravel(y)])
    rotated = rot @ stacked
    x = rotated[0].reshape(np.shape(x))
    y = rotated[1].reshape(np.shape(y))

    nonlinear = float(error.get("nonlinear", 0.0))
    if nonlinear:
        r2 = x**2 + y**2
        x = x * (1.0 + nonlinear * r2)
        y = y * (1.0 + nonlinear * r2)
    return x, y


def map_visual_degrees(
    azimuth_deg: np.ndarray | float,
    elevation_deg: np.ndarray | float,
    gaze_shift: tuple[float, float] = (0.0, 0.0),
    mapping_config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map RF centers into retinal/stimulus coordinates for a global gaze shift."""
    mapping_config = mapping_config or {"mode": "exact", "error": {}}
    rel_az = np.asarray(azimuth_deg, dtype=float) - float(gaze_shift[0])
    rel_el = np.asarray(elevation_deg, dtype=float) - float(gaze_shift[1])
    mode = mapping_config.get("mode", "exact")
    method = "exact" if mode == "exact" else "planar"
    x, y = spherical_to_plane(rel_az, rel_el, method=method)
    return apply_mapping_error(x, y, mapping_config.get("error", {}))


def mapping_error_field(
    azimuth_grid: np.ndarray,
    elevation_grid: np.ndarray,
    mapping_config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact-vs-configured mapping displacement magnitude over a grid."""
    exact_x, exact_y = spherical_to_plane(azimuth_grid, elevation_grid, method="exact")
    mapped_x, mapped_y = map_visual_degrees(azimuth_grid, elevation_grid, (0.0, 0.0), mapping_config)
    dx = mapped_x - exact_x
    dy = mapped_y - exact_y
    return dx, dy, np.sqrt(dx**2 + dy**2)
