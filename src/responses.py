"""Response engines for grating and natural-image stimuli."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .noise import apply_noise, trial_average
from .coordinate_mapping import map_visual_degrees
from .rf_models import analytic_grating_responses, gabor_kernel_bank


def grating_response_trials(
    population: pd.DataFrame,
    stimuli: dict[str, np.ndarray],
    config: dict,
    rng: np.random.Generator,
    *,
    gaze_shift: tuple[float, float] = (0.0, 0.0),
    mapping_config: dict | None = None,
    noise_config: dict | None = None,
    response_model: str | None = None,
) -> np.ndarray:
    """Compute repeated grating responses with shape ``(repeats, n, ori, sf)``."""
    local_config = dict(config)
    if noise_config is not None:
        local_config = {**config, "noise": noise_config}
    noiseless = analytic_grating_responses(
        population,
        stimuli,
        local_config,
        gaze_shift=gaze_shift,
        mapping_config=mapping_config,
        response_model=response_model,
    )
    noisy = apply_noise(noiseless, local_config, rng)
    return trial_average(noisy).mean(axis=-1)[None, ...] if noisy.shape[0] == 0 else noisy.mean(axis=-1)


def grating_response_mean(
    population: pd.DataFrame,
    stimuli: dict[str, np.ndarray],
    config: dict,
    rng: np.random.Generator,
    *,
    gaze_shift: tuple[float, float] = (0.0, 0.0),
    mapping_config: dict | None = None,
    noise_config: dict | None = None,
    response_model: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return trial-averaged grating responses and the full repeated array."""
    trials = grating_response_trials(
        population,
        stimuli,
        config,
        rng,
        gaze_shift=gaze_shift,
        mapping_config=mapping_config,
        noise_config=noise_config,
        response_model=response_model,
    )
    return trials.mean(axis=0), trials


def natural_response_matrix(
    population: pd.DataFrame,
    images: np.ndarray,
    config: dict,
    *,
    gaze_shift: tuple[float, float] = (0.0, 0.0),
    kernels: tuple[np.ndarray, np.ndarray] | None = None,
    mapping_config: dict | None = None,
    response_model: str | None = None,
) -> np.ndarray:
    """Compute image-by-neuron natural-image responses for one gaze shift."""
    img_cfg = config["stimuli"]["natural_images"]
    size_px = int(img_cfg["image_size_px"])
    extent_deg = float(img_cfg["extent_deg"])
    response_model = response_model or config["rf"].get("response_model", "simple")
    if kernels is None:
        kernels = gabor_kernel_bank(population, image_size_px=size_px, extent_deg=extent_deg)
    cos_kernels, sin_kernels = kernels

    px_per_deg = size_px / extent_deg
    if mapping_config is not None:
        bx, by = map_visual_degrees(0.0, 0.0, (0.0, 0.0), mapping_config)
        sx, sy = map_visual_degrees(0.0, 0.0, gaze_shift, mapping_config)
        effective_gaze = (float(np.asarray(bx - sx)), float(np.asarray(by - sy)))
    else:
        effective_gaze = gaze_shift
    # Positive gaze shift is modeled as translating the retinal image relative to fixed RFs.
    shift_px = (float(effective_gaze[1]) * px_per_deg, -float(effective_gaze[0]) * px_per_deg)
    shifted_vectors = []
    for image in np.asarray(images, dtype=float):
        shifted = ndimage.shift(image, shift=shift_px, order=1, mode="nearest", prefilter=False)
        shifted_vectors.append(shifted.ravel())
    image_matrix = np.asarray(shifted_vectors)

    cos_drive = image_matrix @ cos_kernels.T
    gain = population["gain"].to_numpy(float)[None, :]
    baseline = population["baseline"].to_numpy(float)[None, :]
    gamma = population["gamma"].to_numpy(float)[None, :]
    if response_model == "energy":
        sin_drive = image_matrix @ sin_kernels.T
        energy = np.sqrt(cos_drive**2 + sin_drive**2)
        return baseline + gain * np.power(np.maximum(energy, 0.0), gamma)
    if response_model == "simple":
        return baseline + np.power(np.maximum(gain * cos_drive, 0.0), gamma)
    raise ValueError(f"Unknown response model: {response_model}")


def natural_response_trials(
    population: pd.DataFrame,
    images: np.ndarray,
    config: dict,
    rng: np.random.Generator,
    *,
    gaze_shift: tuple[float, float] = (0.0, 0.0),
    kernels: tuple[np.ndarray, np.ndarray] | None = None,
    mapping_config: dict | None = None,
    noise_config: dict | None = None,
    response_model: str | None = None,
) -> np.ndarray:
    """Compute repeated natural-image responses with shape ``(repeats, images, neurons)``."""
    local_config = dict(config)
    if noise_config is not None:
        local_config = {**config, "noise": noise_config}
    noiseless = natural_response_matrix(
        population,
        images,
        local_config,
        gaze_shift=gaze_shift,
        kernels=kernels,
        mapping_config=mapping_config,
        response_model=response_model,
    )
    return apply_noise(noiseless, local_config, rng)
