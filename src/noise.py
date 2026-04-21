"""Output noise models and repeated-trial simulation."""

from __future__ import annotations

import numpy as np


def apply_noise(
    responses: np.ndarray,
    config: dict,
    rng: np.random.Generator,
    *,
    repeats: int | None = None,
) -> np.ndarray:
    """Return repeated noisy responses with shape ``(repeats, *responses.shape)``."""
    noise_cfg = config.get("noise", config)
    n_repeats = int(repeats if repeats is not None else noise_cfg.get("repeats", 1))
    base = np.broadcast_to(np.asarray(responses, dtype=float), (n_repeats,) + np.shape(responses)).copy()
    model = str(noise_cfg.get("model", "none")).lower()

    if model in {"none", "deterministic", "noiseless"}:
        noisy = base
    elif model == "gaussian":
        noisy = base + rng.normal(0.0, float(noise_cfg.get("gaussian_sd", 0.05)), size=base.shape)
    elif model == "poisson":
        scale = float(noise_cfg.get("poisson_scale", 25.0))
        rates = np.maximum(base, 0.0) * scale
        noisy = rng.poisson(rates) / max(scale, 1e-12)
    elif model in {"multiplicative", "gain"}:
        gain_sd = float(noise_cfg.get("gain_sd", 0.1))
        noisy = base * rng.lognormal(mean=-0.5 * gain_sd**2, sigma=gain_sd, size=base.shape)
    elif model == "gaussian_multiplicative":
        gain_sd = float(noise_cfg.get("gain_sd", 0.1))
        noisy = base * rng.lognormal(mean=-0.5 * gain_sd**2, sigma=gain_sd, size=base.shape)
        noisy = noisy + rng.normal(0.0, float(noise_cfg.get("gaussian_sd", 0.05)), size=base.shape)
    else:
        raise ValueError(f"Unknown noise model: {model}")

    if bool(noise_cfg.get("clip_nonnegative", True)):
        noisy = np.maximum(noisy, 0.0)
    return noisy


def trial_average(noisy_responses: np.ndarray) -> np.ndarray:
    """Average repeated responses across the first axis."""
    return np.mean(noisy_responses, axis=0)


def response_snr(noisy_responses: np.ndarray) -> float:
    """Estimate a simple signal-to-noise ratio from repeated responses."""
    means = np.mean(noisy_responses, axis=0)
    variances = np.var(noisy_responses, axis=0, ddof=1) if noisy_responses.shape[0] > 1 else np.zeros_like(means)
    return float(np.nanmean(np.abs(means) / np.sqrt(variances + 1e-12)))
