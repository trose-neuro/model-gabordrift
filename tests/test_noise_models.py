import numpy as np

from src.noise import apply_noise


def test_noiseless_repeats_are_identical():
    rng = np.random.default_rng(0)
    response = np.ones((3, 4))
    cfg = {"model": "none", "repeats": 3}
    noisy = apply_noise(response, cfg, rng)
    assert noisy.shape == (3, 3, 4)
    np.testing.assert_allclose(noisy[0], noisy[1])


def test_gaussian_noise_changes_repeats_and_clips():
    rng = np.random.default_rng(0)
    response = np.ones((3, 4))
    cfg = {"model": "gaussian", "repeats": 3, "gaussian_sd": 0.1, "clip_nonnegative": True}
    noisy = apply_noise(response, cfg, rng)
    assert noisy.shape == (3, 3, 4)
    assert not np.allclose(noisy[0], noisy[1])
    assert np.all(noisy >= 0)
