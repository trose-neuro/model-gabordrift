import numpy as np

from src.tuning import circular_difference_deg, circular_mean_deg


def test_orientation_circular_difference_wraps_at_180():
    assert circular_difference_deg(5, 175) == 10
    assert circular_difference_deg(175, 5) == -10
    np.testing.assert_allclose(circular_difference_deg([0, 90, 179], [179, 0, 1]), [1, -90, -2])


def test_orientation_circular_mean_uses_180_period():
    mean = circular_mean_deg(np.array([175.0, 5.0]), period=180.0)
    assert mean < 10 or mean > 170
