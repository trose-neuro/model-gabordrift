import numpy as np

from src.coordinate_mapping import map_visual_degrees, spherical_to_plane


def test_exact_projection_matches_planar_near_origin():
    x_exact, y_exact = spherical_to_plane(np.array([0.1]), np.array([0.1]), method="exact")
    x_planar, y_planar = spherical_to_plane(np.array([0.1]), np.array([0.1]), method="planar")
    np.testing.assert_allclose(x_exact, x_planar, atol=1e-4)
    np.testing.assert_allclose(y_exact, y_planar, atol=1e-4)


def test_mapping_error_scale_changes_coordinates():
    mapping = {"mode": "approximate", "error": {"scale": [2.0, 1.0], "anisotropy": [1.0, 1.0]}}
    x, y = map_visual_degrees(np.array([1.0]), np.array([1.0]), (0.0, 0.0), mapping)
    assert np.isclose(x[0], 2.0)
    assert np.isclose(y[0], 1.0)
