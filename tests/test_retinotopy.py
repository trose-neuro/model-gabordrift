import numpy as np

from src.retinotopy import apply_retinotopy, retinotopy_matrix
from src.utils import load_config


def test_retinotopy_center_maps_to_visual_center_without_jitter():
    cfg = load_config("configs/default.yaml", debug=True)
    size = cfg["cortical_field"]["size_um"]
    xy = np.array([[size / 2, size / 2]])
    az, el, jitter = apply_retinotopy(xy, cfg, jitter=False)
    assert np.isclose(az[0], cfg["retinotopy"]["center_azimuth_deg"])
    assert np.isclose(el[0], cfg["retinotopy"]["center_elevation_deg"])
    np.testing.assert_allclose(jitter, 0.0)


def test_retinotopy_matrix_shape():
    cfg = load_config("configs/default.yaml", debug=True)
    assert retinotopy_matrix(cfg).shape == (2, 2)
