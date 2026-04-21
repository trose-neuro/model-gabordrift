"""Natural image loading, procedural fallback images, and preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

try:
    from skimage import io, transform
except Exception:  # pragma: no cover - optional import fallback
    io = None
    transform = None


def normalize_image(image: np.ndarray, mode: str = "zscore") -> np.ndarray:
    """Normalize an image using the configured convention."""
    arr = np.asarray(image, dtype=float)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    if mode == "zscore":
        return (arr - np.mean(arr)) / (np.std(arr) + 1e-9)
    if mode == "minmax":
        return 2.0 * (arr - np.min(arr)) / (np.ptp(arr) + 1e-9) - 1.0
    if mode == "center":
        return arr - np.mean(arr)
    raise ValueError(f"Unknown normalization mode: {mode}")


def _resize_image(image: np.ndarray, size_px: int) -> np.ndarray:
    if transform is not None:
        return transform.resize(image, (size_px, size_px), anti_aliasing=True, preserve_range=True)
    zoom = (size_px / image.shape[0], size_px / image.shape[1])
    return ndimage.zoom(image, zoom, order=1)


def generate_procedural_images(n_images: int, size_px: int, rng: np.random.Generator) -> np.ndarray:
    """Generate naturalistic 1/f-like grayscale images when no dataset is present."""
    images = []
    yy, xx = np.mgrid[-1.0:1.0 : complex(size_px), -1.0:1.0 : complex(size_px)]
    radius = np.sqrt(xx**2 + yy**2)
    for idx in range(int(n_images)):
        white = rng.normal(0.0, 1.0, size=(size_px, size_px))
        smooth_large = ndimage.gaussian_filter(white, sigma=rng.uniform(2.0, 7.0))
        smooth_small = ndimage.gaussian_filter(rng.normal(size=(size_px, size_px)), sigma=rng.uniform(0.6, 1.8))
        grating_angle = rng.uniform(0.0, np.pi)
        texture = np.sin(18.0 * (xx * np.cos(grating_angle) + yy * np.sin(grating_angle)) + rng.uniform(0, 2 * np.pi))
        vignette = np.exp(-0.8 * radius**2)
        image = (1.2 * smooth_large + 0.35 * smooth_small + 0.15 * texture) * vignette
        if idx % 3 == 0:
            image += 0.8 * np.exp(-((xx - rng.uniform(-0.4, 0.4)) ** 2 + (yy - rng.uniform(-0.4, 0.4)) ** 2) / 0.04)
        images.append(normalize_image(image, "zscore"))
    return np.asarray(images, dtype=float)


def load_natural_images(config: dict, rng: np.random.Generator | None = None) -> np.ndarray:
    """Load grayscale natural images from a folder or create procedural fallback images."""
    if rng is None:
        rng = np.random.default_rng()
    img_cfg = config["stimuli"]["natural_images"]
    size_px = int(img_cfg["image_size_px"])
    n_images = int(img_cfg["n_images"])
    normalize = str(img_cfg.get("normalize", "zscore"))
    folder = img_cfg.get("folder")

    loaded = []
    if folder:
        folder_path = Path(folder).expanduser()
        if folder_path.exists():
            paths = sorted(
                [
                    p
                    for p in folder_path.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
                ]
            )
            if io is None and paths:
                raise RuntimeError("scikit-image is required to load image files")
            for path in paths[:n_images]:
                image = io.imread(path)
                if image.ndim == 3:
                    image = image[..., :3].mean(axis=-1)
                loaded.append(normalize_image(_resize_image(image, size_px), normalize))

    if loaded:
        return np.asarray(loaded, dtype=float)
    return generate_procedural_images(n_images, size_px, rng)
