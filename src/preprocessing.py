from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _normalize_size(size: int | tuple[int, int]) -> tuple[int, int]:
    """Convert an integer or width-height tuple into a validated size tuple."""
    if isinstance(size, int):
        if size <= 0:
            raise ValueError("size must be a positive integer.")
        return (size, size)

    if (
        isinstance(size, tuple)
        and len(size) == 2
        and all(isinstance(value, int) and value > 0 for value in size)
    ):
        return size

    raise ValueError("size must be a positive int or tuple of two positive ints.")


def load_image(image_path: str | Path) -> Image.Image:
    """Load an image from disk and convert it to RGB mode."""
    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")

    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except OSError as exc:
        raise ValueError(f"Could not load image file: {path}") from exc


def load_mask(mask_path: str | Path) -> Image.Image:
    """Load a mask from disk and convert it to grayscale mode."""
    path = Path(mask_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Mask file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Mask path is not a file: {path}")

    try:
        with Image.open(path) as mask:
            return mask.convert("L")
    except OSError as exc:
        raise ValueError(f"Could not load mask file: {path}") from exc


def resize_image(image: Image.Image, size: int | tuple[int, int]) -> Image.Image:
    """Resize an image with high-quality interpolation."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image.Image instance.")
    return image.resize(_normalize_size(size), Image.LANCZOS)


def resize_mask(mask: Image.Image, size: int | tuple[int, int]) -> Image.Image:
    """Resize a mask with nearest-neighbor interpolation."""
    if not isinstance(mask, Image.Image):
        raise TypeError("mask must be a PIL Image.Image instance.")
    return mask.resize(_normalize_size(size), Image.NEAREST)


def binarize_mask(mask: Image.Image, threshold: int = 127) -> Image.Image:
    """Convert a grayscale mask into a binary keep/inpaint mask."""
    if not isinstance(mask, Image.Image):
        raise TypeError("mask must be a PIL Image.Image instance.")
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be in the range [0, 255].")

    mask_array = np.array(mask.convert("L"))
    binary_array = np.where(mask_array > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary_array, mode="L")


def prepare_image_and_mask(
    image: Image.Image,
    mask: Image.Image,
    image_size: int = 512,
) -> tuple[Image.Image, Image.Image]:
    """Convert, resize, binarize, and validate an image-mask pair."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image.Image instance.")
    if not isinstance(mask, Image.Image):
        raise TypeError("mask must be a PIL Image.Image instance.")

    prepared_image = resize_image(image.convert("RGB"), image_size)
    prepared_mask = resize_mask(mask.convert("L"), image_size)
    prepared_mask = binarize_mask(prepared_mask)

    if prepared_image.size != prepared_mask.size:
        raise ValueError(
            "Prepared image and mask sizes differ: "
            f"{prepared_image.size} != {prepared_mask.size}"
        )

    return prepared_image, prepared_mask


def create_masked_image(
    image: Image.Image,
    mask: Image.Image,
    fill_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Create an RGB preview where white mask pixels are filled."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image.Image instance.")
    if not isinstance(mask, Image.Image):
        raise TypeError("mask must be a PIL Image.Image instance.")
    if len(fill_color) != 3 or any(not 0 <= value <= 255 for value in fill_color):
        raise ValueError("fill_color must contain three values in the range [0, 255].")

    rgb_image = image.convert("RGB")
    grayscale_mask = mask.convert("L")
    if rgb_image.size != grayscale_mask.size:
        raise ValueError(
            f"Image and mask sizes differ: {rgb_image.size} != {grayscale_mask.size}"
        )

    image_array = np.array(rgb_image).copy()
    mask_array = np.array(grayscale_mask)
    image_array[mask_array == 255] = fill_color
    return Image.fromarray(image_array, mode="RGB")
