from __future__ import annotations

import random
from math import sqrt

import numpy as np
from PIL import Image, ImageDraw


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


def generate_rectangle_mask(
    size: int | tuple[int, int],
    mask_ratio: float = 0.25,
) -> Image.Image:
    """Generate a single random rectangular binary mask."""
    if not 0 < mask_ratio < 1:
        raise ValueError("mask_ratio must be between 0 and 1.")

    width, height = _normalize_size(size)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    target_area = width * height * mask_ratio
    aspect_ratio = random.uniform(0.5, 2.0)
    rect_width = max(1, min(width, int(round(sqrt(target_area * aspect_ratio)))))
    rect_height = max(1, min(height, int(round(target_area / rect_width))))

    left = random.randint(0, width - rect_width)
    top = random.randint(0, height - rect_height)
    draw.rectangle((left, top, left + rect_width - 1, top + rect_height - 1), fill=255)
    return mask


def generate_random_rectangles_mask(
    size: int | tuple[int, int],
    num_rectangles: int = 3,
    min_ratio: float = 0.05,
    max_ratio: float = 0.2,
) -> Image.Image:
    """Generate a binary mask made from multiple random rectangles."""
    if num_rectangles <= 0:
        raise ValueError("num_rectangles must be positive.")
    if not 0 < min_ratio <= max_ratio < 1:
        raise ValueError("min_ratio and max_ratio must satisfy 0 < min <= max < 1.")

    width, height = _normalize_size(size)
    mask = Image.new("L", (width, height), 0)

    for _ in range(num_rectangles):
        ratio = random.uniform(min_ratio, max_ratio)
        rectangle = generate_rectangle_mask((width, height), ratio)
        mask_array = np.maximum(np.array(mask), np.array(rectangle)).astype(np.uint8)
        mask = Image.fromarray(mask_array, mode="L")

    return mask


def generate_free_form_mask(
    size: int | tuple[int, int],
    num_strokes: int = 8,
    max_vertices: int = 8,
    max_width: int = 40,
) -> Image.Image:
    """Generate a free-form binary mask with randomized line strokes."""
    if num_strokes <= 0:
        raise ValueError("num_strokes must be positive.")
    if max_vertices <= 1:
        raise ValueError("max_vertices must be greater than 1.")
    if max_width <= 0:
        raise ValueError("max_width must be positive.")

    width, height = _normalize_size(size)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for _ in range(num_strokes):
        vertex_count = random.randint(2, max_vertices)
        points = [
            (random.randint(0, width - 1), random.randint(0, height - 1))
            for _ in range(vertex_count)
        ]
        stroke_width = random.randint(1, max_width)
        draw.line(points, fill=255, width=stroke_width, joint="curve")

        radius = max(1, stroke_width // 2)
        for x_coord, y_coord in points:
            draw.ellipse(
                (
                    x_coord - radius,
                    y_coord - radius,
                    x_coord + radius,
                    y_coord + radius,
                ),
                fill=255,
            )

    return mask


def generate_mask(
    size: int | tuple[int, int],
    mask_type: str = "rectangle",
    **kwargs: object,
) -> Image.Image:
    """Generate a binary mask using one of the supported mask types."""
    generators = {
        "rectangle": generate_rectangle_mask,
        "rectangles": generate_random_rectangles_mask,
        "free_form": generate_free_form_mask,
    }

    if mask_type not in generators:
        valid_types = ", ".join(sorted(generators))
        raise ValueError(f"Invalid mask_type '{mask_type}'. Expected one of: {valid_types}")

    return generators[mask_type](size, **kwargs)
