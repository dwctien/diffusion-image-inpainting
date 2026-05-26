from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return its parsed dictionary."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Config path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse YAML config file: {path}") from exc
    except OSError as exc:
        raise OSError(f"Could not read config file: {path}") from exc

    if not isinstance(config, dict) or not config:
        raise ValueError(f"Config file is empty or invalid: {path}")

    return config


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return its resolved path."""
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve()


def ensure_output_dirs(config: dict[str, Any]) -> None:
    """Create the output directories declared in the config dictionary."""
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise KeyError("Config must contain a 'paths' dictionary.")

    required_keys = (
        "output_dir",
        "image_output_dir",
        "mask_output_dir",
        "visualization_output_dir",
        "metrics_output_dir",
    )
    for key in required_keys:
        if key not in paths:
            raise KeyError(f"Missing required paths config key: {key}")
        ensure_dir(paths[key])


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and torch when available."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_image_files(input_dir: str | Path) -> list[Path]:
    """Return sorted non-recursive image file paths from a directory."""
    directory = Path(input_dir).expanduser()
    if not directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {directory}")

    image_files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(image_files)


def save_image(image: Image.Image, path: str | Path) -> Path:
    """Save a PIL image to disk and return the resolved output path."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image.Image instance.")

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path.resolve()
