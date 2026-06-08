"""Command-line utility for generating binary inpainting masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.mask_generator import generate_mask
from src.preprocessing import create_masked_image, load_image
from src.utils import ensure_dir, get_image_files, save_image, set_seed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for mask generation."""
    parser = argparse.ArgumentParser(description="Generate inpainting masks for images.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Path to a single input image.")
    input_group.add_argument("--input_dir", help="Folder containing input images.")
    parser.add_argument(
        "--output_dir",
        default="outputs/masks",
        help="Folder for generated masks.",
    )
    parser.add_argument(
        "--output_suffix",
        default="_mask",
        help=(
            "Suffix appended to each image stem before .png. "
            "Use an empty string for batch inference."
        ),
    )
    parser.add_argument(
        "--preview_dir",
        default="outputs/visualizations",
        help="Folder for optional masked previews.",
    )
    parser.add_argument(
        "--mask_type",
        choices=("rectangle", "rectangles", "free_form"),
        default="rectangle",
        help="Type of random mask to generate.",
    )
    parser.add_argument("--num_samples", type=int, default=None, help="Optional image limit.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--save_preview",
        action="store_true",
        help="Save a preview image where the masked region is filled.",
    )

    parser.add_argument("--mask_ratio", type=float, default=0.25, help="Rectangle mask ratio.")
    parser.add_argument(
        "--num_rectangles",
        type=int,
        default=3,
        help="Number of rectangles for the rectangles mask type.",
    )
    parser.add_argument(
        "--min_ratio",
        type=float,
        default=0.05,
        help="Minimum rectangle ratio for the rectangles mask type.",
    )
    parser.add_argument(
        "--max_ratio",
        type=float,
        default=0.2,
        help="Maximum rectangle ratio for the rectangles mask type.",
    )
    parser.add_argument(
        "--num_strokes",
        type=int,
        default=8,
        help="Number of strokes for the free_form mask type.",
    )
    parser.add_argument(
        "--max_vertices",
        type=int,
        default=8,
        help="Maximum vertices per stroke for the free_form mask type.",
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=40,
        help="Maximum stroke width for the free_form mask type.",
    )
    return parser.parse_args()


def get_mask_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build mask generation keyword arguments from CLI options."""
    if args.mask_type == "rectangle":
        return {"mask_ratio": args.mask_ratio}
    if args.mask_type == "rectangles":
        return {
            "num_rectangles": args.num_rectangles,
            "min_ratio": args.min_ratio,
            "max_ratio": args.max_ratio,
        }
    return {
        "num_strokes": args.num_strokes,
        "max_vertices": args.max_vertices,
        "max_width": args.max_width,
    }


def collect_image_files(args: argparse.Namespace) -> list[Path]:
    """Return image files from either a single image path or an input folder."""
    if args.image:
        image_path = Path(args.image).expanduser()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        return [image_path]

    image_files = get_image_files(args.input_dir)
    if args.num_samples is not None:
        if args.num_samples <= 0:
            raise ValueError("--num_samples must be a positive integer.")
        image_files = image_files[: args.num_samples]
    if not image_files:
        raise ValueError(f"No supported image files found in: {args.input_dir}")
    return image_files


def iter_with_progress(items: list[Path]) -> Any:
    """Wrap image paths with tqdm when it is installed."""
    if tqdm is None:
        return items
    return tqdm(items, desc="Generating masks", unit="image")


def generate_mask_for_image(
    image_path: Path,
    output_dir: Path,
    preview_dir: Path,
    args: argparse.Namespace,
) -> Path:
    """Generate and save one mask for an image."""
    image = load_image(image_path)
    mask = generate_mask(image.size, args.mask_type, **get_mask_kwargs(args))
    mask_path = save_image(
        mask,
        output_dir / f"{image_path.stem}{args.output_suffix}.png",
    )

    if args.save_preview:
        preview = create_masked_image(image, mask)
        save_image(preview, preview_dir / f"{image_path.stem}_masked.png")

    return mask_path


def main() -> None:
    """Generate binary masks for one image or an image folder."""
    args = parse_args()
    set_seed(args.seed)

    output_dir = ensure_dir(args.output_dir)
    preview_dir = ensure_dir(args.preview_dir)
    image_files = collect_image_files(args)

    for image_path in iter_with_progress(image_files):
        mask_path = generate_mask_for_image(image_path, output_dir, preview_dir, args)
        print(f"Saved mask: {mask_path}")


if __name__ == "__main__":
    main()
