"""Command-line entry point for folder-based inpainting inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model_loader import load_pipeline_from_config
from src.pipeline import run_inpainting_from_paths
from src.preprocessing import (
    create_masked_image,
    load_image,
    load_mask,
    prepare_image_and_mask,
)
from src.utils import (
    SUPPORTED_IMAGE_EXTENSIONS,
    ensure_dir,
    ensure_output_dirs,
    get_image_files,
    load_config,
    save_image,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch inference."""
    parser = argparse.ArgumentParser(
        description="Run Stable Diffusion inpainting on a folder."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--image_dir", required=True, help="Folder containing input images.")
    parser.add_argument(
        "--mask_dir",
        required=True,
        help="Folder containing masks matched by filename stem.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/batch_inference",
        help="Root folder for generated images and optional previews.",
    )
    parser.add_argument("--num_samples", type=int, default=None, help="Optional image limit.")
    parser.add_argument("--prompt", default=None, help="Optional prompt override.")
    parser.add_argument("--negative_prompt", default=None, help="Optional negative prompt.")
    parser.add_argument(
        "--save_masked_preview",
        action="store_true",
        help="Save masked input previews for every successful image-mask pair.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop immediately when an image fails instead of continuing.",
    )
    return parser.parse_args()


def get_inference_settings(config: dict[str, Any], prompt: str | None) -> dict[str, Any]:
    """Read inference settings from config and apply prompt override."""
    inference_config = config.get("inference", {})
    if not isinstance(inference_config, dict):
        raise ValueError("config['inference'] must be a dictionary.")

    return {
        "prompt": prompt or inference_config.get("prompt", "a realistic photo"),
        "image_size": inference_config.get("image_size", 512),
        "num_inference_steps": inference_config.get("num_inference_steps", 30),
        "guidance_scale": inference_config.get("guidance_scale", 7.5),
        "seed": inference_config.get("seed", 42),
    }


def find_matching_mask(image_path: Path, mask_dir: str | Path) -> Path:
    """Find a mask in mask_dir with the same stem as the image path."""
    directory = Path(mask_dir).expanduser()
    if not directory.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Mask path is not a directory: {directory}")

    for suffix in sorted(SUPPORTED_IMAGE_EXTENSIONS):
        candidate = directory / f"{image_path.stem}{suffix}"
        if candidate.is_file():
            return candidate

    supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
    raise FileNotFoundError(
        f"No mask found for image '{image_path.name}' in {directory}. "
        f"Expected same stem with one of: {supported}"
    )


def iter_with_progress(items: list[Path]) -> Any:
    """Wrap image paths with tqdm when it is installed."""
    if tqdm is None:
        return items
    return tqdm(items, desc="Running inference", unit="image")


def save_masked_preview(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
    image_size: int,
) -> Path:
    """Create and save a masked preview for one image-mask pair."""
    image = load_image(image_path)
    mask = load_mask(mask_path)
    prepared_image, prepared_mask = prepare_image_and_mask(image, mask, image_size)
    masked_preview = create_masked_image(prepared_image, prepared_mask)
    return save_image(masked_preview, output_path)


def main() -> None:
    """Run folder-based inpainting inference."""
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)

    image_files = get_image_files(args.image_dir)
    if args.num_samples is not None:
        if args.num_samples <= 0:
            raise ValueError("--num_samples must be a positive integer.")
        image_files = image_files[: args.num_samples]
    if not image_files:
        raise ValueError(f"No supported image files found in: {args.image_dir}")

    settings = get_inference_settings(config, args.prompt)
    output_root = ensure_dir(args.output_dir)
    image_output_dir = ensure_dir(output_root / "images")
    preview_output_dir = ensure_dir(output_root / "visualizations")

    pipe = load_pipeline_from_config(config)
    success_count = 0
    failed_count = 0

    for image_path in iter_with_progress(image_files):
        try:
            mask_path = find_matching_mask(image_path, args.mask_dir)
            result = run_inpainting_from_paths(
                pipe=pipe,
                image_path=image_path,
                mask_path=mask_path,
                prompt=settings["prompt"],
                negative_prompt=args.negative_prompt,
                image_size=settings["image_size"],
                num_inference_steps=settings["num_inference_steps"],
                guidance_scale=settings["guidance_scale"],
                seed=settings["seed"],
            )
            result_path = save_image(
                result,
                image_output_dir / f"{image_path.stem}_inpainted.png",
            )

            if args.save_masked_preview:
                preview_path = preview_output_dir / f"{image_path.stem}_masked.png"
                save_masked_preview(
                    image_path,
                    mask_path,
                    preview_path,
                    settings["image_size"],
                )

            print(f"Saved: {result_path}")
            success_count += 1
        except Exception as exc:
            failed_count += 1
            print(f"Failed: {image_path} ({exc})")
            if args.fail_fast:
                raise

    print(f"Batch inference finished. Success: {success_count}, failed: {failed_count}")


if __name__ == "__main__":
    main()
