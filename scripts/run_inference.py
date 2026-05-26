"""Command-line entry point for single-image inpainting."""

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
from src.preprocessing import create_masked_image, load_image, load_mask, prepare_image_and_mask
from src.utils import ensure_output_dirs, load_config, save_image


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for single-image inference."""
    parser = argparse.ArgumentParser(description="Run Stable Diffusion inpainting.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--mask", required=True, help="Path to the input mask.")
    parser.add_argument("--prompt", default=None, help="Optional prompt override.")
    parser.add_argument("--negative_prompt", default=None, help="Optional negative prompt.")
    parser.add_argument(
        "--output",
        default="outputs/images/result.png",
        help="Path for the inpainted output image.",
    )
    parser.add_argument(
        "--save_masked_preview",
        action="store_true",
        help="Save the masked input preview.",
    )
    parser.add_argument(
        "--masked_preview_output",
        default="outputs/visualizations/masked_preview.png",
        help="Path for the masked preview image.",
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


def save_masked_preview(
    image_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
    image_size: int,
) -> Path:
    """Create and save a masked preview for the input image-mask pair."""
    image = load_image(image_path)
    mask = load_mask(mask_path)
    prepared_image, prepared_mask = prepare_image_and_mask(image, mask, image_size)
    masked_preview = create_masked_image(prepared_image, prepared_mask)
    return save_image(masked_preview, output_path)


def main() -> None:
    """Run single-image inpainting from command-line arguments."""
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)

    settings = get_inference_settings(config, args.prompt)
    pipe = load_pipeline_from_config(config)
    result = run_inpainting_from_paths(
        pipe=pipe,
        image_path=args.image,
        mask_path=args.mask,
        prompt=settings["prompt"],
        negative_prompt=args.negative_prompt,
        image_size=settings["image_size"],
        num_inference_steps=settings["num_inference_steps"],
        guidance_scale=settings["guidance_scale"],
        seed=settings["seed"],
    )
    result_path = save_image(result, args.output)
    print(f"Saved inpainted result: {result_path}")

    if args.save_masked_preview:
        preview_path = save_masked_preview(
            image_path=args.image,
            mask_path=args.mask,
            output_path=args.masked_preview_output,
            image_size=settings["image_size"],
        )
        print(f"Saved masked preview: {preview_path}")


if __name__ == "__main__":
    main()
