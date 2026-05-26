"""Command-line entry point for batch inpainting evaluation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.mask_generator import generate_mask
from src.metrics import compute_all_metrics, get_lpips_model, measure_runtime
from src.model_loader import load_pipeline_from_config
from src.pipeline import run_inpainting
from src.preprocessing import create_masked_image, load_image, prepare_image_and_mask
from src.utils import ensure_dir, ensure_output_dirs, get_image_files, load_config, save_image, set_seed

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _require_pandas() -> Any:
    """Return pandas or raise a clear dependency error."""
    if pd is None:
        raise ImportError(
            "pandas is required to save evaluation metrics. Install it with the "
            "project requirements before running evaluation."
        )
    return pd


def _iter_with_progress(items: list[Path]) -> Any:
    """Wrap image paths with tqdm progress display when available."""
    if tqdm is None:
        return items
    return tqdm(items, desc="Evaluating", unit="image")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch evaluation."""
    parser = argparse.ArgumentParser(description="Run batch inpainting evaluation.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--input_dir", default=None, help="Optional image folder override.")
    parser.add_argument("--output_dir", default=None, help="Optional output folder override.")
    parser.add_argument("--num_samples", type=int, default=None, help="Optional sample limit.")
    parser.add_argument(
        "--mask_type",
        choices=("rectangle", "rectangles", "free_form"),
        default=None,
        help="Mask type to generate.",
    )
    parser.add_argument("--prompt", default=None, help="Optional prompt override.")
    parser.add_argument("--negative_prompt", default=None, help="Optional negative prompt.")
    parser.add_argument("--lpips_device", default="cuda", help="Device for LPIPS metrics.")
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save side-by-side visualization images.",
    )
    return parser.parse_args()


def apply_output_dir_override(config: dict[str, Any], output_dir: str | Path | None) -> None:
    """Update output paths in config when an output directory override is provided."""
    if output_dir is None:
        return

    paths_config = config.setdefault("paths", {})
    if not isinstance(paths_config, dict):
        raise ValueError("config['paths'] must be a dictionary.")

    root = Path(output_dir)
    paths_config["output_dir"] = str(root)
    paths_config["image_output_dir"] = str(root / "images")
    paths_config["mask_output_dir"] = str(root / "masks")
    paths_config["visualization_output_dir"] = str(root / "visualizations")
    paths_config["metrics_output_dir"] = str(root / "metrics")


def get_paths_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the validated paths section from config."""
    paths_config = config.get("paths", {})
    if not isinstance(paths_config, dict):
        raise ValueError("config['paths'] must be a dictionary.")
    return paths_config


def get_inference_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the validated inference section from config."""
    inference_config = config.get("inference", {})
    if not isinstance(inference_config, dict):
        raise ValueError("config['inference'] must be a dictionary.")
    return inference_config


def get_mask_kwargs(config: dict[str, Any], mask_type: str) -> dict[str, Any]:
    """Build mask generation keyword arguments from config."""
    mask_config = config.get("mask", {})
    if not isinstance(mask_config, dict):
        raise ValueError("config['mask'] must be a dictionary.")

    min_ratio = mask_config.get("min_mask_ratio", 0.1)
    max_ratio = mask_config.get("max_mask_ratio", 0.4)
    if mask_type == "rectangle":
        return {"mask_ratio": (float(min_ratio) + float(max_ratio)) / 2.0}
    if mask_type == "rectangles":
        return {"min_ratio": float(min_ratio), "max_ratio": float(max_ratio)}
    return {}


def create_side_by_side_visualization(
    original: Image.Image,
    mask: Image.Image,
    masked: Image.Image,
    result: Image.Image,
) -> Image.Image:
    """Create a horizontal grid of original, mask, masked preview, and result."""
    panels = [
        original.convert("RGB"),
        mask.convert("RGB"),
        masked.convert("RGB"),
        result.convert("RGB"),
    ]
    width, height = panels[0].size
    grid = Image.new("RGB", (width * len(panels), height))
    for index, panel in enumerate(panels):
        if panel.size != (width, height):
            panel = panel.resize((width, height), Image.LANCZOS)
        grid.paste(panel, (index * width, 0))
    return grid


def save_metrics_csv(rows: list[dict[str, Any]], metrics_path: Path) -> None:
    """Save accumulated evaluation rows to a CSV file."""
    pandas_module = _require_pandas()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pandas_module.DataFrame(rows).to_csv(metrics_path, index=False)


def build_error_row(
    filename: str,
    mask_type: str,
    error: Exception,
) -> dict[str, Any]:
    """Create a metrics row that records a failed sample."""
    return {
        "filename": filename,
        "mask_type": mask_type,
        "psnr": None,
        "ssim": None,
        "lpips": None,
        "runtime_sec": None,
        "result_path": None,
        "mask_path": None,
        "masked_preview_path": None,
        "error": str(error),
    }


def evaluate_image(
    image_path: Path,
    pipe: Any,
    lpips_model: Any,
    lpips_device: str,
    config: dict[str, Any],
    mask_type: str,
    prompt: str,
    negative_prompt: str | None,
    output_paths: dict[str, Path],
    save_visualizations: bool,
) -> dict[str, Any]:
    """Run mask generation, inference, saving, and metrics for one image."""
    inference_config = get_inference_config(config)
    image_size = inference_config.get("image_size", 512)
    seed = inference_config.get("seed", 42)
    stem = image_path.stem

    original = load_image(image_path)
    generated_mask = generate_mask(original.size, mask_type, **get_mask_kwargs(config, mask_type))
    prepared_image, prepared_mask = prepare_image_and_mask(original, generated_mask, image_size)
    masked_preview = create_masked_image(prepared_image, prepared_mask)

    start_time = time.time()
    result = run_inpainting(
        pipe=pipe,
        image=prepared_image,
        mask=prepared_mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_size=image_size,
        num_inference_steps=inference_config.get("num_inference_steps", 30),
        guidance_scale=inference_config.get("guidance_scale", 7.5),
        seed=seed,
    )
    end_time = time.time()

    mask_path = save_image(prepared_mask, output_paths["masks"] / f"{stem}_mask.png")
    result_path = save_image(result, output_paths["images"] / f"{stem}_inpainted.png")
    preview_path = save_image(masked_preview, output_paths["visualizations"] / f"{stem}_masked.png")

    if save_visualizations:
        visualization = create_side_by_side_visualization(
            prepared_image,
            prepared_mask,
            masked_preview,
            result,
        )
        save_image(visualization, output_paths["visualizations"] / f"{stem}_comparison.png")

    metrics = compute_all_metrics(prepared_image, result, lpips_model, lpips_device)
    return {
        "filename": image_path.name,
        "mask_type": mask_type,
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "lpips": metrics["lpips"],
        "runtime_sec": measure_runtime(start_time, end_time),
        "result_path": str(result_path),
        "mask_path": str(mask_path),
        "masked_preview_path": str(preview_path),
        "error": None,
    }


def main() -> None:
    """Run batch evaluation from command-line arguments."""
    args = parse_args()
    config = load_config(args.config)
    apply_output_dir_override(config, args.output_dir)
    ensure_output_dirs(config)

    paths_config = get_paths_config(config)
    inference_config = get_inference_config(config)
    mask_config = config.get("mask", {})
    input_dir = args.input_dir or paths_config.get("input_dir")
    if input_dir is None:
        raise ValueError("Input directory must be provided by --input_dir or config paths.")

    set_seed(inference_config.get("seed", 42))
    image_files = get_image_files(input_dir)
    if args.num_samples is not None:
        if args.num_samples <= 0:
            raise ValueError("--num_samples must be a positive integer.")
        image_files = image_files[: args.num_samples]
    if not image_files:
        raise ValueError(f"No supported image files found in: {input_dir}")

    mask_type = args.mask_type or mask_config.get("default_type", "rectangle")
    prompt = args.prompt or inference_config.get("prompt", "a realistic photo")

    output_paths = {
        "images": ensure_dir(paths_config["image_output_dir"]),
        "masks": ensure_dir(paths_config["mask_output_dir"]),
        "visualizations": ensure_dir(paths_config["visualization_output_dir"]),
        "metrics": ensure_dir(paths_config["metrics_output_dir"]),
    }
    metrics_csv_path = output_paths["metrics"] / "metrics.csv"

    pipe = load_pipeline_from_config(config)
    lpips_model, resolved_lpips_device = get_lpips_model(args.lpips_device)

    rows: list[dict[str, Any]] = []
    for image_path in _iter_with_progress(image_files):
        try:
            row = evaluate_image(
                image_path=image_path,
                pipe=pipe,
                lpips_model=lpips_model,
                lpips_device=resolved_lpips_device,
                config=config,
                mask_type=mask_type,
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                output_paths=output_paths,
                save_visualizations=args.save_visualizations,
            )
        except Exception as exc:
            print(f"Failed to evaluate {image_path}: {exc}")
            row = build_error_row(image_path.name, mask_type, exc)

        rows.append(row)
        save_metrics_csv(rows, metrics_csv_path)

    print(f"Saved metrics CSV: {metrics_csv_path.resolve()}")


if __name__ == "__main__":
    main()
