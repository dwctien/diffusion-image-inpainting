from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from src.preprocessing import (
    create_masked_image,
    load_image,
    load_mask,
    prepare_image_and_mask,
)

try:
    import torch
except ImportError:
    torch = None


def _require_torch() -> Any:
    """Return torch or raise a clear missing-dependency error."""
    if torch is None:
        raise ImportError(
            "PyTorch is required for inpainting inference. Install it with the "
            "project requirements before calling this function."
        )
    return torch


def build_generator(seed: int | None, device: str) -> "torch.Generator | None":
    """Create a seeded torch generator for reproducible inference."""
    if seed is None:
        return None

    torch_module = _require_torch()
    generator_device = (
        "cuda"
        if device == "cuda" and torch_module.cuda.is_available()
        else "cpu"
    )
    generator = torch_module.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def validate_inference_inputs(image: Image.Image, mask: Image.Image) -> None:
    """Validate that image and mask are PIL images with matching sizes."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image.Image instance.")
    if not isinstance(mask, Image.Image):
        raise TypeError("mask must be a PIL Image.Image instance.")
    if image.size != mask.size:
        raise ValueError(f"Image and mask sizes differ: {image.size} != {mask.size}")


def _get_pipeline_device(pipe: Any) -> str:
    """Infer the active device from a Diffusers pipeline."""
    torch_module = _require_torch()
    pipe_device = getattr(pipe, "device", None)

    if pipe_device is not None:
        device_type = getattr(pipe_device, "type", str(pipe_device))
        if str(device_type).startswith("cuda"):
            return "cuda"
        if str(device_type) == "cpu":
            return "cpu"

    return "cuda" if torch_module.cuda.is_available() else "cpu"


def run_inpainting(
    pipe: Any,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str | None = None,
    image_size: int = 512,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int | None = 42,
) -> Image.Image:
    """Run single-image Stable Diffusion inpainting with a loaded pipeline."""
    if pipe is None:
        raise ValueError("pipe must be a loaded inpainting pipeline.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")

    prepared_image, prepared_mask = prepare_image_and_mask(image, mask, image_size)
    validate_inference_inputs(prepared_image, prepared_mask)

    device = _get_pipeline_device(pipe)
    generator = build_generator(seed, device)

    try:
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=prepared_image,
            mask_image=prepared_mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
    except Exception as exc:
        raise RuntimeError("Inpainting inference failed.") from exc

    images = getattr(output, "images", None)
    if not images or not isinstance(images[0], Image.Image):
        raise RuntimeError("Inpainting pipeline did not return a PIL output image.")

    return images[0]


def run_inpainting_from_paths(
    pipe: Any,
    image_path: str | Path,
    mask_path: str | Path,
    prompt: str,
    negative_prompt: str | None = None,
    image_size: int = 512,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int | None = 42,
) -> Image.Image:
    """Load image and mask paths, then run single-image inpainting."""
    image = load_image(image_path)
    mask = load_mask(mask_path)
    return run_inpainting(
        pipe=pipe,
        image=image,
        mask=mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_size=image_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )


def run_inpainting_from_config(
    pipe: Any,
    image: Image.Image,
    mask: Image.Image,
    config: dict[str, Any],
    prompt: str | None = None,
    negative_prompt: str | None = None,
) -> Image.Image:
    """Run single-image inpainting with inference settings from config."""
    inference_config = config.get("inference", {})
    if not isinstance(inference_config, dict):
        raise ValueError("config['inference'] must be a dictionary.")

    resolved_prompt = prompt or inference_config.get("prompt", "a realistic photo")
    return run_inpainting(
        pipe=pipe,
        image=image,
        mask=mask,
        prompt=resolved_prompt,
        negative_prompt=negative_prompt,
        image_size=inference_config.get("image_size", 512),
        num_inference_steps=inference_config.get("num_inference_steps", 30),
        guidance_scale=inference_config.get("guidance_scale", 7.5),
        seed=inference_config.get("seed", 42),
    )


__all__ = [
    "build_generator",
    "create_masked_image",
    "run_inpainting",
    "run_inpainting_from_config",
    "run_inpainting_from_paths",
    "validate_inference_inputs",
]
