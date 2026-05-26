from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError:
    torch = None

try:
    from diffusers import StableDiffusionInpaintPipeline
except ImportError:
    StableDiffusionInpaintPipeline = None


def _require_torch() -> Any:
    """Return torch or raise a clear missing-dependency error."""
    if torch is None:
        raise ImportError(
            "PyTorch is required for model loading. Install it with the project "
            "requirements before calling this function."
        )
    return torch


def _require_diffusers_pipeline() -> Any:
    """Return the Diffusers inpainting pipeline class or raise a clear error."""
    if StableDiffusionInpaintPipeline is None:
        raise ImportError(
            "Diffusers is required for model loading. Install it with the project "
            "requirements before calling this function."
        )
    return StableDiffusionInpaintPipeline


def resolve_device(device: str = "cuda") -> str:
    """Resolve the requested compute device to either cuda or cpu."""
    if device not in {"cuda", "cpu"}:
        raise ValueError("Unsupported device. Expected 'cuda' or 'cpu'.")

    torch_module = _require_torch()
    if device == "cuda" and not torch_module.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        return "cpu"

    return device


def resolve_torch_dtype(torch_dtype: str = "float16", device: str = "cuda") -> Any:
    """Resolve a dtype string into a torch dtype for the selected device."""
    torch_module = _require_torch()
    if device == "cpu":
        return torch_module.float32

    if torch_dtype == "auto":
        return torch_module.float16 if device == "cuda" else torch_module.float32
    if torch_dtype == "float16":
        return torch_module.float16
    if torch_dtype == "float32":
        return torch_module.float32

    raise ValueError("Unsupported torch_dtype. Expected 'float16', 'float32', or 'auto'.")


def load_inpainting_pipeline(
    model_id: str,
    device: str = "cuda",
    torch_dtype: str = "float16",
    enable_attention_slicing: bool = True,
    enable_vae_slicing: bool = True,
) -> "StableDiffusionInpaintPipeline":
    """Load a pretrained Diffusers Stable Diffusion inpainting pipeline."""
    if not model_id:
        raise ValueError("model_id must be a non-empty string.")

    pipeline_class = _require_diffusers_pipeline()
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_torch_dtype(torch_dtype, resolved_device)

    try:
        pipe = pipeline_class.from_pretrained(model_id, torch_dtype=resolved_dtype)
        pipe = pipe.to(resolved_device)

        if enable_attention_slicing:
            pipe.enable_attention_slicing()
        if enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=False)

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load inpainting pipeline '{model_id}' on {resolved_device}."
        ) from exc

    return pipe


def load_pipeline_from_config(config: dict[str, Any]) -> "StableDiffusionInpaintPipeline":
    """Load the inpainting pipeline using model settings from a config dict."""
    model_config = config.get("model", {})
    if not isinstance(model_config, dict):
        raise ValueError("config['model'] must be a dictionary.")

    model_id = model_config.get(
        "model_id",
        "stabilityai/stable-diffusion-2-inpainting",
    )
    device = model_config.get("device", "cuda")
    dtype = model_config.get("torch_dtype", "float16")

    return load_inpainting_pipeline(
        model_id=model_id,
        device=device,
        torch_dtype=dtype,
    )
