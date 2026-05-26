from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except ImportError:
    peak_signal_noise_ratio = None
    structural_similarity = None

try:
    import torch
except ImportError:
    torch = None

try:
    import lpips
except ImportError:
    lpips = None


def _require_skimage() -> tuple[Any, Any]:
    """Return skimage metric functions or raise a clear dependency error."""
    if peak_signal_noise_ratio is None or structural_similarity is None:
        raise ImportError(
            "scikit-image is required for PSNR and SSIM. Install it with the "
            "project requirements before computing these metrics."
        )
    return peak_signal_noise_ratio, structural_similarity


def _require_torch() -> Any:
    """Return torch or raise a clear dependency error."""
    if torch is None:
        raise ImportError(
            "PyTorch is required for LPIPS. Install it with the project "
            "requirements before computing LPIPS."
        )
    return torch


def _require_lpips() -> Any:
    """Return lpips or raise a clear dependency error."""
    if lpips is None:
        raise ImportError(
            "LPIPS is required for perceptual distance. Install it with the "
            "project requirements before computing LPIPS."
        )
    return lpips


def _validate_same_size(original: Image.Image, restored: Image.Image) -> None:
    """Ensure two PIL images have matching sizes."""
    if not isinstance(original, Image.Image):
        raise TypeError("original must be a PIL Image.Image instance.")
    if not isinstance(restored, Image.Image):
        raise TypeError("restored must be a PIL Image.Image instance.")
    if original.size != restored.size:
        raise ValueError(
            f"Image sizes must match for metrics: {original.size} != {restored.size}"
        )


def _resolve_device(device: str) -> str:
    """Resolve the LPIPS torch device with CUDA fallback."""
    torch_module = _require_torch()
    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be 'cuda' or 'cpu'.")
    if device == "cuda" and not torch_module.cuda.is_available():
        print("Warning: CUDA requested for LPIPS but not available. Falling back to CPU.")
        return "cpu"
    return device


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a float32 RGB NumPy array in [0, 1]."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image.Image instance.")
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    return array / 255.0


def compute_psnr(original: Image.Image, restored: Image.Image) -> float:
    """Compute PSNR between original and restored PIL images."""
    psnr_fn, _ = _require_skimage()
    _validate_same_size(original, restored)
    return float(
        psnr_fn(
            pil_to_numpy(original),
            pil_to_numpy(restored),
            data_range=1.0,
        )
    )


def compute_ssim(original: Image.Image, restored: Image.Image) -> float:
    """Compute SSIM between original and restored PIL images."""
    _, ssim_fn = _require_skimage()
    _validate_same_size(original, restored)
    return float(
        ssim_fn(
            pil_to_numpy(original),
            pil_to_numpy(restored),
            channel_axis=-1,
            data_range=1.0,
        )
    )


def get_lpips_model(device: str = "cuda") -> tuple[Any, str]:
    """Load the LPIPS AlexNet model and return it with the resolved device."""
    torch_module = _require_torch()
    lpips_module = _require_lpips()
    resolved_device = _resolve_device(device)

    try:
        model = lpips_module.LPIPS(net="alex")
        model = model.to(resolved_device)
        model.eval()
    except Exception as exc:
        raise RuntimeError("Failed to load LPIPS model.") from exc

    return model, resolved_device


def pil_to_lpips_tensor(image: Image.Image, device: str) -> "torch.Tensor":
    """Convert a PIL image to a batched LPIPS tensor in [-1, 1]."""
    torch_module = _require_torch()
    resolved_device = _resolve_device(device)
    array = pil_to_numpy(image)
    tensor = torch_module.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor * 2.0) - 1.0
    return tensor.to(resolved_device)


def compute_lpips(
    original: Image.Image,
    restored: Image.Image,
    lpips_model: Any | None = None,
    device: str = "cuda",
) -> float:
    """Compute LPIPS distance between original and restored PIL images."""
    torch_module = _require_torch()
    _validate_same_size(original, restored)

    resolved_device = _resolve_device(device)
    model = lpips_model
    if model is None:
        model, resolved_device = get_lpips_model(resolved_device)

    original_tensor = pil_to_lpips_tensor(original, resolved_device)
    restored_tensor = pil_to_lpips_tensor(restored, resolved_device)

    with torch_module.no_grad():
        distance = model(original_tensor, restored_tensor)

    return float(distance.detach().cpu().item())


def compute_all_metrics(
    original: Image.Image,
    restored: Image.Image,
    lpips_model: Any | None = None,
    device: str = "cuda",
) -> dict[str, float]:
    """Compute PSNR, SSIM, and LPIPS for an image pair."""
    return {
        "psnr": compute_psnr(original, restored),
        "ssim": compute_ssim(original, restored),
        "lpips": compute_lpips(original, restored, lpips_model, device),
    }


def measure_runtime(start_time: float, end_time: float) -> float:
    """Compute elapsed runtime in seconds."""
    return float(end_time - start_time)
