"""Gradio demo application for Diffusion Image Inpainting."""

from __future__ import annotations

import base64
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.mask_generator import generate_mask
from src.model_loader import load_pipeline_from_config
from src.pipeline import run_inpainting
from src.preprocessing import create_masked_image
from src.utils import load_config

# ---------------------------------------------------------------------------
# Global pipeline state (lazy-loaded)
# ---------------------------------------------------------------------------
_pipeline: Any = None
_config: dict[str, Any] | None = None


def _get_config() -> dict[str, Any]:
    """Load and cache the default YAML config."""
    global _config
    if _config is None:
        _config = load_config(ROOT_DIR / "configs" / "default.yaml")
    return _config


def _get_pipeline() -> Any:
    """Lazy-load the Stable Diffusion inpainting pipeline."""
    global _pipeline
    if _pipeline is None:
        gr.Info("⏳ Đang tải model... Vui lòng chờ.")
        _pipeline = load_pipeline_from_config(_get_config())
        gr.Info("✅ Model đã sẵn sàng!")
    return _pipeline


# ---------------------------------------------------------------------------
# Helpers: base64 bridge for the JS canvas
# ---------------------------------------------------------------------------

def _pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _b64_to_pil(b64: str) -> Image.Image | None:
    if not b64:
        return None
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(b64))).convert("L")


# ---------------------------------------------------------------------------
# Handler: Random mask
# ---------------------------------------------------------------------------

def generate_random_mask_handler(
    image: Image.Image | None,
    mask_type: str,
    mask_ratio: float,
    num_rects: int,
    num_strokes: int,
) -> tuple[Image.Image | None, Image.Image | None]:
    """Generate a random mask and return (mask, preview)."""
    if image is None:
        gr.Warning("⚠️ Vui lòng upload ảnh trước!")
        return None, None

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")

    w, h = image.size
    kw: dict[str, Any] = {}
    if mask_type == "rectangle":
        kw["mask_ratio"] = mask_ratio
    elif mask_type == "rectangles":
        kw["num_rectangles"] = int(num_rects)
        kw["min_ratio"] = max(0.01, mask_ratio * 0.3)
        kw["max_ratio"] = min(0.95, mask_ratio)
    elif mask_type == "free_form":
        kw["num_strokes"] = int(num_strokes)

    mask = generate_mask((w, h), mask_type=mask_type, **kw)
    preview = create_masked_image(image, mask, fill_color=(231, 76, 60))
    return mask, preview


def update_slider_visibility(mask_type: str):
    """Show/hide sliders based on selected mask type."""
    return (
        gr.update(visible=mask_type in ("rectangle", "rectangles")),  # ratio
        gr.update(visible=mask_type == "rectangles"),                  # num_rects
        gr.update(visible=mask_type == "free_form"),                  # num_strokes
    )


# ---------------------------------------------------------------------------
# Handler: Canvas drawn mask
# ---------------------------------------------------------------------------

def load_image_to_canvas(image: Image.Image | None) -> str:
    """Convert uploaded image to base64 for the JS canvas."""
    if image is None:
        gr.Warning("⚠️ Vui lòng upload ảnh trước!")
        return ""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    return _pil_to_b64(image)


def apply_canvas_mask(
    mask_b64: str,
    image: Image.Image | None,
) -> tuple[Image.Image | None, Image.Image | None]:
    """Decode the canvas mask and create preview."""
    if not mask_b64 or image is None:
        gr.Warning("⚠️ Chưa có mask! Hãy vẽ mask trước.")
        return None, None
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")

    mask = _b64_to_pil(mask_b64)
    if mask is None:
        return None, None
    mask = mask.resize(image.size, Image.NEAREST)
    preview = create_masked_image(image, mask, fill_color=(231, 76, 60))
    return mask, preview


# ---------------------------------------------------------------------------
# Handler: Inpainting
# ---------------------------------------------------------------------------

def run_inpainting_handler(
    image: Image.Image | None,
    mask: Image.Image | None,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    seed: int,
    size: int,
) -> Image.Image | None:
    """Run inpainting and return the result image."""
    if image is None:
        gr.Warning("⚠️ Vui lòng upload ảnh!")
        return None
    if mask is None:
        gr.Warning("⚠️ Vui lòng tạo mask trước!")
        return None
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask).convert("L")

    pipe = _get_pipeline()
    return run_inpainting(
        pipe=pipe,
        image=image,
        mask=mask,
        prompt=prompt.strip() or "a realistic photo",
        negative_prompt=negative_prompt.strip() or None,
        image_size=size,
        num_inference_steps=steps,
        guidance_scale=guidance,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Canvas HTML + JS (loaded from external file)
# ---------------------------------------------------------------------------

_CANVAS_JS_PATH = Path(__file__).with_name("canvas.js")
_CANVAS_JS = _CANVAS_JS_PATH.read_text(encoding="utf-8") if _CANVAS_JS_PATH.exists() else ""

CANVAS_HTML = f"""
<div id="mp-container">
  <div class="mp-toolbar">
    <button class="mp-tool active" data-tool="brush" title="Brush">🖌️ Brush</button>
    <button class="mp-tool" data-tool="rect" title="Rectangle">▬ Rect</button>
    <button class="mp-tool" data-tool="circle" title="Circle">⬤ Circle</button>
    <button class="mp-tool" data-tool="eraser" title="Eraser">🧹 Eraser</button>
    <span class="mp-sep"></span>
    <label class="mp-size-wrap">
      Size: <span id="mp-size-label">30px</span>
      <input type="range" id="mp-brush-size" min="3" max="120" value="30">
    </label>
    <span class="mp-sep"></span>
    <button id="mp-clear" class="mp-clear-btn" title="Clear all">🗑️ Xóa</button>
  </div>
  <canvas id="mask-draw-canvas" style="cursor:crosshair;border-radius:10px;display:block;margin:0 auto;"></canvas>
</div>
<script>{_CANVAS_JS}</script>
"""


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.gradio-container { font-family:'Inter',sans-serif!important; max-width:1420px!important; margin:auto!important; }

/* header */
#app-header { text-align:center; padding:26px 20px 16px;
  background:linear-gradient(135deg,#0f0c29,#302b63 50%,#24243e);
  border-radius:16px; margin-bottom:18px;
  border:1px solid rgba(255,255,255,.06);
  box-shadow:0 8px 32px rgba(0,0,0,.35); }
#app-header h1 { background:linear-gradient(90deg,#a78bfa,#818cf8,#6366f1);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  font-size:2rem; font-weight:700; margin:0 0 6px; }
#app-header p { color:#94a3b8; font-size:.92rem; margin:0; font-weight:300; }

/* tabs */
.tab-nav button { font-weight:600!important; border-radius:10px!important; transition:all .25s!important; }
.tab-nav button.selected { background:linear-gradient(135deg,#6366f1,#818cf8)!important;
  color:#fff!important; box-shadow:0 4px 15px rgba(99,102,241,.35)!important; }

/* primary buttons */
#gen-mask-btn, #apply-mask-btn {
  background:linear-gradient(135deg,#6366f1,#8b5cf6)!important; color:#fff!important;
  font-weight:600!important; border-radius:10px!important; border:none!important;
  transition:all .3s!important; box-shadow:0 4px 12px rgba(99,102,241,.3)!important; }
#gen-mask-btn:hover, #apply-mask-btn:hover {
  transform:translateY(-1px)!important; box-shadow:0 6px 18px rgba(99,102,241,.45)!important; }

#run-btn { background:linear-gradient(135deg,#059669,#10b981)!important; color:#fff!important;
  font-weight:700!important; border-radius:12px!important; border:none!important;
  font-size:1.05rem!important; min-height:50px!important; transition:all .3s!important;
  box-shadow:0 4px 16px rgba(16,185,129,.35)!important; }
#run-btn:hover { transform:translateY(-2px)!important;
  box-shadow:0 8px 24px rgba(16,185,129,.5)!important; }

/* canvas toolbar */
#mp-container { background:#1a1a2e; border-radius:14px; padding:12px; }
.mp-toolbar { display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin-bottom:10px; }
.mp-tool { padding:7px 14px; border-radius:8px; border:1px solid rgba(255,255,255,.1);
  background:rgba(255,255,255,.06); color:#cbd5e1; font-size:.82rem; cursor:pointer;
  transition:all .2s; font-weight:500; }
.mp-tool:hover { background:rgba(255,255,255,.12); }
.mp-tool.active { background:linear-gradient(135deg,#6366f1,#818cf8);
  color:#fff; border-color:transparent; box-shadow:0 2px 10px rgba(99,102,241,.4); }
.mp-sep { width:1px; height:26px; background:rgba(255,255,255,.12); margin:0 4px; }
.mp-size-wrap { color:#94a3b8; font-size:.82rem; display:flex; align-items:center; gap:6px; }
.mp-size-wrap input[type=range] { width:100px; accent-color:#818cf8; }
.mp-clear-btn { padding:7px 14px; border-radius:8px; border:1px solid rgba(239,68,68,.3);
  background:rgba(239,68,68,.12); color:#f87171; cursor:pointer; font-size:.82rem;
  transition:all .2s; }
.mp-clear-btn:hover { background:rgba(239,68,68,.25); }

#mask-draw-canvas { background:#111; max-width:100%; }

/* result */
#result-img { border:2px solid rgba(99,102,241,.25)!important; border-radius:14px!important;
  box-shadow:0 4px 18px rgba(99,102,241,.12)!important; }

/* footer */
#app-footer { text-align:center; padding:12px; color:#64748b; font-size:.78rem; }
"""


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def create_demo() -> gr.Blocks:
    """Create the Gradio Blocks demo."""

    with gr.Blocks(title="Diffusion Image Inpainting") as demo:

        # Header
        gr.HTML("""
        <div id="app-header">
          <h1>🎨 Diffusion Image Inpainting</h1>
          <p>Upload ảnh → Tạo mask (random / tự vẽ) → Inpainting bằng Stable Diffusion</p>
        </div>""")

        current_mask = gr.State(value=None)

        with gr.Row(equal_height=False):
            # ── LEFT COLUMN ───────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Bước 1 — Upload ảnh")
                input_image = gr.Image(
                    label="Ảnh gốc", type="pil", height=320,
                    sources=["upload", "clipboard"],
                )

                gr.Markdown("### 🎭 Bước 2 — Tạo Mask")
                with gr.Tabs():
                    # ── Tab: Random Mask ──────────────────────────
                    with gr.TabItem("🎲 Random Mask", id=0):
                        gr.Markdown("*Tự động sinh mask ngẫu nhiên*")
                        mask_type = gr.Dropdown(
                            label="Loại mask",
                            choices=["rectangle", "rectangles", "free_form"],
                            value="rectangle",
                        )
                        mask_ratio = gr.Slider(
                            label="Tỉ lệ mask", minimum=0.05, maximum=0.8,
                            value=0.25, step=0.05, visible=True,
                        )
                        num_rects = gr.Slider(
                            label="Số hình chữ nhật", minimum=1, maximum=15,
                            value=3, step=1, visible=False,
                        )
                        num_strokes = gr.Slider(
                            label="Số nét vẽ", minimum=1, maximum=30,
                            value=8, step=1, visible=False,
                        )
                        gen_btn = gr.Button(
                            "🎲 Tạo Random Mask", elem_id="gen-mask-btn",
                            variant="primary",
                        )
                        # conditional visibility
                        mask_type.change(
                            fn=update_slider_visibility,
                            inputs=[mask_type],
                            outputs=[mask_ratio, num_rects, num_strokes],
                        )

                    # ── Tab: Draw Mask ────────────────────────────
                    with gr.TabItem("✏️ Vẽ Mask", id=1):
                        gr.Markdown(
                            "*Vẽ trực tiếp lên ảnh — vùng đỏ sẽ được inpaint. "
                            "Upload ảnh ở Bước 1, ảnh sẽ tự động hiện ở canvas.*"
                        )
                        # Hidden textboxes for JS ↔ Python bridge
                        img_b64 = gr.Textbox(visible=False, elem_id="img-b64")
                        mask_b64 = gr.Textbox(visible=False, elem_id="mask-b64")

                        gr.HTML(CANVAS_HTML)
                        apply_mask_btn = gr.Button(
                            "✅ Áp dụng Mask đã vẽ", elem_id="apply-mask-btn",
                            variant="primary",
                        )

                # ── Preview ───────────────────────────────────────
                gr.Markdown("### 👁️ Preview")
                with gr.Row():
                    mask_display = gr.Image(
                        label="Mask", type="pil", height=180, interactive=False,
                    )
                    preview_display = gr.Image(
                        label="Ảnh + Mask (đỏ = inpaint)", type="pil",
                        height=180, interactive=False,
                    )

            # ── RIGHT COLUMN ──────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Bước 3 — Cài đặt")
                with gr.Group():
                    prompt = gr.Textbox(
                        label="Prompt", value="a realistic photo",
                        placeholder="Mô tả nội dung muốn tạo…", lines=2,
                    )
                    neg_prompt = gr.Textbox(
                        label="Negative Prompt", value="",
                        placeholder="Nội dung KHÔNG muốn có…", lines=1,
                    )
                with gr.Row():
                    steps = gr.Slider(
                        label="Steps", minimum=10, maximum=100, value=30, step=5,
                    )
                    guidance = gr.Slider(
                        label="Guidance", minimum=1.0, maximum=20.0,
                        value=7.5, step=0.5,
                    )
                with gr.Row():
                    seed = gr.Number(label="Seed", value=42, precision=0)
                    img_size = gr.Dropdown(
                        label="Size", choices=[256, 512, 768, 1024], value=512,
                    )

                run_btn = gr.Button(
                    "🚀 Chạy Inpainting", elem_id="run-btn", variant="primary",
                )

                gr.Markdown("### 🖼️ Bước 4 — Kết quả")
                result_img = gr.Image(
                    label="Prediction", type="pil", height=380,
                    interactive=False, elem_id="result-img",
                )

        gr.HTML('<div id="app-footer">Stable Diffusion Inpainting · Gradio</div>')

        # ══════════════════════════════════════════════════════════
        # Event wiring
        # ══════════════════════════════════════════════════════════

        # Random mask
        gen_btn.click(
            fn=generate_random_mask_handler,
            inputs=[input_image, mask_type, mask_ratio, num_rects, num_strokes],
            outputs=[current_mask, preview_display],
        ).then(lambda m: m, inputs=[current_mask], outputs=[mask_display])

        # Auto-load image into canvas whenever user uploads
        input_image.change(
            fn=load_image_to_canvas,
            inputs=[input_image],
            outputs=[img_b64],
        ).then(
            fn=None, inputs=[img_b64], outputs=[],
            js="(b64) => { if(window.MaskPainter) window.MaskPainter.loadImage(b64); }",
        )

        # Apply drawn mask (JS exports → Python decodes)
        apply_mask_btn.click(
            fn=None, inputs=[], outputs=[mask_b64],
            js="() => { return window.MaskPainter ? window.MaskPainter.exportMask() : ''; }",
        ).then(
            fn=apply_canvas_mask,
            inputs=[mask_b64, input_image],
            outputs=[current_mask, preview_display],
        ).then(lambda m: m, inputs=[current_mask], outputs=[mask_display])

        # Run inpainting
        run_btn.click(
            fn=run_inpainting_handler,
            inputs=[input_image, current_mask, prompt, neg_prompt,
                    steps, guidance, seed, img_size],
            outputs=[result_img],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = create_demo()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
    )
