# Source Modules

The `src/` directory contains the core project logic reused by scripts,
notebooks, and the app.

## Current Responsibilities

- `utils.py`: config loading, directory creation, seed setting, image file
  discovery, and image saving.
- `preprocessing.py`: image/mask loading, resizing, normalization,
  binarization, and masked preview creation.
- `mask_generator.py`: mask generation for evaluation.
- `model_loader.py`: loads the pretrained Stable Diffusion inpainting pipeline
  from Hugging Face Diffusers, resolves device and dtype, and applies
  memory-saving options.
- `pipeline.py`: runs single-image inpainting using a loaded pipeline, input
  image, mask, prompt, and inference settings.
- `metrics.py`: computes PSNR, SSIM, LPIPS, and runtime-related values for
  evaluation.
- `scripts/run_inference.py`: runs single-image inpainting from command line.
- `scripts/run_evaluation.py`: runs batch evaluation on a test image folder,
  generates masks, saves outputs, and writes metrics CSV.

## Expected Dependency Flow

```text
scripts/run_inference.py
-> utils.py
-> preprocessing.py
-> model_loader.py
-> pipeline.py

scripts/run_evaluation.py
-> utils.py
-> preprocessing.py
-> mask_generator.py
-> model_loader.py
-> pipeline.py
-> metrics.py

app/gradio_app.py
-> preprocessing.py
-> model_loader.py
-> pipeline.py
```

The Gradio app will be implemented in a later step.
