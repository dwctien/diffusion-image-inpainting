# Image Inpainting with Diffusion Models

This repository is a starter skeleton for an image inpainting project using diffusion models. The current stage only defines the project layout, placeholder files, and expected workflow without implementing model loading, mask generation, metrics, inference, or UI logic.

## Goals

- Prepare a clean and extensible codebase for diffusion-based image inpainting.
- Separate model, preprocessing, pipeline, evaluation, and app concerns from the beginning.
- Keep data, generated outputs, configs, scripts, and notebooks organized for later experiments.

## Project Structure

```text
src/        Core Python modules for the future inpainting pipeline.
scripts/    Command-line entry points for inference and evaluation.
app/        Placeholder for a future Gradio demo.
notebooks/  Experiment and demo notebooks for Colab or Kaggle.
data/       Raw, test, and sample image folders.
outputs/    Generated images, masks, visualizations, and metric reports.
configs/    YAML configuration files.
```

## Planned Workflow

1. Prepare input images and masks under `data/`.
2. Configure model and inference settings in `configs/default.yaml`.
3. Run inference scripts to generate inpainted outputs.
4. Run evaluation scripts to compute image quality metrics.
5. Explore results through notebooks and a future Gradio app.

Implementation will be added in later steps.
