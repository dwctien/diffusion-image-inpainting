# Image Inpainting with Diffusion Models

This project builds and evaluates an end-to-end image inpainting pipeline based on a pretrained diffusion inpainting model. It supports masks from random occlusions or user-provided/user-drawn masks, generates plausible image completions, and evaluates results with PSNR, SSIM, LPIPS, and runtime measurements.

The project currently does **not** train or fine-tune a model. Fine-tuning is optional future work and is not implemented.

## Project Structure

```text
image-inpainting-diffusion/
|-- configs/
|   `-- default.yaml              Default model, inference, path, and mask settings.
|-- src/
|   |-- utils.py                  Config loading, output directories, seeding, image discovery, and image saving.
|   |-- preprocessing.py          Image/mask loading, resizing, binarization, preparation, and masked previews.
|   |-- mask_generator.py         Random rectangle, multi-rectangle, and free-form mask generation.
|   |-- model_loader.py           Hugging Face Diffusers Stable Diffusion inpainting pipeline loader.
|   |-- pipeline.py               Reusable single-image inpainting pipeline.
|   |-- metrics.py                PSNR, SSIM, LPIPS, and runtime-related metric functions.
|   `-- README.md                 Internal documentation for the `src/` modules.
|-- scripts/
|   |-- run_inference.py          Command-line single-image inpainting script.
|   `-- run_evaluation.py         Command-line batch evaluation script for a folder of test images.
|-- notebooks/
|   |-- test_model.ipynb          Current Kaggle testing notebook.
|   |-- demo_colab.ipynb          Planned Colab notebook for launching the Gradio demo.
|   `-- evaluation_kaggle.ipynb   Planned official evaluation notebook for final experiments.
|-- app/
|   `-- gradio_app.py             Planned Gradio demo app; not implemented yet.
|-- data/                         Placeholder folders for local sample data only.
`-- outputs/                      Placeholder folders for generated images, masks, visualizations, and metrics.
```

## Current Progress

Implemented locally and tested on Kaggle:

- Config loading and utility functions.
- Image and mask preprocessing.
- Random mask generation.
- Pretrained model loading.
- Single-image inpainting pipeline.
- Metrics: PSNR, SSIM, LPIPS, and runtime.
- `scripts/run_inference.py` for single-image inference.
- `scripts/run_evaluation.py` for batch evaluation.

## Current Model

The current default model is:

```text
runwayml/stable-diffusion-inpainting
```

This model is loaded through Hugging Face Diffusers. It replaced the original default model because the previous model could not be loaded successfully in the testing environment.

Inference and evaluation are intended to run on Kaggle or Colab GPU. Local CPU inference is not recommended because diffusion-based inpainting is slow and memory-intensive.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

### Single-image inference

Run inpainting on one image-mask pair:

```bash
python scripts/run_inference.py --config configs/default.yaml --image data/samples/input.jpg --mask data/samples/mask.png
```

Useful arguments:

- `--config`: path to the YAML configuration file. Defaults to `configs/default.yaml`.
- `--image`: path to the input image. Required.
- `--mask`: path to the binary or grayscale mask image. Required. The masked region is the area to be inpainted.
- `--prompt`: optional text prompt override. If omitted, the prompt from the config file is used.
- `--negative_prompt`: optional negative prompt for the diffusion model.
- `--output`: path for the generated inpainted image. Defaults to `outputs/images/result.png`.
- `--save_masked_preview`: save a preview of the masked input image.
- `--masked_preview_output`: path for the masked preview image. Defaults to `outputs/visualizations/masked_preview.png`.

Example with custom prompt and preview output:

```bash
python scripts/run_inference.py \
  --config configs/default.yaml \
  --image data/samples/input.jpg \
  --mask data/samples/mask.png \
  --prompt "a realistic photo" \
  --negative_prompt "blurry, distorted" \
  --output outputs/images/sample_result.png \
  --save_masked_preview \
  --masked_preview_output outputs/visualizations/sample_masked_preview.png
```

### Batch evaluation

Run evaluation on a folder of test images:

```bash
python scripts/run_evaluation.py --config configs/default.yaml --input_dir data/test
```

Useful arguments:

- `--config`: path to the YAML configuration file. Defaults to `configs/default.yaml`.
- `--input_dir`: folder containing test images. If omitted, the input path from the config file is used.
- `--output_dir`: optional root output folder override. When provided, generated images, masks, visualizations, and metrics are saved under this folder.
- `--num_samples`: optional limit on the number of images to evaluate. Useful for quick tests.
- `--mask_type`: random mask type to generate. Supported values are `rectangle`, `rectangles`, and `free_form`.
- `--prompt`: optional text prompt override. If omitted, the prompt from the config file is used.
- `--negative_prompt`: optional negative prompt for the diffusion model.
- `--lpips_device`: device used for LPIPS computation. Defaults to `cuda`.
- `--save_visualizations`: save side-by-side comparison images containing the original image, mask, masked preview, and inpainted result.

Example quick evaluation on 5 images:

```bash
python scripts/run_evaluation.py \
  --config configs/default.yaml \
  --input_dir data/test \
  --output_dir outputs/eval_debug \
  --num_samples 5 \
  --mask_type rectangle \
  --prompt "a realistic photo" \
  --lpips_device cuda \
  --save_visualizations
```

Use a small number of images first to verify paths, GPU availability, and output settings before running a larger evaluation.

## Notebooks

- `notebooks/test_model.ipynb`: current Kaggle testing notebook used to verify model loading, single-image inference, and small evaluation.
- `notebooks/demo_colab.ipynb`: planned notebook for launching the Gradio demo on Colab.
- `notebooks/evaluation_kaggle.ipynb`: planned official evaluation notebook for final experiments.

## Pending Work

- Finalize experiment design.
- Run official evaluation on a larger test subset.
- Summarize metrics and select qualitative examples.
- Implement the Gradio app.
- Create and finalize the demo notebook.
- Update the report with experiment results and limitations.
