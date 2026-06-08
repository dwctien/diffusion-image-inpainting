# Image Inpainting with Diffusion Models

Đồ án xây dựng pipeline Image Inpainting sử dụng mô hình diffusion đã huấn luyện sẵn. Hệ thống nhận ảnh gốc và vùng mask cần khôi phục, sau đó sinh phần ảnh bị che bằng Stable Diffusion Inpainting. Repo hỗ trợ chạy suy luận cho một ảnh, đánh giá hàng loạt trên CelebA-HQ/CelebAMask-HQ, sinh mask ngẫu nhiên và chạy website demo bằng Gradio.

> Trạng thái hiện tại: repo dùng pretrained model từ Hugging Face Diffusers.

## Nội dung chính

- Tải mô hình `runwayml/stable-diffusion-inpainting` bằng Diffusers.
- Tiền xử lý ảnh và mask về cùng kích thước, mặc định `512x512`.
- Sinh mask ngẫu nhiên theo 3 kiểu: `rectangle`, `rectangles`, `free_form`.
- Tạo mask độc lập cho một ảnh hoặc nhiều ảnh trong thư mục.
- Chạy inference cho một cặp ảnh-mask hoặc nhiều ảnh-mask trong cùng thư mục.
- Chạy batch evaluation trên tập ảnh CelebA-HQ.
- Tính các chỉ số PSNR, SSIM, LPIPS và thời gian chạy.
- Lưu ảnh kết quả, mask, ảnh preview và file metrics CSV.
- Website demo Gradio hỗ trợ upload ảnh, tạo/vẽ mask và chạy inpainting.

## Cấu trúc thư mục

```text
diffusion-image-inpainting/
|-- app/
|   |-- gradio_app.py          Website demo Gradio.
|   `-- canvas.js              Canvas vẽ mask trong demo.
|-- configs/
|   `-- default.yaml           Cấu hình model, inference, đường dẫn và mask.
|-- data/
|   |-- raw/                   Dataset gốc tải về, không commit lên git.
|   `-- prepared/              Dataset đã chọn/lọc để chạy pipeline.
|-- docs/                      Tài liệu bổ sung cho báo cáo/đồ án.
|-- notebooks/                 Notebook thử nghiệm trên Kaggle/Colab.
|-- outputs/
|   |-- images/                Ảnh inpainted.
|   |-- masks/                 Mask được dùng khi đánh giá.
|   |-- visualizations/        Preview và ảnh so sánh.
|   `-- metrics/               File metrics CSV.
|-- scripts/
|   |-- generate_masks.py      Tạo mask ngẫu nhiên cho một ảnh hoặc thư mục ảnh.
|   |-- run_inference.py       Chạy inpainting cho một ảnh.
|   |-- run_batch_inference.py Chạy inpainting cho nhiều ảnh trong thư mục.
|   `-- run_evaluation.py      Chạy đánh giá hàng loạt.
|-- src/
|   |-- mask_generator.py      Sinh mask ngẫu nhiên.
|   |-- metrics.py             PSNR, SSIM, LPIPS.
|   |-- model_loader.py        Load Stable Diffusion Inpainting pipeline.
|   |-- pipeline.py            Logic inpainting dùng lại.
|   |-- preprocessing.py       Load, resize, chuẩn hóa ảnh và mask.
|   `-- utils.py               Config, seed, đường dẫn, lưu ảnh.
|-- requirements.txt
`-- README.md
```

## Yêu cầu môi trường

Khuyến nghị chạy trên GPU NVIDIA, đặc biệt khi đánh giá nhiều ảnh hoặc chạy demo. CPU vẫn có thể chạy nhưng rất chậm.

- Python 3.10 hoặc 3.11.
- CUDA GPU nếu dùng `model.device: "cuda"` trong `configs/default.yaml`.
- Dung lượng trống cho model cache Hugging Face và dataset CelebA-HQ.

Cài thư viện:

```bash
pip install -r requirements.txt
```

Nếu cài PyTorch riêng theo CUDA version của máy, cài PyTorch trước theo hướng dẫn chính thức rồi chạy lại:

```bash
pip install -r requirements.txt
```

## Cấu hình mặc định

File cấu hình chính nằm ở `configs/default.yaml`.

```yaml
model:
  model_id: "runwayml/stable-diffusion-inpainting"
  device: "cuda"
  torch_dtype: "float16"

inference:
  image_size: 512
  num_inference_steps: 30
  guidance_scale: 7.5
  seed: 42
  prompt: "a realistic photo"

paths:
  input_dir: "data/test"
  output_dir: "outputs"
```

Nếu máy không có GPU, đổi:

```yaml
model:
  device: "cpu"
  torch_dtype: "float32"
```

## Cách đặt dataset CelebA-HQ

Đặt dataset trong `data/raw/` hoặc `data/prepared/` tùy mục đích.

Nguồn tải CelebA/CelebA-HQ: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

### Cấu trúc khuyến nghị

Sau khi tải CelebA-HQ hoặc CelebAMask-HQ, đặt ảnh gốc theo cấu trúc:

```text
data/
|-- raw/
|   `-- CelebAMask-HQ/
|       `-- CelebA-HQ-img/
|           |-- 0.jpg
|           |-- 1.jpg
|           |-- 2.jpg
|           `-- ...
`-- prepared/
    `-- celeba_hq/
        `-- test/
            |-- 0.jpg
            |-- 1.jpg
            |-- 2.jpg
            `-- ...
```

Pipeline evaluation hiện đọc ảnh trong `input_dir` theo kiểu không đệ quy. Vì vậy thư mục truyền vào `--input_dir` phải chứa trực tiếp các file ảnh `.jpg`, `.jpeg`, `.png`, `.bmp` hoặc `.webp`.

Ví dụ hợp lệ:

```text
data/prepared/celeba_hq/test/000001.jpg
data/prepared/celeba_hq/test/000002.jpg
data/prepared/celeba_hq/test/000003.jpg
```

Ví dụ không hợp lệ nếu truyền `--input_dir data/prepared/celeba_hq`:

```text
data/prepared/celeba_hq/test/000001.jpg
```

Trong trường hợp này phải truyền đúng thư mục con:

```bash
python scripts/run_evaluation.py --input_dir data/prepared/celeba_hq/test
```

### Chuẩn bị tập test nhỏ

Có thể copy một số ảnh CelebA-HQ vào thư mục test để chạy thử nhanh:

```powershell
New-Item -ItemType Directory -Force data\prepared\celeba_hq\test
Copy-Item data\raw\CelebAMask-HQ\CelebA-HQ-img\0.jpg data\prepared\celeba_hq\test\0.jpg
Copy-Item data\raw\CelebAMask-HQ\CelebA-HQ-img\1.jpg data\prepared\celeba_hq\test\1.jpg
Copy-Item data\raw\CelebAMask-HQ\CelebA-HQ-img\2.jpg data\prepared\celeba_hq\test\2.jpg
```

Trên Linux/Colab/Kaggle:

```bash
mkdir -p data/prepared/celeba_hq/test
cp data/raw/CelebAMask-HQ/CelebA-HQ-img/0.jpg data/prepared/celeba_hq/test/0.jpg
cp data/raw/CelebAMask-HQ/CelebA-HQ-img/1.jpg data/prepared/celeba_hq/test/1.jpg
cp data/raw/CelebAMask-HQ/CelebA-HQ-img/2.jpg data/prepared/celeba_hq/test/2.jpg
```

Nếu muốn dùng toàn bộ ảnh, có thể đặt trực tiếp thư mục ảnh gốc làm `--input_dir`:

```bash
python scripts/run_evaluation.py \
  --config configs/default.yaml \
  --input_dir data/raw/CelebAMask-HQ/CelebA-HQ-img \
  --output_dir outputs/celeba_hq_eval \
  --num_samples 100 \
  --mask_type rectangle \
  --save_visualizations
```

## Chạy pipeline với CelebA-HQ

### 1. Chạy thử nhanh trên vài ảnh

```bash
python scripts/run_evaluation.py \
  --config configs/default.yaml \
  --input_dir data/prepared/celeba_hq/test \
  --output_dir outputs/celeba_hq_debug \
  --num_samples 5 \
  --mask_type rectangle \
  --prompt "a realistic photo" \
  --lpips_device cuda \
  --save_visualizations
```

Kết quả được lưu tại:

```text
outputs/celeba_hq_debug/
|-- images/             Ảnh inpainted.
|-- masks/              Mask sinh ngẫu nhiên.
|-- visualizations/     Ảnh masked preview và comparison.
`-- metrics/
    `-- metrics.csv     PSNR, SSIM, LPIPS, runtime_sec.
```

### 2. Chạy evaluation cho nhiều ảnh

Sau khi chạy thử thành công, tăng số lượng ảnh:

```bash
python scripts/run_evaluation.py \
  --config configs/default.yaml \
  --input_dir data/prepared/celeba_hq/test \
  --output_dir outputs/celeba_hq_eval \
  --num_samples 1000 \
  --mask_type rectangle \
  --prompt "a realistic photo" \
  --lpips_device cuda \
  --save_visualizations
```

Các kiểu mask được hỗ trợ:

- `rectangle`: một vùng chữ nhật.
- `rectangles`: nhiều vùng chữ nhật.
- `free_form`: mask dạng nét vẽ tự do.

Ví dụ chạy với free-form mask:

```bash
python scripts/run_evaluation.py \
  --config configs/default.yaml \
  --input_dir data/prepared/celeba_hq/test \
  --output_dir outputs/celeba_hq_free_form \
  --num_samples 100 \
  --mask_type free_form \
  --prompt "a realistic photo" \
  --lpips_device cuda \
  --save_visualizations
```

### 3. Tạo mask để chạy inference

Nếu chưa có mask sẵn, dùng `scripts/generate_masks.py` để tạo mask ngẫu nhiên. Script hỗ trợ 3 kiểu mask: `rectangle`, `rectangles`, `free_form`.

Tạo mask cho một ảnh:

```bash
python scripts/generate_masks.py \
  --image data/samples/input.jpg \
  --output_dir data/samples/masks \
  --mask_type rectangle \
  --mask_ratio 0.25 \
  --save_preview \
  --preview_dir outputs/visualizations
```

Kết quả:

```text
data/samples/masks/input_mask.png
outputs/visualizations/input_masked.png
```

Tạo mask cho nhiều ảnh trong thư mục:

```bash
python scripts/generate_masks.py \
  --input_dir data/batch/images \
  --output_dir data/batch/masks \
  --output_suffix "" \
  --mask_type rectangles \
  --num_rectangles 3 \
  --min_ratio 0.05 \
  --max_ratio 0.2 \
  --save_preview \
  --preview_dir outputs/batch_masks_preview
```

Tạo free-form mask:

```bash
python scripts/generate_masks.py \
  --input_dir data/batch/images \
  --output_dir data/batch/masks \
  --output_suffix "" \
  --mask_type free_form \
  --num_strokes 8 \
  --max_vertices 8 \
  --max_width 40
```

Quy ước file mask mặc định:

```text
input.jpg -> input_mask.png
```

Nếu dùng mask này cho `run_inference.py`, truyền trực tiếp đường dẫn mask:

```bash
python scripts/run_inference.py \
  --image data/samples/input.jpg \
  --mask data/samples/masks/input_mask.png \
  --output outputs/images/input_result.png
```

Nếu dùng mask này cho `run_batch_inference.py`, ảnh và mask cần cùng filename stem. Khi tạo mask cho batch inference, truyền `--output_suffix ""` để tạo mask khớp tên ảnh:

```text
data/batch/images/input.jpg
data/batch/masks/input.png
```

### 4. Chạy inference cho một ảnh-mask

Lệnh này dùng khi đã có ảnh và mask riêng:

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

Quy ước mask:

- Vùng trắng hoặc giá trị cao là vùng cần inpaint.
- Vùng đen là phần ảnh được giữ lại.
- Mask sẽ được resize và nhị phân hóa trong bước preprocessing.

### 5. Chạy inference cho nhiều ảnh

Lệnh này dùng khi đã có một thư mục ảnh và một thư mục mask tương ứng. Script sẽ load model một lần, sau đó chạy lần lượt từng ảnh.

Quy ước đặt file:

```text
data/batch/images/
|-- 000001.jpg
|-- 000002.jpg
`-- 000003.jpg

data/batch/masks/
|-- 000001.png
|-- 000002.png
`-- 000003.png
```

Ảnh và mask được ghép theo cùng tên file không tính phần mở rộng. Ví dụ `000001.jpg` có thể đi với `000001.png`, `000001.jpg` hoặc `000001.webp`.

Chạy batch inference:

```bash
python scripts/run_batch_inference.py \
  --config configs/default.yaml \
  --image_dir data/batch/images \
  --mask_dir data/batch/masks \
  --output_dir outputs/batch_inference \
  --prompt "a realistic photo" \
  --negative_prompt "blurry, distorted" \
  --save_masked_preview
```

Chạy thử nhanh một số ảnh:

```bash
python scripts/run_batch_inference.py \
  --config configs/default.yaml \
  --image_dir data/batch/images \
  --mask_dir data/batch/masks \
  --output_dir outputs/batch_debug \
  --num_samples 5 \
  --save_masked_preview
```

Kết quả được lưu tại:

```text
outputs/batch_inference/
|-- images/
|   |-- 000001_inpainted.png
|   `-- 000002_inpainted.png
`-- visualizations/
    |-- 000001_masked.png
    `-- 000002_masked.png
```

## Chạy website demo

Demo nằm trong `app/gradio_app.py`. Ứng dụng sẽ lazy-load model trong lần chạy inpainting đầu tiên.

Chạy demo:

```bash
python app/gradio_app.py
```

Mở trình duyệt tại:

```text
http://localhost:7860
```

Luồng sử dụng:

1. Upload ảnh gốc.
2. Tạo mask bằng tab `Random Mask` hoặc vẽ trực tiếp trong tab `Vẽ Mask`.
3. Chỉnh prompt, negative prompt, số step, guidance, seed và kích thước ảnh.
4. Bấm chạy inpainting để sinh kết quả.

Nếu chạy trên máy từ xa, Colab hoặc Kaggle, có thể chỉnh cuối file `app/gradio_app.py`:

```python
app.launch(server_name="0.0.0.0", server_port=7860, share=True)
```

Trong repo hiện tại `share=False`, phù hợp khi chạy local.

## Tham số dòng lệnh quan trọng

### `scripts/run_evaluation.py`

- `--config`: đường dẫn file YAML, mặc định `configs/default.yaml`.
- `--input_dir`: thư mục chứa ảnh test.
- `--output_dir`: thư mục gốc để lưu kết quả.
- `--num_samples`: giới hạn số ảnh cần chạy.
- `--mask_type`: `rectangle`, `rectangles` hoặc `free_form`.
- `--prompt`: prompt cho mô hình diffusion.
- `--negative_prompt`: nội dung không mong muốn.
- `--lpips_device`: thiết bị tính LPIPS, thường là `cuda` hoặc `cpu`.
- `--save_visualizations`: lưu ảnh so sánh original/mask/masked/result.

### `scripts/generate_masks.py`

- `--image`: đường dẫn một ảnh đầu vào.
- `--input_dir`: thư mục chứa nhiều ảnh đầu vào.
- `--output_dir`: thư mục lưu mask, mặc định `outputs/masks`.
- `--output_suffix`: hậu tố tên file mask, mặc định `_mask`; dùng `""` để khớp batch inference.
- `--preview_dir`: thư mục lưu preview nếu bật `--save_preview`.
- `--mask_type`: `rectangle`, `rectangles` hoặc `free_form`.
- `--num_samples`: giới hạn số ảnh khi dùng `--input_dir`.
- `--seed`: seed sinh mask.
- `--save_preview`: lưu ảnh preview vùng bị mask.
- `--mask_ratio`: tỉ lệ mask cho kiểu `rectangle`.
- `--num_rectangles`, `--min_ratio`, `--max_ratio`: tham số cho kiểu `rectangles`.
- `--num_strokes`, `--max_vertices`, `--max_width`: tham số cho kiểu `free_form`.

### `scripts/run_inference.py`

- `--image`: ảnh đầu vào.
- `--mask`: mask đầu vào.
- `--prompt`: prompt tùy chọn.
- `--negative_prompt`: negative prompt tùy chọn.
- `--output`: đường dẫn ảnh kết quả.
- `--save_masked_preview`: lưu ảnh preview vùng bị mask.
- `--masked_preview_output`: đường dẫn preview.

### `scripts/run_batch_inference.py`

- `--image_dir`: thư mục chứa ảnh đầu vào.
- `--mask_dir`: thư mục chứa mask, ghép với ảnh theo cùng filename stem.
- `--output_dir`: thư mục gốc lưu kết quả, mặc định `outputs/batch_inference`.
- `--num_samples`: giới hạn số ảnh cần chạy.
- `--prompt`: prompt tùy chọn.
- `--negative_prompt`: negative prompt tùy chọn.
- `--save_masked_preview`: lưu preview vùng bị mask cho từng ảnh.
- `--fail_fast`: dừng ngay khi một ảnh lỗi.

## Ghi chú khi chạy trên Kaggle hoặc Colab

- Bật GPU trước khi chạy.
- Cài dependencies bằng `pip install -r requirements.txt`.
- Nếu dataset được mount ở đường dẫn khác, truyền trực tiếp vào `--input_dir`.
- Model Hugging Face sẽ được tải trong lần chạy đầu tiên, cần kết nối mạng.
- Nên chạy `--num_samples 5` trước để kiểm tra đường dẫn, CUDA và output.

Ví dụ trên Kaggle khi dataset nằm trong `/kaggle/input`:

```bash
python scripts/run_evaluation.py \
  --config configs/default.yaml \
  --input_dir /kaggle/input/celebamaskhq/CelebAMask-HQ/CelebA-HQ-img \
  --output_dir /kaggle/working/outputs/celeba_hq_eval \
  --num_samples 100 \
  --mask_type rectangle \
  --lpips_device cuda \
  --save_visualizations
```

## Lỗi thường gặp

### Không tìm thấy ảnh trong `input_dir`

Kiểm tra thư mục truyền vào có chứa trực tiếp file ảnh không. Script không quét đệ quy qua thư mục con.

### CUDA không khả dụng

Nếu thấy cảnh báo fallback về CPU, kiểm tra lại GPU/CUDA hoặc đổi config:

```yaml
model:
  device: "cpu"
  torch_dtype: "float32"
```

### LPIPS lỗi trên GPU

Thử chuyển LPIPS sang CPU:

```bash
python scripts/run_evaluation.py --lpips_device cpu ...
```

### Hết bộ nhớ GPU

Giảm các tham số:

- `inference.image_size`: từ `512` xuống `256`.
- `inference.num_inference_steps`: từ `30` xuống `20`.
- `--num_samples`: chạy ít ảnh hơn cho mỗi lần.

## Kết quả đầu ra

Sau evaluation, các file quan trọng là:

- `outputs/<run_name>/metrics/metrics.csv`: bảng PSNR, SSIM, LPIPS, runtime.
- `outputs/<run_name>/images/*_inpainted.png`: ảnh kết quả.
- `outputs/<run_name>/visualizations/*_comparison.png`: ảnh ghép để minh họa định tính.

Các kết quả này có thể dùng để lập bảng so sánh định lượng và chọn ví dụ trực quan cho báo cáo đồ án.

## License

Repo sử dụng license trong file `LICENSE`. Dataset CelebA-HQ/CelebAMask-HQ và model Hugging Face có điều khoản sử dụng riêng, cần tuân thủ khi chia sẻ hoặc công bố kết quả.
