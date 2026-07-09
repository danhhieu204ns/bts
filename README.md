# BTS Digital Twin - 3DGS trên Ubuntu

Repo này dùng cho pipeline Novel View Synthesis / Digital Twin với official 3D Gaussian Splatting (3DGS): prepare data, train, render, evaluate public set và đóng gói submission cho private set.

`README.md` là tài liệu hướng dẫn chính của repo. Nội dung từ `docs.md` cũ đã được gộp vào đây để chỉ còn một nguồn tham chiếu.

Toàn bộ hướng dẫn bên dưới giả định chạy trên Ubuntu và đứng tại thư mục gốc repo:

```bash
cd /home/jovyan/bts
```

## 1. Cấu trúc dữ liệu

Data contest được đặt trong `phase1/`:

```text
phase1/
  public_set/
    hcm0031/
    hcm0034/
    HCM0181/
    HCM0193/
    HCM0204/
  private_set1/
    HCM0249/
    HCM0254/
    HCM0276/
    HCM1439/
    HNI0131/
    HNI0265/
    HNI0366/
    HNI0437/
```

Mỗi scene có cấu trúc:

```text
train/images/
train/sparse/0/{cameras.bin,images.bin,points3D.bin}
test/test_poses.csv
test/images/              # chỉ có ở public set để evaluate local
```

Lưu ý: `train/sparse/0/images.bin` của contest có thể chứa cả train và test pose. Repo này có bước prepare riêng để lọc lại chỉ còn train images trước khi train với official 3DGS.

## 2. Cài môi trường Ubuntu

Cài gói hệ thống:

```bash
apt update
apt install -y \
  git \
  curl \
  build-essential \
  cmake \
  ninja-build \
  python3.11 \
  python3.11-venv \
  python3-pip
```

Tạo virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Cài dependency core:

```bash
python -m pip install -r requirements.txt
```

Cài PyTorch CUDA phù hợp với driver/GPU. Ví dụ CUDA 12.6:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Cài dependency phụ cho 3DGS helper script:

```bash
python -m pip install -r requirements-3dgs.txt
```

Cài LPIPS nếu muốn tính local final score trên public set:

```bash
python -m pip install -r requirements-lpips.txt
```

Kiểm tra Python, Torch và GPU:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
```

## 3. Cài official 3DGS

Clone repo chính thức vào `external/gaussian-splatting`:

```bash
mkdir -p external
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting external/gaussian-splatting
git -C external/gaussian-splatting submodule update --init --recursive
```

Trên Ubuntu, cần CUDA Toolkit tương thích với `torch.version.cuda`. Sau đó build extension:

```bash
source .venv/bin/activate
./scripts/build_3dgs_extensions.sh
```

Nếu máy có nhiều loại GPU, có thể set kiến trúc CUDA trước khi build. Ví dụ L40S:

```bash
export TORCH_CUDA_ARCH_LIST="8.9"
```

Một vài mốc thường gặp:

```text
8.6  RTX 30xx / A10
8.9  RTX 40xx / L4 / L40S
9.0  H100
```

## 4. Prepare data cho 3DGS

Các script shell trong repo sẽ tự tìm `.venv/bin/python`. Nếu muốn ép dùng đúng interpreter, export biến môi trường:

```bash
export PYTHON="$(pwd)/.venv/bin/python"
```

Prepare toàn bộ public set:

```bash
./scripts/prepare_3dgs_scene.sh \
  --data-root phase1/public_set \
  --out-root prepared/3dgs_public
```

Prepare toàn bộ private set:

```bash
./scripts/prepare_3dgs_scene.sh \
  --data-root phase1/private_set1 \
  --out-root prepared/3dgs_private
```

Nếu prepare trên một máy rồi train trên máy khác, cần copy cả repo, `phase1/` và `prepared/` sang máy train. Bước render/evaluate cần giữ nguyên `test/test_poses.csv` trong `phase1/*`.

Prepare một scene để smoke test:

```bash
./scripts/prepare_3dgs_scene.sh \
  --data-root phase1/public_set \
  --scene hcm0031 \
  --out-root prepared/3dgs_public
```

Output mỗi scene:

```text
prepared/3dgs_public/hcm0031/
  images/
  sparse/0/
    cameras.bin
    cameras_original.bin
    images.bin
    points3D.bin
```

Mặc định script chuyển camera sang `PINHOLE` approximation vì official 3DGS loader đọc ổn định hơn với `PINHOLE` / `SIMPLE_PINHOLE`.

## 5. Preset train 3DGS

Preset được định nghĩa trong `scripts/_3dgs_train_profiles.sh`.

| Preset | Khi dùng | Resolution | Iterations | Optimizer | Antialiasing | Ghi chú |
|---|---|---:|---:|---|---|---|
| `local-r2-7k` | Smoke test hoặc GPU local nhỏ | `2` | `7000` | `default` | off | Nhanh để kiểm tra pipeline. |
| `l40s-fast` | Chạy nhanh trên L40S | `1` | `15000` | `sparse_adam` | on | Cân bằng tốc độ và chất lượng. |
| `l40s-quality` | Preset khuyến nghị đầu tiên | `1` | `30000` | `sparse_adam` | on | Mode benchmark chính. |
| `l40s-bts-quality` | Ưu tiên chi tiết mảnh | `1` | `30000` | `sparse_adam` | off | Densify dày hơn, chậm hơn. |
| `custom` | Tự override tham số | tùy chọn | tùy chọn | tùy chọn | tùy chọn | Dùng khi sweep tham số. |

Nếu CUDA extension không hỗ trợ `sparse_adam`, truyền thêm:

```text
--optimizer-type default
```

Các script sẽ tự fallback về `default` nếu phát hiện môi trường hiện tại không dùng được accelerated rasterizer, nhưng chạy chậm hơn so với `sparse_adam`.

## 6. Train một scene

Smoke test:

```bash
./scripts/train_3dgs_scene.sh \
  --prepared-scene prepared/3dgs_public/hcm0031 \
  --model-dir outputs/3dgs_models_public_smoke/hcm0031 \
  --preset local-r2-7k
```

Train chất lượng cao:

```bash
./scripts/train_3dgs_scene.sh \
  --prepared-scene prepared/3dgs_public/hcm0031 \
  --model-dir outputs/3dgs_models_public_quality/hcm0031 \
  --preset l40s-quality
```

Chọn GPU cụ thể:

```bash
./scripts/train_3dgs_scene.sh \
  --prepared-scene prepared/3dgs_public/hcm0031 \
  --model-dir outputs/3dgs_models_public_quality/hcm0031 \
  --preset l40s-quality \
  --cuda-visible-devices 0
```

Output chính:

```text
outputs/3dgs_models_public_quality/hcm0031/
  point_cloud/iteration_30000/point_cloud.ply
  logs/train_30000_r1_l40s-quality.log
```

## 7. Train batch nhiều scene

Train toàn bộ public set:

```bash
./scripts/train_3dgs_batch.sh \
  --prepared-root prepared/3dgs_public \
  --model-root outputs/3dgs_models_public_quality \
  --preset l40s-quality \
  --force
```

Train một nhóm scene:

```bash
./scripts/train_3dgs_batch.sh \
  --prepared-root prepared/3dgs_public \
  --model-root outputs/3dgs_models_public_quality \
  --preset l40s-quality \
  --scene hcm0031,hcm0034
```

Train BTS-sharpness profile:

```bash
./scripts/train_3dgs_batch.sh \
  --prepared-root prepared/3dgs_public \
  --model-root outputs/3dgs_models_public_bts_quality \
  --preset l40s-bts-quality \
  --force
```

Train local trên GPU 6 GB ở half resolution:

```bash
./scripts/train_3dgs_batch.sh \
  --prepared-root prepared/3dgs_public \
  --model-root outputs/3dgs_models_public_local_bts_r2 \
  --preset l40s-bts-quality \
  --resolution 2 \
  --force
```

Nếu buộc phải train full resolution trên GPU nhỏ, có thể tắt densification và bỏ train-time eval để giảm áp lực VRAM:

```bash
./scripts/train_3dgs_batch.sh \
  --prepared-root prepared/3dgs_public \
  --model-root outputs/3dgs_models_public_local_r1 \
  --preset l40s-bts-quality \
  --resolution 1 \
  --optimizer-type default \
  --densify-until-iter 0 \
  --skip-train-eval \
  --force
```

Nếu không truyền `-Force`, script sẽ tự skip scene đã có `point_cloud/iteration_<iterations>/point_cloud.ply`.

## 8. Train custom và resume

Ví dụ train 20k iterations, full resolution, optimizer mặc định:

```bash
./scripts/train_3dgs_batch.sh \
  --prepared-root prepared/3dgs_public \
  --model-root outputs/3dgs_models_public_custom20k \
  --preset custom \
  --iterations 20000 \
  --resolution 1 \
  --optimizer-type default \
  --antialiasing \
  --save-iterations 10000,20000 \
  --test-iterations 20000 \
  --checkpoint-iterations 20000 \
  --densify-until-iter 12000 \
  --densify-grad-threshold 0.0002 \
  --densification-interval 100 \
  --opacity-reset-interval 3000
```

Truyền raw args cho official `train.py`:

```bash
./scripts/train_3dgs_scene.sh \
  --prepared-scene prepared/3dgs_public/hcm0031 \
  --model-dir outputs/3dgs_models_public_custom/hcm0031 \
  --preset custom \
  --iterations 30000 \
  --extra-arg --lambda_dssim \
  --extra-arg 0.15
```

Resume từ checkpoint:

```bash
./scripts/train_3dgs_scene.sh \
  --prepared-scene prepared/3dgs_public/hcm0031 \
  --model-dir outputs/3dgs_models_public_quality/hcm0031 \
  --preset l40s-quality \
  --start-checkpoint outputs/3dgs_models_public_quality/hcm0031/chkpnt15000.pth
```

## 9. Render, validate và evaluate public

Render public:

```bash
./scripts/render_3dgs_submission.sh \
  --data-root phase1/public_set \
  --model-root outputs/3dgs_models_public_quality \
  --out-dir outputs/3dgs_public_quality \
  --antialiasing
```

Validate format output:

```bash
./scripts/validate_submission.sh \
  --data-root phase1/public_set \
  --pred-dir outputs/3dgs_public_quality
```

Evaluate public:

```bash
./scripts/evaluate_public.sh \
  --data-root phase1/public_set \
  --pred-dir outputs/3dgs_public_quality \
  --lpips on \
  --output-csv reports/3dgs_public_quality_metrics.csv
```

Nếu model train bằng `l40s-bts-quality`, nên thử render thêm với splat nhỏ hơn:

```bash
./scripts/render_3dgs_submission.sh \
  --data-root phase1/public_set \
  --model-root outputs/3dgs_models_public_bts_quality \
  --out-dir outputs/3dgs_public_bts_quality_scale09 \
  --scaling-modifier 0.9
```

`--scaling-modifier 0.8` hoặc `0.9` đôi khi cải thiện chi tiết mảnh, nhưng cần evaluate lại để tránh thủng hình.

## 10. Train private và đóng gói submission

Train private:

```bash
./scripts/train_3dgs_batch.sh \
  --prepared-root prepared/3dgs_private \
  --model-root outputs/3dgs_models_private_quality \
  --preset l40s-quality \
  --force
```

Render private và tạo zip:

```bash
./scripts/render_3dgs_submission.sh \
  --data-root phase1/private_set1 \
  --model-root outputs/3dgs_models_private_quality \
  --out-dir outputs/3dgs_private_quality \
  --antialiasing \
  --zip submissions/3dgs_private_quality.zip
```

Validate private output:

```bash
./scripts/validate_submission.sh \
  --data-root phase1/private_set1 \
  --pred-dir outputs/3dgs_private_quality
```

## 11. Workflow khuyến nghị

1. Cài Python, Torch CUDA và official 3DGS.
2. Prepare `public_set` và `private_set1`.
3. Smoke test `hcm0031` bằng `local-r2-7k`.
4. Render/evaluate smoke test để xác nhận pipeline đúng.
5. Train toàn bộ public bằng `l40s-quality`.
6. Nếu còn thời gian, thử `l40s-bts-quality` trên public.
7. So sánh metric trong `reports/*.csv`.
8. Train private bằng preset thắng trên public.
9. Render private, validate và zip submission.

## 12. Tóm tắt training logs

`logs/` hiện chứa log train thô cho các run `l40s-quality` đã hoàn tất. Bảng dưới gom các mốc chính vào một chỗ để không cần đọc từng file log.

### Public runs

| Scene | Init points | Iter | Train L1 | Train PSNR |
|---|---:|---:|---:|---:|
| `HCM0181` | 224928 | 30000 | 0.036418 | 24.1099 |
| `HCM0193` | 214593 | 30000 | 0.037278 | 24.4261 |
| `HCM0204` | 251715 | 30000 | 0.029069 | 25.9170 |
| `hcm0031` | 211262 | 30000 | 0.033580 | 25.4861 |
| `hcm0034` | 181413 | 30000 | 0.036673 | 24.0958 |

### Private runs

| Scene | Init points | Iter | Train L1 | Train PSNR |
|---|---:|---:|---:|---:|
| `HCM0249` | 165726 | 30000 | 0.040433 | 23.2628 |
| `HCM0254` | 176590 | 30000 | 0.039068 | 23.7500 |
| `HCM0276` | 220581 | 30000 | 0.041036 | 23.2403 |
| `HCM1439` | 48347 | 30000 | 0.031182 | 25.7154 |
| `HNI0131` | 147464 | 30000 | 0.064013 | 19.8893 |
| `HNI0265` | 83992 | 30000 | 0.070282 | 19.6725 |
| `HNI0366` | 145502 | 30000 | 0.032694 | 25.4328 |
| `HNI0437` | 115026 | 30000 | 0.033225 | 25.2412 |

## 13. Debug nhanh

`official 3DGS repo not found`

: Clone repo vào `external/gaussian-splatting` và init submodule.

`torch.cuda.is_available() == False`

: Kiểm tra lại NVIDIA driver, build PyTorch CUDA và quyền truy cập GPU.

Build extension fail

: Đảm bảo CUDA Toolkit khớp với `torch.version.cuda`, đã cài `build-essential`, và đã kích hoạt `.venv`.

Train lỗi ở `sparse_adam`

: Chạy lại với `--optimizer-type default`.

Hết VRAM

: Dùng `local-r2-7k`, hoặc override `--resolution 2`, giảm iterations, tắt antialiasing, hoặc train từng scene.

Render thiếu file hoặc sai kích thước

: Luôn chạy `validate_submission.sh` sau render.

Metric public không có LPIPS

: Cài `requirements-lpips.txt` và chạy `evaluate_public.sh --lpips on`.

## 14. Baseline phụ

Khi 3DGS chưa build xong, có thể chạy baseline để kiểm tra format và metric:

```bash
./scripts/run_nearest_baseline.sh \
  --data-root phase1/public_set \
  --out-dir outputs/nearest_public

./scripts/run_point_splat_baseline.sh \
  --data-root phase1/public_set \
  --out-dir outputs/point_splat_public \
  --model-dir outputs/point_splat_models_public
```

Các baseline này không thay thế 3DGS, nhưng rất hữu ích để kiểm tra data, validation, packaging và flow đánh giá trước khi chạy train GPU dài.
