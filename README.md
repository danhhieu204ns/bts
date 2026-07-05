# BTS Digital Twin - hướng dẫn train 3DGS

Repo này phục vụ bài toán Novel View Synthesis / Digital Twin. Pipeline chính trong
repo là train và render bằng official 3D Gaussian Splatting (3DGS), sau đó validate,
evaluate trên public set và đóng gói submission cho private set.

Các lệnh bên dưới mặc định chạy từ thư mục gốc repo:

```powershell
cd E:\WORKSPACE\bts_digital_twin
```

## 1. Cấu trúc data cần có

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
test/images/              # chỉ có ở public set, dùng để evaluate local
```

Lưu ý quan trọng: `train/sparse/0/images.bin` của contest có thể chứa cả train và
test pose. Vì official 3DGS loader mở ảnh theo toàn bộ `images.bin`, repo này có
bước prepare riêng để lọc lại chỉ còn train images trước khi train.

## 2. Cài môi trường Python

Tạo virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

Cài dependency core để đọc data, validate và evaluate:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Cài PyTorch CUDA phù hợp với máy train trước khi cài các package phụ. Ví dụ nếu
dùng Torch CUDA 12.6:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Cài dependency phụ cho 3DGS helper script:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-3dgs.txt
```

Cài LPIPS nếu muốn tính local final score đầy đủ trên public set:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lpips.txt
```

Kiểm tra Python, Torch và GPU:

```powershell
@"
import torch
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
"@ | .\.venv\Scripts\python.exe -
```

## 3. Cài official 3DGS

Official repo nằm ở `external/gaussian-splatting`. Nếu thư mục này chưa có:

```powershell
New-Item -ItemType Directory -Force external | Out-Null
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting external/gaussian-splatting
git -C external/gaussian-splatting submodule update --init --recursive
```

3DGS cần build CUDA extension. Trên Windows, cấu hình dễ ổn định nhất là:

- GPU NVIDIA có CUDA.
- PyTorch CUDA runtime khớp với CUDA Toolkit để build extension.
- Visual Studio 2022 Build Tools có workload C++.
- NVIDIA CUDA Toolkit chỉ là một phần; vẫn cần VS Build Tools để có `cl.exe` và `VsDevCmd.bat`.
- CUDA Toolkit tương ứng, ví dụ `v12.6` nếu Torch là `cu126`.

Build extension trên Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_3dgs_extensions_windows.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -CudaPath "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" `
  -ForceReinstall
```

Script sẽ tự detect `TORCH_CUDA_ARCH_LIST` từ GPU đầu tiên. Chỉ truyền tay nếu cần override,
ví dụ RTX 3050 / RTX 30xx dùng `-TorchCudaArchList "8.6"`, RTX 40xx dùng `"8.9"`.

Nếu gặp lỗi `running scripts is disabled`, dùng đúng mẫu lệnh `powershell -ExecutionPolicy Bypass -File`
ở trên thay vì chạy trực tiếp file `.ps1`.

Nếu script báo Python 3.13/3.14, tạo lại `.venv` bằng Python 3.11 hoặc 3.12 rồi cài lại Torch `cu126`.
Nếu script báo chỉ thấy Visual Studio Build Tools 2026/18.x, cài Visual Studio 2022 Build Tools với workload
`Desktop development with C++`. CUDA 12.6 có thể crash `cudafe++` với MSVC quá mới.

`TorchCudaArchList` phụ thuộc GPU:

```text
8.6  RTX 30xx / A10
8.9  RTX 40xx / L4 / L40S
9.0  H100
```

Nếu train trên Linux/L40S, có thể dùng `pwsh` để chạy các script `.ps1`. Trước khi
chạy, trỏ Python một lần:

```powershell
$env:PYTHON = ".venv/bin/python"
```

Hoặc truyền trực tiếp `-Python .venv/bin/python` vào từng script.

## 4. Prepare data cho 3DGS

Prepare toàn bộ public set:

```powershell
.\scripts\prepare_3dgs_scene.ps1 `
  -DataRoot phase1\public_set `
  -OutRoot prepared\3dgs_public
```

Prepare toàn bộ private set:

```powershell
.\scripts\prepare_3dgs_scene.ps1 `
  -DataRoot phase1\private_set1 `
  -OutRoot prepared\3dgs_private
```

Prepare một scene để smoke test nhanh:

```powershell
.\scripts\prepare_3dgs_scene.ps1 `
  -DataRoot phase1\public_set `
  -Scene hcm0031 `
  -OutRoot prepared\3dgs_public
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

Mặc định script chuyển camera sang `PINHOLE` approximation vì official 3DGS loader
nhận tốt `PINHOLE` / `SIMPLE_PINHOLE`. Bản camera gốc được giữ ở
`cameras_original.bin`.

## 5. Các mode train 3DGS

Preset được định nghĩa trong `scripts/_3dgs_train_profiles.ps1`.

| Preset | Khi dùng | Resolution | Iterations | Optimizer | Antialiasing | Ghi chú |
|---|---|---:|---:|---|---|---|
| `local-r2-7k` | Smoke test/local GPU vừa phải | `2` | `7000` | `default` | off | Nhanh, dùng để kiểm tra pipeline. |
| `l40s-fast` | Chạy nhanh trên L40S | `1` | `15000` | `sparse_adam` | on | Cân bằng tốc độ và chất lượng. |
| `l40s-quality` | Mode khuyến nghị đầu tiên | `1` | `30000` | `sparse_adam` | on | Chạy full public/private để lấy benchmark chính. |
| `l40s-bts-quality` | Ưu tiên chi tiết mảnh BTS | `1` | `30000` | `sparse_adam` | off | Densify dày hơn, chậm hơn, thử sau `l40s-quality`. |
| `custom` | Tự override tham số | tùy chọn | tùy chọn | tùy chọn | tùy chọn | Dùng khi cần sweep tham số. |

Nếu CUDA extension không hỗ trợ `sparse_adam`, truyền thêm:

```powershell
-OptimizerType default
```

## 6. Train một scene

Smoke test một scene public:

```powershell
.\scripts\train_3dgs_scene.ps1 `
  -PreparedScene prepared\3dgs_public\hcm0031 `
  -ModelDir outputs\3dgs_models_public_smoke\hcm0031 `
  -Preset local-r2-7k
```

Train một scene chất lượng cao:

```powershell
.\scripts\train_3dgs_scene.ps1 `
  -PreparedScene prepared\3dgs_public\hcm0031 `
  -ModelDir outputs\3dgs_models_public_quality\hcm0031 `
  -Preset l40s-quality
```

Train trên GPU cụ thể:

```powershell
.\scripts\train_3dgs_scene.ps1 `
  -PreparedScene prepared\3dgs_public\hcm0031 `
  -ModelDir outputs\3dgs_models_public_quality\hcm0031 `
  -Preset l40s-quality `
  -CudaVisibleDevices 0
```

Output model chính:

```text
outputs/3dgs_models_public_quality/hcm0031/
  point_cloud/iteration_30000/point_cloud.ply
  logs/train_30000_r1_l40s-quality.log
```

## 7. Train batch nhiều scene

Train toàn bộ public set bằng mode quality:

```powershell
.\scripts\train_3dgs_batch.ps1 `
  -PreparedRoot prepared\3dgs_public `
  -ModelRoot outputs\3dgs_models_public_quality `
  -Preset l40s-quality `
  -Force
```

Train chỉ một vài scene trong prepared root:

```powershell
.\scripts\train_3dgs_batch.ps1 `
  -PreparedRoot prepared\3dgs_public `
  -ModelRoot outputs\3dgs_models_public_quality `
  -Preset l40s-quality `
  -Scene hcm0031,hcm0034
```

Train BTS-sharpness profile:

```powershell
.\scripts\train_3dgs_batch.ps1 `
  -PreparedRoot prepared\3dgs_public `
  -ModelRoot outputs\3dgs_models_public_bts_quality `
  -Preset l40s-bts-quality `
  -Force
```

Nếu không truyền `-Force`, script sẽ skip scene đã có:

```text
point_cloud/iteration_<iterations>/point_cloud.ply
```

## 8. Train custom

Ví dụ train full resolution 20k iterations, dùng optimizer default:

```powershell
.\scripts\train_3dgs_batch.ps1 `
  -PreparedRoot prepared\3dgs_public `
  -ModelRoot outputs\3dgs_models_public_custom20k `
  -Preset custom `
  -Iterations 20000 `
  -Resolution 1 `
  -OptimizerType default `
  -Antialiasing `
  -SaveIterations 10000,20000 `
  -TestIterations 20000 `
  -CheckpointIterations 20000 `
  -DensifyUntilIter 12000 `
  -DensifyGradThreshold 0.0002 `
  -DensificationInterval 100 `
  -OpacityResetInterval 3000
```

Truyền tham số raw của official `train.py` qua `-ExtraArgs`:

```powershell
.\scripts\train_3dgs_scene.ps1 `
  -PreparedScene prepared\3dgs_public\hcm0031 `
  -ModelDir outputs\3dgs_models_public_custom\hcm0031 `
  -Preset custom `
  -Iterations 30000 `
  -ExtraArgs "--lambda_dssim","0.15"
```

Resume từ checkpoint:

```powershell
.\scripts\train_3dgs_scene.ps1 `
  -PreparedScene prepared\3dgs_public\hcm0031 `
  -ModelDir outputs\3dgs_models_public_quality\hcm0031 `
  -Preset l40s-quality `
  -StartCheckpoint outputs\3dgs_models_public_quality\hcm0031\chkpnt15000.pth
```

## 9. Render public để evaluate

Render bằng model vừa train:

```powershell
.\scripts\render_3dgs_submission.ps1 `
  -DataRoot phase1\public_set `
  -ModelRoot outputs\3dgs_models_public_quality `
  -OutDir outputs\3dgs_public_quality `
  -Antialiasing
```

Validate format output:

```powershell
.\scripts\validate_submission.ps1 `
  -DataRoot phase1\public_set `
  -PredDir outputs\3dgs_public_quality
```

Evaluate public:

```powershell
.\scripts\evaluate_public.ps1 `
  -DataRoot phase1\public_set `
  -PredDir outputs\3dgs_public_quality `
  -Lpips on `
  -OutputCsv reports\3dgs_public_quality_metrics.csv
```

Nếu model train bằng `l40s-quality`, render với `-Antialiasing` trước. Nếu train
bằng `l40s-bts-quality`, nên render không antialiasing trước, sau đó thử thêm:

```powershell
.\scripts\render_3dgs_submission.ps1 `
  -DataRoot phase1\public_set `
  -ModelRoot outputs\3dgs_models_public_bts_quality `
  -OutDir outputs\3dgs_public_bts_quality_scale09 `
  -ScalingModifier 0.9
```

`-ScalingModifier 0.8` hoặc `0.9` đôi khi giúp splat mảnh hơn, nhưng cần evaluate
public để kiểm tra có bị thủng hình hay không.

## 10. Train private và đóng gói submission

Sau khi chọn preset tốt nhất trên public, train private:

```powershell
.\scripts\train_3dgs_batch.ps1 `
  -PreparedRoot prepared\3dgs_private `
  -ModelRoot outputs\3dgs_models_private_quality `
  -Preset l40s-quality `
  -Force
```

Render private và tạo zip:

```powershell
.\scripts\render_3dgs_submission.ps1 `
  -DataRoot phase1\private_set1 `
  -ModelRoot outputs\3dgs_models_private_quality `
  -OutDir outputs\3dgs_private_quality `
  -Antialiasing `
  -ZipPath submissions\3dgs_private_quality.zip
```

Validate private output:

```powershell
.\scripts\validate_submission.ps1 `
  -DataRoot phase1\private_set1 `
  -PredDir outputs\3dgs_private_quality
```

Zip submission có cấu trúc:

```text
3dgs_private_quality.zip
  HCM0249/
    ...
  HCM0254/
    ...
```

## 11. Workflow khuyến nghị

1. Cài Python, Torch CUDA, official 3DGS và build CUDA extension.
2. Prepare public/private:

```powershell
.\scripts\prepare_3dgs_scene.ps1 -DataRoot phase1\public_set -OutRoot prepared\3dgs_public
.\scripts\prepare_3dgs_scene.ps1 -DataRoot phase1\private_set1 -OutRoot prepared\3dgs_private
```

3. Smoke test `hcm0031` bằng `local-r2-7k`.
4. Render/evaluate smoke test để chắc chắn pose, camera, output size đúng.
5. Train toàn bộ public bằng `l40s-quality`.
6. Thử `l40s-bts-quality` trên public nếu còn thời gian/GPU.
7. So sánh metric trong `reports/*.csv`.
8. Train private bằng preset thắng trên public.
9. Render private, validate, zip submission.

## 12. Debug nhanh

`official 3DGS repo not found`

: Clone repo vào `external/gaussian-splatting` và init submodule.

`torch.cuda.is_available() == False`

: Cài sai PyTorch build, driver CUDA chưa đúng, hoặc máy train không thấy GPU.

Build extension fail vì CUDA / Torch / Visual Studio lệch version

: Dùng CUDA Toolkit khớp với `torch.version.cuda`. Với Windows, ưu tiên VS 2022
Build Tools cho CUDA 12.x. Chạy lại `scripts/build_3dgs_extensions_windows.ps1`.

Train lỗi ở `sparse_adam`

: Dùng `-OptimizerType default`. Chạy sẽ chậm hơn nhưng ít phụ thuộc accelerated
rasterizer hơn.

Hết VRAM

: Dùng `-Preset local-r2-7k`, hoặc override `-Resolution 2`, giảm iterations,
train từng scene bằng `train_3dgs_scene.ps1`, hoặc tắt antialiasing.

Render thiếu file hoặc sai kích thước

: Luôn chạy `validate_submission.ps1`. Script sẽ báo scene/file nào thiếu hoặc
ảnh nào sai size so với `test/test_poses.csv`.

Metric public không có LPIPS

: Cài `requirements-lpips.txt` và chạy `evaluate_public.ps1 -Lpips on`.

## 13. Baseline phụ

Repo vẫn có baseline để kiểm tra format khi 3DGS toolchain chưa build được:

```powershell
.\scripts\run_nearest_baseline.ps1 -DataRoot phase1\public_set -OutDir outputs\nearest_public
.\scripts\run_point_splat_baseline.ps1 -DataRoot phase1\public_set -OutDir outputs\point_splat_public -ModelDir outputs\point_splat_models_public
```

Các baseline này không thay thế 3DGS, nhưng hữu ích để kiểm tra data, validation,
packaging và metric trước khi chạy train GPU dài.
