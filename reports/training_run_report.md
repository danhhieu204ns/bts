# Training Run Report

## Toolchain Attempt

Requested next step was full 3DGS training. I attempted to unlock the CUDA extension path first:

- Docker Desktop was started successfully.
- CUDA container image `nvidia/cuda:12.6.3-base-ubuntu22.04` was pulled.
- Docker GPU containers hung at create/start, so Linux-container training was not usable.
- Visual Studio 2022 Build Tools bootstrapper was tried, but no VS2022 instance appeared in `vswhere` afterward.
- Windows native CUDA extension builds remain blocked by CUDA/Visual Studio mismatch.

Because of that, I implemented and trained a runnable 3D fallback baseline that does not require custom CUDA extensions.

## Implemented Trainable 3D Baseline

Script:

```text
src/bts_baseline/point_splat_baseline.py
scripts/run_point_splat_baseline.ps1
```

Method:

1. Read COLMAP `points3D.bin`, `images.bin`, `cameras.bin`.
2. Fit point colors using train-image observations only.
3. Render test poses with vectorized z-buffer point splatting.
4. Composite splats over nearest-pose RGB background.
5. Save exact requested filenames and dimensions.

Best tested public config:

```text
SplatRadius = 3
Alpha = 0.5
Background = nearest
```

## Public Metrics

| Method | PSNR | SSIM | LPIPS | Score@30 | Score@35 | Score@40 |
|---|---:|---:|---:|---:|---:|---:|
| nearest_pose | 10.3889 | 0.1364 | 0.6042 | 0.3032 | 0.2883 | 0.2772 |
| point_splat alpha=0.85 r=3 | 12.2347 | 0.1821 | 0.6384 | 0.3216 | 0.3041 | 0.2910 |
| point_splat alpha=0.50 r=3 | 11.9955 | 0.1710 | 0.6118 | 0.3265 | 0.3094 | 0.2965 |

The official `PSNR_max` is not known, so Score@30/35/40 are local estimates.

## Outputs

Public:

```text
outputs/point_splat_public_a05_r3
reports/point_splat_public_a05_r3_metrics.csv
```

Private:

```text
outputs/point_splat_private_a05_r3
outputs/point_splat_models_private_a05_r3
submissions/point_splat_private_a05_r3.zip
```

Private validation:

```text
8 scenes, 434 images, OK
zip size: 319.52 MB
```
