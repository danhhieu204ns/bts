# L40S 3DGS training runbook

This repo now has GPU-sized 3DGS training presets in `scripts/train_3dgs_batch.ps1`
and `scripts/train_3dgs_scene.ps1`.

## Presets

- `local-r2-7k`: reproduces the previous local run, `-r 2`, 7000 iterations.
- `l40s-fast`: full resolution, 15000 iterations, sparse Adam, antialiasing on.
- `l40s-quality`: full resolution, 30000 iterations, sparse Adam, antialiasing on.
- `l40s-bts-quality`: full resolution, 30000 iterations, denser foreground-friendly
  settings for thin BTS structures. This is slower and can create more Gaussians.

`l40s-quality` is the recommended first remote run. If L40S VRAM is comfortable,
run `l40s-bts-quality` on public `hcm0031` and compare the `0059/0060/0061` cases.

## Prepare data on this machine

```powershell
.\scripts\prepare_3dgs_scene.ps1 `
  -DataRoot phase1\public_set `
  -OutRoot prepared\3dgs_public

.\scripts\prepare_3dgs_scene.ps1 `
  -DataRoot phase1\private_set1 `
  -OutRoot prepared\3dgs_private
```

Copy the repo, `prepared/`, and `phase1/` to the L40S machine. The render step
needs `phase1/*/test/test_poses.csv`.

On Linux with `pwsh`, either export Python once:

```powershell
$env:PYTHON = ".venv/bin/python"
```

or pass `-Python .venv/bin/python` to the scripts below. On Windows the scripts
auto-detect `.venv\Scripts\python.exe`. On Linux, use `./scripts/...` paths
instead of `.\scripts\...` in the examples.

## Train on L40S

PowerShell or `pwsh`:

```powershell
.\scripts\train_3dgs_batch.ps1 `
  -PreparedRoot prepared\3dgs_public `
  -ModelRoot outputs\3dgs_models_public_l40s_quality `
  -Preset l40s-quality `
  -Force
```

For the BTS-sharpness profile:

```powershell
.\scripts\train_3dgs_batch.ps1 `
  -PreparedRoot prepared\3dgs_public `
  -ModelRoot outputs\3dgs_models_public_l40s_bts `
  -Preset l40s-bts-quality `
  -Force
```

On a local RTX 3050 / 6 GB GPU, keep the BTS profile at half resolution:

```powershell
.\scripts\train_3dgs_batch.ps1 `
  -PreparedRoot prepared\3dgs_public `
  -ModelRoot outputs\3dgs_models_public_local_bts_r2 `
  -Preset l40s-bts-quality `
  -Resolution 2 `
  -Force
```

If you specifically want full resolution on a 6 GB GPU, disable densification
and skip train-time evaluation. Render/evaluate after training instead:

```powershell
.\scripts\train_3dgs_batch.ps1 `
  -PreparedRoot prepared\3dgs_public `
  -ModelRoot outputs\3dgs_models_public_local_r1 `
  -Preset l40s-bts-quality `
  -Resolution 1 `
  -OptimizerType default `
  -DensifyUntilIter 0 `
  -SkipTrainEval `
  -Force
```

If the accelerated rasterizer is not installed, use:

```powershell
-OptimizerType default
```

but expect slower training. The scripts also detect this case and fall back to
`default` automatically before launching training. The L40S presets default to
`sparse_adam` because that is the right speed target for a large GPU with the
accelerated 3DGS extensions.

## Render and evaluate public

Render with antialiasing if the model was trained with `l40s-quality`:

```powershell
.\scripts\render_3dgs_submission.ps1 `
  -DataRoot phase1\public_set `
  -ModelRoot outputs\3dgs_models_public_l40s_quality `
  -OutDir outputs\3dgs_public_l40s_quality `
  -Antialiasing

.\scripts\evaluate_public.ps1 `
  -DataRoot phase1\public_set `
  -PredDir outputs\3dgs_public_l40s_quality `
  -Lpips on `
  -OutputCsv reports\3dgs_public_l40s_quality_metrics.csv
```

For `l40s-bts-quality`, render without `-Antialiasing` first. Then optionally test
`-ScalingModifier 0.9` and `0.8` on public to see whether thinner splats improve
BTS detail without adding holes.

## Train private and package

After selecting the best public preset:

```powershell
.\scripts\train_3dgs_batch.ps1 `
  -PreparedRoot prepared\3dgs_private `
  -ModelRoot outputs\3dgs_models_private_l40s_quality `
  -Preset l40s-quality `
  -Force

.\scripts\render_3dgs_submission.ps1 `
  -DataRoot phase1\private_set1 `
  -ModelRoot outputs\3dgs_models_private_l40s_quality `
  -OutDir outputs\3dgs_private_l40s_quality `
  -Antialiasing `
  -ZipPath submissions\3dgs_private_l40s_quality.zip

.\scripts\validate_submission.ps1 `
  -DataRoot phase1\private_set1 `
  -PredDir outputs\3dgs_private_l40s_quality
```
