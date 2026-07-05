param(
    [Parameter(Mandatory=$true)][string]$PreparedScene,
    [Parameter(Mandatory=$true)][string]$ModelDir,
    [ValidateSet("local-r2-7k", "l40s-fast", "l40s-quality", "l40s-bts-quality", "custom")]
    [string]$Preset = "l40s-quality",
    [int]$Iterations = 0,
    [int]$Resolution = 0,
    [ValidateSet("profile", "default", "sparse_adam")]
    [string]$OptimizerType = "profile",
    [switch]$Antialiasing,
    [switch]$NoAntialiasing,
    [int[]]$SaveIterations = @(),
    [int[]]$TestIterations = @(),
    [int[]]$CheckpointIterations = @(),
    [int]$DensifyUntilIter = 0,
    [double]$DensifyGradThreshold = -1.0,
    [int]$DensificationInterval = 0,
    [int]$OpacityResetInterval = 0,
    [string]$StartCheckpoint = "",
    [string[]]$ExtraArgs = @(),
    [string]$Python = "",
    [string]$CudaVisibleDevices = "",
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path "external" "gaussian-splatting"

if ($CudaVisibleDevices -ne "") {
    $env:CUDA_VISIBLE_DEVICES = $CudaVisibleDevices
}

$profileHelper = Join-Path $PSScriptRoot "_3dgs_train_profiles.ps1"
. $profileHelper
$pythonExe = Resolve-3DgsPython -Python $Python
$trainPy = (Resolve-Path -LiteralPath (Join-Path (Join-Path "external" "gaussian-splatting") "train.py")).Path

$settings = Resolve-3DgsTrainSettings `
    -Preset $Preset `
    -Iterations $Iterations `
    -Resolution $Resolution `
    -OptimizerType $OptimizerType `
    -AntialiasingSpecified $PSBoundParameters.ContainsKey("Antialiasing") `
    -AntialiasingValue ([bool]$Antialiasing) `
    -NoAntialiasing ([bool]$NoAntialiasing) `
    -SaveIterations $SaveIterations `
    -TestIterations $TestIterations `
    -CheckpointIterations $CheckpointIterations `
    -DensifyUntilIter $DensifyUntilIter `
    -DensifyGradThreshold $DensifyGradThreshold `
    -DensificationInterval $DensificationInterval `
    -OpacityResetInterval $OpacityResetInterval `
    -ExtraArgs $ExtraArgs

if ($settings.OptimizerType -eq "sparse_adam" -and -not (Test-3DgsSparseAdamAvailable -Python $pythonExe)) {
    Write-Warning "Sparse Adam is not available in diff_gaussian_rasterization. Falling back to optimizer=default."
    $settings.OptimizerType = "default"
}

Write-Host (
    "TRAIN scene preset={0} iterations={1} resolution={2} optimizer={3} antialiasing={4}" -f `
    $settings.Preset, $settings.Iterations, $settings.Resolution, $settings.OptimizerType, $settings.Antialiasing
)

$argsList = Add-3DgsTrainOptions `
    -ArgsList @("-s", $PreparedScene, "-m", $ModelDir) `
    -Settings $settings `
    -StartCheckpoint $StartCheckpoint `
    -Quiet:$Quiet

& $pythonExe $trainPy @argsList
