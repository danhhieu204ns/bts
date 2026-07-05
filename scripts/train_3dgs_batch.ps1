param(
    [Parameter(Mandatory=$true)][string]$PreparedRoot,
    [Parameter(Mandatory=$true)][string]$ModelRoot,
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
    [string[]]$Scene = @(),
    [switch]$Force
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
    "TRAIN batch preset={0} iterations={1} resolution={2} optimizer={3} antialiasing={4}" -f `
    $settings.Preset, $settings.Iterations, $settings.Resolution, $settings.OptimizerType, $settings.Antialiasing
)
if ($settings.ExtraArgs.Count -gt 0) {
    Write-Host ("Extra train args: {0}" -f ($settings.ExtraArgs -join " "))
}

$preparedRootPath = Resolve-Path $PreparedRoot
$sceneDirs = Get-ChildItem -LiteralPath $preparedRootPath -Directory | Sort-Object Name
if ($Scene.Count -gt 0) {
    $wanted = @{}
    foreach ($name in $Scene) {
        $wanted[$name.ToLowerInvariant()] = $true
    }
    $sceneDirs = $sceneDirs | Where-Object { $wanted.ContainsKey($_.Name.ToLowerInvariant()) }
}

if (-not $sceneDirs) {
    throw "No prepared scenes found under $PreparedRoot"
}

foreach ($sceneDir in $sceneDirs) {
    $modelDir = Join-Path $ModelRoot $sceneDir.Name
    $donePath = Join-Path $modelDir "point_cloud\iteration_$($settings.Iterations)\point_cloud.ply"
    $logDir = Join-Path $modelDir "logs"
    $logName = "train_{0}_r{1}_{2}.log" -f $settings.Iterations, $settings.Resolution, $settings.Preset
    $logPath = Join-Path $logDir $logName

    if ((-not $Force) -and (Test-Path $donePath)) {
        Write-Host "SKIP $($sceneDir.Name): $donePath exists"
        continue
    }

    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    Write-Host "TRAIN $($sceneDir.Name): log=$logPath"

    $argsList = Add-3DgsTrainOptions `
        -ArgsList @("-s", $sceneDir.FullName, "-m", $modelDir) `
        -Settings $settings `
        -StartCheckpoint $StartCheckpoint `
        -Quiet

    $exitCode = Invoke-3DgsLoggedProcess `
        -FilePath $pythonExe `
        -ArgumentList (@($trainPy) + $argsList) `
        -LogPath $logPath
    if ($exitCode -ne 0) {
        Get-Content -LiteralPath $logPath -Tail 80
        throw "3DGS train failed for $($sceneDir.Name). Log: $logPath"
    }

    if (-not (Test-Path $donePath)) {
        Get-Content -LiteralPath $logPath -Tail 80
        throw "3DGS train finished but output is missing for $($sceneDir.Name): $donePath"
    }

    Write-Host "DONE $($sceneDir.Name): $donePath"
}
