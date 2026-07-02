param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
    [string]$TorchCudaArchList = "8.9",
    [switch]$Clean,
    [switch]$ForceReinstall
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

function Resolve-RepoPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return (Resolve-Path $Path).Path
    }
    return (Resolve-Path (Join-Path $RepoRoot $Path)).Path
}

function Import-VsDevEnv([string]$VsPath) {
    $devCmd = Join-Path $VsPath "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path $devCmd)) {
        throw "VsDevCmd.bat not found at: $devCmd"
    }

    $envLines = & cmd.exe /s /c "`"$devCmd`" -arch=x64 -host_arch=x64 >nul && set"
    foreach ($line in $envLines) {
        if ($line -match "^(.*?)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
}

$pythonPath = Resolve-RepoPath $Python
if (-not (Test-Path $CudaPath)) {
    throw "CUDA Toolkit 12.6 was not found. Expected: $CudaPath"
}
$cudaRoot = (Resolve-Path $CudaPath).Path
$nvccPath = Join-Path $cudaRoot "bin\nvcc.exe"
if (-not (Test-Path $nvccPath)) {
    throw "CUDA 12.6 nvcc not found. Expected: $nvccPath"
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found. Install Visual Studio Installer / VS 2022 Build Tools first."
}

$vsPath = & $vswhere `
    -latest `
    -version "[17.0,18.0)" `
    -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath

if ([string]::IsNullOrWhiteSpace($vsPath)) {
    throw "VS 2022 with C++ build tools was not found. Install VS 2022 Build Tools + Desktop development with C++."
}

Import-VsDevEnv $vsPath.Trim()

$env:CUDA_HOME = $cudaRoot
$env:CUDA_PATH = $cudaRoot
$env:DISTUTILS_USE_SDK = "1"
$env:MSSdk = "1"
$env:TORCH_CUDA_ARCH_LIST = $TorchCudaArchList
$env:PYTHONPATH = "src"
$env:PATH = "$cudaRoot\bin;$cudaRoot\libnvvp;$RepoRoot\.venv\Scripts;$env:PATH"

Write-Host "Using VS: $($vsPath.Trim())"
Write-Host "Using CUDA: $cudaRoot"
Write-Host "Using Python: $pythonPath"
Write-Host "TORCH_CUDA_ARCH_LIST=$env:TORCH_CUDA_ARCH_LIST"

& $nvccPath --version
& cl.exe /Bv

@"
import torch
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
"@ | & $pythonPath -

if ($LASTEXITCODE -ne 0) {
    throw "Python/Torch CUDA check failed."
}

$extensionDirs = @(
    "external\gaussian-splatting\submodules\simple-knn",
    "external\gaussian-splatting\submodules\diff-gaussian-rasterization",
    "external\gaussian-splatting\submodules\fused-ssim"
)

foreach ($extensionDir in $extensionDirs) {
    $extensionPath = Resolve-RepoPath $extensionDir
    Write-Host ""
    Write-Host "Building $extensionDir"

    if ($Clean) {
        $buildPath = Join-Path $extensionPath "build"
        if (Test-Path $buildPath) {
            Remove-Item -LiteralPath $buildPath -Recurse -Force
        }
        Get-ChildItem -LiteralPath $extensionPath -Filter "*.egg-info" -Directory |
            Remove-Item -Recurse -Force
    }

    $pipArgs = @("install", "--no-build-isolation", "--no-cache-dir", "-v")
    if ($ForceReinstall) {
        $pipArgs += "--force-reinstall"
    }
    $pipArgs += "."

    Push-Location $extensionPath
    try {
        & $pythonPath -m pip @pipArgs
        if ($LASTEXITCODE -ne 0) {
            throw "pip build failed for $extensionDir"
        }
    }
    finally {
        Pop-Location
    }
}

@"
import diff_gaussian_rasterization
import simple_knn._C
import fused_ssim
print("3DGS CUDA extensions import OK")
"@ | & $pythonPath -

if ($LASTEXITCODE -ne 0) {
    throw "3DGS extension import check failed."
}
