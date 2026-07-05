param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
    [string]$TorchCudaArchList = "",
    [switch]$Clean,
    [switch]$ForceReinstall,
    [switch]$AllowUnsupportedVs
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

function Resolve-CudaRoot([string]$RequestedPath) {
    if (-not [string]::IsNullOrWhiteSpace($RequestedPath) -and (Test-Path $RequestedPath)) {
        return (Resolve-Path $RequestedPath).Path
    }

    $cudaParent = Join-Path ${env:ProgramFiles} "NVIDIA GPU Computing Toolkit\CUDA"
    $availableCudaRoots = @()
    if (Test-Path $cudaParent) {
        $availableCudaRoots = Get-ChildItem -LiteralPath $cudaParent -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^v\d+(?:\.\d+)?$' } |
            Sort-Object { [version]($_.Name.TrimStart('v')) } -Descending
    }

    if ($availableCudaRoots.Count -gt 0) {
        return $availableCudaRoots[0].FullName
    }

    $availableText = if ($availableCudaRoots.Count -gt 0) {
        ($availableCudaRoots | ForEach-Object { $_.FullName }) -join ", "
    }
    else {
        "none"
    }

    throw "CUDA Toolkit was not found. Requested: $RequestedPath. Checked: $cudaParent. Available: $availableText"
}

function Resolve-VsPath([bool]$AllowUnsupported) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "vswhere.exe not found. Install Visual Studio Installer / VS 2022 Build Tools first."
    }

    $versionRange = if ($AllowUnsupported) { "[17.0,19.0)" } else { "[17.0,18.0)" }

    $vsPath = & $vswhere `
        -latest `
        -version $versionRange `
        -products * `
        -requires "Microsoft.VisualStudio.Component.VC.Tools.x86.x64" `
        -property installationPath

    if (-not [string]::IsNullOrWhiteSpace($vsPath)) {
        return $vsPath.Trim()
    }

    $allVsPaths = & $vswhere `
        -all `
        -version $versionRange `
        -products * `
        -property installationPath

    foreach ($candidate in $allVsPaths) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }

        $candidatePath = $candidate.Trim()
        $devCmd = Join-Path $candidatePath "Common7\Tools\VsDevCmd.bat"
        $msvcRoot = Join-Path $candidatePath "VC\Tools\MSVC"

        if ((Test-Path $devCmd) -and (Test-Path $msvcRoot)) {
            $cl = Get-ChildItem -LiteralPath $msvcRoot -Directory -ErrorAction SilentlyContinue |
                ForEach-Object { Join-Path $_.FullName "bin\Hostx64\x64\cl.exe" } |
                Where-Object { Test-Path $_ } |
                Select-Object -First 1

            if ($cl) {
                return $candidatePath
            }
        }
    }

    $allInstalled = & $vswhere -all -products * -property installationPath
    $installedInstances = if ($allInstalled) { ($allInstalled | ForEach-Object { $_.Trim() }) -join ", " } else { "none" }
    throw "VS 2022 Build Tools with C++ compiler was not found. CUDA 12.6 is safest with VS 2022/MSVC 14.4x. Install Visual Studio 2022 Build Tools with the 'Desktop development with C++' workload. Checked instances: $installedInstances. To try a newer unsupported VS anyway, rerun with -AllowUnsupportedVs."
}

function Resolve-TorchCudaArchList([string]$PythonPath, [string]$RequestedArchList) {
    if (-not [string]::IsNullOrWhiteSpace($RequestedArchList)) {
        return $RequestedArchList
    }

    $arch = @"
import torch
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False; pass -TorchCudaArchList manually or fix CUDA first.")
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
"@ | & $PythonPath -

    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($arch)) {
        throw "Could not auto-detect TORCH_CUDA_ARCH_LIST from PyTorch."
    }

    return $arch.Trim()
}

function Assert-SupportedPython([string]$PythonPath) {
    $versionText = @"
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
"@ | & $PythonPath -

    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($versionText)) {
        throw "Could not determine Python version from: $PythonPath"
    }

    $version = [version]$versionText.Trim()
    if ($version.Major -ne 3 -or $version.Minor -lt 10 -or $version.Minor -gt 12) {
        throw "Unsupported Python $versionText for these 3DGS CUDA extensions on Windows. Use Python 3.11 or 3.12, then recreate .venv and reinstall Torch cu126."
    }
}

$pythonPath = Resolve-RepoPath $Python
Assert-SupportedPython $pythonPath
$cudaRoot = Resolve-CudaRoot $CudaPath
$nvccPath = Join-Path $cudaRoot "bin\nvcc.exe"
if (-not (Test-Path $nvccPath)) {
    throw "CUDA nvcc not found. Expected: $nvccPath"
}

$vsPath = Resolve-VsPath $AllowUnsupportedVs.IsPresent
$resolvedTorchCudaArchList = Resolve-TorchCudaArchList $pythonPath $TorchCudaArchList

Import-VsDevEnv $vsPath.Trim()

$env:CUDA_HOME = $cudaRoot
$env:CUDA_PATH = $cudaRoot
$env:DISTUTILS_USE_SDK = "1"
$env:MSSdk = "1"
$env:TORCH_CUDA_ARCH_LIST = $resolvedTorchCudaArchList
$env:PYTHONPATH = "src"
$env:PATH = "$cudaRoot\bin;$cudaRoot\libnvvp;$RepoRoot\.venv\Scripts;$env:PATH"

$vsMajor = [int]($env:VisualStudioVersion.Split(".")[0])
if ($vsMajor -ge 18 -and $AllowUnsupportedVs -and [string]::IsNullOrWhiteSpace($env:NVCC_PREPEND_FLAGS)) {
    $env:NVCC_PREPEND_FLAGS = "-allow-unsupported-compiler"
    Write-Warning "Visual Studio $env:VisualStudioVersion is newer than CUDA 12.x officially validates. Setting NVCC_PREPEND_FLAGS=-allow-unsupported-compiler."
}

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
