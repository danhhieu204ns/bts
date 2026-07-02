param(
    [Parameter(Mandatory=$true)][string[]]$DataRoot,
    [string]$OutRoot = "prepared\3dgs_data",
    [string]$Scene = "",
    [ValidateSet("hardlink", "copy")][string]$CopyMode = "hardlink",
    [ValidateSet("pinhole", "copy")][string]$CameraMode = "pinhole",
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

$profileHelper = Join-Path $PSScriptRoot "_3dgs_train_profiles.ps1"
. $profileHelper
$pythonExe = Resolve-3DgsPython -Python $Python

$argsList = @(
    "--data-root"
) + $DataRoot + @(
    "--out-root", $OutRoot,
    "--copy-mode", $CopyMode,
    "--camera-mode", $CameraMode
)

if ($Scene -ne "") {
    $argsList += @("--scene", $Scene)
}

& $pythonExe -m bts_baseline.prepare_3dgs_scene @argsList
