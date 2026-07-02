param(
    [Parameter(Mandatory=$true)][string[]]$DataRoot,
    [Parameter(Mandatory=$true)][string]$ModelRoot,
    [Parameter(Mandatory=$true)][string]$OutDir,
    [string]$Scene = "",
    [int]$Iteration = -1,
    [int]$ShDegree = 3,
    [ValidateSet("black", "white")][string]$Background = "black",
    [switch]$Antialiasing,
    [double]$ScalingModifier = 1.0,
    [string]$Python = "",
    [string]$ZipPath = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

$profileHelper = Join-Path $PSScriptRoot "_3dgs_train_profiles.ps1"
. $profileHelper
$pythonExe = Resolve-3DgsPython -Python $Python

$argsList = @(
    "--data-root"
) + $DataRoot + @(
    "--model-root", $ModelRoot,
    "--out-dir", $OutDir,
    "--iteration", "$Iteration",
    "--sh-degree", "$ShDegree",
    "--background", $Background
)

if ($Scene -ne "") {
    $argsList += @("--scene", $Scene)
}

if ($ZipPath -ne "") {
    $argsList += @("--zip", $ZipPath)
}

if ($Antialiasing) {
    $argsList += "--antialiasing"
}

if ($ScalingModifier -ne 1.0) {
    $argsList += @("--scaling-modifier", "$ScalingModifier")
}

& $pythonExe -m bts_baseline.render_3dgs_submission @argsList
