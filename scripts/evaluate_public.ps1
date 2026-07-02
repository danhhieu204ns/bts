param(
    [Parameter(Mandatory=$true)][string[]]$DataRoot,
    [Parameter(Mandatory=$true)][string]$PredDir,
    [ValidateSet("auto", "on", "off")][string]$Lpips = "auto",
    [ValidateSet("alex", "vgg", "squeeze")][string]$LpipsNet = "alex",
    [ValidateSet("auto", "cuda", "cpu")][string]$Device = "auto",
    [double[]]$PsnrMax = @(30.0, 35.0, 40.0),
    [string]$OutputCsv = "",
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
    "--pred-dir", $PredDir,
    "--lpips", $Lpips,
    "--lpips-net", $LpipsNet,
    "--device", $Device,
    "--psnr-max"
) + ($PsnrMax | ForEach-Object { "$_" })

if ($OutputCsv -ne "") {
    $argsList += @("--output-csv", $OutputCsv)
}

& $pythonExe -m bts_baseline.evaluate_public @argsList
