param(
    [Parameter(Mandatory=$true)][string[]]$DataRoot,
    [Parameter(Mandatory=$true)][string]$PredDir,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

$profileHelper = Join-Path $PSScriptRoot "_3dgs_train_profiles.ps1"
. $profileHelper
$pythonExe = Resolve-3DgsPython -Python $Python

$argsList = @("--data-root") + $DataRoot + @("--pred-dir", $PredDir)
& $pythonExe -m bts_baseline.validate_submission @argsList
