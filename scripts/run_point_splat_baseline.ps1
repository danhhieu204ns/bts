param(
    [Parameter(Mandatory=$true)][string[]]$DataRoot,
    [Parameter(Mandatory=$true)][string]$OutDir,
    [string]$ModelDir = "outputs\point_splat_models",
    [string]$Scene = "",
    [ValidateSet("nearest", "solid")][string]$Background = "nearest",
    [int]$SplatRadius = 3,
    [double]$Alpha = 0.85,
    [int]$MaxPoints = 0,
    [string]$ZipPath = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

$argsList = @(
    "--data-root"
) + $DataRoot + @(
    "--out-dir", $OutDir,
    "--model-dir", $ModelDir,
    "--background", $Background,
    "--splat-radius", "$SplatRadius",
    "--alpha", "$Alpha",
    "--max-points", "$MaxPoints"
)

if ($Scene -ne "") {
    $argsList += @("--scene", $Scene)
}

if ($ZipPath -ne "") {
    $argsList += @("--zip", $ZipPath)
}

.\.venv\Scripts\python.exe -m bts_baseline.point_splat_baseline @argsList
