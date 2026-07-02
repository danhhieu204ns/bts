param(
    [Parameter(Mandatory=$true)][string[]]$DataRoot,
    [Parameter(Mandatory=$true)][string]$OutDir,
    [string]$ZipPath = "",
    [double]$OrientationWeight = 0.5
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

$argsList = @(
    "--data-root"
) + $DataRoot + @(
    "--out-dir", $OutDir,
    "--orientation-weight", "$OrientationWeight"
)

if ($ZipPath -ne "") {
    $argsList += @("--zip", $ZipPath)
}

.\.venv\Scripts\python.exe -m bts_baseline.nearest_pose_baseline @argsList
