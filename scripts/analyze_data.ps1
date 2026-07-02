param(
    [string[]]$DataRoot = @("phase1\public_set", "phase1\private_set1"),
    [string]$Output = "reports\data_report.md"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m bts_baseline.analyze_data --data-root @DataRoot --output $Output
