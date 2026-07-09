function Get-3DgsTrainProfile {
    param(
        [ValidateSet("local-r2-7k", "l40s-fast", "l40s-quality", "l40s-bts-quality", "custom")]
        [string]$Preset
    )

    switch ($Preset) {
        "local-r2-7k" {
            return @{
                Iterations = 7000
                Resolution = 2
                OptimizerType = "default"
                Antialiasing = $false
                SaveIterations = @(7000)
                TestIterations = @(7000)
                CheckpointIterations = @()
                DensifyUntilIter = 0
                DensifyGradThreshold = -1.0
                DensificationInterval = 0
                OpacityResetInterval = 0
                ExtraArgs = @()
            }
        }
        "l40s-fast" {
            return @{
                Iterations = 15000
                Resolution = 1
                OptimizerType = "sparse_adam"
                Antialiasing = $true
                SaveIterations = @(7000, 15000)
                TestIterations = @(7000, 15000)
                CheckpointIterations = @(15000)
                DensifyUntilIter = 12000
                DensifyGradThreshold = 0.00025
                DensificationInterval = 100
                OpacityResetInterval = 3000
                ExtraArgs = @()
            }
        }
        "l40s-quality" {
            return @{
                Iterations = 30000
                Resolution = 1
                OptimizerType = "sparse_adam"
                Antialiasing = $true
                SaveIterations = @(7000, 15000, 30000)
                TestIterations = @(7000, 15000, 30000)
                CheckpointIterations = @(15000, 30000)
                DensifyUntilIter = 15000
                DensifyGradThreshold = 0.0002
                DensificationInterval = 100
                OpacityResetInterval = 3000
                ExtraArgs = @()
            }
        }
        "l40s-bts-quality" {
            return @{
                Iterations = 30000
                Resolution = 1
                OptimizerType = "sparse_adam"
                Antialiasing = $false
                SaveIterations = @(7000, 15000, 20000, 30000)
                TestIterations = @(7000, 15000, 20000, 30000)
                CheckpointIterations = @(15000, 30000)
                DensifyUntilIter = 20000
                DensifyGradThreshold = 0.00012
                DensificationInterval = 75
                OpacityResetInterval = 3000
                ExtraArgs = @("--lambda_dssim", "0.15")
            }
        }
        "custom" {
            return @{
                Iterations = 30000
                Resolution = 1
                OptimizerType = "default"
                Antialiasing = $false
                SaveIterations = @(30000)
                TestIterations = @(30000)
                CheckpointIterations = @()
                DensifyUntilIter = 0
                DensifyGradThreshold = -1.0
                DensificationInterval = 0
                OpacityResetInterval = 0
                ExtraArgs = @()
            }
        }
    }
}

function Resolve-3DgsPython {
    param([string]$Python = "")

    if ($Python -ne "") {
        return $Python
    }
    if ($env:PYTHON -and $env:PYTHON -ne "") {
        return $env:PYTHON
    }

    $candidates = @(
        ".\.venv\Scripts\python.exe",
        ".\.venv\bin\python",
        "./.venv/bin/python",
        "./.venv/bin/python3",
        "python"
    )

    foreach ($candidate in $candidates) {
        if ($candidate -eq "python") {
            return $candidate
        }
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
}

function Normalize-3DgsIterationList {
    param(
        [int[]]$Values,
        [int[]]$DefaultValues,
        [int]$FinalIteration,
        [switch]$IncludeFinal
    )

    $items = @()
    if ($Values.Count -gt 0) {
        $items += $Values
    } else {
        $items += $DefaultValues
    }
    if ($IncludeFinal -and $FinalIteration -gt 0) {
        $items += $FinalIteration
    }

    return @($items | Where-Object { $_ -gt 0 } | Sort-Object -Unique)
}

function Resolve-3DgsTrainSettings {
    param(
        [ValidateSet("local-r2-7k", "l40s-fast", "l40s-quality", "l40s-bts-quality", "custom")]
        [string]$Preset,
        [int]$Iterations,
        [int]$Resolution,
        [ValidateSet("profile", "default", "sparse_adam")]
        [string]$OptimizerType,
        [bool]$AntialiasingSpecified,
        [bool]$AntialiasingValue,
        [bool]$NoAntialiasing,
        [int[]]$SaveIterations,
        [int[]]$TestIterations,
        [int[]]$CheckpointIterations,
        [int]$DensifyUntilIter,
        [double]$DensifyGradThreshold,
        [int]$DensificationInterval,
        [int]$OpacityResetInterval,
        [string[]]$ExtraArgs
    )

    $profile = Get-3DgsTrainProfile -Preset $Preset

    $effectiveIterations = if ($Iterations -gt 0) { $Iterations } else { [int]$profile.Iterations }
    $effectiveResolution = if ($Resolution -gt 0) { $Resolution } else { [int]$profile.Resolution }
    $effectiveOptimizer = if ($OptimizerType -eq "profile") { [string]$profile.OptimizerType } else { $OptimizerType }

    if ($NoAntialiasing) {
        $effectiveAntialiasing = $false
    } elseif ($AntialiasingSpecified) {
        $effectiveAntialiasing = $AntialiasingValue
    } else {
        $effectiveAntialiasing = [bool]$profile.Antialiasing
    }

    $effectiveDensifyUntil = if ($DensifyUntilIter -gt 0) { $DensifyUntilIter } else { [int]$profile.DensifyUntilIter }
    $effectiveDensifyGrad = if ($DensifyGradThreshold -ge 0) { $DensifyGradThreshold } else { [double]$profile.DensifyGradThreshold }
    $effectiveDensificationInterval = if ($DensificationInterval -gt 0) { $DensificationInterval } else { [int]$profile.DensificationInterval }
    $effectiveOpacityReset = if ($OpacityResetInterval -gt 0) { $OpacityResetInterval } else { [int]$profile.OpacityResetInterval }

    $effectiveSave = Normalize-3DgsIterationList -Values $SaveIterations -DefaultValues $profile.SaveIterations -FinalIteration $effectiveIterations -IncludeFinal
    $effectiveTest = Normalize-3DgsIterationList -Values $TestIterations -DefaultValues $profile.TestIterations -FinalIteration $effectiveIterations -IncludeFinal
    $effectiveCheckpoint = Normalize-3DgsIterationList -Values $CheckpointIterations -DefaultValues $profile.CheckpointIterations -FinalIteration $effectiveIterations

    $effectiveExtraArgs = @()
    $effectiveExtraArgs += $profile.ExtraArgs
    $effectiveExtraArgs += $ExtraArgs

    return [pscustomobject]@{
        Preset = $Preset
        Iterations = $effectiveIterations
        Resolution = $effectiveResolution
        OptimizerType = $effectiveOptimizer
        Antialiasing = $effectiveAntialiasing
        SaveIterations = @($effectiveSave)
        TestIterations = @($effectiveTest)
        CheckpointIterations = @($effectiveCheckpoint)
        DensifyUntilIter = $effectiveDensifyUntil
        DensifyGradThreshold = $effectiveDensifyGrad
        DensificationInterval = $effectiveDensificationInterval
        OpacityResetInterval = $effectiveOpacityReset
        ExtraArgs = @($effectiveExtraArgs)
    }
}

function Add-3DgsTrainOptions {
    param(
        [string[]]$ArgsList,
        [pscustomobject]$Settings,
        [string]$StartCheckpoint,
        [switch]$Quiet
    )

    $args = @($ArgsList)
    $args += @(
        "-r", "$($Settings.Resolution)",
        "--iterations", "$($Settings.Iterations)",
        "--optimizer_type", "$($Settings.OptimizerType)",
        "--test_iterations"
    )
    $args += $Settings.TestIterations | ForEach-Object { "$_" }
    $args += "--save_iterations"
    $args += $Settings.SaveIterations | ForEach-Object { "$_" }
    $args += "--disable_viewer"

    if ($Settings.Antialiasing) {
        $args += "--antialiasing"
    }
    if ($Settings.DensifyUntilIter -ge 0) {
        $args += @("--densify_until_iter", "$($Settings.DensifyUntilIter)")
    }
    if ($Settings.DensifyGradThreshold -ge 0) {
        $args += @("--densify_grad_threshold", "$($Settings.DensifyGradThreshold)")
    }
    if ($Settings.DensificationInterval -gt 0) {
        $args += @("--densification_interval", "$($Settings.DensificationInterval)")
    }
    if ($Settings.OpacityResetInterval -gt 0) {
        $args += @("--opacity_reset_interval", "$($Settings.OpacityResetInterval)")
    }
    if ($Settings.CheckpointIterations.Count -gt 0) {
        $args += "--checkpoint_iterations"
        $args += $Settings.CheckpointIterations | ForEach-Object { "$_" }
    }
    if ($StartCheckpoint -ne "") {
        $args += @("--start_checkpoint", $StartCheckpoint)
    }
    if ($Quiet) {
        $args += "--quiet"
    }
    if ($Settings.ExtraArgs.Count -gt 0) {
        $args += $Settings.ExtraArgs
    }

    return @($args)
}

function Test-3DgsSparseAdamAvailable {
    param([string]$Python)

    $checkScript = @"
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
"@

    $checkScript | & $Python -
    return ($LASTEXITCODE -eq 0)
}

function ConvertTo-3DgsQuotedCommand {
    param([string[]]$ArgsList)

    return (($ArgsList | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }) -join " ")
}

function Invoke-3DgsLoggedProcess {
    param(
        [Parameter(Mandatory=$true)][string]$FilePath,
        [Parameter(Mandatory=$true)][string[]]$ArgumentList,
        [Parameter(Mandatory=$true)][string]$LogPath
    )

    $logParent = Split-Path -Parent $LogPath
    if ($logParent -ne "") {
        New-Item -ItemType Directory -Force -Path $logParent | Out-Null
    }

    $IsWinOS = [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT
    if ($IsWinOS) {
        $rawCommandArgs = @($FilePath) + $ArgumentList
        $command = (ConvertTo-3DgsQuotedCommand -ArgsList $rawCommandArgs) + ' > "' + $LogPath + '" 2>&1'
        & cmd.exe /d /s /c $command
        return $LASTEXITCODE
    }

    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = $FilePath
    $processInfo.Arguments = ConvertTo-3DgsQuotedCommand -ArgsList $ArgumentList
    $processInfo.UseShellExecute = $false
    $processInfo.RedirectStandardOutput = $true
    $processInfo.RedirectStandardError = $true
    $processInfo.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $processInfo

    $writer = New-Object System.IO.StreamWriter($LogPath, $false, [System.Text.Encoding]::UTF8)
    $sync = New-Object object

    $outputHandler = [System.Diagnostics.DataReceivedEventHandler]{
        param($sender, $eventArgs)
        if ($null -ne $eventArgs.Data) {
            [System.Threading.Monitor]::Enter($sync)
            try {
                $writer.WriteLine($eventArgs.Data)
                $writer.Flush()
            } finally {
                [System.Threading.Monitor]::Exit($sync)
            }
        }
    }
    $errorHandler = [System.Diagnostics.DataReceivedEventHandler]{
        param($sender, $eventArgs)
        if ($null -ne $eventArgs.Data) {
            [System.Threading.Monitor]::Enter($sync)
            try {
                $writer.WriteLine($eventArgs.Data)
                $writer.Flush()
            } finally {
                [System.Threading.Monitor]::Exit($sync)
            }
        }
    }

    $process.add_OutputDataReceived($outputHandler)
    $process.add_ErrorDataReceived($errorHandler)

    try {
        [void]$process.Start()
        $process.BeginOutputReadLine()
        $process.BeginErrorReadLine()
        $process.WaitForExit()
        return $process.ExitCode
    } finally {
        $writer.Dispose()
        $process.Dispose()
    }
}
