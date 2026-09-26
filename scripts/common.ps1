# ==============================================================================
# Business Entity Resolution — PowerShell Pipeline Orchestration: Common Helpers
# ==============================================================================
# Provides common path resolution, colorized logging, error handling,
# and Python execution wrappers for all phase scripts.
# ==============================================================================

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Base directory layout
$SCRIPT_DIR   = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = (Resolve-Path "$SCRIPT_DIR\..").Path
$CODE_DIR     = Join-Path $PROJECT_ROOT "code\business_entity_resolution"
$DATASET_DIR  = Join-Path $PROJECT_ROOT "dataset"
$DEFAULT_OUT  = Join-Path $PROJECT_ROOT "output"

# ------------------------------------------------------------------------------
# Python Interpreter Resolution
# ------------------------------------------------------------------------------
function Get-PythonExecutable {
    param(
        [string]$ExplicitPath = ""
    )

    if ($ExplicitPath -and (Test-Path $ExplicitPath)) {
        return (Resolve-Path $ExplicitPath).Path
    }

    if ($env:PYTHON_BIN -and (Test-Path $env:PYTHON_BIN)) {
        return (Resolve-Path $env:PYTHON_BIN).Path
    }

    # 1. Project root virtualenv (preferred: holds CUDA-enabled PyTorch for GPU training)
    $rootVenv = Join-Path $PROJECT_ROOT ".venv\Scripts\python.exe"
    if (Test-Path $rootVenv) {
        return (Resolve-Path $rootVenv).Path
    }

    # 2. Package subdirectory virtualenv
    $codeVenv = Join-Path $CODE_DIR ".venv\Scripts\python.exe"
    if (Test-Path $codeVenv) {
        return (Resolve-Path $codeVenv).Path
    }

    $cmdPython = Get-Command "python" -ErrorAction SilentlyContinue
    if ($cmdPython) {
        return $cmdPython.Source
    }

    throw "Python interpreter not found. Please activate virtualenv or set `$env:PYTHON_BIN."
}

# ------------------------------------------------------------------------------
# Colorized Console Logging
# ------------------------------------------------------------------------------
function Write-Header {
    param([string]$Title)
    $line = "=" * 80
    Write-Host ""
    Write-Host $line -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor White
    Write-Host $line -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step {
    param(
        [string]$Phase,
        [string]$Description
    )
    Write-Host "[$Phase] " -ForegroundColor Yellow -NoNewline
    Write-Host $Description -ForegroundColor White
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] " -ForegroundColor Green -NoNewline
    Write-Host $Message -ForegroundColor Green
}

function Write-Info {
    param([string]$Message)
    Write-Host "  -> " -ForegroundColor Gray -NoNewline
    Write-Host $Message -ForegroundColor Gray
}

function Write-WarningMessage {
    param([string]$Message)
    Write-Host "[WARN] " -ForegroundColor Yellow -NoNewline
    Write-Host $Message -ForegroundColor Yellow
}

function Write-ErrorMessage {
    param([string]$Message)
    Write-Host "[ERROR] " -ForegroundColor Red -NoNewline
    Write-Host $Message -ForegroundColor Red
}

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
        Write-Info "Created directory: $Path"
    }
}

# ------------------------------------------------------------------------------
# Safe Python Command Invocation
# ------------------------------------------------------------------------------
function Invoke-PythonModule {
    param(
        [string]$ModuleName,
        [string[]]$Arguments,
        [string]$StepName = "",
        [switch]$DryRun = $false,
        [string]$PythonExe = ""
    )

    if (-not $PythonExe) {
        $PythonExe = Get-PythonExecutable
    }

    $allArgs = @("-m", $ModuleName) + $Arguments

    if ($StepName) {
        Write-Step "EXEC" $StepName
    }
    Write-Info "Command: $PythonExe $($allArgs -join ' ')"
    Write-Info "Working Directory: $CODE_DIR"

    if ($DryRun) {
        Write-Host "  [DRY RUN] Skipped execution." -ForegroundColor Magenta
        return $true
    }

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    # Set PYTHONPATH so src modules resolve reliably
    $oldPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = "$CODE_DIR;$PROJECT_ROOT"

    try {
        Push-Location $CODE_DIR
        & $PythonExe $allArgs
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
        $env:PYTHONPATH = $oldPythonPath
        $stopwatch.Stop()
    }

    $elapsed = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 2)

    if ($exitCode -ne 0) {
        Write-ErrorMessage "Step failed with exit code $exitCode ($($elapsed)s): $StepName"
        throw "Python execution of $ModuleName failed (exit code $exitCode)."
    }

    Write-Success "Completed in $($elapsed)s: $StepName"
    return $true
}

# ------------------------------------------------------------------------------
# Logging: Persistent Transcript Capture
# ------------------------------------------------------------------------------

# Central log directory (sibling of output/)
$LOG_DIR = Join-Path $PROJECT_ROOT "logs"

function Initialize-Logging {
    <#
    .SYNOPSIS
        Opens a timestamped transcript file under logs/ and starts capturing
        all stdout + stderr for the calling script.
    .PARAMETER ScriptName
        Short label used in the log filename (e.g. "01_run_blocking_train").
    .PARAMETER LogDir
        Override log directory (default: <project_root>/logs).
    .OUTPUTS
        Absolute path to the opened log file.
    #>
    param(
        [string]$ScriptName = "pipeline",
        [string]$LogDir     = ""
    )

    if (-not $LogDir) { $LogDir = $LOG_DIR }

    if (-not (Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }

    $ts      = Get-Date -Format "yyyyMMdd_HHmmss"
    $logFile = Join-Path $LogDir "${ts}_${ScriptName}.log"

    try {
        # PS 5.0+ supports multiple concurrent transcripts
        Start-Transcript -Path $logFile -Append | Out-Null
        Write-Host "[LOG] Transcript -> $logFile" -ForegroundColor DarkGray
    }
    catch {
        # Transcript already running from parent orchestrator — that transcript
        # captures all output anyway; just return the path for reference.
        Write-Host "[LOG] Nested transcript skipped (parent transcript active): $logFile" -ForegroundColor DarkGray
    }

    return $logFile
}

function Close-Logging {
    <#
    .SYNOPSIS
        Stops the active transcript started by Initialize-Logging.
        Safe to call even if no transcript is running.
    #>
    try {
        Stop-Transcript | Out-Null
    }
    catch {
        # Already stopped or never started; silently ignore.
    }
}

function Invoke-PythonScript {
    param(
        [string]$ScriptPath,
        [string[]]$Arguments,
        [string]$StepName = "",
        [switch]$DryRun = $false,
        [string]$PythonExe = ""
    )

    if (-not $PythonExe) {
        $PythonExe = Get-PythonExecutable
    }

    if ($StepName) {
        Write-Step "EXEC" $StepName
    }
    Write-Info "Command: $PythonExe $ScriptPath $($Arguments -join ' ')"
    Write-Info "Working Directory: $PROJECT_ROOT"

    if ($DryRun) {
        Write-Host "  [DRY RUN] Skipped execution." -ForegroundColor Magenta
        return $true
    }

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    $oldPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = "$CODE_DIR;$PROJECT_ROOT"

    try {
        Push-Location $PROJECT_ROOT
        & $PythonExe $ScriptPath $Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
        $env:PYTHONPATH = $oldPythonPath
        $stopwatch.Stop()
    }

    $elapsed = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 2)

    if ($exitCode -ne 0) {
        Write-ErrorMessage "Script failed with exit code $exitCode ($($elapsed)s): $StepName"
        throw "Python execution of $ScriptPath failed (exit code $exitCode)."
    }

    Write-Success "Completed in $($elapsed)s: $StepName"
    return $true
}
