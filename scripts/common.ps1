# ==============================================================================
# Business Entity Resolution -- PowerShell Pipeline Orchestration: Common Helpers
# ==============================================================================
# Provides common path resolution, colorized logging, error handling,
# GPU detection/teardown, and Python execution wrappers for all phase scripts.
# ==============================================================================

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Base directory layout
$SCRIPT_DIR   = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = (Resolve-Path "$SCRIPT_DIR\..").Path
$CODE_DIR     = Join-Path $PROJECT_ROOT "code\business_entity_resolution"
$DATASET_DIR  = Join-Path $PROJECT_ROOT "dataset"
$DEFAULT_OUT  = Join-Path $PROJECT_ROOT "output"

function Resolve-FullPath {
    <#
    .SYNOPSIS
        Converts any relative or rooted path to an absolute path rooted at $PROJECT_ROOT.
    #>
    param([string]$Path)
    if (-not $Path) { return "" }
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $PROJECT_ROOT $Path))
}

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
# GPU Detection & Memory Management
# ------------------------------------------------------------------------------

# Script-level GPU state -- populated by Initialize-Gpu
$script:_GpuAvailable = $false
$script:_GpuDevice    = "cpu"
$script:_GpuName      = "CPU"
$script:_GpuVramGb    = 0.0

function Initialize-Gpu {
    <#
    .SYNOPSIS
        Probe CUDA availability via the current Python interpreter and cache results
        in script-level variables. Must be called once after $python is resolved.
    .PARAMETER PythonExe
        Path to the Python executable to use for the probe.
    #>
    param([string]$PythonExe)

    try {
        $raw = & $PythonExe -c @"
import sys
try:
    import torch
    avail = torch.cuda.is_available()
    if avail:
        p = torch.cuda.get_device_properties(0)
        print(f"1|{p.name}|{round(p.total_memory/1e9,2)}")
    else:
        print("0|CPU|0")
except Exception as e:
    print(f"0|CPU|0")
"@ 2>$null

        if ($raw -match "^(\d)\|(.+)\|(.+)$") {
            $script:_GpuAvailable = ($Matches[1] -eq "1")
            $script:_GpuName      = $Matches[2]
            $script:_GpuVramGb    = [double]$Matches[3]
            $script:_GpuDevice    = if ($script:_GpuAvailable) { "cuda" } else { "cpu" }
        }
    } catch {
        # Probe failed -- default to CPU
    }

    if ($script:_GpuAvailable) {
        Write-Host ""
        Write-Host "  [GPU] $($script:_GpuName) -- $($script:_GpuVramGb) GB VRAM -- Sequential GPU mode active." -ForegroundColor Green
        Write-Host "        Each model loaded exclusively. VRAM fully released between stages." -ForegroundColor DarkGreen
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "  [CPU] No CUDA GPU detected. All stages will use CPU paths." -ForegroundColor Yellow
        Write-Host ""
    }
}

function Release-GpuMemory {
    <#
    .SYNOPSIS
        Flush CUDA memory between GPU stages to free VRAM for the next model.
        Safe no-op when GPU is unavailable.
    .PARAMETER Tag
        Short label to print in the release log line (e.g. "BGE-M3").
    .PARAMETER PythonExe
        Path to the Python executable.
    #>
    param(
        [string]$Tag      = "",
        [string]$PythonExe = ""
    )

    if (-not $script:_GpuAvailable) { return }
    if (-not $PythonExe) {
        Write-WarningMessage "Release-GpuMemory: no PythonExe supplied, skipping flush."
        return
    }

    $tagMsg = if ($Tag) { " after $Tag" } else { "" }
    Write-Info "Flushing VRAM${tagMsg}..."

    & $PythonExe -c @"
import torch, gc
if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
gc.collect()
print('[GPU] VRAM cache cleared${tagMsg}.')
"@ 2>&1 | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
}

# ------------------------------------------------------------------------------
# Fast Artifact Line Counter (.NET StreamReader -- avoids full file load)
# ------------------------------------------------------------------------------
function Get-FastLineCount {
    <#
    .SYNOPSIS
        Count lines in a large TSV without loading the entire file into memory.
        Subtracts 1 for the header row.
    #>
    param([string]$FilePath)

    if (-not (Test-Path $FilePath)) { return 0 }
    $count = 0
    $reader = [System.IO.StreamReader]::new($FilePath)
    try {
        while ($null -ne $reader.ReadLine()) { $count++ }
    } finally {
        $reader.Dispose()
    }
    return [Math]::Max(0, $count - 1)  # subtract header
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
        [bool]$DryRun = $false,
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
    Write-Info "Working Directory: $PROJECT_ROOT"

    if ($DryRun) {
        Write-Host "  [DRY RUN] Skipped execution." -ForegroundColor Magenta
        return $true
    }

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    # Preserve existing env values
    $oldPythonPath  = $env:PYTHONPATH
    $oldUnbuffered  = $env:PYTHONUNBUFFERED
    $oldCudaAlloc   = $env:PYTORCH_CUDA_ALLOC_CONF
    $oldCudaVisible = $env:CUDA_VISIBLE_DEVICES

    $env:PYTHONPATH           = "$CODE_DIR;$PROJECT_ROOT"
    $env:PYTHONUNBUFFERED     = "1"
    # Limit CUDA fragmentation -- safe no-op on CPU-only machines
    if (-not $env:PYTORCH_CUDA_ALLOC_CONF) {
        $env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"
    }

    $prevEAP = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        Push-Location $PROJECT_ROOT
        & $PythonExe -u $allArgs 2>&1 | ForEach-Object {
            Write-Host "$_"
        }
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $prevEAP
        Pop-Location
        $env:PYTHONPATH              = $oldPythonPath
        $env:PYTHONUNBUFFERED        = $oldUnbuffered
        $env:PYTORCH_CUDA_ALLOC_CONF = $oldCudaAlloc
        $env:CUDA_VISIBLE_DEVICES    = $oldCudaVisible
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
        # Transcript already running from parent orchestrator -- that transcript
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
        [bool]$DryRun = $false,
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

    $oldPythonPath  = $env:PYTHONPATH
    $oldUnbuffered  = $env:PYTHONUNBUFFERED
    $oldCudaAlloc   = $env:PYTORCH_CUDA_ALLOC_CONF

    $env:PYTHONPATH           = "$CODE_DIR;$PROJECT_ROOT"
    $env:PYTHONUNBUFFERED     = "1"
    if (-not $env:PYTORCH_CUDA_ALLOC_CONF) {
        $env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"
    }

    $prevEAP = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        Push-Location $PROJECT_ROOT
        & $PythonExe -u $ScriptPath $Arguments 2>&1 | ForEach-Object {
            Write-Host "$_"
        }
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $prevEAP
        Pop-Location
        $env:PYTHONPATH              = $oldPythonPath
        $env:PYTHONUNBUFFERED        = $oldUnbuffered
        $env:PYTORCH_CUDA_ALLOC_CONF = $oldCudaAlloc
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

