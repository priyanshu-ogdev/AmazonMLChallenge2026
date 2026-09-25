<#
.SYNOPSIS
    Phase 0: Environment Verification, Hardware Audit, and Module Smoke Test.
.DESCRIPTION
    Verifies Python environment, required dependencies (numpy, pandas, sklearn, xgboost,
    torch, sentence-transformers, peft), CUDA/GPU VRAM capacity, dataset file existence,
    and runs module contracts and unit test suites.
.PARAMETER PythonPath
    Optional explicit path to python.exe.
.PARAMETER SkipTests
    Skip running the unit test discovery suite.
#>

param(
    [string]$PythonPath = "",
    [switch]$SkipTests = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\common.ps1"

Write-Header "PHASE 0: ENVIRONMENT VERIFICATION & HARDWARE AUDIT"

$python = Get-PythonExecutable -ExplicitPath $PythonPath
Write-Step "0.1" "Resolved Python Interpreter: $python"

# 1. Check Python Version & Platform
$checkVersionCmd = "import sys; print(f'Python {sys.version.split()[0]} on {sys.platform}')"
$pyVer = & $python -c $checkVersionCmd
Write-Info "$pyVer"

# 2. Check Package Availability & Versions
Write-Step "0.2" "Checking Core & ML Dependencies..."
$checkDepsScript = @'
import sys

packages = [
    ("numpy", "Core Array Library"),
    ("pandas", "DataFrames / TSV Parsing"),
    ("scipy", "Scientific Computing"),
    ("sklearn", "Scikit-Learn (Metrics / Splits)"),
    ("xgboost", "Stage 3 GBM"),
    ("torch", "PyTorch (GPU / Tensor Operations)"),
    ("sentence_transformers", "Stage 2a/2b Bi-Encoders"),
    ("peft", "LoRA Adapter Support"),
    ("datasets", "HuggingFace Datasets"),
]

missing = []
for pkg, desc in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'available')
        print(f"  [OK] {pkg:<22} ({ver}) - {desc}")
    except ImportError:
        print(f"  [--] {pkg:<22} (NOT INSTALLED) - {desc}")
        missing.append(pkg)

if missing:
    print(f"\nNote: Missing optional/specialized packages: {', '.join(missing)}")
'@

$checkDepsScript | & $python

# 3. Check GPU / CUDA & Hardware Budget
Write-Step "0.3" "Auditing GPU & CUDA Acceleration..."
$checkGpuScript = @'
try:
    import torch
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        bf16 = torch.cuda.is_bf16_supported()
        print(f"  CUDA Available: True ({count} device(s))")
        print(f"  Primary Device: {name}")
        print(f"  Total VRAM:     {vram_gb:.2f} GB")
        print(f"  bfloat16 Native: {bf16}")
        if vram_gb >= 11.5:
            print(f"  VRAM Budget:    PASSED (>= 12GB target budget for physical batch 48)")
        else:
            print(f"  VRAM Warning:   VRAM < 12GB; reduce batch_size to 32 or mini_batch_size to 8 if OOM occurs")
    else:
        print("  CUDA Available: False (Running in CPU mode)")
except ImportError:
    print("  PyTorch not installed; GPU audit skipped (CPU-only tools available).")
'@

$checkGpuScript | & $python

# 4. Check Dataset Directory & Files
Write-Step "0.4" "Verifying Dataset Paths..."
$trainFiles = @(
    "dataset\train\train_source1.tsv",
    "dataset\train\train_source2.tsv",
    "dataset\train\train_source3.tsv",
    "dataset\train\train_ground_truth.tsv"
)
$testFiles = @(
    "dataset\test\test_source1.tsv",
    "dataset\test\test_source2.tsv",
    "dataset\test\test_source3.tsv"
)

$allFound = $true
foreach ($relPath in ($trainFiles + $testFiles)) {
    $fullPath = Join-Path $PROJECT_ROOT $relPath
    if (Test-Path $fullPath) {
        $sizeMB = [Math]::Round(((Get-Item $fullPath).Length / 1MB), 2)
        Write-Info "$relPath ($sizeMB MB)"
    } else {
        Write-WarningMessage "Missing file: $relPath"
        $allFound = $false
    }
}

if ($allFound) {
    Write-Success "All training and test dataset files verified."
} else {
    Write-WarningMessage "Some dataset files are absent. Verify dataset layout before running full pipeline."
}

# 5. Run Module Architecture & Interface Validator
Write-Step "0.5" "Running Module Interface & Syntax Validation..."
Invoke-PythonScript "code\business_entity_resolution\validate_modules.py" @() "validate_modules.py" -PythonExe $python

# 6. Run Unit Test Suite
if (-not $SkipTests) {
    Write-Step "0.6" "Executing Unit Test Discovery Suite..."
    Invoke-PythonModule "unittest" @("discover", "-s", "tests") "Unit Tests" -PythonExe $python
}

Write-Header "PHASE 0 COMPLETE: ENVIRONMENT IS READY"
