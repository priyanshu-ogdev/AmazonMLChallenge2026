<#
.SYNOPSIS
    Phase 2c: Pair Feature Extraction with GPU/CPU Parallelism.
.DESCRIPTION
    Extracts multi-modal signals for every candidate pair generated in Phase 1.
    Execution order is designed for maximum hardware utilization:

      [CPU background]  Stage 2c deterministic pair features   (all CPU cores, ThreadPoolExecutor)
      [GPU foreground]  Stage 2a-i   BGE-M3 cosine features    (GPU, sequential)
                        [VRAM flush]
      [GPU foreground]  Stage 2a-ii  Qwen cosine features      (GPU, sequential, if enabled)
                        [VRAM flush]
      [SYNC]            Wait for CPU pair-features job to finish

    This means GPU is 100% busy while CPU runs in parallel -- total wall time is
    max(CPU_time, GPU_time) instead of their sum.

.PARAMETER CandidateFile
    Path to candidate_pairs.tsv generated in Phase 1.
.PARAMETER ProvenanceFile
    Optional path to candidate_provenance.tsv for blocker rank and score margin features.
.PARAMETER Split
    Dataset split: "train" or "test" (default: "train").
.PARAMETER BgeModel
    BGE-M3 model path or HuggingFace ID (default: "BAAI/bge-m3").
.PARAMETER BgeBatchSize
    BGE-M3 encoding batch size (default: 128 for GPU, 32 for CPU).
.PARAMETER IncludeQwen
    Extract auxiliary Qwen3-Embedding-0.6B features (default: true).
.PARAMETER QwenModel
    Qwen embedding model path or HuggingFace ID (default: "Qwen/Qwen3-Embedding-0.6B").
.PARAMETER QwenBatchSize
    Qwen encoding batch size (default: 64 for GPU, 16 for CPU).
.PARAMETER IncludeQwenMatcher
    Extract Stage 2b Qwen3-0.6B generative-matcher probabilities (stretch goal).
.PARAMETER QwenMatcherAdapter
    Path to fine-tuned LoRA adapter checkpoint that passed the held-out-country gate.
.PARAMETER QwenMatcherModel
    Base model name for generative matcher (default: "Qwen/Qwen3-0.6B").
.PARAMETER OutputDir
    Output directory for feature TSVs.
.PARAMETER DryRun
    Display execution commands without executing them.
#>

param(
    [string]$CandidateFile = "",
    [string]$ProvenanceFile = "",
    [ValidateSet("train", "test")]
    [string]$Split = "train",
    [string]$BgeModel = "BAAI/bge-m3",
    [int]$BgeBatchSize = 0,
    [switch]$IncludeQwen,
    [string]$QwenModel = "Qwen/Qwen3-Embedding-0.6B",
    [int]$QwenBatchSize = 0,
    [switch]$IncludeQwenMatcher = $false,
    [string]$QwenMatcherAdapter = "",
    [string]$QwenMatcherModel = "Qwen/Qwen3-0.6B",
    [string]$OutputDir = "",
    [switch]$DryRun = $false,
    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\common.ps1"

Write-Header "PHASE 2c: PAIR FEATURES EXTRACTION ($($Split.ToUpper()))"

$_log = Initialize-Logging -ScriptName "02c_extract_pair_features_$Split"

$python = Get-PythonExecutable -ExplicitPath $PythonPath

# Probe GPU once -- populates $script:_GpuAvailable, $script:_GpuDevice, etc.
Initialize-Gpu -PythonExe $python

if (-not $OutputDir) {
    $OutputDir = Join-Path $DEFAULT_OUT "phase2_features_$Split"
}
$OutputDir = Resolve-FullPath $OutputDir
Ensure-Directory $OutputDir

# Auto-locate candidate pairs if not specified
if (-not $CandidateFile) {
    $autoCand = Join-Path $DEFAULT_OUT "phase1_blocking_$Split\candidate_pairs.tsv"
    if (Test-Path $autoCand) {
        $CandidateFile = $autoCand
        Write-Info "Auto-detected Candidate Pairs: $CandidateFile"
    } elseif ($DryRun) {
        $CandidateFile = $autoCand
        Write-Info "DryRun: Using planned Candidate Pairs: $CandidateFile"
    } else {
        throw "CandidateFile not specified and not found at $autoCand. Run Phase 1 first."
    }
}
$CandidateFile = Resolve-FullPath $CandidateFile

$candParent = Split-Path -Parent $CandidateFile
if (-not $candParent) { $candParent = "." }

if (-not $ProvenanceFile) {
    $autoProv = Join-Path $candParent "candidate_provenance.tsv"
    if (Test-Path $autoProv) {
        $ProvenanceFile = $autoProv
        Write-Info "Auto-detected Candidate Provenance: $ProvenanceFile"
    }
}
if ($ProvenanceFile) {
    $ProvenanceFile = Resolve-FullPath $ProvenanceFile
}

# Resolve Source TSVs -- prefer Stage 0 normalized files when present
$s1File = Join-Path $DATASET_DIR "$Split\${Split}_source1.tsv"
$s2File = Join-Path $DATASET_DIR "$Split\${Split}_source2.tsv"
$s3File = Join-Path $DATASET_DIR "$Split\${Split}_source3.tsv"

$stage0Candidates = @(
    (Join-Path $candParent "stage0_normalized\${Split}_source1_normalized.tsv"),
    (Join-Path $candParent "stage0_normalized\$Split\${Split}_source1_normalized.tsv"),
    (Join-Path $DEFAULT_OUT "phase1_blocking_$Split\stage0_normalized\${Split}_source1_normalized.tsv"),
    (Join-Path $DEFAULT_OUT "phase1_blocking_$Split\stage0_normalized\$Split\${Split}_source1_normalized.tsv")
)
foreach ($s0s1 in $stage0Candidates) {
    $s0dir   = Split-Path -Parent $s0s1
    $candS1  = $s0s1
    $candS2  = Join-Path $s0dir "${Split}_source2_normalized.tsv"
    $candS3  = Join-Path $s0dir "${Split}_source3_normalized.tsv"
    if ((Test-Path $candS1) -and (Test-Path $candS2) -and (Test-Path $candS3)) {
        $s1File = $candS1
        $s2File = $candS2
        $s3File = $candS3
        Write-Info "Using Stage 0 normalized files: $s0dir"
        break
    }
}
$s1File = Resolve-FullPath $s1File
$s2File = Resolve-FullPath $s2File
$s3File = Resolve-FullPath $s3File

# Resolve batch sizes -- GPU vs CPU defaults
if ($BgeBatchSize -le 0)  { $BgeBatchSize  = if ($script:_GpuAvailable) { 256 } else { 32  } }
if ($QwenBatchSize -le 0) { $QwenBatchSize = if ($script:_GpuAvailable) { 256 } else { 16  } }

$pairFeaturesOut        = Join-Path $OutputDir "pair_features.tsv"
$bgeFeaturesOut         = Join-Path $OutputDir "bge_pair_features.tsv"
$qwenFeaturesOut        = Join-Path $OutputDir "qwen_pair_features.tsv"
$qwenMatcherFeaturesOut = Join-Path $OutputDir "qwen_matcher_features.tsv"

Write-Host ""
Write-Host "  Execution Strategy:" -ForegroundColor Cyan
Write-Host "  - [CPU background] Stage 2c pair features  (all cores, ThreadPoolExecutor)" -ForegroundColor White
Write-Host "  - [GPU foreground] BGE-M3 encoding         (device=$($script:_GpuDevice), batch=$BgeBatchSize)" -ForegroundColor White
if ($IncludeQwen) {
    Write-Host "  - [GPU foreground] Qwen encoding           (device=$($script:_GpuDevice), batch=$QwenBatchSize)" -ForegroundColor White
}
Write-Host "  - [SYNC]           Wait for CPU job" -ForegroundColor White
Write-Host ""

# ==============================================================================
# STEP 1 -- Launch Stage 2c (CPU) as a background job immediately
# ==============================================================================
Write-Step "2c.1" "Launching Stage 2c Deterministic Pair Features in background (CPU all-cores)..."

$pairArgs = @("-m", "src.pair_features",
    "--source1", $s1File,
    "--source2", $s2File,
    "--source3", $s3File,
    "--candidate-file", $CandidateFile,
    "--output-file", $pairFeaturesOut
)
if ($ProvenanceFile -and (Test-Path $ProvenanceFile)) {
    $pairArgs += @("--provenance-file", $ProvenanceFile)
}

$pairJob = $null
if (-not $DryRun) {
    # Run Stage 2c synchronously to prevent memory competition with BGE-M3
    $env:PYTHONPATH = "$CODE_DIR;$PROJECT_ROOT"
    $env:PYTHONUNBUFFERED = "1"
    Push-Location $PROJECT_ROOT
    try {
        & $python -u @pairArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Stage 2c pair features failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
    Write-Success "Pair-features job completed successfully."
} else {
    Write-Host "  [DRY RUN] Would launch: $python $($pairArgs -join ' ')" -ForegroundColor Magenta
}

# ==============================================================================
# STEP 2 -- BGE-M3 Dense Features (GPU, sequential foreground)
# ==============================================================================
Write-Step "2c.2" "Extracting Stage 2a-i BGE-M3 Cosine Similarity Features (GPU foreground)..."
Write-Info "Model:      $BgeModel"
Write-Info "Device:     $($script:_GpuDevice)"
Write-Info "Batch size: $BgeBatchSize"

$bgeArgs = @(
    "--source1", $s1File,
    "--candidate-sources", $s2File, $s3File,
    "--candidate-file", $CandidateFile,
    "--output", $bgeFeaturesOut,
    "--model-name", $BgeModel,
    "--batch-size", $BgeBatchSize.ToString(),
    "--device", $script:_GpuDevice
)

Invoke-PythonModule "src.bge_features" $bgeArgs "BGE-M3 Dense Features" -DryRun $DryRun -PythonExe $python

# Flush VRAM before loading Qwen
Release-GpuMemory -Tag "BGE-M3" -PythonExe $python

# ==============================================================================
# STEP 3 -- Qwen3 Embedding Features (GPU, sequential foreground)
# ==============================================================================
if ($IncludeQwen) {
    Write-Step "2c.3" "Extracting Stage 2a-ii Qwen3-Embedding Cosine Features (GPU foreground)..."
    Write-Info "Model:      $QwenModel"
    Write-Info "Device:     $($script:_GpuDevice)"
    Write-Info "Batch size: $QwenBatchSize"

    $qwenArgs = @(
        "--source1", $s1File,
        "--candidate-sources", $s2File, $s3File,
        "--candidate-file", $CandidateFile,
        "--output-file", $qwenFeaturesOut,
        "--model-name", $QwenModel,
        "--batch-size", $QwenBatchSize.ToString(),
        "--device", $script:_GpuDevice
    )

    Invoke-PythonModule "src.qwen_features" $qwenArgs "Qwen3 Auxiliary Features" -DryRun $DryRun -PythonExe $python

    Release-GpuMemory -Tag "Qwen" -PythonExe $python
}

# ==============================================================================
# STEP 4 -- Qwen Generative Matcher (GPU, sequential, stretch goal)
# ==============================================================================
if ($IncludeQwenMatcher) {
    if (-not $QwenMatcherAdapter) {
        throw "IncludeQwenMatcher was specified, but QwenMatcherAdapter path was not provided."
    }
    Write-Step "2c.4" "Extracting Stage 2b Qwen3-0.6B Generative-Matcher Features..."
    $matcherArgs = @(
        "--source1", $s1File,
        "--candidate-sources", $s2File, $s3File,
        "--candidate-file", $CandidateFile,
        "--output", $qwenMatcherFeaturesOut,
        "--adapter-path", $QwenMatcherAdapter,
        "--base-model-name", $QwenMatcherModel
    )
    Invoke-PythonModule "src.qwen_matcher_features" $matcherArgs "Qwen3 Generative Matcher Features" -DryRun $DryRun -PythonExe $python
    Release-GpuMemory -Tag "QwenMatcher" -PythonExe $python
}

# ==============================================================================
# STEP 5 -- (Skipped: Stage 2c now runs synchronously at Step 1)
# ==============================================================================

# ==============================================================================
# STEP 6 -- Artifact Verification (fast .NET line counter)
# ==============================================================================
if (-not $DryRun) {
    Write-Step "2c.6" "Verifying Feature Extraction Artifacts..."

    if (Test-Path $pairFeaturesOut) {
        $pLines = Get-FastLineCount $pairFeaturesOut
        Write-Success "Deterministic features: $pLines pairs -> $pairFeaturesOut"
    } else {
        Write-ErrorMessage "pair_features.tsv not found at $pairFeaturesOut!"
    }

    if (Test-Path $bgeFeaturesOut) {
        $bLines = Get-FastLineCount $bgeFeaturesOut
        Write-Success "BGE features:           $bLines pairs -> $bgeFeaturesOut"
    } else {
        Write-ErrorMessage "bge_pair_features.tsv not found at $bgeFeaturesOut!"
    }

    if ($IncludeQwen) {
        if (Test-Path $qwenFeaturesOut) {
            $qLines = Get-FastLineCount $qwenFeaturesOut
            Write-Success "Qwen features:          $qLines pairs -> $qwenFeaturesOut"
        } else {
            Write-ErrorMessage "qwen_pair_features.tsv not found at $qwenFeaturesOut!"
        }
    }

    if ($IncludeQwenMatcher -and (Test-Path $qwenMatcherFeaturesOut)) {
        $mLines = Get-FastLineCount $qwenMatcherFeaturesOut
        Write-Success "Qwen Matcher features:  $mLines pairs -> $qwenMatcherFeaturesOut"
    }
}

Write-Header "PHASE 2c COMPLETE ($($Split.ToUpper()))"
Close-Logging
