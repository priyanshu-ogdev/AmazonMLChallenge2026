<#
.SYNOPSIS
    Phase 1: Stage 0 Normalization & Stage 1 Multi-Channel Candidate Generation (Blocking).
.DESCRIPTION
    Runs high-recall candidate generation across 7 blocking channels.
    Emits candidate_pairs.tsv, candidate_provenance.tsv, and blocking_summary.json.
.PARAMETER Split
    Target dataset split: "train" or "test" (default: "train").
.PARAMETER RunStage0
    Pre-run Stage 0 streaming normalization on raw dataset files.
.PARAMETER MaxCandidates
    Maximum candidates retained per S1 entity (default: 50).
.PARAMETER TopKSparse
    Top-K retrieval depth for sparse and token channels (default: 50).
.PARAMETER TopKDense
    Top-K retrieval depth for dense embedding channel (default: 50).
.PARAMETER SimilarityFloor
    Minimum cosine similarity floor for dense candidates (default: 0.3).
.PARAMETER DenseEmbeddings
    Optional path to .npz file containing precomputed dense embeddings.
.PARAMETER OutputDir
    Output directory for candidate pairs and blocking summary.
.PARAMETER NumWorkers
    Number of parallel ThreadPoolExecutor workers for the query phase.
    Default: 0 = auto-select based on available RAM:
      < 28 GB  -> 1 worker (Kaggle / low-RAM)
      28-44 GB -> 2 workers
      >= 45 GB -> 4 workers
    All workers share the same in-process index (no RAM duplication on Windows).
.PARAMETER NoResume
    If set, overwrite existing output and re-run all S1 entities (disables checkpoint resume).
.PARAMETER NoCacheIndex
    If set, force a full index rebuild even if a valid cache exists.
.PARAMETER CheckpointInterval
    Flush output and write checkpoint.json every N entities (default: 10000).
.PARAMETER DryRun
    Display execution commands without executing them.
#>

param(
    [ValidateSet("train", "test")]
    [string]$Split = "train",
    [switch]$RunStage0 = $false,
    [int]$MaxCandidates = 50,
    [int]$TopKSparse = 50,
    [int]$TopKDense = 50,
    [float]$SimilarityFloor = 0.3,
    [string]$DenseEmbeddings = "",
    [string]$OutputDir = "",
    [int]$NumWorkers = 0,
    [switch]$NoResume = $false,
    [switch]$NoCacheIndex = $false,
    [int]$CheckpointInterval = 10000,
    [switch]$DryRun = $false,
    [string]$PythonPath = "",
    [switch]$UseFast = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\common.ps1"

Write-Header "PHASE 1: STAGE 0 NORMALIZATION & STAGE 1 BLOCKING ($($Split.ToUpper()))"

$_log = Initialize-Logging -ScriptName "01_run_blocking_$Split"

$python = Get-PythonExecutable -ExplicitPath $PythonPath

# Determine default output directory
if (-not $OutputDir) {
    $OutputDir = Join-Path $DEFAULT_OUT "phase1_blocking_$Split"
}
$OutputDir = Resolve-FullPath $OutputDir
Ensure-Directory $OutputDir

# ------------------------------------------------------------------------------
# Worker count (Windows ThreadPoolExecutor -- memory is shared, not duplicated)
# Workers share the same blocking index in-process; more workers = more CPU utilisation,
# NOT more RAM. We can safely use all available CPU cores.
# ------------------------------------------------------------------------------
if ($NumWorkers -le 0) {
    $NumWorkers = [Environment]::ProcessorCount
    # Cap at 16 to avoid extreme GIL contention overhead on massive threadrippers, 
    # but for most consumer machines this will just use all cores.
    if ($NumWorkers -gt 16) { $NumWorkers = 16 }
    Write-Info "Auto-selected NumWorkers=$NumWorkers based on CPU cores (ThreadPoolExecutor shares RAM)."
}

# Determine source file paths
if ($Split -eq "train") {
    $s1Raw   = Join-Path $DATASET_DIR "train\train_source1.tsv"
    $s2Raw   = Join-Path $DATASET_DIR "train\train_source2.tsv"
    $s3Raw   = Join-Path $DATASET_DIR "train\train_source3.tsv"
    $gtFile  = Join-Path $DATASET_DIR "train\train_ground_truth.tsv"
} else {
    $s1Raw   = Join-Path $DATASET_DIR "test\test_source1.tsv"
    $s2Raw   = Join-Path $DATASET_DIR "test\test_source2.tsv"
    $s3Raw   = Join-Path $DATASET_DIR "test\test_source3.tsv"
    $gtFile  = ""
}

$s1File = $s1Raw
$s2File = $s2Raw
$s3File = $s3Raw

# ------------------------------------------------------------------------------
# Stage 0 normalized file discovery -- single ordered search
# ------------------------------------------------------------------------------
$stage0Out = Join-Path $OutputDir "stage0_normalized"
$s0Roots = @(
    $stage0Out,
    (Join-Path $stage0Out $Split)
)
foreach ($root in $s0Roots) {
    $cS1 = Join-Path $root "${Split}_source1_normalized.tsv"
    $cS2 = Join-Path $root "${Split}_source2_normalized.tsv"
    $cS3 = Join-Path $root "${Split}_source3_normalized.tsv"
    # Also accept _norm.tsv suffix produced by some Stage 0 variants
    if (-not (Test-Path $cS1)) { $cS1 = Join-Path $root "${Split}_source1_norm.tsv" }
    if (-not (Test-Path $cS2)) { $cS2 = Join-Path $root "${Split}_source2_norm.tsv" }
    if (-not (Test-Path $cS3)) { $cS3 = Join-Path $root "${Split}_source3_norm.tsv" }

    if ((Test-Path $cS1) -and (Test-Path $cS2) -and (Test-Path $cS3)) {
        # Integrity check: normalized files must not be truncated relative to raw
        $s2RawLen = (Get-Item $s2Raw).Length
        $s2NormLen = (Get-Item $cS2).Length
        if ($s2NormLen -lt $s2RawLen) {
            Write-WarningMessage "Stage 0 file $cS2 appears truncated (${s2NormLen} bytes vs raw ${s2RawLen} bytes). Skipping corrupted normalized cache."
            continue
        }
        $s1File = $cS1
        $s2File = $cS2
        $s3File = $cS3
        Write-Info "Using Stage 0 normalized files: $root"
        break
    }
}

# Optional Stage 0 Preprocessing
if ($RunStage0) {
    Write-Step "1.0" "Running Stage 0 Streaming Normalization on $Split..."
    Ensure-Directory $stage0Out
    $stage0Args = @(
        "--mode", "stage0_normalize",
        "--data_dir", $DATASET_DIR,
        "--output_dir", $stage0Out,
        "--splits", $Split
    )
    Invoke-PythonModule "src.data_builder" $stage0Args "Stage 0 Normalization" -DryRun $DryRun -PythonExe $python
    # Re-probe normalized file locations after normalization completes
    foreach ($root in $s0Roots) {
        $cS1 = Join-Path $root "${Split}_source1_normalized.tsv"
        $cS2 = Join-Path $root "${Split}_source2_normalized.tsv"
        $cS3 = Join-Path $root "${Split}_source3_normalized.tsv"
        if ((Test-Path $cS1) -and (Test-Path $cS2) -and (Test-Path $cS3)) {
            $s1File = $cS1; $s2File = $cS2; $s3File = $cS3
            Write-Info "Using freshly normalized files: $root"
            break
        }
    }
}

$s1File = Resolve-FullPath $s1File
$s2File = Resolve-FullPath $s2File
$s3File = Resolve-FullPath $s3File
if ($gtFile) { $gtFile = Resolve-FullPath $gtFile }
if ($DenseEmbeddings) { $DenseEmbeddings = Resolve-FullPath $DenseEmbeddings }

Write-Info "Source 1: $s1File"
Write-Info "Source 2: $s2File"
Write-Info "Source 3: $s3File"
Write-Info "Workers:  $NumWorkers (ThreadPoolExecutor -- shared index, no RAM duplication)"

# Run Stage 1 Multi-Channel Blocker
Write-Step "1.1" "Executing Multi-Channel Blocker ($NumWorkers workers, resume=$((-not $NoResume)))..."
$blockerArgs = @(
    "--source1", $s1File,
    "--candidates", $s2File, $s3File,
    "--output-dir", $OutputDir,
    "--max-candidates", $MaxCandidates.ToString(),
    "--top-k-sparse", $TopKSparse.ToString(),
    "--top-k-dense", $TopKDense.ToString(),
    "--similarity-floor", $SimilarityFloor.ToString(),
    "--num-workers", $NumWorkers.ToString(),
    "--checkpoint-interval", $CheckpointInterval.ToString()
)

if ($NoCacheIndex)  { $blockerArgs += "--no-cache-index" }
if ($NoResume)      { $blockerArgs += "--no-resume" }

if ($gtFile -and (Test-Path $gtFile)) {
    $blockerArgs += @("--ground-truth", $gtFile)
}

if ($DenseEmbeddings -and (Test-Path $DenseEmbeddings)) {
    $blockerArgs += @("--dense-embeddings", $DenseEmbeddings)
}

$moduleName = if ($UseFast) { "src.fast_blocking" } else { "src.blocking" }
Invoke-PythonModule $moduleName $blockerArgs "Stage 1 Blocking" -DryRun $DryRun -PythonExe $python

# Summary verification
if (-not $DryRun) {
    $candPairs = Join-Path $OutputDir "candidate_pairs.tsv"
    $summary   = Join-Path $OutputDir "blocking_summary.json"

    if ((Test-Path $candPairs) -and (Test-Path $summary)) {
        Write-Step "1.2" "Auditing Blocking Artifacts..."
        $summaryJson = Get-Content $summary -Raw | ConvertFrom-Json
        Write-Info "S1 Entities:         $($summaryJson.s1_count)"
        Write-Info "Total Pairs:         $($summaryJson.total_pairs)"
        Write-Info "Candidates/S1 Mean:  $($summaryJson.mean_candidates_per_s1)"
        Write-Info "Candidates/S1 P90:   $($summaryJson.p90_candidates_per_s1)"
        Write-Info "True Singletons:     $($summaryJson.singleton_s1_count)"

        if ($summaryJson.PSObject.Properties["recall_audit"] -and $summaryJson.recall_audit) {
            $ra = $summaryJson.recall_audit
            Write-Host ""
            Write-Host "  RECALL AUDIT REPORT (Train Ground Truth):" -ForegroundColor Cyan
            Write-Host "  Pair Recall:     $([Math]::Round($ra.pair_recall * 100, 2))%"   -ForegroundColor Green
            Write-Host "  Entity Recall:   $([Math]::Round($ra.entity_recall * 100, 2))%" -ForegroundColor Green
            Write-Host "  Any-Hit Rate:    $([Math]::Round($ra.any_hit_rate * 100, 2))%"  -ForegroundColor Green
        }
        Write-Success "Phase 1 Blocking completed -> $OutputDir"
    } else {
        Write-ErrorMessage "Blocking failed to produce candidate_pairs.tsv or blocking_summary.json."
    }
}

Write-Header "PHASE 1 COMPLETE"
Close-Logging
