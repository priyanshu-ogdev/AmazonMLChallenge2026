<#
.SYNOPSIS
    Phase 1: Stage 0 Normalization & Stage 1 Multi-Channel Candidate Generation (Blocking).
.DESCRIPTION
    Runs high-recall candidate generation across 7 blocking channels:
    1. Exact normalized name keys
    2. Character 3-gram TF-IDF inverted index
    3. Structural address tokens and building/street matching
    4. Phonetic Soundex/Metaphone keys
    5. Token permutation & set inverted index
    6. City & region partition matching
    7. Optional dense embedding cosine retrieval
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
    [switch]$DryRun = $false,
    [string]$PythonPath = ""
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
Ensure-Directory $OutputDir

# Determine source and ground truth paths
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

# Optional Stage 0 Preprocessing
$s1File = $s1Raw
$s2File = $s2Raw
$s3File = $s3Raw

if ($RunStage0) {
    Write-Step "1.0" "Running Stage 0 Streaming Normalization on $Split..."
    $stage0Out = Join-Path $OutputDir "stage0_normalized"
    Ensure-Directory $stage0Out

    $stage0Args = @(
        "--mode", "stage0_normalize",
        "--data_dir", $DATASET_DIR,
        "--output_dir", $stage0Out,
        "--splits", $Split
    )
    Invoke-PythonModule "src.data_builder" $stage0Args "Stage 0 Normalization" -DryRun $DryRun -PythonExe $python

    $s1Norm = Join-Path $stage0Out "${Split}_source1_normalized.tsv"
    if (-not (Test-Path $s1Norm)) { $s1Norm = Join-Path $stage0Out "${Split}_source1_norm.tsv" }
    $s2Norm = Join-Path $stage0Out "${Split}_source2_normalized.tsv"
    if (-not (Test-Path $s2Norm)) { $s2Norm = Join-Path $stage0Out "${Split}_source2_norm.tsv" }
    $s3Norm = Join-Path $stage0Out "${Split}_source3_normalized.tsv"
    if (-not (Test-Path $s3Norm)) { $s3Norm = Join-Path $stage0Out "${Split}_source3_norm.tsv" }
    if (Test-Path $s1Norm) { $s1File = $s1Norm }
    if (Test-Path $s2Norm) { $s2File = $s2Norm }
    if (Test-Path $s3Norm) { $s3File = $s3Norm }
}

# Run Stage 1 Multi-Channel Blocker
Write-Step "1.1" "Executing Multi-Channel Blocker..."
$blockerArgs = @(
    "--source1", $s1File,
    "--candidates", $s2File, $s3File,
    "--output-dir", $OutputDir,
    "--max-candidates", $MaxCandidates.ToString(),
    "--top-k-sparse", $TopKSparse.ToString(),
    "--top-k-dense", $TopKDense.ToString(),
    "--similarity-floor", $SimilarityFloor.ToString()
)

if ($gtFile -and (Test-Path $gtFile)) {
    $blockerArgs += @("--ground-truth", $gtFile)
}

if ($DenseEmbeddings -and (Test-Path $DenseEmbeddings)) {
    $blockerArgs += @("--dense-embeddings", $DenseEmbeddings)
}

Invoke-PythonModule "src.blocking" $blockerArgs "Stage 1 Blocking" -DryRun $DryRun -PythonExe $python

# Summary verification
if (-not $DryRun) {
    $candPairs = Join-Path $OutputDir "candidate_pairs.tsv"
    $candProv  = Join-Path $OutputDir "candidate_provenance.tsv"
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
            Write-Host "  Pair Recall:     $([Math]::Round($ra.pair_recall * 100, 2))%" -ForegroundColor Green
            Write-Host "  Entity Recall:   $([Math]::Round($ra.entity_recall * 100, 2))%" -ForegroundColor Green
            Write-Host "  Any-Hit Rate:    $([Math]::Round($ra.any_hit_rate * 100, 2))%" -ForegroundColor Green
        }
        Write-Success "Phase 1 Blocking completed -> $OutputDir"
    } else {
        Write-ErrorMessage "Blocking failed to produce candidate_pairs.tsv or blocking_summary.json."
    }
}

Write-Header "PHASE 1 COMPLETE"
Close-Logging
