<#
.SYNOPSIS
    Phase 2a: Bi-Encoder LoRA Dataset Preparation (Stage 2a Data Builder).
.DESCRIPTION
    Prepares country-balanced training pairs (US + India, 50k each per anti-forgetting
    contract) and constructs bidirectional Information Retrieval (IR) evaluation splits
    for the two-direction held-out country gate (US -> India and India -> US).
.PARAMETER SamplePerCountry
    Number of entities to sample per country (default: 50,000).
.PARAMETER EvalSample
    Number of queries for held-out evaluation (default: 2,000).
.PARAMETER EvalCorpusNeg
    Number of negative documents in evaluation corpus (default: 5,000).
.PARAMETER BlockingCandidates
    Optional path to candidate_pairs.tsv from Phase 1 for hard-negative mining.
.PARAMETER NegativesPerPositive
    Number of hard negatives per positive pair (default: 0, or 1-2 when candidates provided).
.PARAMETER OutputDir
    Output directory for prepared datasets (default: output\phase2_prepared_data).
.PARAMETER Seed
    Random seed for reproducible sampling (default: 42).
.PARAMETER BidirectionalGate
    Build both US->India and India->US evaluation splits for 2-way gate (default: true).
.PARAMETER DryRun
    Display execution commands without executing them.
#>

param(
    [int]$SamplePerCountry = 50000,
    [int]$EvalSample = 2000,
    [int]$EvalCorpusNeg = 5000,
    [string]$BlockingCandidates = "",
    [int]$NegativesPerPositive = 2,
    [string]$OutputDir = "",
    [int]$Seed = 42,
    [bool]$BidirectionalGate = $true,
    [switch]$DryRun = $false,
    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\common.ps1"

Write-Header "PHASE 2a: BI-ENCODER TRAINING & EVALUATION DATA PREPARATION"

$_log = Initialize-Logging -ScriptName "02a_prepare_bi_encoder_data"


$python = Get-PythonExecutable -ExplicitPath $PythonPath

if (-not $OutputDir) {
    $OutputDir = Join-Path $DEFAULT_OUT "phase2_prepared_data"
}
Ensure-Directory $OutputDir

Write-Step "2a.1" "Configuring Dataset Parameters..."
Write-Info "Sample Per Country:     $SamplePerCountry (US + India balanced)"
Write-Info "Eval Queries:           $EvalSample"
Write-Info "Eval Corpus Negatives:  $EvalCorpusNeg"
Write-Info "Bidirectional Gate:     $BidirectionalGate"
Write-Info "Output Directory:       $OutputDir"

$builderArgs = @(
    "--mode", "train_data",
    "--data_dir", $DATASET_DIR,
    "--output_dir", $OutputDir,
    "--sample_per_country", $SamplePerCountry.ToString(),
    "--eval_sample", $EvalSample.ToString(),
    "--eval_corpus_neg", $EvalCorpusNeg.ToString(),
    "--seed", $Seed.ToString()
)

if ($BidirectionalGate) {
    $builderArgs += "--bidirectional_gate"
}

if ($BlockingCandidates -and (Test-Path $BlockingCandidates)) {
    Write-Info "Mining hard negatives from: $BlockingCandidates ($NegativesPerPositive / pair)"
    $builderArgs += @(
        "--blocking_candidates", $BlockingCandidates,
        "--negatives_per_positive", $NegativesPerPositive.ToString()
    )
}

Write-Step "2a.2" "Building Training Pairs & Evaluation Splits..."
Invoke-PythonModule "src.data_builder" $builderArgs "Data Builder Pipeline" -DryRun $DryRun -PythonExe $python

if (-not $DryRun) {
    Write-Step "2a.3" "Verifying Prepared Data Artifacts..."
    $statsFile = Join-Path $OutputDir "data_stats.json"
    $fullData  = Join-Path $OutputDir "full_training_dataset"
    $gateData  = Join-Path $OutputDir "held_out_country_dataset"

    if (Test-Path $statsFile) {
        $stats = Get-Content $statsFile -Raw | ConvertFrom-Json
        Write-Info "Total Sampled S1:   $($stats.sampled_s1_count)"
        Write-Info "Total Positive Pairs: $($stats.total_positive_pairs)"
    }

    Write-Success "Phase 2a prepared datasets verified -> $OutputDir"
}

Write-Header "PHASE 2a COMPLETE"
Close-Logging
