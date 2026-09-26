<#
.SYNOPSIS
    Phase 2c: Representation & Deterministic Pair Feature Extraction.
.DESCRIPTION
    Extracts multi-modal signals for every candidate pair generated in Phase 1:
    - Stage 2c: Deterministic lexical, address, phonetic, postal, conflict, rank & provenance features (src.pair_features)
    - Stage 2a-i: Dense BGE-M3 cosine similarity features (src.bge_features)
    - Stage 2a-ii: Optional Qwen3-Embedding-0.6B cosine similarity features (src.qwen_features)
.PARAMETER CandidateFile
    Path to candidate_pairs.tsv generated in Phase 1.
.PARAMETER ProvenanceFile
    Optional path to candidate_provenance.tsv for blocker rank and score margin features.
.PARAMETER Split
    Dataset split: "train" or "test" (default: "train").
.PARAMETER BgeModel
    BGE-M3 model path or HuggingFace ID (default: "BAAI/bge-m3" or merged fine-tuned model).
.PARAMETER IncludeQwen
    Extract auxiliary Qwen3-Embedding-0.6B features (requires GPU or PyTorch).
.PARAMETER QwenModel
    Qwen embedding model path or HuggingFace ID (default: "Qwen/Qwen3-Embedding-0.6B").
.PARAMETER IncludeQwenMatcher
    Extract Stage 2b Qwen3-0.6B generative-matcher probabilities (stretch goal).
.PARAMETER QwenMatcherAdapter
    Path to fine-tuned LoRA adapter checkpoint that passed the held-out-country gate.
.PARAMETER QwenMatcherModel
    Base model name for generative matcher (default: "Qwen/Qwen3-0.6B").
.PARAMETER OutputDir
    Output directory for feature TSVs (default: output\phase2_features_<split>).
.PARAMETER DryRun
    Display execution commands without executing them.
#>

param(
    [string]$CandidateFile = "",
    [string]$ProvenanceFile = "",
    [ValidateSet("train", "test")]
    [string]$Split = "train",
    [string]$BgeModel = "BAAI/bge-m3",
    [switch]$IncludeQwen = $false,
    [string]$QwenModel = "Qwen/Qwen3-Embedding-0.6B",
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

$python = Get-PythonExecutable -ExplicitPath $PythonPath

if (-not $OutputDir) {
    $OutputDir = Join-Path $DEFAULT_OUT "phase2_features_$Split"
}
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

$candParent = Split-Path -Parent $CandidateFile
if (-not $candParent) { $candParent = "." }

if (-not $ProvenanceFile) {
    $autoProv = Join-Path $candParent "candidate_provenance.tsv"
    if (Test-Path $autoProv) {
        $ProvenanceFile = $autoProv
        Write-Info "Auto-detected Candidate Provenance: $ProvenanceFile"
    }
}

# Resolve Source TSVs
$s1File = Join-Path $DATASET_DIR "$Split\${Split}_source1.tsv"
$s2File = Join-Path $DATASET_DIR "$Split\${Split}_source2.tsv"
$s3File = Join-Path $DATASET_DIR "$Split\${Split}_source3.tsv"

$pairFeaturesOut        = Join-Path $OutputDir "pair_features.tsv"
$bgeFeaturesOut         = Join-Path $OutputDir "bge_pair_features.tsv"
$qwenFeaturesOut        = Join-Path $OutputDir "qwen_pair_features.tsv"
$qwenMatcherFeaturesOut = Join-Path $OutputDir "qwen_matcher_features.tsv"

# ------------------------------------------------------------------------------
# 1. Deterministic Pair Features (Stage 2c)
# ------------------------------------------------------------------------------
Write-Step "2c.1" "Extracting Stage 2c Deterministic Pair Features..."
$pairArgs = @(
    "--source1", $s1File,
    "--source2", $s2File,
    "--source3", $s3File,
    "--candidate-file", $CandidateFile,
    "--output-file", $pairFeaturesOut
)
if ($ProvenanceFile -and (Test-Path $ProvenanceFile)) {
    $pairArgs += @("--provenance-file", $ProvenanceFile)
}

Invoke-PythonModule "src.pair_features" $pairArgs "Deterministic Pair Features" -DryRun $DryRun -PythonExe $python

# ------------------------------------------------------------------------------
# 2. BGE-M3 Dense Cosine Features (Stage 2a)
# ------------------------------------------------------------------------------
Write-Step "2c.2" "Extracting Stage 2a BGE-M3 Cosine Similarity Features..."
$bgeArgs = @(
    "--source1", $s1File,
    "--candidate-sources", $s2File, $s3File,
    "--candidate-file", $CandidateFile,
    "--output", $bgeFeaturesOut,
    "--model-name", $BgeModel
)

Invoke-PythonModule "src.bge_features" $bgeArgs "BGE-M3 Dense Features" -DryRun $DryRun -PythonExe $python

# ------------------------------------------------------------------------------
# 3. Optional Qwen3 Embedding Features (Stage 2a-ii)
# ------------------------------------------------------------------------------
if ($IncludeQwen) {
    Write-Step "2c.3" "Extracting Stage 2a-ii Qwen3-Embedding Cosine Features..."
    $qwenArgs = @(
        "--source1", $s1File,
        "--source2", $s2File,
        "--source3", $s3File,
        "--candidate-file", $CandidateFile,
        "--output-file", $qwenFeaturesOut,
        "--model-name", $QwenModel
    )

    Invoke-PythonModule "src.qwen_features" $qwenArgs "Qwen3 Auxiliary Features" -DryRun $DryRun -PythonExe $python
}

# ------------------------------------------------------------------------------
# 4. Optional Qwen3 Generative Matcher Features (Stage 2b Stretch)
# ------------------------------------------------------------------------------
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
}

if (-not $DryRun) {
    Write-Step "2c.5" "Verifying Feature Extraction Artifacts..."
    if (Test-Path $pairFeaturesOut) {
        $pLines = (Get-Content $pairFeaturesOut | Measure-Object -Line).Lines - 1
        Write-Success "Deterministic features: $pLines pairs -> $pairFeaturesOut"
    }
    if (Test-Path $bgeFeaturesOut) {
        $bLines = (Get-Content $bgeFeaturesOut | Measure-Object -Line).Lines - 1
        Write-Success "BGE features:           $bLines pairs -> $bgeFeaturesOut"
    }
    if ($IncludeQwen -and (Test-Path $qwenFeaturesOut)) {
        $qLines = (Get-Content $qwenFeaturesOut | Measure-Object -Line).Lines - 1
        Write-Success "Qwen features:          $qLines pairs -> $qwenFeaturesOut"
    }
    if ($IncludeQwenMatcher -and (Test-Path $qwenMatcherFeaturesOut)) {
        $mLines = (Get-Content $qwenMatcherFeaturesOut | Measure-Object -Line).Lines - 1
        Write-Success "Qwen Matcher features:  $mLines pairs -> $qwenMatcherFeaturesOut"
    }
}

Write-Header "PHASE 2c COMPLETE"
