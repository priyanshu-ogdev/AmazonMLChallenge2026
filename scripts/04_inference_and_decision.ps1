<#
.SYNOPSIS
    Phase 4: Test Candidate Scoring & Stage 4 Submission Assembly.
.DESCRIPTION
    Scores test candidate pairs using the trained Stage 3 model and leak-safe calibrator,
    then executes the Stage 4 deterministic decision policy:
    1. Scores candidates -> scored_candidates.tsv
    2. Applies optimal macro-F0.5 threshold
    3. Emits exactly one row per test S1 entity in matching_results.tsv
    4. Preserves empty match strings for singletons (guaranteeing 1.0 macro-F0.5 credit)
    5. Packages candidate_pairs.tsv alongside matching_results.tsv for final submission
.PARAMETER ArtifactDir
    Directory containing Stage 3 model artifacts (gbm.json, stage3_metadata.json).
.PARAMETER TestFeatures
    Path to test pair_features.tsv from Phase 2c.
.PARAMETER TestBgeFeatures
    Optional path to test bge_pair_features.tsv.
.PARAMETER TestQwenFeatures
    Optional path to test qwen_pair_features.tsv.
.PARAMETER TestQwenMatcherFeatures
    Optional path to test qwen_matcher_features.tsv (Stage 2b stretch generative-matcher features).
.PARAMETER TestCandidateFile
    Path to test candidate_pairs.tsv from Phase 1.
.PARAMETER OutputDir
    Final output directory for submission package (default: output\phase4_submission).
.PARAMETER DryRun
    Display execution commands without executing them.
#>

param(
    [string]$ArtifactDir = "",
    [string]$TestFeatures = "",
    [string]$TestBgeFeatures = "",
    [string]$TestQwenFeatures = "",
    [string]$TestQwenMatcherFeatures = "",
    [string]$TestCandidateFile = "",
    [string]$OutputDir = "",
    [switch]$DryRun = $false,
    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\common.ps1"

Write-Header "PHASE 4: TEST CANDIDATE SCORING & STAGE 4 DECISION ASSEMBLY"

$python = Get-PythonExecutable -ExplicitPath $PythonPath

if (-not $OutputDir) {
    $OutputDir = Join-Path $DEFAULT_OUT "phase4_submission"
}
Ensure-Directory $OutputDir

if (-not $ArtifactDir) {
    $autoArt = Join-Path $DEFAULT_OUT "phase3_gbm"
    if (Test-Path (Join-Path $autoArt "stage3_metadata.json")) {
        $ArtifactDir = $autoArt
        Write-Info "Auto-detected Stage 3 Artifacts: $ArtifactDir"
    } else {
        throw "ArtifactDir not specified and not found at $autoArt. Run Phase 3 first."
    }
}

if (-not $TestFeatures) {
    $autoFeat = Join-Path $DEFAULT_OUT "phase2_features_test\pair_features.tsv"
    if (Test-Path $autoFeat) {
        $TestFeatures = $autoFeat
        Write-Info "Auto-detected Test Features: $TestFeatures"
    } else {
        throw "TestFeatures not specified and not found at $autoFeat. Run Phase 2c on test split first."
    }
}

if (-not $TestBgeFeatures) {
    $autoBge = Join-Path (Split-Path -Parent $TestFeatures) "bge_pair_features.tsv"
    if (Test-Path $autoBge) {
        $TestBgeFeatures = $autoBge
        Write-Info "Auto-detected Test BGE Features: $TestBgeFeatures"
    }
}

if (-not $TestQwenFeatures) {
    $autoQwen = Join-Path (Split-Path -Parent $TestFeatures) "qwen_pair_features.tsv"
    if (Test-Path $autoQwen) {
        $TestQwenFeatures = $autoQwen
        Write-Info "Auto-detected Test Qwen Features: $TestQwenFeatures"
    }
}

if (-not $TestQwenMatcherFeatures) {
    $autoMatcher = Join-Path (Split-Path -Parent $TestFeatures) "qwen_matcher_features.tsv"
    if (Test-Path $autoMatcher) {
        $TestQwenMatcherFeatures = $autoMatcher
        Write-Info "Auto-detected Test Qwen Matcher Features: $TestQwenMatcherFeatures"
    }
}

if (-not $TestCandidateFile) {
    $autoCand = Join-Path $DEFAULT_OUT "phase1_blocking_test\candidate_pairs.tsv"
    if (Test-Path $autoCand) {
        $TestCandidateFile = $autoCand
        Write-Info "Auto-detected Test Candidates: $TestCandidateFile"
    }
}

$scoredCandidatesOut = Join-Path $OutputDir "scored_candidates.tsv"
$matchingResultsOut  = Join-Path $OutputDir "matching_results.tsv"
$metaFile            = Join-Path $ArtifactDir "stage3_metadata.json"
$s1TestFile          = Join-Path $DATASET_DIR "test\test_source1.tsv"

# ------------------------------------------------------------------------------
# 1. Score Candidates
# ------------------------------------------------------------------------------
Write-Step "4.1" "Applying Stage 3 GBM & Calibration to Test Candidates..."
if ($TestBgeFeatures -and (Test-Path $TestBgeFeatures)) {
    Write-Info "BGE Features:           $TestBgeFeatures"
}
if ($TestQwenFeatures -and (Test-Path $TestQwenFeatures)) {
    Write-Info "Qwen Features:          $TestQwenFeatures"
}
if ($TestQwenMatcherFeatures -and (Test-Path $TestQwenMatcherFeatures)) {
    Write-Info "Qwen Matcher Features:  $TestQwenMatcherFeatures"
}
$scoreArgs = @(
    "--mode", "score",
    "--features", $TestFeatures,
    "--artifact-dir", $ArtifactDir,
    "--output-file", $scoredCandidatesOut
)
if ($TestBgeFeatures -and (Test-Path $TestBgeFeatures)) {
    $scoreArgs += @("--bge-features", $TestBgeFeatures)
}
if ($TestQwenFeatures -and (Test-Path $TestQwenFeatures)) {
    $scoreArgs += @("--qwen-features", $TestQwenFeatures)
}
if ($TestQwenMatcherFeatures -and (Test-Path $TestQwenMatcherFeatures)) {
    $scoreArgs += @("--qwen-matcher-features", $TestQwenMatcherFeatures)
}

# Pass test source files if TF-IDF vectorizer was fitted in Stage 3
$tfidfPath = Join-Path $ArtifactDir "tfidf_vectorizer.joblib"
if (Test-Path $tfidfPath) {
    $s2TestFile = Join-Path $DATASET_DIR "test\test_source2.tsv"
    $s3TestFile = Join-Path $DATASET_DIR "test\test_source3.tsv"
    $scoreArgs += @(
        "--source1", $s1TestFile,
        "--candidate-sources", $s2TestFile, $s3TestFile
    )
}

Invoke-PythonModule "src.scoring" $scoreArgs "Score Test Candidates" -DryRun $DryRun -PythonExe $python

# ------------------------------------------------------------------------------
# 2. Assemble Stage 4 Decision
# ------------------------------------------------------------------------------
Write-Step "4.2" "Assembling Stage 4 Final Matching Results..."
$decisionArgs = @(
    "--scored", $scoredCandidatesOut,
    "--source1", $s1TestFile,
    "--metadata", $metaFile,
    "--output", $matchingResultsOut
)

Invoke-PythonModule "src.decision" $decisionArgs "Assemble Matching Results" -DryRun $DryRun -PythonExe $python

# ------------------------------------------------------------------------------
# 3. Package Candidate Pairs
# ------------------------------------------------------------------------------
if ($TestCandidateFile -and (Test-Path $TestCandidateFile)) {
    Write-Step "4.3" "Packaging candidate_pairs.tsv for Submission..."
    $destCand = Join-Path $OutputDir "candidate_pairs.tsv"
    Copy-Item -Path $TestCandidateFile -Destination $destCand -Force
    Write-Success "Copied candidate pairs -> $destCand"
}

if (-not $DryRun) {
    Write-Step "4.4" "Auditing Output Completeness..."
    if (Test-Path $matchingResultsOut) {
        $mCount = (Get-Content $matchingResultsOut | Measure-Object -Line).Lines - 1
        $s1Expected = (Get-Content $s1TestFile | Measure-Object -Line).Lines - 1
        Write-Info "Emitted S1 Rows:     $mCount / $s1Expected expected"

        if ($mCount -eq $s1Expected) {
            Write-Success "Exact 1-to-1 coverage of Test Source 1 entities confirmed."
        } else {
            Write-ErrorMessage "Row count mismatch: emitted $mCount rows, expected $s1Expected!"
        }
    }
}

Write-Header "PHASE 4 COMPLETE: SUBMISSION PACKAGE ASSEMBLED -> $OutputDir"
