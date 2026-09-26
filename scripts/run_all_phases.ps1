<#
.SYNOPSIS
    Master Pipeline Orchestrator: End-to-End Business Entity Resolution Pipeline.
.DESCRIPTION
    Automates and coordinates all execution phases from raw TSVs to a fully validated submission:
      Phase 0: Environment verification, hardware audit, and test suite execution
      Phase 1: Stage 0 Normalization & Stage 1 Multi-Channel Candidate Generation (Blocking)
      Phase 2: Representation & Feature Engineering:
               - 2a: Bi-encoder LoRA dataset preparation (50k balanced + bidirectional splits)
               - 2b: BGE-M3 LoRA training and bidirectional held-out country gate evaluation
               - 2c: Stage 2a dense, 2b Qwen (optional), and 2c deterministic pair feature extraction
      Phase 3: Stage 3 Grouped-OOF GBM training, leak-safe calibration & macro-F0.5 threshold search
      Phase 4: Test candidate scoring and Stage 4 singleton-safe decision assembly
      Phase 5: Competition format and containment validation via utils/validate_submission.py
.PARAMETER FromPhase
    Starting phase index (0 to 5, default: 0).
.PARAMETER ToPhase
    Ending phase index (0 to 5, default: 5).
.PARAMETER RunMode
    Execution profile: "Full", "FastSample", "GateOnly", or "InferenceOnly" (default: "Full").
.PARAMETER SkipGPU
    Run in CPU-only mode, using off-the-shelf BAAI/bge-m3 embeddings without fine-tuning.
.PARAMETER IncludeQwen
    Enable auxiliary Qwen3-Embedding-0.6B feature extraction.
.PARAMETER IncludeQwenMatcher
    Enable Stage 2b Qwen3-0.6B generative-matcher feature extraction (stretch goal).
.PARAMETER QwenMatcherAdapter
    Path to fine-tuned LoRA adapter checkpoint that passed the held-out-country gate.
.PARAMETER RunStage0
    Pre-run Stage 0 streaming normalization.
.PARAMETER DryRun
    Display execution commands without executing them.
#>

param(
    [ValidateRange(0, 5)]
    [int]$FromPhase = 0,
    [ValidateRange(0, 5)]
    [int]$ToPhase = 5,
    [ValidateSet("Full", "FastSample", "GateOnly", "InferenceOnly")]
    [string]$RunMode = "Full",
    [switch]$SkipGPU = $false,
    [bool]$IncludeQwen = $true,
    [bool]$IncludeQwenMatcher = $false,
    [string]$QwenMatcherAdapter = "",
    [switch]$RunStage0 = $false,
    [int]$NegativesPerPositive = 2,
    [ValidateSet("gbtree", "dart")]
    [string]$Booster = "gbtree",
    [bool]$CompareDART = $false,
    [bool]$Injective = $true,
    [switch]$DryRun = $false,
    [string]$OutputDir = "",
    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\common.ps1"

$pipelineStopwatch = [System.Diagnostics.Stopwatch]::StartNew()

Write-Header "AMAZON ML CHALLENGE 2026: END-TO-END PIPELINE ORCHESTRATOR"

$python = Get-PythonExecutable -ExplicitPath $PythonPath

if (-not $OutputDir) {
    $OutputDir = $DEFAULT_OUT
}
Ensure-Directory $OutputDir

# Display Run Plan
Write-Host "  Pipeline Configuration:" -ForegroundColor Cyan
Write-Host "  - From Phase:            $FromPhase" -ForegroundColor White
Write-Host "  - To Phase:              $ToPhase" -ForegroundColor White
Write-Host "  - Run Mode:              $RunMode" -ForegroundColor White
Write-Host "  - Skip GPU:              $SkipGPU (Use BGE-M3 base if true)" -ForegroundColor White
Write-Host "  - Include Qwen:          $IncludeQwen" -ForegroundColor White
Write-Host "  - Include Qwen Matcher:  $IncludeQwenMatcher" -ForegroundColor White
Write-Host "  - Run Stage 0:           $RunStage0" -ForegroundColor White
Write-Host "  - Dry Run:               $DryRun" -ForegroundColor White
Write-Host "  - Python:                $python" -ForegroundColor White
Write-Host "  - Output Root:           $OutputDir" -ForegroundColor White
Write-Host ""

$phaseTimings = @{}

function Run-PipelinePhase {
    param(
        [int]$PhaseNum,
        [string]$PhaseName,
        [scriptblock]$Action
    )

    if ($PhaseNum -lt $FromPhase -or $PhaseNum -gt $ToPhase) {
        Write-Host ">>> Skipping Phase $PhaseNum ($PhaseName) per execution range [$FromPhase..$ToPhase]" -ForegroundColor DarkGray
        return
    }

    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Yellow
    Write-Host "  STARTING PHASE ${PhaseNum} - $PhaseName" -ForegroundColor Yellow
    Write-Host "================================================================================" -ForegroundColor Yellow

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        & $Action
        $sw.Stop()
        $sec = [Math]::Round($sw.Elapsed.TotalSeconds, 2)
        $phaseTimings["Phase ${PhaseNum} - $PhaseName"] = "$($sec)s"
        Write-Success "Phase ${PhaseNum} ($PhaseName) succeeded in $($sec)s."
    }
    catch {
        $sw.Stop()
        $sec = [Math]::Round($sw.Elapsed.TotalSeconds, 2)
        $phaseTimings["Phase ${PhaseNum} - $PhaseName"] = "FAILED after $($sec)s"
        Write-ErrorMessage "Phase ${PhaseNum} ($PhaseName) failed: $_"
        throw $_
    }
}

# ------------------------------------------------------------------------------
# Phase 0: Environment & Smoke Test
# ------------------------------------------------------------------------------
Run-PipelinePhase 0 "Environment Verification & Smoke Test" {
    & "$PSScriptRoot\00_verify_environment.ps1" -PythonPath $python
}

if ($RunMode -eq "InferenceOnly") {
    $FromPhase = [Math]::Max($FromPhase, 4)
}

# ------------------------------------------------------------------------------
# Phase 1: Candidate Generation (Blocking) - Train & Test
# ------------------------------------------------------------------------------
Run-PipelinePhase 1 "Candidate Generation / Blocking" {
    $sampleMax = if ($RunMode -eq "FastSample") { 20 } else { 50 }

    # Train blocking
    & "$PSScriptRoot\01_run_blocking.ps1" `
        -Split "train" `
        -RunStage0:$RunStage0 `
        -MaxCandidates $sampleMax `
        -OutputDir (Join-Path $OutputDir "phase1_blocking_train") `
        -DryRun:$DryRun `
        -PythonPath $python

    # Test blocking
    & "$PSScriptRoot\01_run_blocking.ps1" `
        -Split "test" `
        -RunStage0:$RunStage0 `
        -MaxCandidates $sampleMax `
        -OutputDir (Join-Path $OutputDir "phase1_blocking_test") `
        -DryRun:$DryRun `
        -PythonPath $python
}

# ------------------------------------------------------------------------------
# Phase 2: Representation & Features
# ------------------------------------------------------------------------------
Run-PipelinePhase 2 "Representation & Feature Engineering" {
    $p2PrepOut = Join-Path $OutputDir "phase2_prepared_data"
    $p2ModelOut = Join-Path $OutputDir "phase2_models"
    $trainCandFile = Join-Path $OutputDir "phase1_blocking_train\candidate_pairs.tsv"
    $testCandFile  = Join-Path $OutputDir "phase1_blocking_test\candidate_pairs.tsv"

    # 2a. Data Prep
    $samples = if ($RunMode -eq "FastSample") { 5000 } else { 50000 }
    & "$PSScriptRoot\02a_prepare_bi_encoder_data.ps1" `
        -SamplePerCountry $samples `
        -BlockingCandidates $trainCandFile `
        -NegativesPerPositive $NegativesPerPositive `
        -OutputDir $p2PrepOut `
        -DryRun:$DryRun `
        -PythonPath $python

    # 2b. LoRA Fine-Tuning & Gate
    $bgeModel = "BAAI/bge-m3"
    if (-not $SkipGPU) {
        $skipFull = ($RunMode -eq "GateOnly")
        & "$PSScriptRoot\02b_train_and_eval_bi_encoder.ps1" `
            -DataDir $p2PrepOut `
            -OutputDir $p2ModelOut `
            -SkipFullTrain:$skipFull `
            -DryRun:$DryRun `
            -PythonPath $python

        $mergedPath = Join-Path $p2ModelOut "bge-m3-merged"
        if (Test-Path $mergedPath) {
            $bgeModel = $mergedPath
        }
    } else {
        Write-Info "SkipGPU set: Using off-the-shelf BAAI/bge-m3 for Stage 2a features without fine-tuning."
    }

    if ($RunMode -eq "GateOnly") {
        Write-Info "GateOnly mode specified: stopping pipeline after Phase 2b bi-encoder gate."
        return
    }

    # 2c. Feature Extraction (Train & Test)
    & "$PSScriptRoot\02c_extract_pair_features.ps1" `
        -CandidateFile $trainCandFile `
        -Split "train" `
        -BgeModel $bgeModel `
        -IncludeQwen:$IncludeQwen `
        -IncludeQwenMatcher:$IncludeQwenMatcher `
        -QwenMatcherAdapter $QwenMatcherAdapter `
        -OutputDir (Join-Path $OutputDir "phase2_features_train") `
        -DryRun:$DryRun `
        -PythonPath $python

    & "$PSScriptRoot\02c_extract_pair_features.ps1" `
        -CandidateFile $testCandFile `
        -Split "test" `
        -BgeModel $bgeModel `
        -IncludeQwen:$IncludeQwen `
        -IncludeQwenMatcher:$IncludeQwenMatcher `
        -QwenMatcherAdapter $QwenMatcherAdapter `
        -OutputDir (Join-Path $OutputDir "phase2_features_test") `
        -DryRun:$DryRun `
        -PythonPath $python
}

if ($RunMode -eq "GateOnly") {
    $pipelineStopwatch.Stop()
    Write-Header "GATE-ONLY RUN COMPLETE"
    return
}

# ------------------------------------------------------------------------------
# Phase 3: Stage 3 Grouped-OOF GBM & Calibration
# ------------------------------------------------------------------------------
Run-PipelinePhase 3 "Grouped-OOF GBM Training & Calibration" {
    $trainFeats       = Join-Path $OutputDir "phase2_features_train\pair_features.tsv"
    $bgeFeats         = Join-Path $OutputDir "phase2_features_train\bge_pair_features.tsv"
    $qwenFeats        = Join-Path $OutputDir "phase2_features_train\qwen_pair_features.tsv"
    $qwenMatcherFeats = Join-Path $OutputDir "phase2_features_train\qwen_matcher_features.tsv"
    $p3ModelOut       = Join-Path $OutputDir "phase3_gbm"

    $p3Params = @{
        FeaturesFile = $trainFeats
        BgeFeatures  = $bgeFeats
        OutputDir    = $p3ModelOut
        Booster      = $Booster
        CompareDART  = $CompareDART
        DryRun       = $DryRun
        PythonPath   = $python
    }
    if ($IncludeQwen -and ($DryRun -or (Test-Path $qwenFeats))) {
        $p3Params["QwenFeatures"] = $qwenFeats
    }
    if ($IncludeQwenMatcher -and ($DryRun -or (Test-Path $qwenMatcherFeats))) {
        $p3Params["QwenMatcherFeatures"] = $qwenMatcherFeats
    }

    & "$PSScriptRoot\03_train_scoring_gbm.ps1" @p3Params
}

# ------------------------------------------------------------------------------
# Phase 4: Test Scoring & Stage 4 Decision Assembly
# ------------------------------------------------------------------------------
Run-PipelinePhase 4 "Test Scoring & Stage 4 Decision Assembly" {
    $testFeats        = Join-Path $OutputDir "phase2_features_test\pair_features.tsv"
    $testBge          = Join-Path $OutputDir "phase2_features_test\bge_pair_features.tsv"
    $testQwen         = Join-Path $OutputDir "phase2_features_test\qwen_pair_features.tsv"
    $testQwenMatcher  = Join-Path $OutputDir "phase2_features_test\qwen_matcher_features.tsv"
    $testCand         = Join-Path $OutputDir "phase1_blocking_test\candidate_pairs.tsv"
    $p3ModelOut       = Join-Path $OutputDir "phase3_gbm"
    $p4SubOut         = Join-Path $OutputDir "phase4_submission"

    $p4Params = @{
        ArtifactDir       = $p3ModelOut
        TestFeatures      = $testFeats
        TestBgeFeatures   = $testBge
        TestCandidateFile = $testCand
        OutputDir         = $p4SubOut
        Injective         = $Injective
        DryRun            = $DryRun
        PythonPath        = $python
    }
    if ($IncludeQwen -and ($DryRun -or (Test-Path $testQwen))) {
        $p4Params["TestQwenFeatures"] = $testQwen
    }
    if ($IncludeQwenMatcher -and ($DryRun -or (Test-Path $testQwenMatcher))) {
        $p4Params["TestQwenMatcherFeatures"] = $testQwenMatcher
    }

    & "$PSScriptRoot\04_inference_and_decision.ps1" @p4Params
}

# ------------------------------------------------------------------------------
# Phase 5: Submission Validation
# ------------------------------------------------------------------------------
Run-PipelinePhase 5 "Competition Submission Validation" {
    $p4SubOut = Join-Path $OutputDir "phase4_submission"
    & "$PSScriptRoot\05_validate_submission.ps1" `
        -SubmissionDir $p4SubOut `
        -DryRun:$DryRun `
        -PythonPath $python
}

$pipelineStopwatch.Stop()
$totalSec = [Math]::Round($pipelineStopwatch.Elapsed.TotalSeconds, 2)

# Write Summary Report
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  PIPELINE EXECUTION SUMMARY" -ForegroundColor White
Write-Host "================================================================================" -ForegroundColor Cyan
foreach ($key in $phaseTimings.Keys) {
    Write-Host "  $($key.PadRight(50)): $($phaseTimings[$key])" -ForegroundColor Green
}
Write-Host "  ----------------------------------------------------------------" -ForegroundColor Gray
Write-Host "  Total Cumulative Time:                             $($totalSec)s" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

# Save Report Markdown
$reportPath = Join-Path $OutputDir "pipeline_execution_summary.md"
$reportLines = New-Object System.Collections.Generic.List[string]
$reportLines.Add("# Amazon ML Challenge 2026 - Pipeline Execution Summary")
$reportLines.Add("")
$reportLines.Add("- **Execution Date:** $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))")
$reportLines.Add("- **Run Mode:** $RunMode")
$reportLines.Add("- **Skip GPU:** $SkipGPU")
$reportLines.Add("- **Include Qwen:** $IncludeQwen")
$reportLines.Add("- **Total Duration:** $($totalSec)s")
$reportLines.Add("")
$reportLines.Add("## Phase Timings")
$reportLines.Add("")
$reportLines.Add("| Phase | Duration / Status |")
$reportLines.Add("|---|---|")
foreach ($key in $phaseTimings.Keys) {
    $reportLines.Add("| $key | $($phaseTimings[$key]) |")
}
$reportLines.Add("")
$reportLines.Add("## Key Artifact Locations")
$reportLines.Add("- **Phase 1 Train Blocking:** output/phase1_blocking_train")
$reportLines.Add("- **Phase 1 Test Blocking:** output/phase1_blocking_test")
$reportLines.Add("- **Phase 2 Prepared Data:** output/phase2_prepared_data")
$reportLines.Add("- **Phase 3 Model & Calibration:** output/phase3_gbm")
$reportLines.Add("- **Phase 4 Final Submission Package:** output/phase4_submission")
$reportLines.Add("  - matching_results.tsv")
$reportLines.Add("  - candidate_pairs.tsv")

Set-Content -Path $reportPath -Value $reportLines
Write-Info "Saved execution summary -> $reportPath"

Write-Header "PIPELINE ORCHESTRATION COMPLETE"
