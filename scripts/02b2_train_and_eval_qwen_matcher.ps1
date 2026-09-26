<#
.SYNOPSIS
    Phase 2b-2: Qwen3-0.6B Causal Generative Matcher Training & Gate Evaluation (Stretch Goal).
.DESCRIPTION
    Executes Stage 2b fine-tuning of Qwen3-0.6B with LoRA for sequence-pair classification:
    1. Prepares candidate pair prompts with [COL]/[VAL] delimiter template
    2. Sliced verdict-token Cross-Entropy loss + sliced KL self-distillation
    3. Evaluates cross-country gate (US -> India) before accepting adapter
.PARAMETER DatasetDir
    Root directory containing train TSVs (default: dataset).
.PARAMETER BlockingCandidates
    Path to Stage 1 blocking candidate pairs TSV for negative mining.
.PARAMETER OutputDir
    Output directory for fine-tuned LoRA adapter (default: output\phase2_qwen_matcher).
.PARAMETER Epochs
    Training epochs (default: 3).
.PARAMETER BatchSize
    Per-device batch size (default: 16).
.PARAMETER GradAccumSteps
    Gradient accumulation steps (default: 2).
.PARAMETER LearningRate
    Optimizer learning rate (default: 5e-5).
.PARAMETER LoraR
    LoRA rank (default: 64 with rsLoRA scaling).
.PARAMETER DistillWeight
    Weight for sliced KL self-distillation against frozen base model (default: 0.10).
.PARAMETER NoDistillation
    Disable self-distillation.
.PARAMETER DryRun
    Display commands without executing them.
#>

param(
    [string]$DatasetDir = "",
    [string]$BlockingCandidates = "",
    [string]$OutputDir = "",
    [int]$Epochs = 3,
    [int]$BatchSize = 16,
    [int]$GradAccumSteps = 2,
    [float]$LearningRate = 5e-5,
    [int]$LoraR = 64,
    [float]$DistillWeight = 0.10,
    [switch]$NoDistillation = $false,
    [switch]$DryRun = $false,
    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\common.ps1"

Write-Header "PHASE 2b-2: QWEN3-0.6B GENERATIVE MATCHER LORA TRAINING (STRETCH)"

$python = Get-PythonExecutable -ExplicitPath $PythonPath

if (-not $DatasetDir) {
    $DatasetDir = $DATASET_DIR
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $DEFAULT_OUT "phase2_qwen_matcher"
}
Ensure-Directory $OutputDir

if (-not $BlockingCandidates) {
    $BlockingCandidates = Join-Path $DEFAULT_OUT "phase1_blocking_train\candidate_pairs.tsv"
}

$gtFile = Join-Path $DatasetDir "train_ground_truth.tsv"
$s1File = Join-Path $DatasetDir "train_source1.tsv"
$s2File = Join-Path $DatasetDir "train_source2.tsv"
$s3File = Join-Path $DatasetDir "train_source3.tsv"

$trainArgs = @(
    "--ground_truth_file", $gtFile,
    "--source1_files", $s1File,
    "--candidate_source_files", $s2File, $s3File,
    "--output_dir", $OutputDir,
    "--epochs", $Epochs.ToString(),
    "--batch_size", $BatchSize.ToString(),
    "--grad_accum_steps", $GradAccumSteps.ToString(),
    "--learning_rate", $LearningRate.ToString(),
    "--lora_r", $LoraR.ToString(),
    "--distill_weight", $DistillWeight.ToString(),
    "--mode", "held_out_country",
    "--train_country", "us",
    "--eval_country", "india"
)

if ($BlockingCandidates -and (Test-Path $BlockingCandidates)) {
    $trainArgs += @("--blocking_candidates_file", $BlockingCandidates)
}

if ($NoDistillation) {
    $trainArgs += "--no_distillation"
}

Write-Step "2b2.1" "Fine-tuning Qwen3-0.6B with LoRA & Sliced Verdict Loss..."
Invoke-PythonModule "src.train_qwen_matcher" $trainArgs "Qwen Matcher Training" -DryRun $DryRun -PythonExe $python

if (-not $DryRun) {
    $metaFile = Join-Path $OutputDir "qwen_matcher_train_metadata.json"
    if (Test-Path $metaFile) {
        $meta = Get-Content $metaFile -Raw | ConvertFrom-Json
        $decision = $meta.gate_decision
        Write-Host ""
        Write-Host "  ========================================================" -ForegroundColor Cyan
        Write-Host "  STAGE 2b QWEN MATCHER GATE DECISION: $decision" -ForegroundColor $(if ($decision -eq "GO") { "Green" } else { "Red" })
        Write-Host "  ========================================================" -ForegroundColor Cyan

        if ($decision -eq "GO") {
            Write-Success "Qwen Matcher Adapter passed cross-country gate! Ready for Stage 2c feature extraction."
        } else {
            Write-WarningMessage "Qwen Matcher failed cross-country gate. Reasons:"
            foreach ($r in $meta.gate_reasons) {
                Write-Host "    - $r" -ForegroundColor Yellow
            }
            Write-WarningMessage "As specified in system architecture: Qwen Matcher will be skipped in downstream GBM."
        }
    }
}

Write-Header "PHASE 2b-2 COMPLETE"
