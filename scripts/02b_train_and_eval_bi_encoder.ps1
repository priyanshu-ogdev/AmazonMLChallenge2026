<#
.SYNOPSIS
    Phase 2b: Bi-Encoder BGE-M3 LoRA Training & Bidirectional Held-Out Gate Evaluation.
.DESCRIPTION
    Executes the anti-forgetting fine-tuning protocol:
    1. Directional gate training on held-out split (e.g. US -> India)
    2. Bidirectional 2-way evaluation (US -> India and India -> US) with hard-negative
       margin analysis against off-the-shelf BAAI/bge-m3 baseline
    3. Strict GO / NO-GO acceptance check:
       - Recall@10 >= 0.80
       - Relative degradation <= 0.10 vs off-the-shelf baseline
       - Margin pass rate at 0.10 >= 0.60
    4. If GO: Full training on both countries and LoRA weight merge
.PARAMETER DataDir
    Path to prepared dataset directory from Phase 2a (default: output\phase2_prepared_data).
.PARAMETER OutputDir
    Base output directory for models and checkpoints (default: output\phase2_models).
.PARAMETER Epochs
    Number of training epochs (default: 3).
.PARAMETER BatchSize
    Physical training batch size (default: 48).
.PARAMETER MiniBatchSize
    GradCache chunk mini-batch size (default: 16).
.PARAMETER LearningRate
    Optimizer learning rate (default: 2e-5).
.PARAMETER LoraR
    LoRA rank (default: 64, with rank-stabilized scaling rsLoRA).
.PARAMETER DistillWeight
    Cosine self-distillation anchor loss weight (default: 0.10).
.PARAMETER NoDistillation
    Disable self-distillation anchor.
.PARAMETER SkipGate
    Skip the held-out country gate and proceed directly to full training.
.PARAMETER SkipFullTrain
    Stop after the held-out country gate evaluation (do not run full training).
.PARAMETER DryRun
    Display execution commands without executing them.
#>

param(
    [string]$DataDir = "",
    [string]$OutputDir = "",
    [int]$Epochs = 3,
    [int]$BatchSize = 48,
    [int]$MiniBatchSize = 16,
    [float]$LearningRate = 2e-5,
    [int]$LoraR = 64,
    [float]$DistillWeight = 0.10,
    [switch]$NoDistillation = $false,
    [switch]$SkipGate = $false,
    [switch]$SkipFullTrain = $false,
    [switch]$DryRun = $false,
    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\common.ps1"

Write-Header "PHASE 2b: BGE-M3 LORA TRAINING & BIDIRECTIONAL GATE EVALUATION"

$python = Get-PythonExecutable -ExplicitPath $PythonPath

if (-not $DataDir) {
    $DataDir = Join-Path $DEFAULT_OUT "phase2_prepared_data"
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $DEFAULT_OUT "phase2_models"
}
Ensure-Directory $OutputDir

$gateModelDir   = Join-Path $OutputDir "bge-m3-lora-gate"
$finalModelDir  = Join-Path $OutputDir "bge-m3-lora-final"
$mergedModelDir = Join-Path $OutputDir "bge-m3-merged"

# ------------------------------------------------------------------------------
# Step 1: Held-out Country Gate Training
# ------------------------------------------------------------------------------
$gatePassed = $true

if (-not $SkipGate) {
    Write-Step "2b.1" "Training BGE-M3 LoRA for Held-Out Country Gate..."
    Write-Info "Mode: held_out_country | LoRA r=$LoraR | Epochs=$Epochs | Batch=$BatchSize (mini=$MiniBatchSize)"

    $trainGateArgs = @(
        "train",
        "--data_dir", $DataDir,
        "--output_dir", $gateModelDir,
        "--mode", "held_out_country",
        "--epochs", $Epochs.ToString(),
        "--batch_size", $BatchSize.ToString(),
        "--mini_batch_size", $MiniBatchSize.ToString(),
        "--learning_rate", $LearningRate.ToString(),
        "--lora_r", $LoraR.ToString(),
        "--distill_weight", $DistillWeight.ToString()
    )
    if ($NoDistillation) {
        $trainGateArgs += "--no_distillation"
    } else {
        $trainGateArgs += "--use_distillation"
    }

    Invoke-PythonModule "src.train_bi_encoder" $trainGateArgs "Held-Out Country Gate Training" -DryRun $DryRun -PythonExe $python

    # Step 2: Evaluate Bidirectional Gate
    Write-Step "2b.2" "Evaluating Bidirectional Gate against BAAI/bge-m3 baseline..."
    $gateCheckpoint = Join-Path $gateModelDir "final"

    $evalArgs = @(
        "--model_path", $gateCheckpoint,
        "--data_dir", $DataDir,
        "--baseline_model", "BAAI/bge-m3",
        "--direction", "bidirectional",
        "--batch_size", "64"
    )
    Invoke-PythonModule "src.eval_bi_encoder" $evalArgs "Bidirectional Gate Evaluation" -DryRun $DryRun -PythonExe $python

    if (-not $DryRun) {
        $evalResultPath = Join-Path $DataDir "eval_results_bidirectional.json"
        if (Test-Path $evalResultPath) {
            $evalData = Get-Content $evalResultPath -Raw | ConvertFrom-Json
            $decision = $evalData.overall_gate_decision
            Write-Host ""
            Write-Host "  ========================================================" -ForegroundColor Cyan
            Write-Host "  HELD-OUT COUNTRY GATE DECISION: $decision" -ForegroundColor $(if ($decision -eq "GO") { "Green" } else { "Red" })
            Write-Host "  ========================================================" -ForegroundColor Cyan

            if ($decision -ne "GO") {
                $gatePassed = $false
                Write-WarningMessage "Bidirectional Gate resulted in NO-GO. Reasons:"
                foreach ($r in $evalData.gate_reasons) {
                    Write-Host "    - $r" -ForegroundColor Yellow
                }
                Write-WarningMessage "Actionable Recovery Policy (per system architecture plan):"
                Write-Host "    1. Retry with stronger distillation anchor: -DistillWeight 0.15" -ForegroundColor White
                Write-Host "    2. Reduce adapter capacity to limit overfitting: -LoraR 32" -ForegroundColor White
                Write-Host "    3. Fall back to off-the-shelf BAAI/bge-m3 for Stage 2a feature extraction." -ForegroundColor White
                Write-Host "       NEVER silently deploy a NO-GO adapter to production." -ForegroundColor Red
            } else {
                Write-Success "Bidirectional Gate PASSED! Proceeding to full training."
            }
        }
    }
}

# ------------------------------------------------------------------------------
# Step 3: Full Training & LoRA Merge (if Gate PASSED)
# ------------------------------------------------------------------------------
if ($gatePassed -and (-not $SkipFullTrain)) {
    Write-Step "2b.3" "Executing Full Training on Combined Countries..."
    $fullTrainArgs = @(
        "train",
        "--data_dir", $DataDir,
        "--output_dir", $finalModelDir,
        "--mode", "full",
        "--epochs", $Epochs.ToString(),
        "--batch_size", $BatchSize.ToString(),
        "--mini_batch_size", $MiniBatchSize.ToString(),
        "--learning_rate", $LearningRate.ToString(),
        "--lora_r", $LoraR.ToString(),
        "--distill_weight", $DistillWeight.ToString()
    )
    if ($NoDistillation) {
        $fullTrainArgs += "--no_distillation"
    } else {
        $fullTrainArgs += "--use_distillation"
    }

    Invoke-PythonModule "src.train_bi_encoder" $fullTrainArgs "Full Dataset Bi-Encoder Training" -DryRun $DryRun -PythonExe $python

    Write-Step "2b.4" "Merging LoRA Adapter into Standalone SentenceTransformer..."
    $finalCheckpoint = Join-Path $finalModelDir "final"
    $mergeArgs = @(
        "merge",
        "--adapter_path", $finalCheckpoint,
        "--output_path", $mergedModelDir
    )
    Invoke-PythonModule "src.train_bi_encoder" $mergeArgs "Merge LoRA Adapter" -DryRun $DryRun -PythonExe $python

    Write-Success "Phase 2b complete: Merged model ready for zero-overhead inference at -> $mergedModelDir"
}

Write-Header "PHASE 2b COMPLETE"
