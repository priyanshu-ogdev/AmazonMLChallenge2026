<#
.SYNOPSIS
    Phase 3: Stage 3 Grouped-OOF Gradient Boosted Decision Tree (GBM) Training & Calibration.
.DESCRIPTION
    Trains an entity-grouped, country-stratified XGBoost matching model:
    - Merges Stage 2 deterministic, BGE-M3, and optional Qwen feature tables
    - Applies monotonic domain constraints (e.g. edit sim +, rank margin -)
    - Applies stochastic country masking (rate 0.15) to prevent shortcut learning
    - Carves fold-safe early stopping holdouts to prevent validation leakage
    - Fits out-of-fold leak-safe Platt or Isotonic calibration
    - Optimizes competition macro-F0.5 threshold with ground-truth match accounting
    - Emits gbm.json, stage3_metadata.json, and fold diagnostics
.PARAMETER FeaturesFile
    Path to pair_features.tsv from Phase 2c.
.PARAMETER GroundTruthFile
    Path to train_ground_truth.tsv (default: dataset\train\train_ground_truth.tsv).
.PARAMETER BgeFeatures
    Optional path to bge_pair_features.tsv.
.PARAMETER QwenFeatures
    Optional path to qwen_pair_features.tsv.
.PARAMETER QwenMatcherFeatures
    Optional path to qwen_matcher_features.tsv (Stage 2b stretch generative-matcher features).
.PARAMETER OutputDir
    Output directory for model artifacts (default: output\phase3_gbm).
.PARAMETER Booster
    XGBoost booster type: "gbtree" or "dart" (default: "gbtree").
.PARAMETER Eta
    Learning rate / shrinkage (default: 0.03).
.PARAMETER CountryMaskRate
    Stochastic country masking rate (default: 0.15).
.PARAMETER UseMonotoneConstraints
    Enforce domain monotonic constraints on features (default: true).
.PARAMETER IncludeTFIDF
    Fit fold-safe TF-IDF cosine feature inside cross-validation folds.
.PARAMETER DryRun
    Display execution commands without executing them.
#>

param(
    [string]$FeaturesFile = "",
    [string]$GroundTruthFile = "",
    [string]$BgeFeatures = "",
    [string]$QwenFeatures = "",
    [string]$QwenMatcherFeatures = "",
    [string]$OutputDir = "",
    [ValidateSet("gbtree", "dart")]
    [string]$Booster = "gbtree",
    [float]$Eta = 0.03,
    [float]$CountryMaskRate = 0.15,
    [bool]$UseMonotoneConstraints = $true,
    [bool]$CompareDART = $false,
    [switch]$IncludeTFIDF = $false,
    [switch]$DryRun = $false,
    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\common.ps1"

Write-Header "PHASE 3: STAGE 3 GROUPED-OOF GBM TRAINING & CALIBRATION"

$python = Get-PythonExecutable -ExplicitPath $PythonPath

if (-not $OutputDir) {
    $OutputDir = Join-Path $DEFAULT_OUT "phase3_gbm"
}
Ensure-Directory $OutputDir

# Auto-locate features from Phase 2c if not specified
if (-not $FeaturesFile) {
    $autoFeat = Join-Path $DEFAULT_OUT "phase2_features_train\pair_features.tsv"
    if (Test-Path $autoFeat) {
        $FeaturesFile = $autoFeat
        Write-Info "Auto-detected Pair Features: $FeaturesFile"
    } elseif ($DryRun) {
        $FeaturesFile = $autoFeat
        Write-Info "DryRun: Using planned Pair Features: $FeaturesFile"
    } else {
        throw "FeaturesFile not specified and not found at $autoFeat. Run Phase 2c first."
    }
}

$featParent = Split-Path -Parent $FeaturesFile
if (-not $featParent) { $featParent = "." }

if (-not $BgeFeatures) {
    $autoBge = Join-Path $featParent "bge_pair_features.tsv"
    if (Test-Path $autoBge) {
        $BgeFeatures = $autoBge
        Write-Info "Auto-detected BGE Features: $BgeFeatures"
    }
}

if (-not $QwenFeatures) {
    $autoQwen = Join-Path $featParent "qwen_pair_features.tsv"
    if (Test-Path $autoQwen) {
        $QwenFeatures = $autoQwen
        Write-Info "Auto-detected Qwen Features: $QwenFeatures"
    }
}

if (-not $QwenMatcherFeatures) {
    $autoMatcher = Join-Path $featParent "qwen_matcher_features.tsv"
    if (Test-Path $autoMatcher) {
        $QwenMatcherFeatures = $autoMatcher
        Write-Info "Auto-detected Qwen Matcher Features: $QwenMatcherFeatures"
    }
}

if (-not $GroundTruthFile) {
    $GroundTruthFile = Join-Path $DATASET_DIR "train\train_ground_truth.tsv"
}

Write-Step "3.1" "Configuring Stage 3 Training Parameters..."
Write-Info "Booster:                $Booster (eta=$Eta)"
Write-Info "Country Masking Rate:   $CountryMaskRate"
Write-Info "Monotonic Constraints:  $UseMonotoneConstraints"
if ($BgeFeatures -and (Test-Path $BgeFeatures)) {
    Write-Info "BGE Features:           $BgeFeatures"
}
if ($QwenFeatures -and (Test-Path $QwenFeatures)) {
    Write-Info "Qwen Features:          $QwenFeatures"
}
if ($QwenMatcherFeatures -and (Test-Path $QwenMatcherFeatures)) {
    Write-Info "Qwen Matcher Features:  $QwenMatcherFeatures"
}
Write-Info "Fold-Safe TF-IDF:       $IncludeTFIDF"
Write-Info "Output Artifacts Dir:   $OutputDir"

$trainArgs = @(
    "--mode", "train",
    "--features", $FeaturesFile,
    "--ground-truth", $GroundTruthFile,
    "--output-dir", $OutputDir,
    "--booster", $Booster,
    "--eta", $Eta.ToString(),
    "--country-mask-rate", $CountryMaskRate.ToString()
)

if ($BgeFeatures -and ($DryRun -or (Test-Path $BgeFeatures))) {
    $trainArgs += @("--bge-features", $BgeFeatures)
}
if ($QwenFeatures -and ($DryRun -or (Test-Path $QwenFeatures))) {
    $trainArgs += @("--qwen-features", $QwenFeatures)
}
if ($QwenMatcherFeatures -and ($DryRun -or (Test-Path $QwenMatcherFeatures))) {
    $trainArgs += @("--qwen-matcher-features", $QwenMatcherFeatures)
}
if ($UseMonotoneConstraints) {
    $trainArgs += "--use-monotone-constraints"
}
if ($CompareDART) {
    $trainArgs += "--compare-dart"
}
if ($IncludeTFIDF) {
    $s1Train = Join-Path $DATASET_DIR "train\train_source1.tsv"
    $s2Train = Join-Path $DATASET_DIR "train\train_source2.tsv"
    $s3Train = Join-Path $DATASET_DIR "train\train_source3.tsv"
    $trainArgs += @(
        "--source1", $s1Train,
        "--candidate-sources", $s2Train, $s3Train
    )
}

Write-Step "3.2" "Training Grouped-OOF XGBoost Scorer..."
Invoke-PythonModule "src.scoring" $trainArgs "Stage 3 GBM Training" -DryRun $DryRun -PythonExe $python

if (-not $DryRun) {
    Write-Step "3.3" "Auditing Trained Model & Calibration Artifacts..."
    $metaPath = Join-Path $OutputDir "stage3_metadata.json"
    $gbmPath  = Join-Path $OutputDir "gbm.json"

    if ((Test-Path $metaPath) -and (Test-Path $gbmPath)) {
        $meta = Get-Content $metaPath -Raw | ConvertFrom-Json
        Write-Host ""
        Write-Host "  ========================================================" -ForegroundColor Cyan
        Write-Host "  STAGE 3 SCORING & CALIBRATION RESULTS" -ForegroundColor White
        Write-Host "  ========================================================" -ForegroundColor Cyan
        Write-Host "  Optimal Macro F0.5:   $([Math]::Round($meta.macro_f05, 4))" -ForegroundColor Green
        Write-Host "  Decision Threshold:  $([Math]::Round($meta.threshold, 4))" -ForegroundColor Green
        Write-Host "  OOF Average Precision: $([Math]::Round($meta.average_precision, 4))" -ForegroundColor Green
        Write-Host "  Calibrator Method:    $($meta.calibrator_parameters.method)" -ForegroundColor White
        Write-Host "  Final Estimators:     $($meta.final_n_estimators)" -ForegroundColor White
        Write-Host ""
        Write-Host "  Top 5 Predictive Features (Gain):" -ForegroundColor Yellow
        $gainObj = if ($meta.feature_importance.gain) { $meta.feature_importance.gain } else { $meta.feature_importance }
        $topFeats = $gainObj.PSObject.Properties | Sort-Object { [double]$_.Value } -Descending | Select-Object -First 5
        foreach ($f in $topFeats) {
            Write-Host "    - $($f.Name): $([Math]::Round([double]$f.Value, 4))" -ForegroundColor Gray
        }
        if ($meta.diagnostics.dart_vs_gbtree_comparison) {
            $cmp = $meta.diagnostics.dart_vs_gbtree_comparison
            Write-Host ""
            Write-Host "  DART vs GBDT Cross-Country Diagnostic:" -ForegroundColor Yellow
            Write-Host "    - GBDT Mean AP:       $([Math]::Round([double]$cmp.gbtree_mean_held_out_country_ap, 4))" -ForegroundColor Gray
            Write-Host "    - DART Mean AP:       $([Math]::Round([double]$cmp.dart_mean_held_out_country_ap, 4))" -ForegroundColor Gray
            Write-Host "    - Winning Booster:    $($cmp.winning_booster) (delta: $([Math]::Round([double]$cmp.delta_ap, 4)))" -ForegroundColor Green
        }
        Write-Success "Phase 3 model artifacts saved -> $OutputDir"
    } else {
        Write-ErrorMessage "Stage 3 failed to produce gbm.json or stage3_metadata.json."
    }
}

Write-Header "PHASE 3 COMPLETE"
