#!/usr/bin/env bash
# ==============================================================================
# Phase 3: Stage 3 Grouped-OOF GBM Training & Calibration
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

FEATURES_FILE=""
GROUND_TRUTH_FILE=""
BGE_FEATURES=""
QWEN_FEATURES=""
QWEN_MATCHER_FEATURES=""
OUTPUT_DIR=""
BOOSTER="gbtree"
ETA="0.03"
COUNTRY_MASK_RATE="0.15"
USE_MONOTONE_CONSTRAINTS=true
COMPARE_DART=false
INCLUDE_TFIDF=false
DRY_RUN=false
PYTHON_PATH=""

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --features-file <path>          Path to pair_features.tsv
    --ground-truth-file <path>      Path to train_ground_truth.tsv
    --bge-features <path>           Optional path to bge_pair_features.tsv
    --qwen-features <path>          Optional path to qwen_pair_features.tsv
    --qwen-matcher-features <path>  Optional path to qwen_matcher_features.tsv
    --output-dir <path>             Output directory
    --booster <gbtree|dart>         XGBoost booster type (default: gbtree)
    --eta <F>                       Learning rate (default: 0.03)
    --country-mask-rate <F>         Stochastic country masking rate (default: 0.15)
    --use-monotone-constraints      Enforce constraints (default: true)
    --no-monotone-constraints       Disable constraints
    --compare-dart                  Compare DART vs GBDT (default: false)
    --include-tfidf                 Fit fold-safe TF-IDF (default: false)
    --dry-run                       Dry run
    --python-path <path>            Path to python executable
    -h, --help                      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --features-file|-FeaturesFile) FEATURES_FILE="$2"; shift 2 ;;
        --ground-truth-file|-GroundTruthFile) GROUND_TRUTH_FILE="$2"; shift 2 ;;
        --bge-features|-BgeFeatures) BGE_FEATURES="$2"; shift 2 ;;
        --qwen-features|-QwenFeatures) QWEN_FEATURES="$2"; shift 2 ;;
        --qwen-matcher-features|-QwenMatcherFeatures) QWEN_MATCHER_FEATURES="$2"; shift 2 ;;
        --output-dir|-OutputDir) OUTPUT_DIR="$2"; shift 2 ;;
        --booster|-Booster) BOOSTER="$2"; shift 2 ;;
        --eta|-Eta) ETA="$2"; shift 2 ;;
        --country-mask-rate|-CountryMaskRate) COUNTRY_MASK_RATE="$2"; shift 2 ;;
        --use-monotone-constraints|-UseMonotoneConstraints) USE_MONOTONE_CONSTRAINTS=true; shift 1 ;;
        --no-monotone-constraints) USE_MONOTONE_CONSTRAINTS=false; shift 1 ;;
        --compare-dart|-CompareDART) COMPARE_DART=true; shift 1 ;;
        --no-compare-dart) COMPARE_DART=false; shift 1 ;;
        --include-tfidf|-IncludeTFIDF) INCLUDE_TFIDF=true; shift 1 ;;
        --dry-run|-DryRun) DRY_RUN=true; shift 1 ;;
        --python-path|-PythonPath) PYTHON_PATH="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) write_error "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

write_header "PHASE 3: STAGE 3 GROUPED-OOF GBM TRAINING & CALIBRATION"
initialize_logging "03_train_scoring_gbm"

python="$(get_python_executable "$PYTHON_PATH")"

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${DEFAULT_OUT}/phase3_gbm"
fi
ensure_directory "$OUTPUT_DIR"

if [ -z "$FEATURES_FILE" ]; then
    auto_feat="${DEFAULT_OUT}/phase2_features_train/pair_features.tsv"
    if [ -f "$auto_feat" ]; then
        FEATURES_FILE="$auto_feat"
        write_info "Auto-detected Pair Features: $FEATURES_FILE"
    elif [ "$DRY_RUN" = true ]; then
        FEATURES_FILE="$auto_feat"
        write_info "DryRun: Using planned Pair Features: $FEATURES_FILE"
    else
        write_error "FeaturesFile not specified and not found at $auto_feat. Run Phase 2c first."
        exit 1
    fi
fi

feat_parent="$(dirname "$FEATURES_FILE")"
if [ -z "$feat_parent" ]; then feat_parent="."; fi

if [ -z "$BGE_FEATURES" ]; then
    auto_bge="${feat_parent}/bge_pair_features.tsv"
    if [ -f "$auto_bge" ]; then
        BGE_FEATURES="$auto_bge"
        write_info "Auto-detected BGE Features: $BGE_FEATURES"
    fi
fi

if [ -z "$QWEN_FEATURES" ]; then
    auto_qwen="${feat_parent}/qwen_pair_features.tsv"
    if [ -f "$auto_qwen" ]; then
        QWEN_FEATURES="$auto_qwen"
        write_info "Auto-detected Qwen Features: $QWEN_FEATURES"
    fi
fi

if [ -z "$QWEN_MATCHER_FEATURES" ]; then
    auto_matcher="${feat_parent}/qwen_matcher_features.tsv"
    if [ -f "$auto_matcher" ]; then
        QWEN_MATCHER_FEATURES="$auto_matcher"
        write_info "Auto-detected Qwen Matcher Features: $QWEN_MATCHER_FEATURES"
    fi
fi

if [ -z "$GROUND_TRUTH_FILE" ]; then
    GROUND_TRUTH_FILE="${DATASET_DIR}/train/train_ground_truth.tsv"
fi

write_step "3.1" "Configuring Stage 3 Training Parameters..."
write_info "Booster:                $BOOSTER (eta=$ETA)"
write_info "Country Masking Rate:   $COUNTRY_MASK_RATE"
write_info "Monotonic Constraints:  $USE_MONOTONE_CONSTRAINTS"
if [ -n "$BGE_FEATURES" ] && [ -f "$BGE_FEATURES" ]; then
    write_info "BGE Features:           $BGE_FEATURES"
fi
if [ -n "$QWEN_FEATURES" ] && [ -f "$QWEN_FEATURES" ]; then
    write_info "Qwen Features:          $QWEN_FEATURES"
fi
if [ -n "$QWEN_MATCHER_FEATURES" ] && [ -f "$QWEN_MATCHER_FEATURES" ]; then
    write_info "Qwen Matcher Features:  $QWEN_MATCHER_FEATURES"
fi
write_info "Fold-Safe TF-IDF:       $INCLUDE_TFIDF"
write_info "Output Artifacts Dir:   $OUTPUT_DIR"

train_args=(
    "--mode" "train"
    "--features" "$FEATURES_FILE"
    "--ground-truth" "$GROUND_TRUTH_FILE"
    "--output-dir" "$OUTPUT_DIR"
    "--booster" "$BOOSTER"
    "--eta" "$ETA"
    "--country-mask-rate" "$COUNTRY_MASK_RATE"
)

if [ -n "$BGE_FEATURES" ] && ( [ "$DRY_RUN" = true ] || [ -f "$BGE_FEATURES" ] ); then
    train_args+=("--bge-features" "$BGE_FEATURES")
fi
if [ -n "$QWEN_FEATURES" ] && ( [ "$DRY_RUN" = true ] || [ -f "$QWEN_FEATURES" ] ); then
    train_args+=("--qwen-features" "$QWEN_FEATURES")
fi
if [ -n "$QWEN_MATCHER_FEATURES" ] && ( [ "$DRY_RUN" = true ] || [ -f "$QWEN_MATCHER_FEATURES" ] ); then
    train_args+=("--qwen-matcher-features" "$QWEN_MATCHER_FEATURES")
fi
if [ "$USE_MONOTONE_CONSTRAINTS" = true ]; then
    train_args+=("--use-monotone-constraints")
fi
if [ "$COMPARE_DART" = true ]; then
    train_args+=("--compare-dart")
fi
if [ "$INCLUDE_TFIDF" = true ]; then
    s1_train="${DATASET_DIR}/train/train_source1.tsv"
    s2_train="${DATASET_DIR}/train/train_source2.tsv"
    s3_train="${DATASET_DIR}/train/train_source3.tsv"
    train_args+=(
        "--source1" "$s1_train"
        "--candidate-sources" "$s2_train" "$s3_train"
    )
fi

write_step "3.2" "Training Grouped-OOF XGBoost Scorer..."
invoke_python_module "src.scoring" "Stage 3 GBM Training" "$DRY_RUN" "$python" "${train_args[@]}"

if [ "$DRY_RUN" = false ]; then
    write_step "3.3" "Auditing Trained Model & Calibration Artifacts..."
    meta_path="${OUTPUT_DIR}/stage3_metadata.json"
    gbm_path="${OUTPUT_DIR}/gbm.json"

    if [ -f "$meta_path" ] && [ -f "$gbm_path" ]; then
        "$python" <<EOF
import json
with open("$meta_path") as f:
    meta = json.load(f)
    print("")
    print("\033[0;36m  ========================================================\033[0m")
    print("\033[1;37m  STAGE 3 SCORING & CALIBRATION RESULTS\033[0m")
    print("\033[0;36m  ========================================================\033[0m")
    print(f"\033[0;32m  Optimal Macro F0.5:   {round(meta.get('macro_f05', 0), 4)}\033[0m")
    print(f"\033[0;32m  Decision Threshold:  {round(meta.get('threshold', 0), 4)}\033[0m")
    print(f"\033[0;32m  OOF Average Precision: {round(meta.get('average_precision', 0), 4)}\033[0m")
    
    cal_method = meta.get('calibrator_parameters', {}).get('method') or meta.get('calibrator_parameters', {}).get('name') or meta.get('calibrator', 'unknown')
    loaded_booster = meta.get('booster') or meta.get('params', {}).get('booster', 'unknown')
    
    print(f"\033[1;37m  Calibrator Method:    {cal_method}\033[0m")
    print(f"\033[1;37m  Model Booster:        {loaded_booster}\033[0m")
    print(f"\033[1;37m  Final Estimators:     {meta.get('final_n_estimators', 'unknown')}\033[0m")
    print("")
    print("\033[1;33m  Top 5 Predictive Features (Gain):\033[0m")
    gain_obj = meta.get('feature_importance', {}).get('gain', meta.get('feature_importance', {}))
    if isinstance(gain_obj, dict):
        top_feats = sorted(gain_obj.items(), key=lambda x: float(x[1]), reverse=True)[:5]
        for name, val in top_feats:
            print(f"\033[0;90m    - {name}: {round(float(val), 4)}\033[0m")
    
    cmp = meta.get('diagnostics', {}).get('dart_vs_gbtree_comparison')
    if cmp:
        print("")
        print("\033[1;33m  DART vs GBDT Cross-Country Diagnostic:\033[0m")
        print(f"\033[0;90m    - GBDT Mean AP:       {round(float(cmp.get('gbtree_mean_held_out_country_ap', 0)), 4)}\033[0m")
        print(f"\033[0;90m    - DART Mean AP:       {round(float(cmp.get('dart_mean_held_out_country_ap', 0)), 4)}\033[0m")
        print(f"\033[0;32m    - Winning Booster:    {cmp.get('winning_booster')} (delta: {round(float(cmp.get('delta_ap', 0)), 4)})\033[0m")
EOF
        write_success "Phase 3 model artifacts saved -> $OUTPUT_DIR"
    else
        write_error "Stage 3 failed to produce gbm.json or stage3_metadata.json."
    fi
fi

write_header "PHASE 3 COMPLETE"
close_logging
