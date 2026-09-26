#!/usr/bin/env bash
# ==============================================================================
# Phase 4: Stage 4 Test Inference & Clustering
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

FEATURES_FILE=""
MODEL_DIR=""
OUTPUT_DIR=""
DRY_RUN=false
PYTHON_PATH=""

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --features-file <path>          Path to test pair_features.tsv
    --model-dir <path>              Directory containing gbm.json and stage3_metadata.json
    --output-dir <path>             Output directory
    --dry-run                       Dry run
    --python-path <path>            Path to python executable
    -h, --help                      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --features-file|-FeaturesFile) FEATURES_FILE="$2"; shift 2 ;;
        --model-dir|-ModelDir) MODEL_DIR="$2"; shift 2 ;;
        --output-dir|-OutputDir) OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run|-DryRun) DRY_RUN=true; shift 1 ;;
        --python-path|-PythonPath) PYTHON_PATH="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) write_error "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

write_header "PHASE 4: STAGE 4 TEST INFERENCE & CLUSTERING"
initialize_logging "04_predict_and_cluster"

python="$(get_python_executable "$PYTHON_PATH")"

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${DEFAULT_OUT}/phase4_predictions"
fi
ensure_directory "$OUTPUT_DIR"

if [ -z "$FEATURES_FILE" ]; then
    auto_feat="${DEFAULT_OUT}/phase2_features_test/pair_features.tsv"
    if [ -f "$auto_feat" ]; then
        FEATURES_FILE="$auto_feat"
        write_info "Auto-detected Test Pair Features: $FEATURES_FILE"
    elif [ "$DRY_RUN" = true ]; then
        FEATURES_FILE="$auto_feat"
        write_info "DryRun: Using planned Test Pair Features: $FEATURES_FILE"
    else
        write_error "Test FeaturesFile not specified and not found at $auto_feat. Run Phase 2c (test) first."
        exit 1
    fi
fi

if [ -z "$MODEL_DIR" ]; then
    auto_model="${DEFAULT_OUT}/phase3_gbm"
    if [ -f "${auto_model}/gbm.json" ]; then
        MODEL_DIR="$auto_model"
        write_info "Auto-detected Model Dir: $MODEL_DIR"
    elif [ "$DRY_RUN" = true ]; then
        MODEL_DIR="$auto_model"
        write_info "DryRun: Using planned Model Dir: $MODEL_DIR"
    else
        write_error "ModelDir not specified and no model found at $auto_model. Run Phase 3 first."
        exit 1
    fi
fi

feat_parent="$(dirname "$FEATURES_FILE")"
bge_feat="${feat_parent}/bge_pair_features.tsv"
qwen_feat="${feat_parent}/qwen_pair_features.tsv"
qwen_matcher_feat="${feat_parent}/qwen_matcher_features.tsv"

meta_file="${MODEL_DIR}/stage3_metadata.json"
if [ "$DRY_RUN" = false ] && [ ! -f "$meta_file" ]; then
    write_error "Missing stage3_metadata.json in $MODEL_DIR"
    exit 1
fi

if [ "$DRY_RUN" = false ]; then
    write_step "4.1" "Resolving Dynamic Inference Configuration..."
    bge_used=$("$python" -c "
import json
try:
    print('true' if 'bge_cosine' in json.load(open('$meta_file')).get('feature_columns', []) else 'false')
except:
    print('false')
")
    qwen_used=$("$python" -c "
import json
try:
    print('true' if 'qwen_cosine' in json.load(open('$meta_file')).get('feature_columns', []) else 'false')
except:
    print('false')
")
    matcher_used=$("$python" -c "
import json
try:
    print('true' if 'qwen_matcher_score' in json.load(open('$meta_file')).get('feature_columns', []) else 'false')
except:
    print('false')
")
    threshold=$("$python" -c "
import json
try:
    print(json.load(open('$meta_file')).get('threshold', 0.5))
except:
    print(0.5)
")
else
    bge_used="true"
    qwen_used="true"
    matcher_used="true"
    threshold="0.5"
fi

infer_args=(
    "--mode" "infer"
    "--features" "$FEATURES_FILE"
    "--model-dir" "$MODEL_DIR"
    "--output-dir" "$OUTPUT_DIR"
)

if [ "$bge_used" = "true" ] && [ -f "$bge_feat" ]; then
    infer_args+=("--bge-features" "$bge_feat")
fi
if [ "$qwen_used" = "true" ] && [ -f "$qwen_feat" ]; then
    infer_args+=("--qwen-features" "$qwen_feat")
fi
if [ "$matcher_used" = "true" ] && [ -f "$qwen_matcher_feat" ]; then
    infer_args+=("--qwen-matcher-features" "$qwen_matcher_feat")
fi

write_step "4.2" "Scoring Candidate Pairs..."
invoke_python_module "src.scoring" "Stage 3 Test Inference" "$DRY_RUN" "$python" "${infer_args[@]}"

write_step "4.3" "Graph Clustering & Hardening to Submission Format..."
cluster_args=(
    "--predictions" "${OUTPUT_DIR}/predictions.tsv"
    "--output" "${OUTPUT_DIR}/submission.csv"
    "--threshold" "$threshold"
)
invoke_python_module "src.decision" "Stage 4 Graph Clustering" "$DRY_RUN" "$python" "${cluster_args[@]}"

if [ "$DRY_RUN" = false ]; then
    write_step "4.4" "Auditing Output..."
    sub_file="${OUTPUT_DIR}/submission.csv"
    if [ -f "$sub_file" ]; then
        s1_count="$(awk -F, 'NR>1 {s[$1]=1} END {print length(s)}' "$sub_file")"
        write_success "Phase 4 complete. Submission generated -> $sub_file"
        write_success "Total Source 1 entities clustered: $s1_count"
    else
        write_error "Failed to generate submission.csv"
    fi
fi

write_header "PHASE 4 COMPLETE"
close_logging
