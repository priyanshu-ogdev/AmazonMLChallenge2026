#!/usr/bin/env bash
# ==============================================================================
# Master Pipeline Orchestrator: End-to-End Business Entity Resolution Pipeline
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

FROM_PHASE=0
TO_PHASE=5
RUN_MODE="Full"
SKIP_GPU=false
BASELINE_ONLY=false
INCLUDE_QWEN=true
INCLUDE_QWEN_MATCHER=false
QWEN_MATCHER_ADAPTER=""
RUN_STAGE0=false
NEGATIVES_PER_POSITIVE=2
BOOSTER="gbtree"
COMPARE_DART=false
INJECTIVE=true
COUNTRY_MASK_RATE="0.15"
USE_MONOTONE_CONSTRAINTS=true
DRY_RUN=false
OUTPUT_DIR=""
PYTHON_PATH=""

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Master Pipeline Orchestrator: End-to-End Business Entity Resolution Pipeline.

Options:
    --from-phase <0-5>            Starting phase index (default: 0)
    --to-phase <0-5>              Ending phase index (default: 5)
    --run-mode <mode>             "Full", "FastSample", "GateOnly", or "InferenceOnly" (default: "Full")
    --skip-gpu                    Run in CPU-only mode (uses BAAI/bge-m3 base)
    --baseline-only               Skip dense embedding extraction entirely for ultra-fast CPU baselines
    --no-qwen                     Disable auxiliary Qwen3-Embedding feature extraction
    --include-qwen-matcher        Enable Stage 2b Qwen3-0.6B generative-matcher (stretch goal)
    --qwen-matcher-adapter <path> Path to fine-tuned LoRA adapter
    --run-stage0                  Pre-run Stage 0 streaming normalization
    --negatives-per-positive <N>  Hard negatives per positive for Phase 2a (default: 2)
    --booster <gbtree|dart>       Booster type for Phase 3 (default: gbtree)
    --compare-dart                Compare DART vs GBDT
    --no-monotone-constraints     Disable monotone constraints
    --country-mask-rate <F>       Country masking rate (default: 0.15)
    --dry-run                     Dry run
    --output-dir <path>           Output root directory
    --python-path <path>          Path to python executable
    -h, --help                    Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from-phase|-FromPhase) FROM_PHASE="$2"; shift 2 ;;
        --to-phase|-ToPhase) TO_PHASE="$2"; shift 2 ;;
        --run-mode|-RunMode) RUN_MODE="$2"; shift 2 ;;
        --skip-gpu|-SkipGPU) SKIP_GPU=true; shift 1 ;;
        --baseline-only) BASELINE_ONLY=true; shift 1 ;;
    --baseline-only               Skip dense embedding extraction entirely for ultra-fast CPU baselines
        --no-qwen|-NoQwen) INCLUDE_QWEN=false; shift 1 ;;
        --include-qwen-matcher|-IncludeQwenMatcher) INCLUDE_QWEN_MATCHER=true; shift 1 ;;
        --qwen-matcher-adapter|-QwenMatcherAdapter) QWEN_MATCHER_ADAPTER="$2"; shift 2 ;;
        --run-stage0|-RunStage0) RUN_STAGE0=true; shift 1 ;;
        --negatives-per-positive|-NegativesPerPositive) NEGATIVES_PER_POSITIVE="$2"; shift 2 ;;
        --booster|-Booster) BOOSTER="$2"; shift 2 ;;
        --compare-dart|-CompareDART) COMPARE_DART=true; shift 1 ;;
        --no-monotone-constraints) USE_MONOTONE_CONSTRAINTS=false; shift 1 ;;
        --country-mask-rate|-CountryMaskRate) COUNTRY_MASK_RATE="$2"; shift 2 ;;
        --dry-run|-DryRun) DRY_RUN=true; shift 1 ;;
        --output-dir|-OutputDir) OUTPUT_DIR="$2"; shift 2 ;;
        --python-path|-PythonPath) PYTHON_PATH="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) write_error "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

if [[ "$RUN_MODE" != "Full" && "$RUN_MODE" != "FastSample" && "$RUN_MODE" != "GateOnly" && "$RUN_MODE" != "InferenceOnly" ]]; then
    write_error "Invalid run mode: $RUN_MODE"
    exit 1
fi

initialize_logging "run_all_phases"
start_time=$(date +%s)

write_header "AMAZON ML CHALLENGE 2026: END-TO-END PIPELINE ORCHESTRATOR"

python="$(get_python_executable "$PYTHON_PATH")"

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$DEFAULT_OUT"
fi
ensure_directory "$OUTPUT_DIR"

write_info "Pipeline Configuration:"
write_info "- From Phase:            $FROM_PHASE"
write_info "- To Phase:              $TO_PHASE"
write_info "- Run Mode:              $RUN_MODE"
write_info "- Skip GPU:              $SKIP_GPU (Use BGE-M3 base if true)"
write_info "- Include Qwen:          $INCLUDE_QWEN"
write_info "- Include Qwen Matcher:  $INCLUDE_QWEN_MATCHER"
write_info "- Qwen Matcher Adapter:  ${QWEN_MATCHER_ADAPTER:-'(none)'}"
write_info "- Run Stage 0:           $RUN_STAGE0"
write_info "- Country Mask Rate:     $COUNTRY_MASK_RATE"
write_info "- Monotone Constraints:  $USE_MONOTONE_CONSTRAINTS"
write_info "- Dry Run:               $DRY_RUN"
write_info "- Python:                $python"
write_info "- Output Root:           $OUTPUT_DIR"
echo ""

declare -A phase_timings

run_pipeline_phase() {
    local phase_num="$1"
    local phase_name="$2"
    local script_cmd="${@:3}"

    if [ "$phase_num" -lt "$FROM_PHASE" ] || [ "$phase_num" -gt "$TO_PHASE" ]; then
        echo -e "${COLOR_DARK_GRAY}>>> Skipping Phase $phase_num ($phase_name) per execution range [$FROM_PHASE..$TO_PHASE]${COLOR_RESET}"
        return 0
    fi

    echo ""
    echo -e "${COLOR_YELLOW}================================================================================${COLOR_RESET}"
    echo -e "${COLOR_YELLOW}  STARTING PHASE ${phase_num} - $phase_name${COLOR_RESET}"
    echo -e "${COLOR_YELLOW}================================================================================${COLOR_RESET}"

    local start_ts=$(date +%s)
    if eval "$script_cmd"; then
        local end_ts=$(date +%s)
        local elapsed=$((end_ts - start_ts))
        phase_timings["Phase ${phase_num} - $phase_name"]="${elapsed}s"
        write_success "Phase ${phase_num} ($phase_name) succeeded in ${elapsed}s."
    else
        local exit_code=$?
        local end_ts=$(date +%s)
        local elapsed=$((end_ts - start_ts))
        phase_timings["Phase ${phase_num} - $phase_name"]="FAILED after ${elapsed}s"
        write_error "Phase ${phase_num} ($phase_name) failed with exit code $exit_code."
        exit $exit_code
    fi
}

# ------------------------------------------------------------------------------
# Phase 0: Environment & Smoke Test
# ------------------------------------------------------------------------------
run_pipeline_phase 0 "Environment Verification & Smoke Test" "/bin/bash \"$SCRIPT_DIR/00_verify_environment.sh\" --python-path \"$python\""

if [ "$RUN_MODE" = "InferenceOnly" ]; then
    if [ "$FROM_PHASE" -lt 4 ]; then
        FROM_PHASE=4
    fi
fi

# ------------------------------------------------------------------------------
# Phase 1: Candidate Generation (Blocking) - Train & Test
# ------------------------------------------------------------------------------
max_candidates=50
if [ "$RUN_MODE" = "FastSample" ]; then
    top_k=20
else
    top_k=50
fi

p1_train_cmd="/bin/bash \"$SCRIPT_DIR/01_run_blocking.sh\" --split train --max-candidates $max_candidates --top-k-sparse $top_k --top-k-dense $top_k --output-dir \"$OUTPUT_DIR/phase1_blocking_train\" --python-path \"$python\""
if [ "$RUN_STAGE0" = true ]; then p1_train_cmd+=" --run-stage0"; fi
if [ "$DRY_RUN" = true ]; then p1_train_cmd+=" --dry-run"; fi

p1_test_cmd="/bin/bash \"$SCRIPT_DIR/01_run_blocking.sh\" --split test --max-candidates $max_candidates --top-k-sparse $top_k --top-k-dense $top_k --output-dir \"$OUTPUT_DIR/phase1_blocking_test\" --python-path \"$python\""
if [ "$RUN_STAGE0" = true ]; then p1_test_cmd+=" --run-stage0"; fi
if [ "$DRY_RUN" = true ]; then p1_test_cmd+=" --dry-run"; fi

run_pipeline_phase 1 "Candidate Generation / Blocking" "$p1_train_cmd && $p1_test_cmd"

# ------------------------------------------------------------------------------
# Phase 2: Representation & Features
# ------------------------------------------------------------------------------
p2_prep_out="$OUTPUT_DIR/phase2_prepared_data"
p2_model_out="$OUTPUT_DIR/phase2_models"
train_cand_file="$OUTPUT_DIR/phase1_blocking_train/candidate_pairs.tsv"
test_cand_file="$OUTPUT_DIR/phase1_blocking_test/candidate_pairs.tsv"

if [ "$RUN_MODE" = "FastSample" ]; then
    samples=5000
    epochs=1
else
    samples=50000
    epochs=3
fi

# 2a Prep
p2a_cmd="/bin/bash \"$SCRIPT_DIR/02a_prepare_bi_encoder_data.sh\" --sample-per-country $samples --blocking-candidates \"$train_cand_file\" --negatives-per-positive $NEGATIVES_PER_POSITIVE --output-dir \"$p2_prep_out\" --python-path \"$python\""
if [ "$DRY_RUN" = true ]; then p2a_cmd+=" --dry-run"; fi

# 2b Train BGE
if [ "$SKIP_GPU" = true ]; then
    p2b_cmd="echo '>>> SkipGPU=true: Skipping BGE-M3 fine-tuning (using off-the-shelf weights).'"
else
    p2b_cmd="/bin/bash \"$SCRIPT_DIR/02b_train_and_eval_bi_encoder.sh\" --data-dir \"$p2_prep_out\" --output-dir \"$p2_model_out\" --epochs $epochs --python-path \"$python\""
    if [ "$RUN_MODE" = "GateOnly" ]; then
        p2b_cmd+=" --skip-full-train"
    fi
    if [ "$DRY_RUN" = true ]; then p2b_cmd+=" --dry-run"; fi
fi

# 2c Extract
bge_model="BAAI/bge-m3"
if [ "$SKIP_GPU" = false ]; then
    bge_model="$p2_model_out/bge-m3-merged"
fi

p2c_train_cmd="/bin/bash \"$SCRIPT_DIR/02c_extract_pair_features.sh\" --split train --candidate-file \"$train_cand_file\" --output-dir \"$OUTPUT_DIR/phase2_features_train\" --bge-model \"$bge_model\" --python-path \"$python\""
p2c_test_cmd="/bin/bash \"$SCRIPT_DIR/02c_extract_pair_features.sh\" --split test --candidate-file \"$test_cand_file\" --output-dir \"$OUTPUT_DIR/phase2_features_test\" --bge-model \"$bge_model\" --python-path \"$python\""

if [ "$BASELINE_ONLY" = true ]; then
    p2c_train_cmd+=" --skip-bge"
    p2c_test_cmd+=" --skip-bge"
    INCLUDE_QWEN=false
    INCLUDE_QWEN_MATCHER=false
fi

if [ "$INCLUDE_QWEN" = true ]; then
    p2c_train_cmd+=" --include-qwen"
    p2c_test_cmd+=" --include-qwen"
fi
if [ "$INCLUDE_QWEN_MATCHER" = true ] && [ -n "$QWEN_MATCHER_ADAPTER" ]; then
    p2c_train_cmd+=" --include-qwen-matcher --qwen-matcher-adapter \"$QWEN_MATCHER_ADAPTER\""
    p2c_test_cmd+=" --include-qwen-matcher --qwen-matcher-adapter \"$QWEN_MATCHER_ADAPTER\""
fi
if [ "$DRY_RUN" = true ]; then
    p2c_train_cmd+=" --dry-run"
    p2c_test_cmd+=" --dry-run"
fi

if [ "$RUN_MODE" = "GateOnly" ]; then
    p2c_train_cmd="echo '>>> GateOnly=true: Skipping Stage 2c Feature Extraction'"
    p2c_test_cmd="echo ''"
fi

run_pipeline_phase 2 "Representation & Feature Engineering" "$p2a_cmd && $p2b_cmd && $p2c_train_cmd && $p2c_test_cmd"

# ------------------------------------------------------------------------------
# Phase 3: Grouped-OOF GBM Training & Calibration
# ------------------------------------------------------------------------------
p3_out="$OUTPUT_DIR/phase3_gbm"
p3_train_feat="$OUTPUT_DIR/phase2_features_train/pair_features.tsv"
p3_train_gt="$DATASET_DIR/train/train_ground_truth.tsv"

p3_cmd="/bin/bash \"$SCRIPT_DIR/03_train_scoring_gbm.sh\" --features-file \"$p3_train_feat\" --ground-truth-file \"$p3_train_gt\" --output-dir \"$p3_out\" --booster \"$BOOSTER\" --country-mask-rate \"$COUNTRY_MASK_RATE\" --python-path \"$python\""
if [ "$BASELINE_ONLY" = true ]; then
    p3_cmd+=" --bge-features none --qwen-features none --qwen-matcher-features none"
fi
if [ "$USE_MONOTONE_CONSTRAINTS" = false ]; then p3_cmd+=" --no-monotone-constraints"; fi
if [ "$COMPARE_DART" = true ]; then p3_cmd+=" --compare-dart"; fi
if [ "$DRY_RUN" = true ]; then p3_cmd+=" --dry-run"; fi

if [ "$RUN_MODE" = "GateOnly" ]; then
    p3_cmd="echo '>>> GateOnly=true: Skipping Stage 3 GBM Training'"
fi

run_pipeline_phase 3 "Stage 3 GBM Training & Calibration" "$p3_cmd"

# ------------------------------------------------------------------------------
# Phase 4: Test Inference & Clustering
# ------------------------------------------------------------------------------
p4_test_feat="$OUTPUT_DIR/phase2_features_test/pair_features.tsv"
p4_out="$OUTPUT_DIR/phase4_predictions"

p4_cmd="/bin/bash \"$SCRIPT_DIR/04_inference_and_decision.sh\" --features-file \"$p4_test_feat\" --model-dir \"$p3_out\" --output-dir \"$p4_out\" --python-path \"$python\""
if [ "$DRY_RUN" = true ]; then p4_cmd+=" --dry-run"; fi

if [ "$RUN_MODE" = "GateOnly" ]; then
    p4_cmd="echo '>>> GateOnly=true: Skipping Stage 4 Test Inference'"
fi

run_pipeline_phase 4 "Stage 4 Test Inference & Clustering" "$p4_cmd"

# ------------------------------------------------------------------------------
# Phase 5: Fast Path Validation
# ------------------------------------------------------------------------------
p5_sub="$p4_out/submission.csv"
p5_cmd="/bin/bash \"$SCRIPT_DIR/05_validate_submission.sh\" --submission-file \"$p5_sub\" --split test --python-path \"$python\""
if [ "$DRY_RUN" = true ]; then p5_cmd+=" --dry-run"; fi

if [ "$RUN_MODE" = "GateOnly" ]; then
    p5_cmd="echo '>>> GateOnly=true: Skipping Phase 5 Validation'"
fi

run_pipeline_phase 5 "Format Validation" "$p5_cmd"

echo ""
write_header "PIPELINE EXECUTION SUMMARY"
for k in "${!phase_timings[@]}"; do
    echo "  $k: ${phase_timings[$k]}"
done

end_time=$(date +%s)
total_time=$((end_time - start_time))
echo ""
write_success "Pipeline completed successfully in ${total_time}s!"

close_logging
