#!/usr/bin/env bash
# ==============================================================================
# Phase 2b-2: Qwen3-0.6B Causal Generative Matcher Training & Gate Evaluation
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

DATASET_DIR=""
BLOCKING_CANDIDATES=""
OUTPUT_DIR=""
EPOCHS=3
BATCH_SIZE=16
GRAD_ACCUM_STEPS=2
LEARNING_RATE="5e-5"
LORA_R=64
DISTILL_WEIGHT="0.10"
NO_DISTILLATION=false
DRY_RUN=false
PYTHON_PATH=""

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --dataset-dir <path>          Root directory containing train TSVs
    --blocking-candidates <path>  Path to Phase 1 candidate pairs TSV
    --output-dir <path>           Output directory
    --epochs <N>                  Training epochs (default: 3)
    --batch-size <N>              Per-device batch size (default: 16)
    --grad-accum-steps <N>        Gradient accumulation steps (default: 2)
    --learning-rate <F>           Learning rate (default: 5e-5)
    --lora-r <N>                  LoRA rank (default: 64)
    --distill-weight <F>          Distillation weight (default: 0.10)
    --no-distillation             Disable distillation
    --dry-run                     Dry run
    --python-path <path>          Path to python executable
    -h, --help                    Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-dir|-DatasetDir) DATASET_DIR="$2"; shift 2 ;;
        --blocking-candidates|-BlockingCandidates) BLOCKING_CANDIDATES="$2"; shift 2 ;;
        --output-dir|-OutputDir) OUTPUT_DIR="$2"; shift 2 ;;
        --epochs|-Epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size|-BatchSize) BATCH_SIZE="$2"; shift 2 ;;
        --grad-accum-steps|-GradAccumSteps) GRAD_ACCUM_STEPS="$2"; shift 2 ;;
        --learning-rate|-LearningRate) LEARNING_RATE="$2"; shift 2 ;;
        --lora-r|-LoraR) LORA_R="$2"; shift 2 ;;
        --distill-weight|-DistillWeight) DISTILL_WEIGHT="$2"; shift 2 ;;
        --no-distillation|-NoDistillation) NO_DISTILLATION=true; shift 1 ;;
        --dry-run|-DryRun) DRY_RUN=true; shift 1 ;;
        --python-path|-PythonPath) PYTHON_PATH="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) write_error "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

write_header "PHASE 2b-2: QWEN3-0.6B GENERATIVE MATCHER LORA TRAINING (STRETCH)"
initialize_logging "02b2_train_and_eval_qwen_matcher"

python="$(get_python_executable "$PYTHON_PATH")"

if [ -z "$DATASET_DIR" ]; then
    DATASET_DIR="$DATASET_DIR" # from common.sh
fi
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${DEFAULT_OUT}/phase2_qwen_matcher"
fi
ensure_directory "$OUTPUT_DIR"

if [ -z "$BLOCKING_CANDIDATES" ]; then
    BLOCKING_CANDIDATES="${DEFAULT_OUT}/phase1_blocking_train/candidate_pairs.tsv"
fi

gt_file="${DATASET_DIR}/train/train_ground_truth.tsv"
s1_file="${DATASET_DIR}/train/train_source1.tsv"
s2_file="${DATASET_DIR}/train/train_source2.tsv"
s3_file="${DATASET_DIR}/train/train_source3.tsv"

train_args=(
    "--ground_truth_file" "$gt_file"
    "--source1_files" "$s1_file"
    "--candidate_source_files" "$s2_file" "$s3_file"
    "--output_dir" "$OUTPUT_DIR"
    "--epochs" "$EPOCHS"
    "--batch_size" "$BATCH_SIZE"
    "--grad_accum_steps" "$GRAD_ACCUM_STEPS"
    "--learning_rate" "$LEARNING_RATE"
    "--lora_r" "$LORA_R"
    "--distill_weight" "$DISTILL_WEIGHT"
    "--mode" "held_out_country"
    "--train_country" "us"
    "--eval_country" "india"
)

if [ -n "$BLOCKING_CANDIDATES" ] && [ -f "$BLOCKING_CANDIDATES" ]; then
    train_args+=("--blocking_candidates_file" "$BLOCKING_CANDIDATES")
fi

if [ "$NO_DISTILLATION" = true ]; then
    train_args+=("--no_distillation")
fi

write_step "2b2.1" "Fine-tuning Qwen3-0.6B with LoRA & Sliced Verdict Loss..."
invoke_python_module "src.train_qwen_matcher" "Qwen Matcher Training" "$DRY_RUN" "$python" "${train_args[@]}"

if [ "$DRY_RUN" = false ]; then
    meta_file="${OUTPUT_DIR}/qwen_matcher_train_metadata.json"
    if [ -f "$meta_file" ]; then
        decision=$("$python" -c "import json; print(json.load(open('$meta_file')).get('gate_decision', 'NO-GO'))")
        echo ""
        echo -e "${COLOR_CYAN}  ========================================================${COLOR_RESET}"
        if [ "$decision" = "GO" ]; then
            echo -e "${COLOR_CYAN}  STAGE 2b QWEN MATCHER GATE DECISION: ${COLOR_GREEN}$decision${COLOR_RESET}"
        else
            echo -e "${COLOR_CYAN}  STAGE 2b QWEN MATCHER GATE DECISION: ${COLOR_RED}$decision${COLOR_RESET}"
        fi
        echo -e "${COLOR_CYAN}  ========================================================${COLOR_RESET}"

        if [ "$decision" = "GO" ]; then
            write_success "Qwen Matcher Adapter passed cross-country gate! Ready for Stage 2c feature extraction."
        else
            write_warning "Qwen Matcher failed cross-country gate. Reasons:"
            "$python" -c "import json; [print(f'    - {r}') for r in json.load(open('$meta_file')).get('gate_reasons', [])]"
            write_warning "As specified in system architecture: Qwen Matcher will be skipped in downstream GBM."
        fi
    fi
fi

write_header "PHASE 2b-2 COMPLETE"
close_logging
