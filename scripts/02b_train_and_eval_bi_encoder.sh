#!/usr/bin/env bash
# ==============================================================================
# Phase 2b: Bi-Encoder BGE-M3 LoRA Training & Bidirectional Held-Out Gate Eval
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

DATA_DIR=""
OUTPUT_DIR=""
EPOCHS=3
BATCH_SIZE=48
MINI_BATCH_SIZE=16
LEARNING_RATE="2e-5"
LORA_R=64
DISTILL_WEIGHT="0.10"
NO_DISTILLATION=false
DISTILL_ALL_COLUMNS=false
SKIP_GATE=false
SKIP_FULL_TRAIN=false
DRY_RUN=false
PYTHON_PATH=""

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --data-dir <path>          Path to prepared dataset directory
    --output-dir <path>        Base output directory
    --epochs <N>               Training epochs (default: 3)
    --batch-size <N>           Physical batch size (default: 48)
    --mini-batch-size <N>      GradCache mini-batch size (default: 16)
    --learning-rate <F>        Learning rate (default: 2e-5)
    --lora-r <N>               LoRA rank (default: 64)
    --distill-weight <F>       Distillation anchor loss weight (default: 0.10)
    --no-distillation          Disable self-distillation anchor
    --distill-all-columns      Distill all columns
    --skip-gate                Skip held-out country gate
    --skip-full-train          Stop after gate eval
    --dry-run                  Dry run
    --python-path <path>       Path to python executable
    -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir|-DataDir) DATA_DIR="$2"; shift 2 ;;
        --output-dir|-OutputDir) OUTPUT_DIR="$2"; shift 2 ;;
        --epochs|-Epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size|-BatchSize) BATCH_SIZE="$2"; shift 2 ;;
        --mini-batch-size|-MiniBatchSize) MINI_BATCH_SIZE="$2"; shift 2 ;;
        --learning-rate|-LearningRate) LEARNING_RATE="$2"; shift 2 ;;
        --lora-r|-LoraR) LORA_R="$2"; shift 2 ;;
        --distill-weight|-DistillWeight) DISTILL_WEIGHT="$2"; shift 2 ;;
        --no-distillation|-NoDistillation) NO_DISTILLATION=true; shift 1 ;;
        --distill-all-columns|-DistillAllColumns) DISTILL_ALL_COLUMNS=true; shift 1 ;;
        --skip-gate|-SkipGate) SKIP_GATE=true; shift 1 ;;
        --skip-full-train|-SkipFullTrain) SKIP_FULL_TRAIN=true; shift 1 ;;
        --dry-run|-DryRun) DRY_RUN=true; shift 1 ;;
        --python-path|-PythonPath) PYTHON_PATH="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) write_error "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

write_header "PHASE 2b: BGE-M3 LORA TRAINING & BIDIRECTIONAL GATE EVALUATION"
initialize_logging "02b_train_and_eval_bi_encoder"

python="$(get_python_executable "$PYTHON_PATH")"

if [ -z "$DATA_DIR" ]; then
    DATA_DIR="${DEFAULT_OUT}/phase2_prepared_data"
fi
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${DEFAULT_OUT}/phase2_models"
fi
ensure_directory "$OUTPUT_DIR"

gate_model_dir="${OUTPUT_DIR}/bge-m3-lora-gate"
final_model_dir="${OUTPUT_DIR}/bge-m3-lora-final"
merged_model_dir="${OUTPUT_DIR}/bge-m3-merged"

gate_passed=true

# Step 1: Held-out Country Gate Training
if [ "$SKIP_GATE" = false ]; then
    write_step "2b.1" "Training BGE-M3 LoRA for Held-Out Country Gate..."
    write_info "Mode: held_out_country | LoRA r=$LORA_R | Epochs=$EPOCHS | Batch=$BATCH_SIZE (mini=$MINI_BATCH_SIZE)"

    train_gate_args=(
        "train"
        "--data_dir" "$DATA_DIR"
        "--output_dir" "$gate_model_dir"
        "--mode" "held_out_country"
        "--epochs" "$EPOCHS"
        "--batch_size" "$BATCH_SIZE"
        "--mini_batch_size" "$MINI_BATCH_SIZE"
        "--learning_rate" "$LEARNING_RATE"
        "--lora_r" "$LORA_R"
        "--distill_weight" "$DISTILL_WEIGHT"
    )
    if [ "$NO_DISTILLATION" = true ]; then
        train_gate_args+=("--no_distillation")
    else
        train_gate_args+=("--use_distillation")
    fi
    if [ "$DISTILL_ALL_COLUMNS" = true ]; then
        train_gate_args+=("--distill_all_columns")
    fi

    invoke_python_module "src.train_bi_encoder" "Held-Out Country Gate Training" "$DRY_RUN" "$python" "${train_gate_args[@]}"

    write_step "2b.2" "Evaluating Bidirectional Gate against BAAI/bge-m3 baseline..."
    gate_checkpoint="${gate_model_dir}/final"

    eval_args=(
        "--model_path" "$gate_checkpoint"
        "--data_dir" "$DATA_DIR"
        "--baseline_model" "BAAI/bge-m3"
        "--direction" "bidirectional"
        "--batch_size" "64"
    )
    invoke_python_module "src.eval_bi_encoder" "Bidirectional Gate Evaluation" "$DRY_RUN" "$python" "${eval_args[@]}"

    if [ "$DRY_RUN" = false ]; then
        eval_result_path="${DATA_DIR}/eval_results_bidirectional.json"
        if [ -f "$eval_result_path" ]; then
            decision=$("$python" -c "import json; print(json.load(open('$eval_result_path')).get('decision', 'NO-GO'))")
            echo ""
            echo -e "${COLOR_CYAN}  ========================================================${COLOR_RESET}"
            if [ "$decision" = "GO" ]; then
                echo -e "${COLOR_CYAN}  HELD-OUT COUNTRY GATE DECISION: ${COLOR_GREEN}$decision${COLOR_RESET}"
            else
                echo -e "${COLOR_CYAN}  HELD-OUT COUNTRY GATE DECISION: ${COLOR_RED}$decision${COLOR_RESET}"
            fi
            echo -e "${COLOR_CYAN}  ========================================================${COLOR_RESET}"

            if [ "$decision" != "GO" ]; then
                gate_passed=false
                write_warning "Bidirectional Gate resulted in NO-GO. Reasons:"
                "$python" -c "import json; [print(f'    - {r}') for r in json.load(open('$eval_result_path')).get('reasons', [])]"
                write_warning "Actionable Recovery Policy (per system architecture plan):"
                echo -e "${COLOR_WHITE}    1. Retry with stronger distillation anchor: --distill-weight 0.15${COLOR_RESET}"
                echo -e "${COLOR_WHITE}    2. Reduce adapter capacity to limit overfitting: --lora-r 32${COLOR_RESET}"
                echo -e "${COLOR_WHITE}    3. Fall back to off-the-shelf BAAI/bge-m3 for Stage 2a feature extraction.${COLOR_RESET}"
                echo -e "${COLOR_RED}       NEVER silently deploy a NO-GO adapter to production.${COLOR_RESET}"
            else
                write_success "Bidirectional Gate PASSED! Proceeding to full training."
            fi
        fi
    fi
fi

# Step 3: Full Training & LoRA Merge
if [ "$gate_passed" = true ] && [ "$SKIP_FULL_TRAIN" = false ]; then
    write_step "2b.3" "Executing Full Training on Combined Countries..."
    full_train_args=(
        "train"
        "--data_dir" "$DATA_DIR"
        "--output_dir" "$final_model_dir"
        "--mode" "full"
        "--epochs" "$EPOCHS"
        "--batch_size" "$BATCH_SIZE"
        "--mini_batch_size" "$MINI_BATCH_SIZE"
        "--learning_rate" "$LEARNING_RATE"
        "--lora_r" "$LORA_R"
        "--distill_weight" "$DISTILL_WEIGHT"
    )
    if [ "$NO_DISTILLATION" = true ]; then
        full_train_args+=("--no_distillation")
    else
        full_train_args+=("--use_distillation")
    fi
    if [ "$DISTILL_ALL_COLUMNS" = true ]; then
        full_train_args+=("--distill_all_columns")
    fi

    invoke_python_module "src.train_bi_encoder" "Full Dataset Bi-Encoder Training" "$DRY_RUN" "$python" "${full_train_args[@]}"

    write_step "2b.4" "Merging LoRA Adapter into Standalone SentenceTransformer..."
    final_checkpoint="${final_model_dir}/final"
    merge_args=(
        "merge"
        "--adapter_path" "$final_checkpoint"
        "--output_path" "$merged_model_dir"
    )
    invoke_python_module "src.train_bi_encoder" "Merge LoRA Adapter" "$DRY_RUN" "$python" "${merge_args[@]}"

    write_success "Phase 2b complete: Merged model ready for zero-overhead inference at -> $merged_model_dir"
fi

write_header "PHASE 2b COMPLETE"
close_logging
