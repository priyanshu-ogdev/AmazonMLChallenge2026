#!/usr/bin/env bash
# ==============================================================================
# Phase 2a: Bi-Encoder LoRA Dataset Preparation (Stage 2a Data Builder)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

SAMPLE_PER_COUNTRY=50000
EVAL_SAMPLE=2000
EVAL_CORPUS_NEG=5000
BLOCKING_CANDIDATES=""
NEGATIVES_PER_POSITIVE=2
OUTPUT_DIR=""
SEED=42
BIDIRECTIONAL_GATE=true
DRY_RUN=false
PYTHON_PATH=""

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --sample-per-country <N>      Number of entities to sample per country (default: 50000)
    --eval-sample <N>             Number of queries for held-out evaluation (default: 2000)
    --eval-corpus-neg <N>         Number of negative docs in eval corpus (default: 5000)
    --blocking-candidates <path>  Path to candidate_pairs.tsv for hard-negative mining
    --negatives-per-positive <N>  Number of hard negatives per positive pair (default: 2)
    --output-dir <path>           Output directory
    --seed <N>                    Random seed (default: 42)
    --bidirectional-gate          Build US->India and India->US eval splits (default: true)
    --no-bidirectional-gate       Disable bidirectional gate
    --dry-run                     Dry run
    --python-path <path>          Path to python executable
    -h, --help                    Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample-per-country|-SamplePerCountry) SAMPLE_PER_COUNTRY="$2"; shift 2 ;;
        --eval-sample|-EvalSample) EVAL_SAMPLE="$2"; shift 2 ;;
        --eval-corpus-neg|-EvalCorpusNeg) EVAL_CORPUS_NEG="$2"; shift 2 ;;
        --blocking-candidates|-BlockingCandidates) BLOCKING_CANDIDATES="$2"; shift 2 ;;
        --negatives-per-positive|-NegativesPerPositive) NEGATIVES_PER_POSITIVE="$2"; shift 2 ;;
        --output-dir|-OutputDir) OUTPUT_DIR="$2"; shift 2 ;;
        --seed|-Seed) SEED="$2"; shift 2 ;;
        --bidirectional-gate|-BidirectionalGate) BIDIRECTIONAL_GATE=true; shift 1 ;;
        --no-bidirectional-gate) BIDIRECTIONAL_GATE=false; shift 1 ;;
        --dry-run|-DryRun) DRY_RUN=true; shift 1 ;;
        --python-path|-PythonPath) PYTHON_PATH="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) write_error "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

write_header "PHASE 2a: BI-ENCODER TRAINING & EVALUATION DATA PREPARATION"
initialize_logging "02a_prepare_bi_encoder_data"

python="$(get_python_executable "$PYTHON_PATH")"

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${DEFAULT_OUT}/phase2_prepared_data"
fi
ensure_directory "$OUTPUT_DIR"

write_step "2a.1" "Configuring Dataset Parameters..."
write_info "Sample Per Country:     $SAMPLE_PER_COUNTRY (US + India balanced)"
write_info "Eval Queries:           $EVAL_SAMPLE"
write_info "Eval Corpus Negatives:  $EVAL_CORPUS_NEG"
write_info "Bidirectional Gate:     $BIDIRECTIONAL_GATE"
write_info "Output Directory:       $OUTPUT_DIR"

builder_args=(
    "--mode" "train_data"
    "--data_dir" "$DATASET_DIR"
    "--output_dir" "$OUTPUT_DIR"
    "--sample_per_country" "$SAMPLE_PER_COUNTRY"
    "--eval_sample" "$EVAL_SAMPLE"
    "--eval_corpus_neg" "$EVAL_CORPUS_NEG"
    "--seed" "$SEED"
)

if [ "$BIDIRECTIONAL_GATE" = true ]; then
    builder_args+=("--bidirectional_gate")
fi

if [ -n "$BLOCKING_CANDIDATES" ] && [ -f "$BLOCKING_CANDIDATES" ]; then
    write_info "Mining hard negatives from: $BLOCKING_CANDIDATES ($NEGATIVES_PER_POSITIVE / pair)"
    builder_args+=(
        "--blocking_candidates" "$BLOCKING_CANDIDATES"
        "--negatives_per_positive" "$NEGATIVES_PER_POSITIVE"
    )
fi

write_step "2a.2" "Building Training Pairs & Evaluation Splits..."
invoke_python_module "src.data_builder" "Data Builder Pipeline" "$DRY_RUN" "$python" "${builder_args[@]}"

if [ "$DRY_RUN" = false ]; then
    write_step "2a.3" "Verifying Prepared Data Artifacts..."
    stats_file="${OUTPUT_DIR}/data_stats.json"

    if [ -f "$stats_file" ]; then
        "$python" <<EOF
import json
with open("$stats_file") as f:
    s = json.load(f)
    print(f"  -> Total Sampled S1:   {s.get('sampled_s1_count')}")
    print(f"  -> Total Positive Pairs: {s.get('total_positive_pairs')}")
EOF
    fi
    write_success "Phase 2a prepared datasets verified -> $OUTPUT_DIR"
fi

write_header "PHASE 2a COMPLETE"
close_logging
