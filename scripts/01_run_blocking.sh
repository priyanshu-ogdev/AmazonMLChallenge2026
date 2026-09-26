#!/usr/bin/env bash
# ==============================================================================
# Phase 1: Stage 0 Normalization & Stage 1 Multi-Channel Candidate Generation
# ==============================================================================
# Runs high-recall candidate generation across 7 blocking channels.
# Emits candidate_pairs.tsv, candidate_provenance.tsv, and blocking_summary.json.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

SPLIT="train"
RUN_STAGE0=false
MAX_CANDIDATES=50
TOP_K_SPARSE=50
TOP_K_DENSE=50
SIMILARITY_FLOOR="0.3"
DENSE_EMBEDDINGS=""
OUTPUT_DIR=""
NUM_WORKERS=$(nproc --all || echo 4)
if [ "$NUM_WORKERS" -gt 14 ]; then
    NUM_WORKERS=14
fi
NO_RESUME=false
NO_CACHE_INDEX=false
CHECKPOINT_INTERVAL=10000
DRY_RUN=false
PYTHON_PATH=""

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Phase 1: Stage 0 Normalization & Stage 1 Multi-Channel Candidate Generation (Blocking).

Options:
    --split <train|test>        Target dataset split (default: train)
    --run-stage0                Pre-run Stage 0 streaming normalization
    --max-candidates <N>        Maximum candidates retained per S1 entity (default: 50)
    --top-k-sparse <N>          Top-K retrieval depth for sparse and token channels (default: 50)
    --top-k-dense <N>           Top-K retrieval depth for dense embedding channel (default: 50)
    --similarity-floor <F>      Minimum cosine similarity floor for dense candidates (default: 0.3)
    --dense-embeddings <path>   Optional path to .npz file containing precomputed dense embeddings
    --output-dir <path>         Output directory for candidate pairs and blocking summary
    --num-workers <N>           Number of parallel workers (default: 1)
    --no-resume                 Overwrite existing output and re-run all S1 entities
    --no-cache-index            Force a full index rebuild even if a valid cache exists
    --checkpoint-interval <N>   Flush output every N entities (default: 10000)
    --dry-run                   Display execution commands without executing them
    --python-path <path>        Optional explicit path to python
    -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split|-Split) SPLIT="$2"; shift 2 ;;
        --split=*) SPLIT="${1#*=}"; shift 1 ;;
        --run-stage0|-RunStage0) RUN_STAGE0=true; shift 1 ;;
        --max-candidates|-MaxCandidates) MAX_CANDIDATES="$2"; shift 2 ;;
        --max-candidates=*) MAX_CANDIDATES="${1#*=}"; shift 1 ;;
        --top-k-sparse|-TopKSparse) TOP_K_SPARSE="$2"; shift 2 ;;
        --top-k-sparse=*) TOP_K_SPARSE="${1#*=}"; shift 1 ;;
        --top-k-dense|-TopKDense) TOP_K_DENSE="$2"; shift 2 ;;
        --top-k-dense=*) TOP_K_DENSE="${1#*=}"; shift 1 ;;
        --similarity-floor|-SimilarityFloor) SIMILARITY_FLOOR="$2"; shift 2 ;;
        --similarity-floor=*) SIMILARITY_FLOOR="${1#*=}"; shift 1 ;;
        --dense-embeddings|-DenseEmbeddings) DENSE_EMBEDDINGS="$2"; shift 2 ;;
        --dense-embeddings=*) DENSE_EMBEDDINGS="${1#*=}"; shift 1 ;;
        --output-dir|-OutputDir) OUTPUT_DIR="$2"; shift 2 ;;
        --output-dir=*) OUTPUT_DIR="${1#*=}"; shift 1 ;;
        --num-workers|-NumWorkers) NUM_WORKERS="$2"; shift 2 ;;
        --num-workers=*) NUM_WORKERS="${1#*=}"; shift 1 ;;
        --no-resume|-NoResume) NO_RESUME=true; shift 1 ;;
        --no-cache-index|-NoCacheIndex) NO_CACHE_INDEX=true; shift 1 ;;
        --checkpoint-interval|-CheckpointInterval) CHECKPOINT_INTERVAL="$2"; shift 2 ;;
        --checkpoint-interval=*) CHECKPOINT_INTERVAL="${1#*=}"; shift 1 ;;
        --dry-run|-DryRun) DRY_RUN=true; shift 1 ;;
        --python-path|-PythonPath) PYTHON_PATH="$2"; shift 2 ;;
        --python-path=*) PYTHON_PATH="${1#*=}"; shift 1 ;;
        -h|--help) show_help; exit 0 ;;
        *) write_error "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

if [[ "$SPLIT" != "train" && "$SPLIT" != "test" ]]; then
    write_error "--split must be 'train' or 'test'"
    exit 1
fi

SPLIT_UPPER="$(tr '[:lower:]' '[:upper:]' <<< "$SPLIT")"
write_header "PHASE 1: STAGE 0 NORMALIZATION & STAGE 1 BLOCKING ($SPLIT_UPPER)"

initialize_logging "01_run_blocking_$SPLIT"

python="$(get_python_executable "$PYTHON_PATH")"

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${DEFAULT_OUT}/phase1_blocking_${SPLIT}"
fi
ensure_directory "$OUTPUT_DIR"

if [ "$SPLIT" = "train" ]; then
    s1_raw="${DATASET_DIR}/train/train_source1.tsv"
    s2_raw="${DATASET_DIR}/train/train_source2.tsv"
    s3_raw="${DATASET_DIR}/train/train_source3.tsv"
    gt_file="${DATASET_DIR}/train/train_ground_truth.tsv"
else
    s1_raw="${DATASET_DIR}/test/test_source1.tsv"
    s2_raw="${DATASET_DIR}/test/test_source2.tsv"
    s3_raw="${DATASET_DIR}/test/test_source3.tsv"
    gt_file=""
fi

s1_file="$s1_raw"
s2_file="$s2_raw"
s3_file="$s3_raw"

stage0_out="${OUTPUT_DIR}/stage0_normalized"
s1_norm="${stage0_out}/${SPLIT}_source1_normalized.tsv"
[ ! -f "$s1_norm" ] && s1_norm="${stage0_out}/${SPLIT}/${SPLIT}_source1_normalized.tsv"
[ ! -f "$s1_norm" ] && s1_norm="${stage0_out}/${SPLIT}_source1_norm.tsv"
[ ! -f "$s1_norm" ] && s1_norm="${stage0_out}/${SPLIT}/${SPLIT}_source1_norm.tsv"

s2_norm="${stage0_out}/${SPLIT}_source2_normalized.tsv"
[ ! -f "$s2_norm" ] && s2_norm="${stage0_out}/${SPLIT}/${SPLIT}_source2_normalized.tsv"
[ ! -f "$s2_norm" ] && s2_norm="${stage0_out}/${SPLIT}_source2_norm.tsv"
[ ! -f "$s2_norm" ] && s2_norm="${stage0_out}/${SPLIT}/${SPLIT}_source2_norm.tsv"

s3_norm="${stage0_out}/${SPLIT}_source3_normalized.tsv"
[ ! -f "$s3_norm" ] && s3_norm="${stage0_out}/${SPLIT}/${SPLIT}_source3_normalized.tsv"
[ ! -f "$s3_norm" ] && s3_norm="${stage0_out}/${SPLIT}_source3_norm.tsv"
[ ! -f "$s3_norm" ] && s3_norm="${stage0_out}/${SPLIT}/${SPLIT}_source3_norm.tsv"

if [ "$RUN_STAGE0" = true ]; then
    write_step "1.0" "Running Stage 0 Streaming Normalization on $SPLIT..."
    ensure_directory "$stage0_out"
    invoke_python_module "src.data_builder" "Stage 0 Normalization" "$DRY_RUN" "$python" \
        "--mode" "stage0_normalize" \
        "--data_dir" "$DATASET_DIR" \
        "--output_dir" "$stage0_out" \
        "--splits" "$SPLIT"
fi

if [ -f "$s1_norm" ] && [ -f "$s2_norm" ] && [ -f "$s3_norm" ]; then
    s1_file="$s1_norm"
    s2_file="$s2_norm"
    s3_file="$s3_norm"
    write_info "Using Stage 0 S1 normalized: $s1_file"
    write_info "Using Stage 0 S2 normalized: $s2_file"
    write_info "Using Stage 0 S3 normalized: $s3_file"
else
    write_info "Using raw input files (Stage 0 normalized files not present or not requested)."
fi

resume_str="resume"
[ "$NO_RESUME" = true ] && resume_str="no resume"
write_step "1.1" "Executing Multi-Channel Blocker ($NUM_WORKERS workers, $resume_str)..."

blocker_args=(
    "--source1" "$s1_file"
    "--candidates" "$s2_file" "$s3_file"
    "--output-dir" "$OUTPUT_DIR"
    "--max-candidates" "$MAX_CANDIDATES"
    "--top-k-sparse" "$TOP_K_SPARSE"
    "--top-k-dense" "$TOP_K_DENSE"
    "--similarity-floor" "$SIMILARITY_FLOOR"
    "--num-workers" "$NUM_WORKERS"
    "--checkpoint-interval" "$CHECKPOINT_INTERVAL"
)

if [ "$NO_CACHE_INDEX" = true ]; then
    blocker_args+=("--no-cache-index")
fi
if [ "$NO_RESUME" = true ]; then
    blocker_args+=("--no-resume")
fi
if [ -n "$gt_file" ] && [ -f "$gt_file" ]; then
    blocker_args+=("--ground-truth" "$gt_file")
fi
if [ -n "$DENSE_EMBEDDINGS" ] && [ -f "$DENSE_EMBEDDINGS" ]; then
    blocker_args+=("--dense-embeddings" "$DENSE_EMBEDDINGS")
fi

invoke_python_module "src.blocking" "Stage 1 Blocking" "$DRY_RUN" "$python" "${blocker_args[@]}"

if [ "$DRY_RUN" = false ]; then
    cand_pairs="${OUTPUT_DIR}/candidate_pairs.tsv"
    summary="${OUTPUT_DIR}/blocking_summary.json"

    if [ -f "$cand_pairs" ] && [ -f "$summary" ]; then
        write_step "1.2" "Auditing Blocking Artifacts..."
        
        "$python" <<EOF
import json, sys
with open("$summary") as f:
    s = json.load(f)
    print(f"  -> S1 Entities:         {s.get('s1_count')}")
    print(f"  -> Total Pairs:         {s.get('total_pairs')}")
    print(f"  -> Candidates/S1 Mean:  {s.get('mean_candidates_per_s1')}")
    print(f"  -> Candidates/S1 P90:   {s.get('p90_candidates_per_s1')}")
    print(f"  -> True Singletons:     {s.get('singleton_s1_count')}")
    
    if s.get('recall_audit'):
        ra = s['recall_audit']
        print("\n\033[0;36m  RECALL AUDIT REPORT (Train Ground Truth):\033[0m")
        print(f"\033[0;32m  Pair Recall:     {round(ra.get('pair_recall', 0) * 100, 2)}%\033[0m")
        print(f"\033[0;32m  Entity Recall:   {round(ra.get('entity_recall', 0) * 100, 2)}%\033[0m")
        print(f"\033[0;32m  Any-Hit Rate:    {round(ra.get('any_hit_rate', 0) * 100, 2)}%\033[0m")
EOF
        write_success "Phase 1 Blocking completed -> $OUTPUT_DIR"
    else
        write_error "Blocking failed to produce candidate_pairs.tsv or blocking_summary.json."
    fi
fi

write_header "PHASE 1 COMPLETE"
close_logging
