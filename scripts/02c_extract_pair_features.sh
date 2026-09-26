#!/usr/bin/env bash
# ==============================================================================
# Phase 2c: Representation & Deterministic Pair Feature Extraction
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

CANDIDATE_FILE=""
PROVENANCE_FILE=""
SPLIT="train"
BGE_MODEL="BAAI/bge-m3"
SKIP_BGE=false
INCLUDE_QWEN=false
QWEN_MODEL="Qwen/Qwen3-Embedding-0.6B"
INCLUDE_QWEN_MATCHER=false
QWEN_MATCHER_ADAPTER=""
QWEN_MATCHER_MODEL="Qwen/Qwen3-0.6B"
OUTPUT_DIR=""
DRY_RUN=false
PYTHON_PATH=""

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --candidate-file <path>         Path to candidate_pairs.tsv
    --provenance-file <path>        Path to candidate_provenance.tsv
    --split <train|test>            Dataset split (default: train)
    --bge-model <path>              BGE-M3 model path or ID
    --skip-bge                      Skip BGE-M3 extraction for baseline
    --include-qwen                  Extract auxiliary Qwen3-Embedding features
    --qwen-model <path>             Qwen embedding model path or ID
    --include-qwen-matcher          Extract Stage 2b generative-matcher features
    --qwen-matcher-adapter <path>   Path to fine-tuned LoRA adapter checkpoint
    --qwen-matcher-model <path>     Base model name for generative matcher
    --output-dir <path>             Output directory
    --dry-run                       Dry run
    --python-path <path>            Path to python executable
    -h, --help                      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --candidate-file|-CandidateFile) CANDIDATE_FILE="$2"; shift 2 ;;
        --provenance-file|-ProvenanceFile) PROVENANCE_FILE="$2"; shift 2 ;;
        --split|-Split) SPLIT="$2"; shift 2 ;;
        --bge-model|-BgeModel) BGE_MODEL="$2"; shift 2 ;;
SKIP_BGE=false
        --skip-bge) SKIP_BGE=true; shift 1 ;;
        --include-qwen|-IncludeQwen) INCLUDE_QWEN=true; shift 1 ;;
        --qwen-model|-QwenModel) QWEN_MODEL="$2"; shift 2 ;;
        --include-qwen-matcher|-IncludeQwenMatcher) INCLUDE_QWEN_MATCHER=true; shift 1 ;;
        --qwen-matcher-adapter|-QwenMatcherAdapter) QWEN_MATCHER_ADAPTER="$2"; shift 2 ;;
        --qwen-matcher-model|-QwenMatcherModel) QWEN_MATCHER_MODEL="$2"; shift 2 ;;
        --output-dir|-OutputDir) OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run|-DryRun) DRY_RUN=true; shift 1 ;;
        --python-path|-PythonPath) PYTHON_PATH="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) write_error "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

if [[ "$SPLIT" != "train" && "$SPLIT" != "test" ]]; then
    write_error "--split must be 'train' or 'test'"
    exit 1
fi

SPLIT_UPPER="$(tr '[:lower:]' '[:upper:]' <<< "$SPLIT")"
write_header "PHASE 2c: PAIR FEATURES EXTRACTION ($SPLIT_UPPER)"
initialize_logging "02c_extract_pair_features_$SPLIT"

python="$(get_python_executable "$PYTHON_PATH")"

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${DEFAULT_OUT}/phase2_features_${SPLIT}"
fi
ensure_directory "$OUTPUT_DIR"

if [ -z "$CANDIDATE_FILE" ]; then
    auto_cand="${DEFAULT_OUT}/phase1_blocking_${SPLIT}/candidate_pairs.tsv"
    if [ -f "$auto_cand" ]; then
        CANDIDATE_FILE="$auto_cand"
        write_info "Auto-detected Candidate Pairs: $CANDIDATE_FILE"
    elif [ "$DRY_RUN" = true ]; then
        CANDIDATE_FILE="$auto_cand"
        write_info "DryRun: Using planned Candidate Pairs: $CANDIDATE_FILE"
    else
        write_error "CandidateFile not specified and not found at $auto_cand. Run Phase 1 first."
        exit 1
    fi
fi

cand_parent="$(dirname "$CANDIDATE_FILE")"
if [ -z "$cand_parent" ]; then cand_parent="."; fi

if [ -z "$PROVENANCE_FILE" ]; then
    auto_prov="${cand_parent}/candidate_provenance.tsv"
    if [ -f "$auto_prov" ]; then
        PROVENANCE_FILE="$auto_prov"
        write_info "Auto-detected Candidate Provenance: $PROVENANCE_FILE"
    fi
fi

s1_file="${DATASET_DIR}/${SPLIT}/${SPLIT}_source1.tsv"
s2_file="${DATASET_DIR}/${SPLIT}/${SPLIT}_source2.tsv"
s3_file="${DATASET_DIR}/${SPLIT}/${SPLIT}_source3.tsv"

stage0_candidates=(
    "${cand_parent}/stage0_normalized"
    "${cand_parent}/stage0_normalized/${SPLIT}"
    "${DEFAULT_OUT}/phase1_blocking_${SPLIT}/stage0_normalized"
    "${DEFAULT_OUT}/phase1_blocking_${SPLIT}/stage0_normalized/${SPLIT}"
)
for st0 in "${stage0_candidates[@]}"; do
    cand_s1="${st0}/${SPLIT}_source1_normalized.tsv"
    cand_s2="${st0}/${SPLIT}_source2_normalized.tsv"
    cand_s3="${st0}/${SPLIT}_source3_normalized.tsv"
    if [ -f "$cand_s1" ] && [ -f "$cand_s2" ] && [ -f "$cand_s3" ]; then
        s1_file="$cand_s1"
        s2_file="$cand_s2"
        s3_file="$cand_s3"
        write_info "Using Stage 0 normalized files for pair features: $st0"
        break
    fi
done

pair_features_out="${OUTPUT_DIR}/pair_features.tsv"
bge_features_out="${OUTPUT_DIR}/bge_pair_features.tsv"
qwen_features_out="${OUTPUT_DIR}/qwen_pair_features.tsv"
qwen_matcher_features_out="${OUTPUT_DIR}/qwen_matcher_features.tsv"

write_step "2c.1" "Extracting Stage 2c Deterministic Pair Features..."
pair_args=(
    "--source1" "$s1_file"
    "--source2" "$s2_file"
    "--source3" "$s3_file"
    "--candidate-file" "$CANDIDATE_FILE"
    "--output-file" "$pair_features_out"
)
if [ -n "$PROVENANCE_FILE" ] && [ -f "$PROVENANCE_FILE" ]; then
    pair_args+=("--provenance-file" "$PROVENANCE_FILE")
fi
invoke_python_module "src.pair_features" "Deterministic Pair Features" "$DRY_RUN" "$python" "${pair_args[@]}"

if [ "$SKIP_BGE" = false ]; then
    write_step "2c.2" "Extracting Stage 2a BGE-M3 Cosine Similarity Features..."
    bge_args=(
        "--source1" "$s1_file"
        "--candidate-sources" "$s2_file" "$s3_file"
        "--candidate-file" "$CANDIDATE_FILE"
        "--output" "$bge_features_out"
        "--model-name" "$BGE_MODEL"
    )
    invoke_python_module "src.bge_features" "BGE-M3 Dense Features" "$DRY_RUN" "$python" "${bge_args[@]}"
else
    write_info "Skipping Stage 2a BGE-M3 Dense Features (--skip-bge specified)"
fi

if [ "$INCLUDE_QWEN" = true ]; then
    write_step "2c.3" "Extracting Stage 2a-ii Qwen3-Embedding Cosine Features..."
    qwen_args=(
        "--source1" "$s1_file"
        "--source2" "$s2_file"
        "--source3" "$s3_file"
        "--candidate-file" "$CANDIDATE_FILE"
        "--output-file" "$qwen_features_out"
        "--model-name" "$QWEN_MODEL"
    )
    invoke_python_module "src.qwen_features" "Qwen3 Auxiliary Features" "$DRY_RUN" "$python" "${qwen_args[@]}"
fi

if [ "$INCLUDE_QWEN_MATCHER" = true ]; then
    if [ -z "$QWEN_MATCHER_ADAPTER" ]; then
        write_error "--include-qwen-matcher was specified, but --qwen-matcher-adapter was not provided."
        exit 1
    fi
    write_step "2c.4" "Extracting Stage 2b Qwen3-0.6B Generative-Matcher Features..."
    matcher_args=(
        "--source1" "$s1_file"
        "--candidate-sources" "$s2_file" "$s3_file"
        "--candidate-file" "$CANDIDATE_FILE"
        "--output" "$qwen_matcher_features_out"
        "--adapter-path" "$QWEN_MATCHER_ADAPTER"
        "--base-model-name" "$QWEN_MATCHER_MODEL"
    )
    invoke_python_module "src.qwen_matcher_features" "Qwen3 Generative Matcher Features" "$DRY_RUN" "$python" "${matcher_args[@]}"
fi

if [ "$DRY_RUN" = false ]; then
    write_step "2c.5" "Verifying Feature Extraction Artifacts..."
    if [ -f "$pair_features_out" ]; then
        p_lines="$(awk 'END {print (NR>1?NR-1:0)}' "$pair_features_out")"
        write_success "Deterministic features: $p_lines pairs -> $pair_features_out"
    fi
    if [ -f "$bge_features_out" ]; then
        b_lines="$(awk 'END {print (NR>1?NR-1:0)}' "$bge_features_out")"
        write_success "BGE features:           $b_lines pairs -> $bge_features_out"
    fi
    if [ "$INCLUDE_QWEN" = true ] && [ -f "$qwen_features_out" ]; then
        q_lines="$(awk 'END {print (NR>1?NR-1:0)}' "$qwen_features_out")"
        write_success "Qwen features:          $q_lines pairs -> $qwen_features_out"
    fi
    if [ "$INCLUDE_QWEN_MATCHER" = true ] && [ -f "$qwen_matcher_features_out" ]; then
        m_lines="$(awk 'END {print (NR>1?NR-1:0)}' "$qwen_matcher_features_out")"
        write_success "Qwen Matcher features:  $m_lines pairs -> $qwen_matcher_features_out"
    fi
fi

write_header "PHASE 2c COMPLETE"
close_logging
