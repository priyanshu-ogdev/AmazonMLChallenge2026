#!/usr/bin/env bash
# ==============================================================================
# Phase 5: Fast Path Submission Validation
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

SUBMISSION_FILE=""
SPLIT="test"
DRY_RUN=false
PYTHON_PATH=""

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --submission-file <path>   Path to submission.csv
    --split <train|test>       Split to validate against (default: test)
    --dry-run                  Dry run
    --python-path <path>       Path to python executable
    -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submission-file|-SubmissionFile) SUBMISSION_FILE="$2"; shift 2 ;;
        --split|-Split) SPLIT="$2"; shift 2 ;;
        --dry-run|-DryRun) DRY_RUN=true; shift 1 ;;
        --python-path|-PythonPath) PYTHON_PATH="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) write_error "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

write_header "PHASE 5: SUBMISSION VALIDATION"
initialize_logging "05_validate_submission"

python="$(get_python_executable "$PYTHON_PATH")"

if [ -z "$SUBMISSION_FILE" ]; then
    auto_sub="${DEFAULT_OUT}/phase4_predictions/submission.csv"
    if [ -f "$auto_sub" ]; then
        SUBMISSION_FILE="$auto_sub"
        write_info "Auto-detected Submission File: $SUBMISSION_FILE"
    elif [ "$DRY_RUN" = true ]; then
        SUBMISSION_FILE="$auto_sub"
        write_info "DryRun: Using planned Submission File: $SUBMISSION_FILE"
    else
        write_error "SubmissionFile not specified and not found at $auto_sub. Run Phase 4 first."
        exit 1
    fi
fi

s1_file="${DATASET_DIR}/${SPLIT}/${SPLIT}_source1.tsv"

val_args=(
    "--submission-file" "$SUBMISSION_FILE"
    "--source1-file" "$s1_file"
)
invoke_python_script "utils/validate_submission.py" "Format Validation" "$DRY_RUN" "$python" "${val_args[@]}"

write_header "PHASE 5 COMPLETE"
close_logging
