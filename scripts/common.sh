#!/usr/bin/env bash
# ==============================================================================
# Business Entity Resolution — Bash Pipeline Orchestration: Common Helpers
# ==============================================================================
# Provides common path resolution, colorized logging, error handling,
# and Python execution wrappers for all phase scripts.
# ==============================================================================

set -euo pipefail

# Base directory layout
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${PROJECT_ROOT}/code/business_entity_resolution"
DATASET_DIR="${PROJECT_ROOT}/dataset"
DEFAULT_OUT="${PROJECT_ROOT}/output"
LOG_DIR="${PROJECT_ROOT}/logs"

# ------------------------------------------------------------------------------
# Terminal Colors
# ------------------------------------------------------------------------------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    COLOR_RESET="\033[0m"
    COLOR_CYAN="\033[0;36m"
    COLOR_WHITE="\033[1;37m"
    COLOR_YELLOW="\033[1;33m"
    COLOR_GREEN="\033[0;32m"
    COLOR_GRAY="\033[0;90m"
    COLOR_RED="\033[0;31m"
    COLOR_MAGENTA="\033[0;35m"
else
    COLOR_RESET=""
    COLOR_CYAN=""
    COLOR_WHITE=""
    COLOR_YELLOW=""
    COLOR_GREEN=""
    COLOR_GRAY=""
    COLOR_RED=""
    COLOR_MAGENTA=""
fi

# ------------------------------------------------------------------------------
# Python Interpreter Resolution
# ------------------------------------------------------------------------------
get_python_executable() {
    local explicit_path="${1:-}"

    if [ -n "$explicit_path" ] && [ -x "$explicit_path" ]; then
        echo "$explicit_path"
        return 0
    fi

    if [ -n "${PYTHON_BIN:-}" ] && [ -x "${PYTHON_BIN}" ]; then
        echo "${PYTHON_BIN}"
        return 0
    fi

    # 1. Project root virtualenv (preferred: holds CUDA-enabled PyTorch for GPU training)
    if [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
        echo "${PROJECT_ROOT}/.venv/bin/python"
        return 0
    fi
    if [ -x "${PROJECT_ROOT}/.venv/Scripts/python.exe" ]; then
        echo "${PROJECT_ROOT}/.venv/Scripts/python.exe"
        return 0
    fi

    # 2. Package subdirectory virtualenv
    if [ -x "${CODE_DIR}/.venv/bin/python" ]; then
        echo "${CODE_DIR}/.venv/bin/python"
        return 0
    fi
    if [ -x "${CODE_DIR}/.venv/Scripts/python.exe" ]; then
        echo "${CODE_DIR}/.venv/Scripts/python.exe"
        return 0
    fi

    # 3. System python3 / python
    local sys_py
    if sys_py="$(command -v python3 2>/dev/null)" && [ -n "$sys_py" ]; then
        echo "$sys_py"
        return 0
    fi
    if sys_py="$(command -v python 2>/dev/null)" && [ -n "$sys_py" ]; then
        echo "$sys_py"
        return 0
    fi

    echo "ERROR: Python interpreter not found. Please activate a virtualenv or set PYTHON_BIN." >&2
    return 1
}

# ------------------------------------------------------------------------------
# Colorized Console Logging
# ------------------------------------------------------------------------------
write_header() {
    local title="$1"
    local line="================================================================================"
    echo ""
    echo -e "${COLOR_CYAN}${line}${COLOR_RESET}"
    echo -e "${COLOR_WHITE}  ${title}${COLOR_RESET}"
    echo -e "${COLOR_CYAN}${line}${COLOR_RESET}"
    echo ""
}

write_step() {
    local phase="$1"
    local desc="$2"
    echo -e "${COLOR_YELLOW}[${phase}]${COLOR_RESET} ${COLOR_WHITE}${desc}${COLOR_RESET}"
}

write_success() {
    local msg="$1"
    echo -e "${COLOR_GREEN}[OK] ${msg}${COLOR_RESET}"
}

write_info() {
    local msg="$1"
    echo -e "${COLOR_GRAY}  -> ${msg}${COLOR_RESET}"
}

write_warning() {
    local msg="$1"
    echo -e "${COLOR_YELLOW}[WARN] ${msg}${COLOR_RESET}"
}

write_error() {
    local msg="$1"
    echo -e "${COLOR_RED}[ERROR] ${msg}${COLOR_RESET}" >&2
}

ensure_directory() {
    local dir_path="$1"
    if [ ! -d "$dir_path" ]; then
        mkdir -p "$dir_path"
        write_info "Created directory: ${dir_path}"
    fi
}

# ------------------------------------------------------------------------------
# Logging: Persistent Transcript Capture
# ------------------------------------------------------------------------------
_LOG_FIFO=""
_LOG_TEE_PID=""
_SAVED_STDOUT_FD=""
_SAVED_STDERR_FD=""
_CURRENT_LOG_FILE=""

initialize_logging() {
    local script_name="${1:-pipeline}"
    local log_dir="${2:-$LOG_DIR}"

    mkdir -p "$log_dir"
    local ts
    ts="$(date +"%Y%m%d_%H%M%S")"
    local log_file="${log_dir}/${ts}_${script_name}.log"
    _CURRENT_LOG_FILE="$log_file"

    # Set up FIFO tee redirect
    local fifo_path="${log_dir}/.fifo_${ts}_$$_${RANDOM}"
    mkfifo "$fifo_path"
    _LOG_FIFO="$fifo_path"

    # Save original stdout / stderr descriptors
    exec 3>&1 4>&2
    _SAVED_STDOUT_FD=3
    _SAVED_STDERR_FD=4

    # Run background tee writing to log_file and original stdout
    tee -a "$log_file" < "$fifo_path" >&3 &
    _LOG_TEE_PID=$!

    # Redirect stdout and stderr into FIFO
    exec > "$fifo_path" 2>&1

    echo -e "${COLOR_GRAY}[LOG] Transcript -> ${log_file}${COLOR_RESET}"
}

close_logging() {
    if [ -n "${_LOG_FIFO}" ] && [ -p "${_LOG_FIFO}" ]; then
        # Restore original descriptors
        if [ -n "${_SAVED_STDOUT_FD}" ]; then
            exec 1>&3 2>&4 2>/dev/null || true
            exec 3>&- 4>&- 2>/dev/null || true
            _SAVED_STDOUT_FD=""
            _SAVED_STDERR_FD=""
        fi

        # Wait for tee to flush and exit
        if [ -n "${_LOG_TEE_PID}" ]; then
            wait "${_LOG_TEE_PID}" 2>/dev/null || true
            _LOG_TEE_PID=""
        fi

        # Remove FIFO
        rm -f "${_LOG_FIFO}" 2>/dev/null || true
        _LOG_FIFO=""
    fi
}

# Ensure cleanup on any shell exit / interruption
_pipeline_exit_trap() {
    local exit_code=$?
    close_logging
    exit "$exit_code"
}
trap _pipeline_exit_trap EXIT INT TERM

# ------------------------------------------------------------------------------
# Safe Python Command Invocation
# ------------------------------------------------------------------------------
invoke_python_module() {
    local module_name="$1"
    local step_name="$2"
    local dry_run="${3:-false}"
    local python_exe="${4:-}"
    shift 4
    local -a module_args=("$@")

    if [ -z "$python_exe" ]; then
        python_exe="$(get_python_executable)"
    fi

    if [ -n "$step_name" ]; then
        write_step "EXEC" "$step_name"
    fi
    write_info "Command: $python_exe -m $module_name ${module_args[*]}"
    write_info "Working Directory: $CODE_DIR"

    if [ "$dry_run" = "true" ] || [ "$dry_run" = "1" ]; then
        echo -e "${COLOR_MAGENTA}  [DRY RUN] Skipped execution.${COLOR_RESET}"
        return 0
    fi

    local start_time end_time elapsed
    start_time="$(date +%s.%N 2>/dev/null || date +%s)"

    local old_pythonpath="${PYTHONPATH:-}"
    export PYTHONPATH="${CODE_DIR}:${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONUNBUFFERED="1"

    local exit_code=0
    (
        cd "$CODE_DIR"
        "$python_exe" -u -m "$module_name" "${module_args[@]}"
    ) || exit_code=$?

    if [ -n "$old_pythonpath" ]; then
        export PYTHONPATH="$old_pythonpath"
    else
        unset PYTHONPATH
    fi

    end_time="$(date +%s.%N 2>/dev/null || date +%s)"
    elapsed="$(awk -v s="$start_time" -v e="$end_time" 'BEGIN { printf "%.2f", (e - s) }' 2>/dev/null || echo "0.00")"

    if [ "$exit_code" -ne 0 ]; then
        write_error "Step failed with exit code $exit_code (${elapsed}s): $step_name"
        return "$exit_code"
    fi

    write_success "Completed in ${elapsed}s: $step_name"
    return 0
}

invoke_python_script() {
    local script_path="$1"
    local step_name="$2"
    local dry_run="${3:-false}"
    local python_exe="${4:-}"
    shift 4
    local -a script_args=("$@")

    if [ -z "$python_exe" ]; then
        python_exe="$(get_python_executable)"
    fi

    if [ -n "$step_name" ]; then
        write_step "EXEC" "$step_name"
    fi
    write_info "Command: $python_exe $script_path ${script_args[*]}"
    write_info "Working Directory: $PROJECT_ROOT"

    if [ "$dry_run" = "true" ] || [ "$dry_run" = "1" ]; then
        echo -e "${COLOR_MAGENTA}  [DRY RUN] Skipped execution.${COLOR_RESET}"
        return 0
    fi

    local start_time end_time elapsed
    start_time="$(date +%s.%N 2>/dev/null || date +%s)"

    local old_pythonpath="${PYTHONPATH:-}"
    export PYTHONPATH="${CODE_DIR}:${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONUNBUFFERED="1"

    local exit_code=0
    (
        cd "$PROJECT_ROOT"
        "$python_exe" -u "$script_path" "${script_args[@]}"
    ) || exit_code=$?

    if [ -n "$old_pythonpath" ]; then
        export PYTHONPATH="$old_pythonpath"
    else
        unset PYTHONPATH
    fi

    end_time="$(date +%s.%N 2>/dev/null || date +%s)"
    elapsed="$(awk -v s="$start_time" -v e="$end_time" 'BEGIN { printf "%.2f", (e - s) }' 2>/dev/null || echo "0.00")"

    if [ "$exit_code" -ne 0 ]; then
        write_error "Script failed with exit code $exit_code (${elapsed}s): $step_name"
        return "$exit_code"
    fi

    write_success "Completed in ${elapsed}s: $step_name"
    return 0
}
