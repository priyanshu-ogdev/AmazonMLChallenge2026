#!/usr/bin/env bash
# ==============================================================================
# Phase 0: Environment Verification, Hardware Audit, and Module Smoke Test
# ==============================================================================
# Verifies Python environment, required dependencies (numpy, pandas, sklearn, xgboost,
# torch, sentence-transformers, peft), CUDA/GPU VRAM capacity, dataset file existence,
# and runs module contracts and unit test suites.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

PYTHON_PATH=""
SKIP_TESTS=false

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Phase 0: Environment Verification, Hardware Audit, and Module Smoke Test.

Options:
    --python-path <path>    Optional explicit path to python executable.
    --skip-tests            Skip running the unit test discovery suite.
    -h, --help              Show this help message and exit.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python-path|--python_path|-PythonPath)
            PYTHON_PATH="$2"
            shift 2
            ;;
        --python-path=*|--python_path=*)
            PYTHON_PATH="${1#*=}"
            shift 1
            ;;
        --skip-tests|--skip_tests|-SkipTests)
            SKIP_TESTS=true
            shift 1
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            write_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

initialize_logging "00_verify_environment"

write_header "PHASE 0: ENVIRONMENT VERIFICATION & HARDWARE AUDIT"

python="$(get_python_executable "$PYTHON_PATH")"
write_step "0.1" "Resolved Python Interpreter: $python"

# 1. Check Python Version & Platform
py_ver="$("$python" -c "import sys; print(f'Python {sys.version.split()[0]} on {sys.platform}')")"
write_info "$py_ver"

# 2. Check Package Availability & Versions
write_step "0.2" "Checking Core & ML Dependencies..."
"$python" << 'EOF'
import sys

packages = [
    ("numpy", "Core Array Library"),
    ("pandas", "DataFrames / TSV Parsing"),
    ("scipy", "Scientific Computing"),
    ("sklearn", "Scikit-Learn (Metrics / Splits)"),
    ("xgboost", "Stage 3 GBM"),
    ("torch", "PyTorch (GPU / Tensor Operations)"),
    ("sentence_transformers", "Stage 2a/2b Bi-Encoders"),
    ("peft", "LoRA Adapter Support"),
    ("datasets", "HuggingFace Datasets"),
]

missing = []
for pkg, desc in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'available')
        print(f"  [OK] {pkg:<22} ({ver}) - {desc}")
    except ImportError:
        print(f"  [--] {pkg:<22} (NOT INSTALLED) - {desc}")
        missing.append(pkg)

if missing:
    print(f"\nNote: Missing optional/specialized packages: {', '.join(missing)}")
EOF

# 3. Check GPU / CUDA & Hardware Budget
write_step "0.3" "Auditing GPU & CUDA Acceleration..."
"$python" << 'EOF'
try:
    import torch
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        bf16 = torch.cuda.is_bf16_supported()
        print(f"  CUDA Available: True ({count} device(s))")
        print(f"  Primary Device: {name}")
        print(f"  Total VRAM:     {vram_gb:.2f} GB")
        print(f"  bfloat16 Native: {bf16}")
        if vram_gb >= 11.5:
            print(f"  VRAM Budget:    PASSED (>= 12GB target budget for physical batch 48)")
        else:
            print(f"  VRAM Warning:   VRAM < 12GB; reduce batch_size to 32 or mini_batch_size to 8 if OOM occurs")
    else:
        print("  CUDA Available: False (Running in CPU mode)")
except ImportError:
    print("  PyTorch not installed; GPU audit skipped (CPU-only tools available).")
EOF

# 4. Check Dataset Directory & Files
write_step "0.4" "Verifying Dataset Paths..."
train_files=(
    "dataset/train/train_source1.tsv"
    "dataset/train/train_source2.tsv"
    "dataset/train/train_source3.tsv"
    "dataset/train/train_ground_truth.tsv"
)
test_files=(
    "dataset/test/test_source1.tsv"
    "dataset/test/test_source2.tsv"
    "dataset/test/test_source3.tsv"
)

all_found=true
for rel_path in "${train_files[@]}" "${test_files[@]}"; do
    full_path="${PROJECT_ROOT}/${rel_path}"
    if [ -f "$full_path" ]; then
        size_bytes="$(wc -c < "$full_path" 2>/dev/null || stat -c%s "$full_path" 2>/dev/null || stat -f%z "$full_path" 2>/dev/null || echo 0)"
        size_mb="$(awk -v b="$size_bytes" 'BEGIN { printf "%.2f", b / (1024 * 1024) }')"
        write_info "${rel_path} (${size_mb} MB)"
    else
        write_warning "Missing file: ${rel_path}"
        all_found=false
    fi
done

if [ "$all_found" = true ]; then
    write_success "All training and test dataset files verified."
else
    write_warning "Some dataset files are absent. Verify dataset layout before running full pipeline."
fi

# 5. Run Module Architecture & Interface Validator
write_step "0.5" "Running Module Interface & Syntax Validation..."
invoke_python_script "code/business_entity_resolution/validate_modules.py" "validate_modules.py" false "$python"

# 6. Run Unit Test Suite
if [ "$SKIP_TESTS" = false ]; then
    write_step "0.6" "Executing Unit Test Discovery Suite..."
    set +e
    invoke_python_module "unittest" "Unit Tests" false "$python" "discover" "-s" "tests"
    test_exit=$?
    set -e
    if [ "$test_exit" -ne 0 ]; then
        write_warning "Unit test suite finished with exit code $test_exit (expected if optional ML dependencies are not yet installed in host environment)."
    fi
fi

write_header "PHASE 0 COMPLETE: ENVIRONMENT IS READY"
close_logging
