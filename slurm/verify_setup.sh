#!/bin/bash
# =============================================================================
# Pocket Agent CLI - Verify Setup
# =============================================================================
# Checks that everything is correctly installed before running benchmarks.
#
# Usage: bash slurm/verify_setup.sh
# =============================================================================

set -e

PROJECT=${PROJECT:-2013932}
PROJECT_DIR="/projappl/project_${PROJECT}/${USER}/pocket-agent-cli"

echo "============================================="
echo "Pocket Agent CLI - Verify Setup"
echo "============================================="

ERRORS=0
pass() { echo -e "  \033[32m✓\033[0m $1"; }
fail() { echo -e "  \033[31m✗\033[0m $1"; ((ERRORS++)); }
warn() { echo -e "  \033[33m⚠\033[0m $1"; }

# 1. Environment
echo ""
echo "[1/4] Environment..."
if [ -f "${PROJECT_DIR}/slurm/activate.sh" ]; then
    pass "Activation script exists"
    source "${PROJECT_DIR}/slurm/activate.sh" 2>/dev/null || fail "Activation failed"
else
    fail "Missing slurm/activate.sh - run setup_mahti.sh first"
fi

# 2. Python packages
echo ""
echo "[2/4] Python packages..."
python -c "import llama_cpp; print('  ✓ llama-cpp-python:', llama_cpp.__version__)" 2>/dev/null || fail "llama-cpp-python not installed"
python -c "import pocket_agent_cli" 2>/dev/null && pass "pocket-agent-cli" || fail "pocket-agent-cli not installed"
command -v pocket-agent &>/dev/null && pass "pocket-agent CLI" || fail "pocket-agent not in PATH"

# 3. Models
echo ""
echo "[3/4] Models..."
MODELS=("llama-3.2-3b-instruct" "gemma-3n-e2b-it" "deepseek-r1-distill-qwen-1.5b" "qwen-3-4b" "qwen-3-0.6b")
for model in "${MODELS[@]}"; do
    if ls "${PROJECT_DIR}/data/models"/*"${model}"*.gguf &>/dev/null 2>&1; then
        pass "$model"
    else
        warn "$model - NOT DOWNLOADED"
    fi
done

# 4. Datasets
echo ""
echo "[4/4] Datasets..."
[ -f "${PROJECT_DIR}/pocket_agent_cli/data/mbpp_full.json" ] && pass "MBPP" || fail "MBPP dataset missing"
[ -f "${PROJECT_DIR}/pocket_agent_cli/data/humaneval.json" ] && pass "HumanEval" || fail "HumanEval dataset missing"

# Summary
echo ""
echo "============================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "\033[32mReady to run benchmarks!\033[0m"
    echo ""
    echo "Run all HumanEval benchmarks:"
    echo "  bash slurm/submit_all_humaneval.sh"
    exit 0
else
    echo -e "\033[31mSetup incomplete ($ERRORS errors)\033[0m"
    echo ""
    echo "Fix issues:"
    echo "  - Missing packages: bash slurm/setup_mahti.sh"
    echo "  - Missing models: bash slurm/download_models.sh"
    exit 1
fi
