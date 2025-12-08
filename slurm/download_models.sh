#!/bin/bash
# =============================================================================
# Pocket Agent CLI - Download Models
# =============================================================================
# Downloads all required models. Run on login node (has internet).
#
# Usage: bash slurm/download_models.sh [--version Q4_K_M|F16|all]
# =============================================================================

set -e

PROJECT=${PROJECT:-2013932}
PROJECT_DIR="/projappl/project_${PROJECT}/${USER}/pocket-agent-cli"

VERSION="${1:-all}"
[ "$1" = "--version" ] && VERSION="${2:-all}"

echo "============================================="
echo "Download Models (version: ${VERSION})"
echo "============================================="

source "${PROJECT_DIR}/slurm/activate.sh"

MODELS=(
    "llama-3.2-3b-instruct"
    "gemma-3n-e2b-it"
    "deepseek-r1-distill-qwen-1.5b"
    "qwen-3-4b"
    "qwen-3-0.6b"
)

for model in "${MODELS[@]}"; do
    echo ""
    echo "Downloading ${model}..."

    if [ "$VERSION" = "all" ] || [ "$VERSION" = "Q4_K_M" ]; then
        pocket-agent model download "${model}" --version Q4_K_M || echo "  (Q4_K_M skipped)"
    fi

    if [ "$VERSION" = "all" ] || [ "$VERSION" = "F16" ]; then
        pocket-agent model download "${model}" --version F16 || echo "  (F16 skipped)"
    fi
done

echo ""
echo "============================================="
echo "Done! Verify with: bash slurm/verify_setup.sh"
