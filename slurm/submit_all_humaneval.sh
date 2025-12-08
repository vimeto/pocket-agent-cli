#!/bin/bash
# =============================================================================
# Pocket Agent CLI - Submit All HumanEval Benchmarks
# =============================================================================
# Submits HumanEval benchmarks for all models (5 models x 2 quantizations).
#
# Usage: bash slurm/submit_all_humaneval.sh [--dry-run] [--version Q4_K_M|F16|all]
# =============================================================================

set -e

PROJECT=${PROJECT:-2013932}
PROJECT_DIR="/projappl/project_${PROJECT}/${USER}/pocket-agent-cli"

DRY_RUN=false
VERSION="all"
MODE="base"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --version) VERSION="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [--dry-run] [--version Q4_K_M|F16|all] [--mode base|tool_submission|full_tool]"
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "============================================="
echo "Submit All HumanEval (version: ${VERSION}, mode: ${MODE})"
echo "============================================="

# Verify setup
bash "${PROJECT_DIR}/slurm/verify_setup.sh" || exit 1

MODELS=("llama-3.2-3b-instruct" "gemma-3n-e2b-it" "deepseek-r1-distill-qwen-1.5b" "qwen-3-4b" "qwen-3-0.6b")
[ "$VERSION" = "all" ] && VERSIONS=("Q4_K_M" "F16") || VERSIONS=("$VERSION")

echo ""
echo "Submitting ${#MODELS[@]} models x ${#VERSIONS[@]} versions = $((${#MODELS[@]} * ${#VERSIONS[@]})) jobs"

for model in "${MODELS[@]}"; do
    for version in "${VERSIONS[@]}"; do
        echo ""
        echo ">>> ${model} (${version})"
        if $DRY_RUN; then
            echo "  [DRY RUN] Would submit: --model ${model} --version ${version} --mode ${MODE}"
        else
            bash "${PROJECT_DIR}/slurm/submit_humaneval_parallel.sh" \
                --model "${model}" --version "${version}" --mode "${MODE}"
            sleep 2
        fi
    done
done

echo ""
echo "============================================="
$DRY_RUN && echo "Dry run complete." || echo "All jobs submitted! Monitor: squeue -u ${USER}"
