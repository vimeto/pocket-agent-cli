#!/bin/bash
# =============================================================================
# Pocket Agent CLI - Mahti Setup
# =============================================================================
# Sets up pocket-agent-cli on CSC Mahti using the PyTorch module (Apptainer).
#
# Usage:
#   bash slurm/setup_mahti.sh
#
# After setup:
#   source slurm/activate.sh
# =============================================================================

set -e

# Configuration
PROJECT=${PROJECT:-2013932}
PROJECT_DIR="/projappl/project_${PROJECT}/${USER}/pocket-agent-cli"

echo "============================================="
echo "Pocket Agent CLI - Setup"
echo "============================================="
echo "Project: project_${PROJECT}"
echo "Directory: ${PROJECT_DIR}"
echo "============================================="

# Verify project exists
if [ ! -f "${PROJECT_DIR}/pyproject.toml" ]; then
    echo "ERROR: Project not found at ${PROJECT_DIR}"
    echo "Clone the repository first:"
    echo "  git clone <repo> ${PROJECT_DIR}"
    exit 1
fi

cd "${PROJECT_DIR}"

# Create directories
echo ""
echo "[1/4] Creating directories..."
mkdir -p data/{models,results,logs,datasets,cache,sandbox}

# Load PyTorch module and create venv
echo ""
echo "[2/4] Setting up Python environment..."
module purge
module load pytorch/2.5

VENV_DIR="${PROJECT_DIR}/venv"
if [ ! -d "${VENV_DIR}" ]; then
    python -m venv --system-site-packages "${VENV_DIR}"
    echo "  Created virtual environment"
else
    echo "  Virtual environment exists"
fi

source "${VENV_DIR}/bin/activate"

# Install packages
echo ""
echo "[3/4] Installing packages..."
pip install --upgrade pip --quiet

# llama-cpp-python with CUDA (prebuilt wheel)
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
    --quiet 2>/dev/null || pip install llama-cpp-python --quiet

# Install project
pip install -e "${PROJECT_DIR}" --quiet
pip install httpx aiofiles tqdm --quiet

# Create activation script
echo ""
echo "[4/4] Creating activation script..."
cat > "${PROJECT_DIR}/slurm/activate.sh" << 'EOF'
#!/bin/bash
# Activate pocket-agent-cli environment on Mahti
export PROJECT=${PROJECT:-2013932}
export PROJECT_DIR="/projappl/project_${PROJECT}/${USER}/pocket-agent-cli"

# Environment paths
export POCKET_AGENT_HOME="${PROJECT_DIR}/data"
export POCKET_AGENT_MODELS_DIR="${PROJECT_DIR}/data/models"
export POCKET_AGENT_DATA_DIR="${PROJECT_DIR}/data/datasets"
export POCKET_AGENT_SANDBOX_DIR="${PROJECT_DIR}/data/sandbox"
export POCKET_AGENT_RESULTS_DIR="${PROJECT_DIR}/data/results"
export HF_HOME="${PROJECT_DIR}/data/cache/huggingface"
export DISABLE_DOCKER=1

# Load module and activate venv
module purge 2>/dev/null
module load pytorch/2.5 2>/dev/null
source "${PROJECT_DIR}/venv/bin/activate"

echo "Environment ready: $(python --version)"
EOF

chmod +x "${PROJECT_DIR}/slurm/activate.sh"

# Verify
echo ""
echo "============================================="
echo "Setup Complete!"
echo "============================================="
echo ""
echo "Activate with:"
echo "  source slurm/activate.sh"
echo ""
echo "Next steps:"
echo "  1. Download models:  bash slurm/download_models.sh"
echo "  2. Verify setup:     bash slurm/verify_setup.sh"
echo "  3. Run benchmarks:   bash slurm/submit_all_humaneval.sh"
