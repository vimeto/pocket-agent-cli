#!/bin/bash
# Setup script for Mahti environment with containerization

# Check current node
CURRENT_NODE=$(hostname -s)
echo "================================="
echo "Pocket Agent CLI - Mahti Setup"
echo "================================="
echo "Current node: $CURRENT_NODE"

# Warn if on login node
if [[ "$CURRENT_NODE" == *"login"* ]]; then
    echo "WARNING: On login node. Full setup requires compute node with local storage."
    echo "To get compute node with local storage:"
    echo "  srun --account=project_$PROJECT --partition=small --time=1:00:00 --mem=12000 --gres=nvme:100 --pty bash"
    echo "OR use GPU node:"
    echo "  srun --account=project_$PROJECT --partition=gputest --gres=gpu:a100:1 --time=0:15:00 --pty bash"
fi

# Check if PROJECT is set
if [ -z "$PROJECT" ]; then
    echo "ERROR: PROJECT environment variable not set"
    echo "Please run: export PROJECT=2013932 (or your project number)"
    exit 1
fi

# Set project directory
export PROJECT_DIR=/projappl/project_$PROJECT/$USER/pocket-agent-cli
export TYKKY_ENV=$PROJECT_DIR/tykky-env

# Check disk quota
echo ""
echo "Checking disk quota..."
if command -v lfs &> /dev/null; then
    lfs quota -hg project_$PROJECT /projappl | grep project_$PROJECT || true
fi

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p $PROJECT_DIR/{models,results,logs}

# Load modules
echo ""
echo "Loading modules..."
module purge
module load gcc/10.4.0 2>/dev/null || module load gcc 2>/dev/null || echo "GCC module not available"

# Load CUDA if on GPU node
if [[ "$CURRENT_NODE" == g* ]]; then
    echo "GPU node detected, loading CUDA..."
    module load cuda 2>/dev/null || echo "CUDA module not available"
fi

# Load Tykky module (required for containerization)
echo "Loading Tykky module..."
if ! module load tykky 2>/dev/null; then
    echo "ERROR: Cannot load Tykky module."
    echo "This is required for containerized Python environments on Mahti."
    echo "Make sure you're on a compute node."
    exit 1
fi

# Check if Tykky environment exists and is valid
if [ ! -d "$TYKKY_ENV/bin" ]; then
    # Clean up incomplete environment
    if [ -d "$TYKKY_ENV" ]; then
        echo "Removing incomplete Tykky environment..."
        rm -rf $TYKKY_ENV
    fi
    
    echo ""
    echo "Creating containerized Python environment..."
    echo "This will take 5-10 minutes on first run..."
    
    # Clean up old temp files
    echo "Cleaning up old temporary files..."
    rm -rf /tmp/$USER/*/cw-* 2>/dev/null || true
    rm -rf /tmp/vtoivone/*/cw-* 2>/dev/null || true
    
    # Check for LOCAL_SCRATCH
    if [ -d "$LOCAL_SCRATCH" ] && [ -w "$LOCAL_SCRATCH" ]; then
        export TMPDIR=$LOCAL_SCRATCH
        export TEMP=$LOCAL_SCRATCH
        export TMP=$LOCAL_SCRATCH
        echo "Using LOCAL_SCRATCH for temp files: $LOCAL_SCRATCH"
        df -h $LOCAL_SCRATCH
    else
        echo "LOCAL_SCRATCH not available, using /tmp"
        echo "Available space in /tmp:"
        df -h /tmp
    fi
    
    # Create conda environment file for Tykky (uses conda-containerize)
    # This ensures we get Python 3.11 and all dependencies
    cat > $PROJECT_DIR/environment.yml << EOF
name: pocket-agent
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - wheel
  - setuptools
  - numpy
  - scipy
  - pip:
    - click
    - rich
    - httpx
    - psutil
    - docker
    - jinja2
    - pydantic
    - tqdm
    - aiofiles
    - huggingface-hub
    - requests
    - scikit-build
    - cmake
    - ninja
EOF

    # Create the containerized environment using conda-containerize
    echo "Running conda-containerize (this ensures Python 3.11)..."
    
    # Check if conda-containerize is available
    if command -v conda-containerize &> /dev/null; then
        conda-containerize new \
            --prefix $TYKKY_ENV \
            $PROJECT_DIR/environment.yml
    else
        echo "conda-containerize not found, falling back to pip-containerize..."
        echo "WARNING: This will use Python 3.6 which is too old!"
        
        # Create requirements file as fallback
        cat > $PROJECT_DIR/requirements-container.txt << EOF
pip
wheel
setuptools
scikit-build
cmake
ninja
click
rich
httpx
psutil
docker
jinja2
pydantic
tqdm
aiofiles
huggingface-hub
requests
numpy
EOF
        
        pip-containerize new \
            --prefix $TYKKY_ENV \
            $PROJECT_DIR/requirements-container.txt
    fi
    
    # Check if successful
    if [ ! -d "$TYKKY_ENV/bin" ]; then
        echo ""
        echo "ERROR: Tykky environment creation failed."
        echo ""
        echo "Most likely cause: Not enough temporary space."
        echo ""
        echo "Solution: Request node with local NVMe storage:"
        echo "  srun --account=project_$PROJECT --partition=small --time=1:00:00 --mem=12000 --gres=nvme:100 --pty bash"
        echo "OR use GPU node (always has local storage):"
        echo "  srun --account=project_$PROJECT --partition=gputest --gres=gpu:a100:1 --time=0:15:00 --pty bash"
        echo ""
        echo "Other troubleshooting:"
        echo "1. Check quota: lfs quota -hg project_$PROJECT /projappl"
        echo "2. Debug mode: CW_LOG_LEVEL=3 conda-containerize new --prefix $TYKKY_ENV $PROJECT_DIR/environment.yml"
        exit 1
    fi
    
    echo "✓ Containerized environment created successfully"
else
    echo "✓ Tykky environment already exists at $TYKKY_ENV"
fi

# Add to PATH
export PATH="$TYKKY_ENV/bin:$PATH"

# Verify Python is available and correct version
echo ""
echo "Checking Python installation..."
if command -v python &> /dev/null; then
    echo "✓ Python found: $(which python)"
    PYTHON_VERSION=$(python --version 2>&1)
    echo "  Version: $PYTHON_VERSION"
    
    # Check if Python version is too old
    if [[ "$PYTHON_VERSION" == *"3.6"* ]] || [[ "$PYTHON_VERSION" == *"3.7"* ]] || [[ "$PYTHON_VERSION" == *"3.8"* ]]; then
        echo ""
        echo "WARNING: Python version is too old for this project!"
        echo "This project requires Python 3.11+"
        echo "Please recreate the environment using conda-containerize"
        echo ""
        echo "To fix:"
        echo "1. Remove old environment: rm -rf $TYKKY_ENV"
        echo "2. Re-run this script"
    fi
else
    echo "ERROR: Python not found in Tykky environment"
    exit 1
fi

# Install llama-cpp-python separately (outside container)
echo ""
echo "Installing llama-cpp-python..."

# Create a separate virtual environment for packages that can't be containerized
VENV_DIR=$PROJECT_DIR/venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment for additional packages..."
    python -m venv $VENV_DIR
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip in the venv
pip install --upgrade pip setuptools wheel scikit-build cmake ninja --quiet

# Install llama-cpp-python with CUDA if available
if [[ "$CURRENT_NODE" == g* ]] && command -v nvcc &> /dev/null; then
    echo "GPU node with CUDA detected, installing with GPU support..."
    CMAKE_ARGS="-DLLAMA_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80" \
    FORCE_CMAKE=1 \
    pip install llama-cpp-python --no-cache-dir
else
    echo "Installing CPU version of llama-cpp-python..."
    pip install llama-cpp-python
fi

# Install pocket-agent-cli
echo ""
echo "Installing pocket-agent-cli..."
cd $PROJECT_DIR
if [ -f "pyproject.toml" ]; then
    pip install -e . --quiet
    
    # Verify installation
    if python -c "import pocket_agent_cli" 2>/dev/null; then
        echo "✓ pocket-agent-cli installed"
    else
        echo "⚠ pocket-agent-cli installation may have failed"
    fi
else
    echo "ERROR: pyproject.toml not found in $PROJECT_DIR"
    echo "Make sure you're in the pocket-agent-cli directory"
fi

# Final summary
echo ""
echo "================================="
echo "Setup Summary"
echo "================================="
echo "Project: project_$PROJECT"
echo "Directory: $PROJECT_DIR"
echo "Container env: $TYKKY_ENV"
echo "Additional packages: $VENV_DIR"
echo "Node type: $CURRENT_NODE"

if command -v python &> /dev/null; then
    echo "Python: $(python --version)"
fi

if command -v nvcc &> /dev/null; then
    echo "CUDA: Available ($(nvcc --version | grep release | cut -d' ' -f5))"
else
    echo "CUDA: Not available (normal on CPU nodes)"
fi

# Test if pocket-agent command is available
if command -v pocket-agent &> /dev/null; then
    echo "✓ pocket-agent command available"
else
    echo "⚠ pocket-agent command not in PATH"
fi

echo ""
echo "Next steps:"
echo "1. Download a model: pocket-agent model download llama-3.2-3b-instruct"
echo "2. Test chat: pocket-agent chat --model llama-3.2-3b-instruct"
echo "3. Run benchmarks: sbatch slurm/run_benchmark.sh"
echo ""
echo "To use this environment in future sessions:"
echo "  export PROJECT=$PROJECT"
echo "  export PATH=$TYKKY_ENV/bin:\$PATH"
echo "  source $VENV_DIR/bin/activate"