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

# Create necessary directories in a data subdirectory to avoid setuptools issues
echo ""
echo "Creating project directories..."
mkdir -p $PROJECT_DIR/data/{models,results,logs}

# Load modules
echo ""
echo "Loading modules..."
module purge
module load gcc/10.4.0 2>/dev/null || module load gcc 2>/dev/null || echo "GCC module not available"

# Load CUDA if on GPU node
if [[ "$CURRENT_NODE" == g* ]]; then
    echo "GPU node detected, loading CUDA..."
    # Try different CUDA module names
    if ! module load cuda 2>/dev/null; then
        if ! module load cuda/11.7.0 2>/dev/null; then
            if ! module load cuda/12.0.0 2>/dev/null; then
                module load nvidia 2>/dev/null || echo "WARNING: Could not load CUDA module"
            fi
        fi
    fi
    
    # Verify CUDA is loaded
    if command -v nvcc &> /dev/null; then
        echo "✓ CUDA loaded: $(nvcc --version | grep release | cut -d' ' -f5,6)"
    else
        echo "⚠ CUDA module loaded but nvcc not found"
        # Try to find nvcc
        if [ -d "/appl/opt/" ]; then
            export PATH=/appl/opt/cuda/*/bin:$PATH
        fi
    fi
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
        echo "WARNING: LOCAL_SCRATCH not available!"
        echo "Available space in /tmp:"
        df -h /tmp
        
        # Check if /tmp is too small
        TMP_SIZE=$(df /tmp | awk 'NR==2 {print $2}')
        if [ "$TMP_SIZE" -lt "1000000" ]; then  # Less than 1GB
            echo ""
            echo "ERROR: /tmp is too small (only $(df -h /tmp | awk 'NR==2 {print $2}'))!"
            echo "Conda-containerize needs at least 1GB of temp space."
            echo ""
            echo "Solution: Request NVMe storage explicitly:"
            echo "  Exit and run: srun --account=project_$PROJECT --partition=gputest --gres=gpu:a100:1,nvme:100 --time=0:30:00 --pty bash"
            echo "  OR: srun --account=project_$PROJECT --partition=small --gres=nvme:100 --time=1:00:00 --mem=16000 --pty bash"
            exit 1
        fi
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

# Check if llama-cpp-python is already installed
if python -c "import llama_cpp" 2>/dev/null; then
    CURRENT_VERSION=$(python -c "import llama_cpp; print(llama_cpp.__version__)")
    echo "llama-cpp-python already installed (version $CURRENT_VERSION)"
    
    # Check if it has CUDA support
    if [[ "$CURRENT_NODE" == g* ]]; then
        # Try to detect CUDA support in existing installation
        python -c "import llama_cpp" 2>&1 | grep -q "CUDA" && HAS_CUDA=true || HAS_CUDA=false
        
        if [ "$HAS_CUDA" = false ]; then
            echo "⚠ Current installation doesn't have CUDA support"
            if command -v nvcc &> /dev/null; then
                echo "Reinstalling with CUDA support..."
                pip uninstall -y llama-cpp-python
                CMAKE_ARGS="-DLLAMA_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80" \
                FORCE_CMAKE=1 \
                pip install llama-cpp-python --no-cache-dir --force-reinstall
            else
                echo "⚠ CUDA compiler not found. To enable GPU:"
                echo "  1. module load cuda"
                echo "  2. pip uninstall llama-cpp-python"
                echo "  3. CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python --no-cache-dir"
            fi
        else
            echo "✓ CUDA support detected"
        fi
    fi
else
    # Fresh installation
    if [[ "$CURRENT_NODE" == g* ]] && command -v nvcc &> /dev/null; then
        echo "GPU node with CUDA detected, installing with GPU support..."
        CMAKE_ARGS="-DLLAMA_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80" \
        FORCE_CMAKE=1 \
        pip install llama-cpp-python --no-cache-dir
    else
        echo "Installing CPU version of llama-cpp-python..."
        if [[ "$CURRENT_NODE" == g* ]]; then
            echo "⚠ On GPU node but CUDA not available. To fix:"
            echo "  1. module load cuda"
            echo "  2. Rerun this script"
        fi
        pip install llama-cpp-python
    fi
fi

# Install pocket-agent-cli
echo ""
echo "Installing pocket-agent-cli..."
cd $PROJECT_DIR
if [ -f "pyproject.toml" ]; then
    # Create setup.cfg to avoid setuptools confusion with directories
    cat > setup.cfg << EOF
[options]
packages = find:
package_dir =
    = .

[options.packages.find]
include = pocket_agent_cli*
exclude = logs*, slurm*, models*, results*, tests*, venv*, tykky-env*
EOF
    
    # Try different installation methods
    echo "Attempting installation..."
    
    # Method 1: With config settings
    if ! pip install -e . --config-settings editable_mode=compat --quiet 2>/dev/null; then
        echo "First method failed, trying alternative..."
        # Method 2: Without build isolation
        if ! pip install --no-build-isolation -e . --quiet 2>/dev/null; then
            echo "Second method failed, trying standard install..."
            # Method 3: Standard editable install
            pip install -e . --quiet || echo "Installation may have issues"
        fi
    fi
    
    # Verify installation
    if python -c "import pocket_agent_cli" 2>/dev/null; then
        echo "✓ pocket-agent-cli installed successfully"
    else
        echo "⚠ pocket-agent-cli installation may have failed"
        echo "Try manually: pip install -e ."
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
    echo "Trying to add to PATH..."
    export PATH="$VENV_DIR/bin:$PATH"
    if command -v pocket-agent &> /dev/null; then
        echo "✓ Fixed! pocket-agent now available"
    fi
fi

# Run tests to verify everything works
echo ""
echo "Running verification tests..."
echo "----------------------------"

# Test 1: Check Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PYTHON_VERSION" == "3.11" ]] || [[ "$PYTHON_VERSION" == "3.12" ]]; then
    echo "✓ Python version OK: $PYTHON_VERSION"
else
    echo "⚠ Python version issue: $PYTHON_VERSION (expected 3.11+)"
fi

# Test 2: Check llama-cpp-python
if python -c "import llama_cpp" 2>/dev/null; then
    LLAMA_VERSION=$(python -c "import llama_cpp; print(llama_cpp.__version__)" 2>/dev/null)
    echo "✓ llama-cpp-python OK: version $LLAMA_VERSION"
    
    # Check if CUDA is enabled
    if [[ "$CURRENT_NODE" == g* ]]; then
        python -c "import llama_cpp" 2>&1 | grep -q "CUDA" && echo "  ✓ CUDA support detected" || echo "  ⚠ CUDA support unclear"
    fi
else
    echo "⚠ llama-cpp-python not found"
fi

# Test 3: Check pocket-agent-cli
if python -c "import pocket_agent_cli" 2>/dev/null; then
    echo "✓ pocket-agent-cli module OK"
else
    echo "⚠ pocket-agent-cli module not found"
fi

# Test 4: Check CLI command
if command -v pocket-agent &> /dev/null; then
    echo "✓ pocket-agent CLI command OK"
else
    echo "⚠ pocket-agent CLI command not found"
fi

echo ""
echo "================================="
echo "Setup Complete!"
echo "================================="

# Check if all tests passed
if command -v pocket-agent &> /dev/null && \
   python -c "import llama_cpp" 2>/dev/null && \
   python -c "import pocket_agent_cli" 2>/dev/null; then
    echo "✓ All components installed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Download a model: pocket-agent model download llama-3.2-3b-instruct"
    echo "2. Test chat: pocket-agent chat --model llama-3.2-3b-instruct"
    echo "3. Run benchmarks: sbatch slurm/run_benchmark.sh"
else
    echo "⚠ Some components may need attention. Check the verification tests above."
fi

echo ""
echo "To use this environment in future sessions:"
echo "----------------------------------------"
echo "source $PROJECT_DIR/slurm/activate_env.sh"