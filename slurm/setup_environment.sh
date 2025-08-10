#!/bin/bash
# Setup script for Mahti environment with containerization

# Check current node
CURRENT_NODE=$(hostname -s)
echo "================================="
echo "Pocket Agent CLI - Mahti Setup"
echo "================================="
echo "Current node: $CURRENT_NODE"

# Warn if on login node but don't exit (some operations can still work)
if [[ "$CURRENT_NODE" == *"login"* ]]; then
    echo "WARNING: On login node. Full setup requires compute node."
    echo "To get compute node: srun --account=project_$PROJECT --partition=test --time=1:00:00 --mem=12000 --pty bash"
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

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p $PROJECT_DIR/{models,results,logs}

# Load modules
echo ""
echo "Loading modules..."
module purge
module load gcc/10.4.0 2>/dev/null || module load gcc 2>/dev/null || echo "GCC module not available"

# Try to load CUDA if on GPU node
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
    
    # Create minimal requirements for containerization
    # llama-cpp-python will be installed separately
    cat > $PROJECT_DIR/requirements-container.txt << EOF
wheel
setuptools
pip
numpy
scipy
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
EOF

    # Create the containerized environment
    echo "Running pip-containerize..."
    pip-containerize new \
        --prefix $TYKKY_ENV \
        $PROJECT_DIR/requirements-container.txt
    
    # Check if successful
    if [ ! -d "$TYKKY_ENV/bin" ]; then
        echo ""
        echo "ERROR: Tykky environment creation failed."
        echo ""
        echo "Troubleshooting:"
        echo "1. Make sure you're on a compute node (not login node)"
        echo "2. Try with debug output:"
        echo "   CW_LOG_LEVEL=3 pip-containerize new --prefix $TYKKY_ENV $PROJECT_DIR/requirements-container.txt"
        echo "3. Check available disk space: lfs quota -hg project_$PROJECT /projappl"
        exit 1
    fi
    
    echo "✓ Containerized environment created successfully"
else
    echo "✓ Tykky environment already exists at $TYKKY_ENV"
fi

# Add to PATH
export PATH="$TYKKY_ENV/bin:$PATH"

# Verify Python is available
echo ""
echo "Checking Python installation..."
if command -v python &> /dev/null; then
    echo "✓ Python found: $(which python)"
    echo "  Version: $(python --version)"
else
    echo "ERROR: Python not found in Tykky environment"
    exit 1
fi

# Install/upgrade pip tools
echo ""
echo "Upgrading pip tools..."
python -m pip install --upgrade pip setuptools wheel --quiet

# Install llama-cpp-python
echo ""
echo "Installing llama-cpp-python..."
if ! python -c "import llama_cpp" 2>/dev/null; then
    if [[ "$CURRENT_NODE" == g* ]] && command -v nvcc &> /dev/null; then
        echo "GPU node with CUDA detected, installing with GPU support..."
        CMAKE_ARGS="-DLLAMA_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80" \
        FORCE_CMAKE=1 \
        python -m pip install llama-cpp-python --no-cache-dir
    else
        echo "Installing CPU version..."
        python -m pip install llama-cpp-python
    fi
    
    # Verify installation
    if python -c "import llama_cpp" 2>/dev/null; then
        version=$(python -c "import llama_cpp; print(llama_cpp.__version__)")
        echo "✓ llama-cpp-python installed (version: $version)"
    else
        echo "⚠ llama-cpp-python installation may have failed"
    fi
else
    version=$(python -c "import llama_cpp; print(llama_cpp.__version__)")
    echo "✓ llama-cpp-python already installed (version: $version)"
fi

# Install pocket-agent-cli
echo ""
echo "Installing pocket-agent-cli..."
cd $PROJECT_DIR
if [ -f "pyproject.toml" ]; then
    python -m pip install -e . --quiet
    
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
echo "Environment: $TYKKY_ENV"
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