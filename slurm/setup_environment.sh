#!/bin/bash
# Setup script for Mahti environment

# Check if running on login node (warn but continue)
# Use hostname -s to get actual node name
CURRENT_NODE=$(hostname -s)
echo "Current node: $CURRENT_NODE"

if [[ "$CURRENT_NODE" == *"login"* ]]; then
    echo "WARNING: Running on login node. Some operations may be limited."
    echo "For full setup, use:"
    echo "  srun --account=project_$PROJECT --partition=test --time=1:00:00 --mem=12000 --pty bash"
elif [[ "$CURRENT_NODE" == c* ]] || [[ "$CURRENT_NODE" == g* ]]; then
    echo "Running on compute node: $CURRENT_NODE - Good!"
fi

# Check if PROJECT is set
if [ -z "$PROJECT" ]; then
    echo "ERROR: PROJECT environment variable not set"
    exit 1
fi

# Set project directory
export PROJECT_DIR=/projappl/project_$PROJECT/$USER/pocket-agent-cli

# Create necessary directories
echo "Creating directories in $PROJECT_DIR..."
mkdir -p $PROJECT_DIR
mkdir -p $PROJECT_DIR/models
mkdir -p $PROJECT_DIR/results
mkdir -p $PROJECT_DIR/logs

# Load necessary modules
echo "Loading modules..."
module purge

# Check available GCC versions
echo "Checking available GCC versions..."
module spider gcc 2>/dev/null | grep "gcc/" | head -5

# Try to load GCC (use default or available version)
if module load gcc/10.4.0 2>/dev/null; then
    echo "Loaded gcc/10.4.0"
elif module load gcc 2>/dev/null; then
    echo "Loaded default gcc"
else
    echo "WARNING: Could not load GCC module. Using system compiler."
fi

# Try to load CUDA (check if we're on compute node)
CURRENT_NODE=$(hostname -s)
if [[ "$CURRENT_NODE" == c* ]] || [[ "$CURRENT_NODE" == g* ]] || [[ -e /tmp/slurmd.pid ]]; then
    echo "On compute node ($CURRENT_NODE), attempting to load CUDA module..."
    if module spider cuda 2>/dev/null | grep -q "cuda"; then
        # Try common CUDA versions
        if module load cuda/12.6.1 2>/dev/null; then
            echo "Loaded cuda/12.6.1"
        elif module load cuda 2>/dev/null; then
            echo "Loaded default cuda"
        else
            echo "WARNING: CUDA module not available. GPU support may be limited."
        fi
    else
        echo "INFO: CUDA modules not available on this node."
    fi
else
    echo "INFO: Not on compute node. CUDA will be loaded when on compute nodes."
fi

# Setup Python environment using Tykky (required for CSC)
export TYKKY_ENV=$PROJECT_DIR/tykky-env

# Load Tykky module
echo "Loading Tykky module for Python environment..."
if ! module load tykky 2>/dev/null; then
    echo "ERROR: Cannot load Tykky module. This is required for Python environments on Mahti."
    echo "Please ensure you're on a compute node or in an interactive session."
    echo "Run: srun --account=project_$PROJECT --partition=test --time=1:00:00 --mem=12000 --pty bash"
    exit 1
fi

# Check if Tykky environment exists and is valid
if [ ! -d "$TYKKY_ENV/bin" ]; then
    # Remove incomplete environment if it exists
    if [ -d "$TYKKY_ENV" ]; then
        echo "Removing incomplete Tykky environment..."
        rm -rf $TYKKY_ENV
    fi
    
    echo "Creating Tykky containerized environment..."
    echo "This may take several minutes on first run..."

    # Create requirements file for Tykky (without llama-cpp-python)
    # llama-cpp-python will be installed separately due to version/CUDA requirements
    cat > $PROJECT_DIR/requirements-base.txt << EOF
click>=8.1.7
rich>=13.7.1
httpx>=0.27.0
psutil>=6.0.0
docker>=7.1.0
jinja2>=3.1.4
pydantic>=2.8.2
tqdm>=4.66.5
aiofiles>=24.1.0
huggingface-hub>=0.24.5
line-profiler>=5.0.0
EOF

    # Create Tykky containerized environment
    if command -v pip-containerize &> /dev/null; then
        echo "Running pip-containerize (this may take 5-10 minutes)..."
        pip-containerize new \
            --prefix $TYKKY_ENV \
            $PROJECT_DIR/requirements-base.txt
        
        # Check if successful
        if [ ! -d "$TYKKY_ENV/bin" ]; then
            echo "ERROR: Tykky environment creation failed."
            echo "Try running: CW_LOG_LEVEL=3 pip-containerize new --prefix $TYKKY_ENV $PROJECT_DIR/requirements-base.txt"
            exit 1
        fi
    else
        echo "ERROR: pip-containerize command not found."
        echo "Make sure Tykky module is loaded: module load tykky"
        exit 1
    fi
else
    echo "Tykky environment already exists at $TYKKY_ENV"
fi

# Add Tykky environment to PATH
export PATH="$TYKKY_ENV/bin:$PATH"

# Check if Python is available
if command -v python &> /dev/null; then
    echo "Python found: $(which python)"
    echo "Python version: $(python --version 2>&1)"
else
    echo "WARNING: Python not found in Tykky environment."
    echo "Environment may not be properly initialized."
fi

# Set CUDA compilation flags for llama-cpp-python
export CMAKE_ARGS="-DLLAMA_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80"
export FORCE_CMAKE=1

# Install llama-cpp-python separately after base environment is ready
if command -v python &> /dev/null; then
    echo ""
    echo "Checking llama-cpp-python installation..."
    
    if ! python -c "import llama_cpp" 2>/dev/null; then
        echo "Installing llama-cpp-python..."
        
        # First upgrade pip to latest version
        pip install --upgrade pip setuptools wheel
        
        CURRENT_NODE=$(hostname -s)
        if [[ "$CURRENT_NODE" == g* ]]; then
            # On GPU node - try CUDA installation
            if command -v nvcc &> /dev/null; then
                echo "GPU node with CUDA detected, installing llama-cpp-python with GPU support..."
                CMAKE_ARGS="-DLLAMA_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80" \
                FORCE_CMAKE=1 \
                pip install llama-cpp-python --no-cache-dir
            else
                echo "GPU node but CUDA not loaded. Installing CPU version..."
                pip install llama-cpp-python
            fi
        else
            # On CPU node - install CPU version
            echo "Installing CPU version of llama-cpp-python..."
            pip install llama-cpp-python
        fi
        
        # Verify installation
        if python -c "import llama_cpp" 2>/dev/null; then
            echo "✓ llama-cpp-python installed successfully"
            python -c "import llama_cpp; print(f'Version: {llama_cpp.__version__}')"
        else
            echo "⚠ llama-cpp-python installation failed"
            echo "You may need to install it manually with: pip install llama-cpp-python"
        fi
    else
        echo "✓ llama-cpp-python is already installed"
        python -c "import llama_cpp; print(f'Version: {llama_cpp.__version__}')"
    fi
fi

echo ""
echo "================================="
echo "Environment setup status:"
echo "================================="
echo "Project: project_$PROJECT"
echo "Project dir: $PROJECT_DIR"
echo "Tykky env: $TYKKY_ENV"

if command -v python &> /dev/null; then
    echo "Python: $(which python)"
else
    echo "Python: NOT FOUND (may need to run on compute node)"
fi

if command -v nvcc &> /dev/null; then
    echo "CUDA: $(nvcc --version | grep release || echo 'version info not available')"
else
    echo "CUDA: Not available (normal on login nodes)"
fi

CURRENT_NODE=$(hostname -s)
if [[ "$CURRENT_NODE" == *"login"* ]] && [[ ! -e /tmp/slurmd.pid ]]; then
    echo ""
    echo "NOTE: You're on a login node. For full setup, run:"
    echo "  srun --account=project_$PROJECT --partition=test --time=1:00:00 --mem=12000 --pty bash"
    echo "  Then: source $PROJECT_DIR/slurm/setup_environment.sh"
else
    echo ""
    echo "Setup complete on compute node: $CURRENT_NODE"
fi
