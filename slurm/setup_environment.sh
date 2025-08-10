#!/bin/bash
# Setup script for Mahti environment

# Check if running on login node (warn but continue)
if [[ "$HOSTNAME" == *"login"* ]]; then
    echo "WARNING: Running on login node. Some operations may be limited."
    echo "For full setup, use: sinteractive --account project_$PROJECT --time 1:00:00 --mem 12000"
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

# Try to load CUDA (only on compute nodes or if available)
if [[ "$HOSTNAME" != *"login"* ]]; then
    echo "Attempting to load CUDA module..."
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
    echo "INFO: Skipping CUDA module on login node. Will be loaded on compute nodes."
fi

# Setup Python environment using Tykky (required for CSC)
export TYKKY_ENV=$PROJECT_DIR/tykky-env

# Load Tykky module
echo "Loading Tykky module for Python environment..."
if ! module load tykky 2>/dev/null; then
    echo "ERROR: Cannot load Tykky module. This is required for Python environments on Mahti."
    echo "Please ensure you're on a compute node or in an interactive session."
    echo "Run: sinteractive --account project_$PROJECT --time 1:00:00 --mem 12000"
    exit 1
fi

# Check if Tykky environment exists
if [ ! -d "$TYKKY_ENV" ]; then
    echo "Creating Tykky containerized environment..."
    echo "This may take several minutes on first run..."

    # Create requirements file for Tykky
    cat > $PROJECT_DIR/requirements.txt << EOF
llama-cpp-python>=0.2.90
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
        pip-containerize new \
            --prefix $TYKKY_ENV \
            $PROJECT_DIR/requirements.txt
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

# Check if llama-cpp-python needs CUDA reinstall
if command -v python &> /dev/null; then
    if ! python -c "import llama_cpp" 2>/dev/null; then
        echo "Installing llama-cpp-python with CUDA support..."
        echo "This requires being on a compute node with CUDA available."

        if [[ "$HOSTNAME" != *"login"* ]] && command -v nvcc &> /dev/null; then
            pip install --force-reinstall llama-cpp-python
        else
            echo "WARNING: Cannot install CUDA-enabled llama-cpp-python on login node."
            echo "Run this script on a compute node for full CUDA support."
        fi
    else
        echo "llama-cpp-python is already installed."
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

if [[ "$HOSTNAME" == *"login"* ]]; then
    echo ""
    echo "NOTE: You're on a login node. For full setup, run:"
    echo "  sinteractive --account project_$PROJECT --time 1:00:00 --mem 12000"
    echo "  source $PROJECT_DIR/slurm/setup_environment.sh"
fi
