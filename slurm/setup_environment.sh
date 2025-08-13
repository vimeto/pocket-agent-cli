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

# Disable Docker (not available on Mahti compute nodes)
export DISABLE_DOCKER=1

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
module --force purge
module load gcc/13.1.0 2>/dev/null || module load gcc 2>/dev/null || echo "GCC module not available"
module load git 2>/dev/null || echo "Git module not needed for build"

# Load CUDA if we are on a GPU node (hostname OR presence of NVIDIA tools)
if [[ "$CURRENT_NODE" == g* ]] || command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU node detected, setting up CUDA..."
    echo "Node: $CURRENT_NODE"

    # On Mahti, load the specific CUDA module requested
    # Using gcc/13.1.0 and cuda/11.5.0 as requested
    echo "Loading Mahti CUDA modules..."
    module load gcc/13.1.0 cuda/11.5.0
    CUDA_LOADED=true
    echo "✓ Loaded gcc/13.1.0 and cuda/11.5.0 modules"

    # Verify CUDA is available
    echo "Checking for CUDA compiler..."

    # Final verification
    if command -v nvcc &> /dev/null; then
        NVCC_VERSION=$(nvcc --version | grep release | cut -d' ' -f5,6)
        echo "✓ CUDA compiler found: $NVCC_VERSION"
        echo "  nvcc location: $(which nvcc)"

        # Set CUDA environment variables for cmake
        export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
        export CMAKE_CUDA_COMPILER=$(which nvcc)
        echo "  CUDA_HOME set to: $CUDA_HOME"
    else
        echo "⚠ CUDA compiler (nvcc) not found in PATH"
        echo "⚠ This should not happen on Mahti GPU nodes"
        echo ""
        echo "To fix:"
        echo "  1. Ensure modules are loaded: module load gcc/13.1.0 cuda/11.5.0"
        echo "  2. Check module list: module list"
        echo "  3. Reinstall llama-cpp-python:"
        echo "     pip uninstall -y llama-cpp-python"
        echo "     python -m pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 llama-cpp-python"
    fi

    # Show GPU information
    echo ""
    echo "GPU Information:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
    else
        echo "nvidia-smi not found"
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
    - uv
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

# Use LOCAL_SCRATCH for build if available (faster compilation)
if [ -d "$LOCAL_SCRATCH" ] && [ -w "$LOCAL_SCRATCH" ]; then
    export TMPDIR=$LOCAL_SCRATCH
    export TEMP=$LOCAL_SCRATCH
    export TMP=$LOCAL_SCRATCH
    # Also use LOCAL_SCRATCH for uv cache to reduce metadata overhead
    export UV_CACHE_DIR="${LOCAL_SCRATCH}/uv-cache"
    echo "Using LOCAL_SCRATCH for build: $LOCAL_SCRATCH"
fi

# Create a separate virtual environment for packages that can't be containerized
VENV_DIR=$PROJECT_DIR/venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment for additional packages..."
    python -m venv $VENV_DIR
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# CRITICAL: Ensure CUDA libraries are in LD_LIBRARY_PATH before installing
if [[ "$CURRENT_NODE" == g* ]] || command -v nvidia-smi >/dev/null 2>&1; then
    if [ -n "$CUDA_HOME" ]; then
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
        echo "LD_LIBRARY_PATH updated for CUDA: $CUDA_HOME/lib64"
    fi
fi

# Install uv for faster package management
python -m pip install -U uv --quiet

# Check uv version and set flags accordingly
echo "Installing build dependencies with uv..."
if uv --version 2>&1 | grep -q "0\.[3-9]"; then
    # Newer uv versions support --link-mode
    UV_FLAGS="--link-mode=copy"
    uv pip install $UV_FLAGS -U pip setuptools wheel scikit-build-core cmake ninja
else
    # Older uv versions don't support --link-mode
    uv pip install -U pip setuptools wheel scikit-build-core cmake ninja
fi

# Create a wheels cache directory for built packages
WHEEL_CACHE_DIR=$PROJECT_DIR/wheel_cache
mkdir -p $WHEEL_CACHE_DIR

# Define the wheel filename based on Python version and CUDA version
PYTHON_VERSION=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
if [[ "$CURRENT_NODE" == g* ]]; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/' | sed 's/\./_/g')
    WHEEL_NAME="llama_cpp_python-0.3.15-${PYTHON_VERSION}-${PYTHON_VERSION}-linux_x86_64_cuda${CUDA_VERSION}_gcc13.whl"
else
    WHEEL_NAME="llama_cpp_python-0.3.15-${PYTHON_VERSION}-${PYTHON_VERSION}-linux_x86_64_cpu.whl"
fi

CACHED_WHEEL="$WHEEL_CACHE_DIR/$WHEEL_NAME"

# Check if we have a cached wheel
if [ -f "$CACHED_WHEEL" ]; then
    echo "Found cached wheel: $WHEEL_NAME"
    echo "Installing from cache..."
    pip install "$CACHED_WHEEL"
    
    # Verify installation
    if python -c "import llama_cpp" 2>/dev/null; then
        CURRENT_VERSION=$(python -c "import llama_cpp; print(llama_cpp.__version__)")
        echo "✓ llama-cpp-python installed from cache (version $CURRENT_VERSION)"
        
        # Verify CUDA support if on GPU node
        if [[ "$CURRENT_NODE" == g* ]]; then
            if python -c "from llama_cpp import llama_backend_init; llama_backend_init()" 2>&1 | grep -q "ggml_cuda"; then
                echo "✓ CUDA support verified"
            else
                echo "⚠ CUDA support not detected, rebuilding..."
                rm -f "$CACHED_WHEEL"
                pip uninstall -y llama-cpp-python
            fi
        fi
    else
        echo "⚠ Installation from cache failed, rebuilding..."
        rm -f "$CACHED_WHEEL"
    fi
fi

# If not installed yet (no cache or cache failed), build from source
if ! python -c "import llama_cpp" 2>/dev/null; then
    echo "Building llama-cpp-python from source..."
    
    if [[ "$CURRENT_NODE" == g* ]]; then
        if command -v nvcc &> /dev/null; then
            echo "Building with CUDA support..."
            echo "Using nvcc from: $(which nvcc)"
            echo "CUDA version: $(nvcc --version | grep release)"
            echo "GCC version: $(gcc --version | head -n1)"
            
            # Set build environment for CUDA
            export FORCE_CMAKE=1
            # Use all available cores for faster compilation
            export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
            export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_CUDA_COMPILER=$(which nvcc)"
            
            # Build the wheel
            echo "This will take 10-15 minutes on first build (using $(nproc) cores)..."
            
            # First, download and build the wheel
            pip wheel --no-deps --wheel-dir="$WHEEL_CACHE_DIR" \
                --no-build-isolation \
                -v 'llama-cpp-python==0.3.15'
            
            # Find the built wheel (it might have a different name)
            BUILT_WHEEL=$(ls -t $WHEEL_CACHE_DIR/llama_cpp_python-0.3.15*.whl 2>/dev/null | head -1)
            
            if [ -f "$BUILT_WHEEL" ]; then
                # Rename to our standard name for easier caching
                if [ "$BUILT_WHEEL" != "$CACHED_WHEEL" ]; then
                    mv "$BUILT_WHEEL" "$CACHED_WHEEL"
                fi
                
                echo "✓ Wheel built successfully and cached"
                echo "Installing the built wheel..."
                pip install "$CACHED_WHEEL"
                
                # Verify CUDA support
                if python -c "from llama_cpp import llama_backend_init; llama_backend_init()" 2>&1 | grep -q "ggml_cuda"; then
                    echo "✓ CUDA support verified in built wheel"
                else
                    echo "⚠ Built wheel doesn't have CUDA support - build may have failed"
                fi
            else
                echo "⚠ Wheel build failed, trying direct installation..."
                export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
                python -m pip install --no-build-isolation --no-cache-dir -v 'llama-cpp-python==0.3.15'
            fi
        else
            echo "ERROR: On GPU node but CUDA compiler not found!"
            echo "This is critical - cannot run GPU benchmarks without CUDA."
            echo "Please ensure modules are loaded: module load gcc/13.1.0 cuda/11.5.0"
            exit 1
        fi
    else
        echo "Building CPU version of llama-cpp-python..."
        
        # Build CPU wheel
        pip wheel --no-deps --wheel-dir="$WHEEL_CACHE_DIR" 'llama-cpp-python==0.3.15'
        
        # Find and rename the built wheel
        BUILT_WHEEL=$(ls -t $WHEEL_CACHE_DIR/llama_cpp_python-0.3.15*.whl 2>/dev/null | head -1)
        if [ -f "$BUILT_WHEEL" ] && [ "$BUILT_WHEEL" != "$CACHED_WHEEL" ]; then
            mv "$BUILT_WHEEL" "$CACHED_WHEEL"
        fi
        
        # Install it
        pip install "$CACHED_WHEEL"
    fi
fi

# Final verification
if python -c "import llama_cpp" 2>/dev/null; then
    CURRENT_VERSION=$(python -c "import llama_cpp; print(llama_cpp.__version__)")
    echo "✓ llama-cpp-python successfully installed (version $CURRENT_VERSION)"
    echo "  Cached wheel: $CACHED_WHEEL"
else
    echo "⚠ llama-cpp-python installation failed"
    echo "Please check the error messages above"
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

# Test 2: Check llama-cpp-python (make sure we're in the venv)
if [ -n "$VIRTUAL_ENV" ] && python -c "import llama_cpp" 2>/dev/null; then
    LLAMA_VERSION=$(python -c "import llama_cpp; print(llama_cpp.__version__)" 2>/dev/null)
    echo "✓ llama-cpp-python OK: version $LLAMA_VERSION"

    # Check if CUDA is enabled
    if [[ "$CURRENT_NODE" == g* ]] || command -v nvidia-smi >/dev/null 2>&1; then
        if python -c "from llama_cpp import llama_backend_init; llama_backend_init()" 2>&1 | grep -q "ggml_cuda"; then
            echo "  ✓ CUDA support detected"
        else
            echo "  ⚠ CUDA support NOT detected - GPU benchmarks will fail!"
            echo "  To fix: Re-run setup_environment.sh"
        fi
    fi
else
    # Try activating venv and checking again
    if [ -d "$VENV_DIR/bin" ]; then
        source $VENV_DIR/bin/activate
        if python -c "import llama_cpp" 2>/dev/null; then
            LLAMA_VERSION=$(python -c "import llama_cpp; print(llama_cpp.__version__)" 2>/dev/null)
            echo "✓ llama-cpp-python OK: version $LLAMA_VERSION (in venv)"
        else
            echo "⚠ llama-cpp-python not found"
        fi
    else
        echo "⚠ llama-cpp-python not found"
    fi
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
echo ""
echo "IMPORTANT: Use 'source' not './' to run activate_env.sh!"
