#!/bin/bash
# Script to force rebuild of llama-cpp-python with CUDA support

echo "======================================="
echo "Rebuilding llama-cpp-python from source"
echo "======================================="

# Check if PROJECT is set
if [ -z "$PROJECT" ]; then
    export PROJECT=2013932
fi

# Set directories
PROJECT_DIR=/projappl/project_$PROJECT/$USER/pocket-agent-cli
WHEEL_CACHE_DIR=$PROJECT_DIR/wheel_cache
VENV_DIR=$PROJECT_DIR/venv

# Check if on GPU node
CURRENT_NODE=$(hostname -s)
if [[ "$CURRENT_NODE" != g* ]]; then
    echo "WARNING: Not on a GPU node. CUDA build will fail!"
    echo "Request a GPU node first:"
    echo "  srun --account=project_$PROJECT --partition=gputest --gres=gpu:a100:1 --time=0:30:00 --pty bash"
    exit 1
fi

# Load CUDA modules
echo "Loading CUDA modules..."
module purge
module load gcc/10.4.0 cuda/12.1.1

# Verify CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA compiler not found!"
    exit 1
fi

echo "Using CUDA: $(nvcc --version | grep release)"
echo "Using GCC: $(gcc --version | head -n1)"

# Activate virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Run setup_environment.sh first"
    exit 1
fi

source $VENV_DIR/bin/activate

# Clean existing installations and cache
echo ""
echo "Cleaning existing installations..."
pip uninstall -y llama-cpp-python 2>/dev/null || true
pip cache remove llama-cpp-python 2>/dev/null || true
rm -rf $WHEEL_CACHE_DIR/llama_cpp_python*.whl

# Set build environment
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export FORCE_CMAKE=1
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
# No need for -allow-unsupported-compiler with gcc/10.4.0 + cuda/12.1.1
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_CUDA_COMPILER=$(which nvcc)"

echo ""
echo "Build configuration:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  CMAKE_BUILD_PARALLEL_LEVEL: $CMAKE_BUILD_PARALLEL_LEVEL"
echo "  CMAKE_ARGS: $CMAKE_ARGS"

# Use LOCAL_SCRATCH if available for faster builds
if [ -d "$LOCAL_SCRATCH" ] && [ -w "$LOCAL_SCRATCH" ]; then
    export TMPDIR=$LOCAL_SCRATCH
    export TEMP=$LOCAL_SCRATCH
    export TMP=$LOCAL_SCRATCH
    echo "  Using LOCAL_SCRATCH: $LOCAL_SCRATCH"
fi

# Build the wheel
echo ""
echo "Building wheel (this will take 10-15 minutes)..."
mkdir -p $WHEEL_CACHE_DIR

# Build with explicit flags to ensure CUDA support
pip wheel --no-deps \
    --wheel-dir="$WHEEL_CACHE_DIR" \
    --no-cache-dir \
    --no-binary llama-cpp-python \
    --no-build-isolation \
    -v 'llama-cpp-python==0.3.15'

# Find the built wheel
BUILT_WHEEL=$(ls -t $WHEEL_CACHE_DIR/llama_cpp_python-0.3.15*.whl 2>/dev/null | head -1)

if [ ! -f "$BUILT_WHEEL" ]; then
    echo "ERROR: Wheel build failed!"
    exit 1
fi

echo ""
echo "Installing built wheel..."
pip install --force-reinstall "$BUILT_WHEEL"

# Verify installation
echo ""
echo "Verifying installation..."
if python -c "import llama_cpp" 2>/dev/null; then
    VERSION=$(python -c "import llama_cpp; print(llama_cpp.__version__)")
    echo "✓ llama-cpp-python installed (version $VERSION)"
    
    # Check for CUDA support
    echo "Checking CUDA support..."
    python -c "
import sys
try:
    from llama_cpp import llama_backend_init
    llama_backend_init()
    print('✓ Backend initialized successfully')
except Exception as e:
    print(f'✗ Backend initialization failed: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        echo "✓ CUDA support verified!"
        echo ""
        echo "Built wheel saved to: $BUILT_WHEEL"
        echo "Installation complete!"
    else
        echo "✗ CUDA support verification failed"
        echo "The build may not have CUDA support enabled"
        exit 1
    fi
else
    echo "✗ Installation failed!"
    exit 1
fi