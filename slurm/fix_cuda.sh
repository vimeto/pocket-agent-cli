#!/bin/bash
# Fix CUDA and reinstall llama-cpp-python with GPU support

echo "================================="
echo "CUDA Fix for Pocket Agent CLI"
echo "================================="

# Check current node
CURRENT_NODE=$(hostname -s)
echo "Current node: $CURRENT_NODE"

if [[ "$CURRENT_NODE" != g* ]]; then
    echo "⚠ WARNING: Not on a GPU node. This script should be run on GPU nodes (g*)"
    echo "To get a GPU node:"
    echo "  srun --account=project_$PROJECT --partition=gputest --gres=gpu:a100:1 --time=0:15:00 --pty bash"
    exit 1
fi

# Set project directory
export PROJECT_DIR=/projappl/project_$PROJECT/$USER/pocket-agent-cli
export VENV_DIR=$PROJECT_DIR/venv

# Activate virtual environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    source $VENV_DIR/bin/activate
else
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Please run setup_environment.sh first"
    exit 1
fi

echo ""
echo "Step 1: Detecting CUDA installations..."
echo "-----------------------------------------"

# Find all CUDA installations
echo "Searching for nvcc..."
NVCC_PATHS=$(find /appl /opt /usr/local -name nvcc 2>/dev/null | head -10)

if [ -z "$NVCC_PATHS" ]; then
    echo "No nvcc found in common locations"
    echo ""
    echo "Trying module system..."
    
    # List all available CUDA modules
    echo "Available CUDA modules:"
    module spider cuda 2>&1 | grep -E "cuda/" || echo "No CUDA modules found"
    
    # Try to load different CUDA versions
    for cuda_version in cuda cuda/11.7.0 cuda/11.8.0 cuda/12.0.0 cuda/12.1.0 cuda/12.2.0; do
        echo "Trying to load $cuda_version..."
        if module load $cuda_version 2>/dev/null; then
            echo "✓ Loaded $cuda_version"
            break
        fi
    done
else
    echo "Found nvcc at:"
    echo "$NVCC_PATHS"
    
    # Use the first one found
    NVCC_PATH=$(echo "$NVCC_PATHS" | head -1)
    CUDA_BIN_DIR=$(dirname "$NVCC_PATH")
    
    echo ""
    echo "Using CUDA from: $CUDA_BIN_DIR"
    export PATH="$CUDA_BIN_DIR:$PATH"
fi

echo ""
echo "Step 2: Verifying CUDA setup..."
echo "-----------------------------------------"

# Check if nvcc is now available
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA compiler found at: $(which nvcc)"
    NVCC_VERSION=$(nvcc --version | grep release)
    echo "  Version: $NVCC_VERSION"
    
    # Set CUDA environment variables
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    export CMAKE_CUDA_COMPILER=$(which nvcc)
    export CUDA_PATH=$CUDA_HOME
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    
    echo "  CUDA_HOME: $CUDA_HOME"
    echo "  CMAKE_CUDA_COMPILER: $CMAKE_CUDA_COMPILER"
else
    echo "ERROR: nvcc still not found in PATH"
    echo ""
    echo "Manual fix required:"
    echo "1. Find CUDA manually: ls -la /appl/opt/"
    echo "2. Add to PATH: export PATH=/appl/opt/cuda/VERSION/bin:\$PATH"
    echo "3. Re-run this script"
    exit 1
fi

# Check GPU availability
echo ""
echo "Step 3: Checking GPU..."
echo "-----------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
else
    echo "⚠ nvidia-smi not found"
fi

echo ""
echo "Step 4: Reinstalling llama-cpp-python with CUDA..."
echo "-----------------------------------------"

# First uninstall existing version
echo "Uninstalling existing llama-cpp-python..."
pip uninstall -y llama-cpp-python

# Clear pip cache
echo "Clearing pip cache..."
pip cache purge 2>/dev/null || true

# Install with CUDA support
echo "Installing llama-cpp-python with CUDA support..."
echo "This may take 5-10 minutes..."

# A100 GPUs use compute capability 8.0
CMAKE_ARGS="-DLLAMA_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80" \
CUDA_DOCKER_ARCH=sm_80 \
FORCE_CMAKE=1 \
LLAMA_CUDA=1 \
pip install llama-cpp-python --no-cache-dir --force-reinstall --verbose

echo ""
echo "Step 5: Verifying installation..."
echo "-----------------------------------------"

# Test import
python -c "import llama_cpp" 2>&1 | tee /tmp/llama_test.log

# Check for CUDA in the output
if grep -q "CUDA" /tmp/llama_test.log; then
    echo "✓ CUDA support detected in llama-cpp-python!"
else
    echo "Checking build info..."
    python -c "
try:
    import llama_cpp
    print('llama-cpp-python version:', llama_cpp.__version__)
    # Try to create a model to see if CUDA is used
    try:
        from llama_cpp import Llama
        # This will fail without a model but shows CUDA info in error
        model = Llama(model_path='dummy', n_gpu_layers=1)
    except Exception as e:
        if 'CUDA' in str(e) or 'GPU' in str(e):
            print('✓ CUDA/GPU references found in error - likely has GPU support')
        else:
            print('⚠ No clear CUDA/GPU support indicators')
except ImportError as e:
    print('ERROR:', e)
"
fi

echo ""
echo "================================="
echo "CUDA Fix Complete!"
echo "================================="
echo ""
echo "Next steps:"
echo "1. Test with a model: pocket-agent benchmark --model llama-3.2-3b-instruct --problems 1"
echo "2. Monitor GPU usage: nvidia-smi -l 1"
echo ""
echo "If CUDA still doesn't work:"
echo "1. Check dmesg for GPU errors: dmesg | grep -i nvidia"
echo "2. Verify CUDA samples work: $CUDA_HOME/samples/1_Utilities/deviceQuery/deviceQuery"
echo "3. Contact CSC support with the error logs"

# Save the working CUDA configuration
echo ""
echo "Saving CUDA configuration..."
cat > $PROJECT_DIR/cuda_env.sh << EOF
#!/bin/bash
# Working CUDA configuration for this system
export PATH=$PATH
export CUDA_HOME=$CUDA_HOME
export CMAKE_CUDA_COMPILER=$CMAKE_CUDA_COMPILER
export CUDA_PATH=$CUDA_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo "CUDA environment loaded from $PROJECT_DIR/cuda_env.sh"
EOF

echo "✓ Configuration saved to $PROJECT_DIR/cuda_env.sh"
echo "  Source this file in future sessions: source $PROJECT_DIR/cuda_env.sh"