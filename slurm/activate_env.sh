#!/bin/bash
# Activation script for pocket-agent-cli environment on Mahti

# Check if PROJECT is set, if not try to detect from path
if [ -z "$PROJECT" ]; then
    # Try to extract project number from current path or default
    if pwd | grep -q "project_[0-9]"; then
        export PROJECT=$(pwd | grep -o "project_[0-9]*" | head -1 | cut -d_ -f2)
    else
        export PROJECT=2013932  # Default project number
    fi
fi

# Set up environment paths
export PROJECT_DIR=/projappl/project_$PROJECT/$USER/pocket-agent-cli
export TYKKY_ENV=$PROJECT_DIR/tykky-env
export VENV_DIR=$PROJECT_DIR/venv

# Disable Docker (not available on Mahti compute nodes)
export DISABLE_DOCKER=1

# Load CUDA module if on GPU node (required for llama-cpp-python with CUDA)
CURRENT_NODE=$(hostname -s)
if [[ "$CURRENT_NODE" == g* ]] || command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU node detected, loading CUDA modules..."
    
    # Load the correct modules for Mahti
    # Using gcc/13.1.0 and cuda/11.5.0 as requested
    module purge
    module load gcc/13.1.0 cuda/11.5.0
    
    # Set CUDA environment variables
    if command -v nvcc &> /dev/null; then
        export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
        export CMAKE_CUDA_COMPILER=$(which nvcc)
        export CUDACXX=$(which nvcc)
        
        # CRITICAL: Add CUDA libraries to LD_LIBRARY_PATH
        # The prebuilt wheel expects to find libcudart.so.12 in these paths
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
        
        # Also add the compat libraries which often contain versioned .so files
        if [ -d "$CUDA_HOME/compat" ]; then
            export LD_LIBRARY_PATH=$CUDA_HOME/compat:$LD_LIBRARY_PATH
        fi
        
        # Add the targets directory which contains architecture-specific libraries
        if [ -d "$CUDA_HOME/targets/x86_64-linux/lib" ]; then
            export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
        fi
        
        echo "  ✓ CUDA loaded successfully"
        echo "  CUDA_HOME: $CUDA_HOME"
        echo "  nvcc: $(which nvcc)"
        echo "  LD_LIBRARY_PATH includes: $CUDA_HOME/lib64"
        
        # Handle CUDA library version mismatch
        # The prebuilt wheel expects libcudart.so.12 but CUDA 12.1.1 might provide .11
        if ls $CUDA_HOME/lib64/libcudart.so.12* >/dev/null 2>&1; then
            echo "  ✓ libcudart.so.12 found in $CUDA_HOME/lib64"
        elif ls $CUDA_HOME/lib64/libcudart.so.11* >/dev/null 2>&1; then
            echo "  ⚠ Found libcudart.so.11 but wheel expects libcudart.so.12"
            echo "  ERROR: CUDA version mismatch. Please use CUDA 12.4 or newer."
        else
            echo "  ⚠ libcudart.so not found in expected location"
        fi
    else
        echo "  ⚠ WARNING: CUDA not properly loaded!"
        echo "  This is critical for GPU benchmarks."
        echo "  Try manually: module load gcc/13.1.0 cuda/11.5.0"
    fi
else
    echo "CPU node detected, CUDA not needed"
fi

# Check if environments exist
if [ ! -d "$TYKKY_ENV" ]; then
    echo "ERROR: Tykky environment not found at $TYKKY_ENV"
    echo "Please run: source $PROJECT_DIR/slurm/setup_environment.sh"
    return 1 2>/dev/null || exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Please run: source $PROJECT_DIR/slurm/setup_environment.sh"
    return 1 2>/dev/null || exit 1
fi

# Activate environments
export PATH=$TYKKY_ENV/bin:$PATH
source $VENV_DIR/bin/activate

# Verify activation
if command -v pocket-agent &> /dev/null; then
    echo "✓ Environment activated successfully!"
    echo "  Project: project_$PROJECT"
    echo "  Python: $(python --version)"
    echo "  pocket-agent is ready to use"
else
    echo "⚠ Environment activated but pocket-agent command not found"
    echo "  Try: pip install -e $PROJECT_DIR"
fi

# Set convenient aliases
alias pa="pocket-agent"
alias pa-chat="pocket-agent chat"
alias pa-bench="pocket-agent benchmark"
alias pa-models="pocket-agent model list"

echo ""
echo "Aliases available:"
echo "  pa         -> pocket-agent"
echo "  pa-chat    -> pocket-agent chat"
echo "  pa-bench   -> pocket-agent benchmark"
echo "  pa-models  -> pocket-agent model list"
