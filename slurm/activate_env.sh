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
if [[ "$CURRENT_NODE" == g* ]]; then
    # Load CUDA module
    if ! module list 2>&1 | grep -q cuda; then
        module load cuda 2>/dev/null || true
    fi
    
    # CRITICAL FIX: llama-cpp-python needs libcudart.so.12 (CUDA 12)
    # Search for CUDA 12 installations
    CUDA_FOUND=false
    for cuda_base in /appl/opt /opt /usr/local; do
        if [ -d "$cuda_base/cuda" ]; then
            # Check subdirectories for CUDA 12
            for cuda_dir in $cuda_base/cuda/12* $cuda_base/cuda/cuda-12*; do
                if [ -d "$cuda_dir/lib64" ] && [ -f "$cuda_dir/lib64/libcudart.so.12" ]; then
                    export LD_LIBRARY_PATH=$cuda_dir/lib64:$cuda_dir/lib:$LD_LIBRARY_PATH
                    export CUDA_HOME=$cuda_dir
                    CUDA_FOUND=true
                    echo "  Found CUDA 12 at: $cuda_dir"
                    break 2
                fi
            done
            # Also check specific versions
            for version in 12.6.0 12.2.0 12.1.0 12.0.0; do
                cuda_path="$cuda_base/cuda/$version"
                if [ -d "$cuda_path/lib64" ] && [ -f "$cuda_path/lib64/libcudart.so.12" ]; then
                    export LD_LIBRARY_PATH=$cuda_path/lib64:$cuda_path/lib:$LD_LIBRARY_PATH
                    export CUDA_HOME=$cuda_path
                    CUDA_FOUND=true
                    echo "  Found CUDA $version at: $cuda_path"
                    break 2
                fi
            done
        fi
    done
    
    # If CUDA 12 not found, warn and suggest CPU reinstall
    if [ "$CUDA_FOUND" = false ]; then
        echo "  ⚠ WARNING: CUDA 12 libraries not found!"
        echo "  llama-cpp-python needs CUDA 12 but it's not available."
        echo "  To fix, either:"
        echo "    1. Find CUDA 12: ls -la /appl/opt/cuda/"
        echo "    2. Reinstall llama-cpp-python without CUDA:"
        echo "       pip uninstall -y llama-cpp-python"
        echo "       pip install llama-cpp-python --no-binary llama-cpp-python"
    fi
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
