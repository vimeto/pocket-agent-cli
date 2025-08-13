#!/bin/bash
# Diagnostic script to check CUDA setup for pocket-agent-cli

echo "==================================="
echo "CUDA Diagnostic Check"
echo "==================================="
echo ""

# Check hostname
echo "Current node: $(hostname -s)"
echo ""

# Check if CUDA modules are loaded
echo "Loaded modules:"
module list 2>&1 | grep -E "(cuda|gcc)" || echo "  No CUDA/GCC modules loaded"
echo ""
echo "Expected: gcc/13.1.0 and cuda/11.5.0"
echo ""

# Check CUDA compiler
echo "CUDA compiler (nvcc):"
if command -v nvcc &> /dev/null; then
    nvcc --version | head -n4
    echo "  Location: $(which nvcc)"
else
    echo "  NOT FOUND - This is critical!"
fi
echo ""

# Check CUDA environment variables
echo "CUDA environment variables:"
echo "  CUDA_HOME: ${CUDA_HOME:-NOT SET}"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-NOT SET}"
echo ""

# Check for CUDA runtime libraries
echo "Checking for CUDA runtime libraries:"
if [ -n "$CUDA_HOME" ]; then
    echo "  In $CUDA_HOME/lib64:"
    ls -la $CUDA_HOME/lib64/libcudart.so* 2>/dev/null | head -5 || echo "    No libcudart.so found"
    echo ""
    echo "  In $CUDA_HOME/lib:"
    ls -la $CUDA_HOME/lib/libcudart.so* 2>/dev/null | head -5 || echo "    No libcudart.so found"
else
    echo "  CUDA_HOME not set - cannot check library locations"
fi
echo ""

# Check if in virtual environment
echo "Python environment:"
if [ -n "$VIRTUAL_ENV" ]; then
    echo "  Virtual env: $VIRTUAL_ENV"
    echo "  Python: $(which python)"
    echo "  Version: $(python --version)"
else
    echo "  NOT in virtual environment"
fi
echo ""

# Try to import llama-cpp-python
echo "Testing llama-cpp-python import:"
python -c "
import sys
try:
    import llama_cpp
    print(f'  ✓ Import successful')
    print(f'  Version: {llama_cpp.__version__}')
    
    # Try to check for CUDA support
    try:
        from llama_cpp import llama_backend_init
        llama_backend_init()
        print('  ✓ Backend initialized')
    except Exception as e:
        print(f'  ✗ Backend init failed: {e}')
        
except ImportError as e:
    print(f'  ✗ Import failed: {e}')
except Exception as e:
    print(f'  ✗ Unexpected error: {e}')
" 2>&1
echo ""

# Check shared library dependencies
echo "Checking shared library dependencies:"
LLAMA_LIB=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)/llama_cpp/lib/libllama.so
if [ -f "$LLAMA_LIB" ]; then
    echo "  libllama.so location: $LLAMA_LIB"
    echo "  Missing libraries:"
    ldd "$LLAMA_LIB" 2>/dev/null | grep "not found" || echo "    None - all libraries found!"
else
    echo "  libllama.so not found in expected location"
fi
echo ""

# GPU check
echo "GPU availability:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "  nvidia-smi not available"
fi
echo ""

echo "==================================="
echo "Diagnostic Summary:"
echo "==================================="

# Summary checks
ISSUES=0

if ! command -v nvcc &> /dev/null; then
    echo "✗ CUDA compiler (nvcc) not found"
    ISSUES=$((ISSUES + 1))
else
    echo "✓ CUDA compiler available"
fi

if [ -z "$CUDA_HOME" ]; then
    echo "✗ CUDA_HOME not set"
    ISSUES=$((ISSUES + 1))
else
    echo "✓ CUDA_HOME set"
fi

if [ -z "$LD_LIBRARY_PATH" ] || [[ "$LD_LIBRARY_PATH" != *"cuda"* ]]; then
    echo "✗ LD_LIBRARY_PATH missing CUDA paths"
    ISSUES=$((ISSUES + 1))
else
    echo "✓ LD_LIBRARY_PATH includes CUDA"
fi

if ! python -c "import llama_cpp" 2>/dev/null; then
    echo "✗ llama-cpp-python not importable"
    ISSUES=$((ISSUES + 1))
else
    echo "✓ llama-cpp-python imports successfully"
fi

echo ""
if [ $ISSUES -eq 0 ]; then
    echo "✅ All checks passed! CUDA setup appears correct."
else
    echo "⚠️ Found $ISSUES issue(s). Please address them before running benchmarks."
    echo ""
    echo "To fix:"
    echo "1. Make sure you're on a GPU node (hostname should start with 'g')"
    echo "2. Run: source /projappl/project_2013932/\$USER/pocket-agent-cli/slurm/activate_env.sh"
    echo "3. If issues persist, re-run: source /projappl/project_2013932/\$USER/pocket-agent-cli/slurm/setup_environment.sh"
    echo "4. Note: First setup will take 10-15 minutes to build llama-cpp-python from source"
fi