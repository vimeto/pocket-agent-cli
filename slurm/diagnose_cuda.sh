#!/bin/bash
# Diagnostic script to help debug CUDA issues on Mahti

echo "================================="
echo "CUDA Diagnostics for Mahti"
echo "================================="
echo "Date: $(date)"
echo "Node: $(hostname -s)"
echo ""

# System information
echo "1. System Information"
echo "---------------------"
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"
echo ""

# Check if on GPU node
echo "2. Node Type Check"
echo "------------------"
CURRENT_NODE=$(hostname -s)
if [[ "$CURRENT_NODE" == g* ]]; then
    echo "✓ On GPU node: $CURRENT_NODE"
else
    echo "⚠ Not on GPU node: $CURRENT_NODE"
    echo "  GPU nodes start with 'g' (e.g., g1101, g1102)"
fi
echo ""

# Check loaded modules
echo "3. Currently Loaded Modules"
echo "---------------------------"
module list 2>&1 || echo "Module system not available"
echo ""

# Search for CUDA modules
echo "4. Available CUDA Modules"
echo "-------------------------"
echo "Searching with 'module spider cuda':"
module spider cuda 2>&1 | grep -E "cuda/|nvidia" || echo "No CUDA modules found"
echo ""
echo "Searching with 'module avail':"
module avail 2>&1 | grep -i cuda || echo "No CUDA modules in avail"
echo ""

# Try to load CUDA
echo "5. Attempting to Load CUDA Modules"
echo "-----------------------------------"
CUDA_VERSIONS="cuda cuda/11.7.0 cuda/11.8.0 cuda/12.0.0 cuda/12.1.0 cuda/12.2.0 nvidia"
for version in $CUDA_VERSIONS; do
    echo -n "Trying $version... "
    if module load $version 2>/dev/null; then
        echo "SUCCESS"
        module list 2>&1 | grep -i cuda
        break
    else
        echo "failed"
    fi
done
echo ""

# Search filesystem for CUDA
echo "6. CUDA Installations in Filesystem"
echo "------------------------------------"
echo "Searching common locations for nvcc:"
for dir in /appl /opt /usr/local /apps; do
    if [ -d "$dir" ]; then
        echo "Checking $dir..."
        find $dir -maxdepth 4 -name nvcc 2>/dev/null | head -5
    fi
done
echo ""

# Check environment variables
echo "7. CUDA Environment Variables"
echo "-----------------------------"
echo "PATH: $PATH" | grep -o "[^:]*cuda[^:]*" || echo "No CUDA in PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" | grep -o "[^:]*cuda[^:]*" || echo "No CUDA in LD_LIBRARY_PATH"
echo "CUDA_HOME: ${CUDA_HOME:-not set}"
echo "CUDA_PATH: ${CUDA_PATH:-not set}"
echo "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER:-not set}"
echo ""

# Check for nvcc
echo "8. CUDA Compiler (nvcc) Check"
echo "------------------------------"
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc found at: $(which nvcc)"
    echo "Version:"
    nvcc --version
else
    echo "✗ nvcc not found in PATH"
fi
echo ""

# Check for nvidia-smi
echo "9. GPU Detection (nvidia-smi)"
echo "------------------------------"
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found at: $(which nvidia-smi)"
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,driver_version,compute_cap --format=csv
    echo ""
    echo "GPU Memory:"
    nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv
else
    echo "✗ nvidia-smi not found"
fi
echo ""

# Check Python and llama-cpp-python
echo "10. Python Environment Check"
echo "----------------------------"
if command -v python &> /dev/null; then
    echo "Python: $(which python)"
    echo "Version: $(python --version)"
    echo ""
    echo "Checking llama-cpp-python:"
    python -c "
import sys
try:
    import llama_cpp
    print(f'✓ llama-cpp-python version: {llama_cpp.__version__}')
    # Check if it was built with CUDA
    import subprocess
    result = subprocess.run(['ldd', llama_cpp._lib_path], capture_output=True, text=True)
    if 'libcuda' in result.stdout or 'libcudart' in result.stdout:
        print('✓ Linked with CUDA libraries')
    else:
        print('✗ Not linked with CUDA libraries (CPU version)')
except ImportError as e:
    print(f'✗ llama-cpp-python not installed: {e}')
except Exception as e:
    print(f'Error checking: {e}')
" 2>&1
else
    echo "Python not found"
fi
echo ""

# Summary and recommendations
echo "================================="
echo "Summary and Recommendations"
echo "================================="

if [[ "$CURRENT_NODE" != g* ]]; then
    echo "⚠ You're not on a GPU node. To use CUDA:"
    echo "  srun --account=project_\$PROJECT --partition=gputest --gres=gpu:a100:1 --time=0:15:00 --pty bash"
elif ! command -v nvcc &> /dev/null; then
    echo "⚠ On GPU node but nvcc not found. Try:"
    echo "  1. module load cuda/11.7.0  (or another version from the list above)"
    echo "  2. export PATH=/appl/opt/cuda/11.7.0/bin:\$PATH"
    echo "  3. source /projappl/project_\$PROJECT/\$USER/pocket-agent-cli/slurm/fix_cuda.sh"
elif ! command -v nvidia-smi &> /dev/null; then
    echo "⚠ CUDA compiler found but GPU not accessible. Check with system admin."
else
    echo "✓ CUDA environment appears to be set up correctly!"
    echo "  If llama-cpp-python shows 'CPU version', reinstall it:"
    echo "  pip uninstall -y llama-cpp-python"
    echo "  CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python --no-cache-dir"
fi

echo ""
echo "For more help, save this output and contact CSC support:"
echo "servicedesk@csc.fi"