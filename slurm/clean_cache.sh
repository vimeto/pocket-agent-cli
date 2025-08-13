#!/bin/bash
# Script to clean cached wheels and pip cache for llama-cpp-python

echo "Cleaning llama-cpp-python cache..."

# Set PROJECT if not set
if [ -z "$PROJECT" ]; then
    export PROJECT=2013932
fi

# Clean wheel cache
WHEEL_CACHE=/projappl/project_$PROJECT/$USER/pocket-agent-cli/wheel_cache
if [ -d "$WHEEL_CACHE" ]; then
    echo "Removing cached wheels from $WHEEL_CACHE..."
    rm -f $WHEEL_CACHE/llama_cpp_python*.whl
    ls -la $WHEEL_CACHE 2>/dev/null || echo "  Wheel cache is empty"
fi

# Clean pip cache
echo "Cleaning pip cache..."
pip cache remove llama-cpp-python 2>/dev/null || true

# Uninstall existing installation
if python -c "import llama_cpp" 2>/dev/null; then
    echo "Uninstalling existing llama-cpp-python..."
    pip uninstall -y llama-cpp-python
fi

echo "Cache cleaned successfully!"
echo ""
echo "Now run: source /projappl/project_$PROJECT/$USER/pocket-agent-cli/slurm/setup_environment.sh"