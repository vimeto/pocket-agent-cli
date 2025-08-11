#!/bin/bash
# Test script to verify pocket-agent-cli installation on Mahti

echo "================================="
echo "Testing Pocket Agent Installation"
echo "================================="
echo "Date: $(date)"
echo "Node: $(hostname -s)"
echo ""

# Set up environment
export PROJECT=${PROJECT:-2013932}
export PROJECT_DIR=/projappl/project_$PROJECT/$USER/pocket-agent-cli
export DISABLE_DOCKER=1

# Activate environment
echo "1. Activating environment..."
source $PROJECT_DIR/slurm/activate_env.sh

echo ""
echo "2. Environment variables:"
echo "   PROJECT: $PROJECT"
echo "   PROJECT_DIR: $PROJECT_DIR"
echo "   DISABLE_DOCKER: $DISABLE_DOCKER"
echo "   PATH includes: $(echo $PATH | grep -o pocket-agent-cli | head -1)"

echo ""
echo "3. Python environment:"
echo "   Python: $(which python)"
echo "   Version: $(python --version)"

echo ""
echo "4. Testing imports:"
echo -n "   pocket_agent_cli: "
python -c "import pocket_agent_cli; print('✓ OK')" 2>&1 || echo "✗ FAILED"

echo -n "   llama_cpp: "
python -c "import llama_cpp; print('✓ OK - version', llama_cpp.__version__)" 2>&1 || echo "✗ FAILED"

echo -n "   Docker disabled: "
python -c "import os; print('✓ OK' if os.environ.get('DISABLE_DOCKER')=='1' else '✗ NOT SET')"

echo ""
echo "5. Testing CLI:"
echo -n "   pocket-agent command: "
which pocket-agent &>/dev/null && echo "✓ Found at $(which pocket-agent)" || echo "✗ NOT FOUND"

echo -n "   pocket-agent --version: "
pocket-agent --version 2>&1 | head -1 || echo "✗ FAILED"

echo ""
echo "6. Testing direct Python execution:"
echo -n "   python -m pocket_agent_cli.cli: "
python -m pocket_agent_cli.cli --help &>/dev/null && echo "✓ OK" || echo "✗ FAILED"

echo ""
echo "7. Testing model operations:"
echo "   Available models:"
pocket-agent model list 2>&1 | head -5 || echo "   ✗ Failed to list models"

echo ""
echo "8. Quick benchmark test (if model available):"
MODEL="gemma-3n-e2b-it"
if pocket-agent model list 2>&1 | grep -q "$MODEL"; then
    echo "   Testing with $MODEL..."
    python -m pocket_agent_cli.cli benchmark \
        --model "$MODEL" \
        --mode "base" \
        --problems "1" \
        --num-samples "1" \
        --output-dir "/tmp/test_$$" \
        --no-monitoring 2>&1 | grep -E "(Success|Failed|Error)" | head -5
else
    echo "   Model $MODEL not found. Download with:"
    echo "   pocket-agent model download $MODEL"
fi

echo ""
echo "================================="
echo "Test Complete"
echo "================================="

# Check for common issues
echo ""
echo "Common Issues Detected:"
ISSUES_FOUND=false

if ! command -v pocket-agent &>/dev/null; then
    echo "⚠ pocket-agent command not in PATH"
    ISSUES_FOUND=true
fi

if ! python -c "import pocket_agent_cli" 2>/dev/null; then
    echo "⚠ pocket_agent_cli module not importable"
    ISSUES_FOUND=true
fi

if [ "$DISABLE_DOCKER" != "1" ]; then
    echo "⚠ DISABLE_DOCKER not set (will cause errors on Mahti)"
    ISSUES_FOUND=true
fi

if [ "$ISSUES_FOUND" = false ]; then
    echo "✓ No issues detected - installation appears correct!"
fi