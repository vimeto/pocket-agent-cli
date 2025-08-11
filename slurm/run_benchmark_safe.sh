#!/bin/bash
# Safe benchmark runner for Mahti that handles Singularity/containerization issues

# Parse command line arguments
MODEL_NAME=${1:-"llama-3.2-3b-instruct"}
MODE=${2:-"base"}
PROBLEMS=${3:-"1-5"}
NUM_SAMPLES=${4:-"5"}

echo "================================="
echo "Safe Benchmark Runner for Mahti"
echo "================================="
echo "Model: $MODEL_NAME"
echo "Mode: $MODE"
echo "Problems: $PROBLEMS"
echo "Samples: $NUM_SAMPLES"
echo ""

# Check if we're on a compute node
CURRENT_NODE=$(hostname -s)
if [[ "$CURRENT_NODE" == *"login"* ]]; then
    echo "ERROR: On login node. Please get a compute node first:"
    echo "  srun --account=project_$PROJECT --partition=gputest --gres=gpu:a100:1 --time=0:15:00 --pty bash"
    exit 1
fi

# Set up environment
export PROJECT=${PROJECT:-2013932}
export PROJECT_DIR=/projappl/project_$PROJECT/$USER/pocket-agent-cli
export TYKKY_ENV=$PROJECT_DIR/tykky-env
export VENV_DIR=$PROJECT_DIR/venv

# CRITICAL: Disable Docker completely
export DISABLE_DOCKER=1

# Set memory limits to prevent bus errors
ulimit -v unlimited 2>/dev/null || true
ulimit -m unlimited 2>/dev/null || true
ulimit -s unlimited 2>/dev/null || true

# Add paths
export PATH=$TYKKY_ENV/bin:$VENV_DIR/bin:$PATH

# Load CUDA if on GPU node
if [[ "$CURRENT_NODE" == g* ]]; then
    echo "GPU node detected, loading CUDA..."
    module load cuda 2>/dev/null || module load cuda/12.0.0 2>/dev/null || true
    
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU available:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    fi
fi

# Activate virtual environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    source $VENV_DIR/bin/activate
else
    echo "ERROR: Virtual environment not found. Run setup_environment.sh first."
    exit 1
fi

# Create output directory
OUTPUT_DIR=$PROJECT_DIR/data/results/bench_${MODEL_NAME}_${MODE}_$(date +%Y%m%d_%H%M%S)
mkdir -p $OUTPUT_DIR

echo ""
echo "Running benchmark..."
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run the benchmark with explicit Python interpreter
# Use the actual Python binary, not the wrapper
PYTHON_BIN=$VENV_DIR/bin/python

# Check if we can use the Python directly
if [ -f "$PYTHON_BIN" ]; then
    # Try to run Python directly, bypassing any wrappers
    $PYTHON_BIN -c "import pocket_agent_cli; print('✓ Direct Python works')" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "Using direct Python interpreter"
        # Run benchmark using direct Python
        $PYTHON_BIN -m pocket_agent_cli.cli benchmark \
            --model "$MODEL_NAME" \
            --mode "$MODE" \
            --problems "$PROBLEMS" \
            --num-samples "$NUM_SAMPLES" \
            --output-dir "$OUTPUT_DIR"
    else
        echo "Direct Python failed, trying pocket-agent command"
        # Fallback to pocket-agent command
        pocket-agent benchmark \
            --model "$MODEL_NAME" \
            --mode "$MODE" \
            --problems "$PROBLEMS" \
            --num-samples "$NUM_SAMPLES" \
            --output-dir "$OUTPUT_DIR"
    fi
else
    echo "Python binary not found, using pocket-agent command"
    pocket-agent benchmark \
        --model "$MODEL_NAME" \
        --mode "$MODE" \
        --problems "$PROBLEMS" \
        --num-samples "$NUM_SAMPLES" \
        --output-dir "$OUTPUT_DIR"
fi

echo ""
echo "================================="
echo "Benchmark Complete"
echo "================================="
echo "Results saved to: $OUTPUT_DIR"