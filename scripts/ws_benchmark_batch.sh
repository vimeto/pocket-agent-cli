#!/bin/bash
#SBATCH --job-name=ws-benchmark
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=ws-benchmark-%j.out
#SBATCH --error=ws-benchmark-%j.err

# ==========================================================================
# Web Search QA Benchmark - All-in-one SLURM job
#
# Runs SGLang servers + benchmark sequentially for each model inside
# a single SLURM job, so it cannot be cancelled by other agents.
#
# Usage:
#   sbatch scripts/ws_benchmark_batch.sh
# ==========================================================================

set -e

source /appl/profile/zz-csc-env.sh
module load pytorch/2.9

export PYTHONUSERBASE=/projappl/project_2013932/vtoivone/sglang-env
export PATH=/projappl/project_2013932/vtoivone/sglang-env/bin:$PATH
export HF_HOME=/scratch/project_2013932/vtoivone/hf_cache
export HF_TOKEN=${HF_TOKEN}
export SGLANG_DISABLE_CUDNN_CHECK=1
export LD_LIBRARY_PATH=/projappl/project_2013932/vtoivone/sglang-env/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export FLASHINFER_WORKSPACE_BASE=/scratch/project_2013932/vtoivone/flashinfer_cache
mkdir -p $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer
export TMPDIR=${LOCAL_SCRATCH:-/tmp}
mkdir -p $TMPDIR
export APPTAINERENV_LD_LIBRARY_PATH=/users/vtoivone/cudnn_override

PORT=30010  # Use a non-standard port to avoid conflicts

# Clone the repo to scratch for running the benchmark
WORK_DIR=/scratch/project_2013932/vtoivone/ws_benchmark
mkdir -p $WORK_DIR

# We need the benchmark code. Copy from home if exists, or use git
if [ -d /users/vtoivone/pocket-agent/cli ]; then
    BENCH_DIR=/users/vtoivone/pocket-agent/cli
elif [ -d $WORK_DIR/cli ]; then
    BENCH_DIR=$WORK_DIR/cli
else
    echo "ERROR: Cannot find pocket-agent/cli"
    exit 1
fi

# Function to run benchmark for a model
run_benchmark() {
    local model_path="$1"
    local model_id="$2"
    local parser="$3"
    local max_tokens="${4:-2048}"

    echo ""
    echo "============================================"
    echo "Benchmarking: $model_id ($model_path)"
    echo "  Parser: ${parser:-none}"
    echo "  Port: $PORT"
    echo "  Time: $(date)"
    echo "============================================"

    # Build server command
    local sglang_cmd="python3 -m sglang.launch_server \
        --model-path $model_path \
        --host 0.0.0.0 \
        --port $PORT \
        --dtype auto \
        --trust-remote-code \
        --served-model-name $model_path"

    if [ -n "$parser" ]; then
        sglang_cmd="$sglang_cmd --tool-call-parser $parser"
    fi

    # Start server in background
    echo "Starting SGLang server..."
    $sglang_cmd &
    SERVER_PID=$!

    # Wait for server to be ready
    echo "Waiting for server to be ready..."
    for i in $(seq 1 60); do
        if curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q "model"; then
            echo "Server ready after ${i}0 seconds"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "ERROR: Server process died"
            return 1
        fi
        sleep 10
    done

    # Verify server
    if ! curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q "model"; then
        echo "ERROR: Server not ready, skipping $model_id"
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
        return 1
    fi

    # Run validation first (5 problems)
    echo "Running validation (5 problems)..."
    cd $BENCH_DIR
    python3 scripts/validate_websearch_models.py \
        --models "$model_id" \
        --problems 5 2>&1 || true

    # Run full benchmark (100 problems)
    echo "Running full benchmark (100 problems)..."
    python3 scripts/run_websearch_benchmark.py \
        --problems 100 \
        --models "$model_id" \
        --network-conditions wifi 4g poor_cellular \
        --port $PORT \
        --concurrency 5 \
        --timeout 180 2>&1

    echo "Benchmark complete for $model_id"

    # Kill server
    echo "Stopping server..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    sleep 5  # Let GPU memory free

    return 0
}

echo "============================================"
echo "Web Search QA Benchmark - Batch Runner"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Time: $(date)"
echo "============================================"

# Run benchmarks for each model
# Order: fastest first (smaller models first)

# 1. Qwen 3 0.6B
run_benchmark "Qwen/Qwen3-0.6B" "qwen-3-0.6b" "qwen" || echo "FAILED: qwen-3-0.6b"

# 2. Llama 3.2 3B
run_benchmark "meta-llama/Llama-3.2-3B-Instruct" "llama-3.2-3b-instruct" "llama3" || echo "FAILED: llama-3.2-3b-instruct"

# 3. Qwen 3.5 4B
run_benchmark "Qwen/Qwen3.5-4B" "qwen-3.5-4b" "qwen3_coder" || echo "FAILED: qwen-3.5-4b"

# 4. Qwen 3 4B (re-run to ensure consistency with new prompts)
# Skip: already have results
# run_benchmark "Qwen/Qwen3-4B" "qwen-3-4b" "qwen" || echo "FAILED: qwen-3-4b"

echo ""
echo "============================================"
echo "All benchmarks complete!"
echo "  Time: $(date)"
echo "============================================"
