#!/bin/bash
#SBATCH --job-name=all-benchmarks
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=all-benchmarks-%j.out
#SBATCH --error=all-benchmarks-%j.err

# ==========================================================================
# All Benchmarks - BFCL + Web Search QA - Single SLURM Job
#
# Runs server + benchmark sequentially for each model inside one job.
# Models:
#   BFCL:       Gemma 3n E2B, Qwen 3.5 4B, Gemma 4 E2B
#   WebSearch:  Qwen 3 0.6B, Llama 3.2 3B
#
# Usage:
#   sbatch scripts/all_benchmarks_batch.sh
# ==========================================================================

# -- Common env --
source /appl/profile/zz-csc-env.sh 2>/dev/null || true
module load pytorch/2.9
set -eo pipefail

export HF_HOME=/scratch/project_2013932/vtoivone/hf_cache
export HF_TOKEN=${HF_TOKEN}
export TMPDIR=${LOCAL_SCRATCH:-/tmp}
mkdir -p $TMPDIR
export FLASHINFER_WORKSPACE_BASE=${LOCAL_SCRATCH:-/tmp}
export XDG_CACHE_HOME=${LOCAL_SCRATCH:-/tmp}/cache
mkdir -p $XDG_CACHE_HOME 2>/dev/null || true

# Export data dir so the scripts can find datasets
export POCKET_AGENT_HOME=/scratch/project_2013932/vtoivone/pocket-agent-cli/.pocket_agent_home
export POCKET_AGENT_DATA_DIR=/scratch/project_2013932/vtoivone/pocket-agent-cli/data
mkdir -p $POCKET_AGENT_HOME 2>/dev/null || true

BENCH_DIR=/scratch/project_2013932/vtoivone/pocket-agent-cli
PORT=30010

echo "============================================"
echo "All Benchmarks Runner"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $(hostname)"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Time:   $(date)"
echo "============================================"

# -- Helper: setup SGLang env --
setup_sglang_env() {
    export PYTHONUSERBASE=/projappl/project_2013932/vtoivone/sglang-env
    export PATH=/projappl/project_2013932/vtoivone/sglang-env/bin:$PATH
    export SGLANG_DISABLE_CUDNN_CHECK=1
    export LD_LIBRARY_PATH=/projappl/project_2013932/vtoivone/sglang-env/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
    export APPTAINERENV_LD_LIBRARY_PATH=/users/vtoivone/cudnn_override
}

# -- Helper: setup vLLM 0.16 env (Gemma 3n) --
setup_vllm16_env() {
    export PYTHONUSERBASE=/projappl/project_2013932/vtoivone/vllm-env
    export PATH=/projappl/project_2013932/vtoivone/vllm-env/bin:$PATH
    export APPTAINERENV_LD_LIBRARY_PATH=/users/vtoivone/cudnn_override
    export LD_LIBRARY_PATH=/users/vtoivone/cudnn_override:$LD_LIBRARY_PATH
    export VLLM_CACHE_DIR=${LOCAL_SCRATCH:-/tmp}/vllm_cache
    mkdir -p $VLLM_CACHE_DIR 2>/dev/null || true
}

# -- Helper: setup vLLM 0.19 env (Gemma 4) --
setup_vllm19_env() {
    export PYTHONUSERBASE=/scratch/project_2013932/vtoivone/vllm19-env
    export PATH=/scratch/project_2013932/vtoivone/vllm19-env/bin:$PATH
    export APPTAINERENV_LD_LIBRARY_PATH=/users/vtoivone/cudnn_override
    export LD_LIBRARY_PATH=/users/vtoivone/cudnn_override:$LD_LIBRARY_PATH
}

# -- Helper: wait for server --
wait_for_server() {
    local port=$1
    local max_wait=${2:-300}  # default 5 min
    echo "  Waiting for server on port $port..."
    local start=$(date +%s)
    while true; do
        if curl -s http://localhost:$port/v1/models 2>/dev/null | grep -q "id"; then
            echo "  Server ready after $(($(date +%s) - start))s"
            return 0
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "  ERROR: Server process died"
            return 1
        fi
        local elapsed=$(($(date +%s) - start))
        if [ $elapsed -ge $max_wait ]; then
            echo "  ERROR: Server didn't start within ${max_wait}s"
            return 1
        fi
        sleep 5
    done
}

# -- Helper: kill server --
kill_server() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 $SERVER_PID 2>/dev/null; then
        echo "  Stopping server (PID $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null || true
        sleep 5
    fi
    SERVER_PID=""
}

# -- Helper: run BFCL benchmark --
run_bfcl() {
    local model_id=$1
    local port=$2
    echo ""
    echo "  Running BFCL benchmark for $model_id..."
    cd $BENCH_DIR

    # Validate with 5 examples first
    echo "  Validation (5 examples):"
    python3 scripts/run_bfcl_benchmark.py \
        --models "$model_id" \
        --port $port \
        --limit 5 \
        --concurrency 5 2>&1 || true

    # Full run
    echo "  Full run (45 examples):"
    python3 scripts/run_bfcl_benchmark.py \
        --models "$model_id" \
        --port $port \
        --concurrency 30 2>&1

    echo "  BFCL complete for $model_id"
}

# -- Helper: run WebSearch QA benchmark --
run_websearch() {
    local model_id=$1
    local port=$2
    echo ""
    echo "  Running WebSearch QA benchmark for $model_id..."
    cd $BENCH_DIR

    python3 scripts/run_websearch_benchmark.py \
        --problems 100 \
        --models "$model_id" \
        --network-conditions wifi 4g poor_cellular \
        --port $port \
        --concurrency 5 \
        --timeout 180 2>&1

    echo "  WebSearch complete for $model_id"
}

# ==========================================================================
# 1. BFCL: Gemma 3n E2B (vLLM 0.16)
# ==========================================================================
echo ""
echo "########################################"
echo "# 1/5: Gemma 3n E2B - BFCL"
echo "########################################"
setup_vllm16_env

python3 -c 'import vllm; print("vLLM:", vllm.__version__)' 2>/dev/null || echo "vLLM: version check failed"

python3 -m vllm.entrypoints.openai.api_server \
    --model google/gemma-3n-E2B-it \
    --host 0.0.0.0 --port $PORT \
    --dtype auto --trust-remote-code \
    --gpu-memory-utilization 0.70 \
    --max-model-len 4096 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048 \
    --served-model-name google/gemma-3n-E2B-it &
SERVER_PID=$!

if wait_for_server $PORT 300; then
    # Reset env for running benchmark scripts (needs httpx etc from sglang-env)
    export PYTHONUSERBASE=/projappl/project_2013932/vtoivone/sglang-env
    export PATH=/projappl/project_2013932/vtoivone/sglang-env/bin:$PATH
    run_bfcl "gemma-3n-e2b-it" $PORT
fi
kill_server

# ==========================================================================
# 2. BFCL: Qwen 3.5 4B (SGLang)
# ==========================================================================
echo ""
echo "########################################"
echo "# 2/5: Qwen 3.5 4B - BFCL"
echo "########################################"
setup_sglang_env

python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-4B \
    --host 0.0.0.0 \
    --port $PORT \
    --dtype auto \
    --trust-remote-code \
    --served-model-name Qwen/Qwen3.5-4B \
    --tool-call-parser qwen3_coder &
SERVER_PID=$!

if wait_for_server $PORT 300; then
    run_bfcl "qwen-3.5-4b" $PORT
fi
kill_server

# ==========================================================================
# 3. BFCL: Gemma 4 E2B (vLLM 0.19)
# ==========================================================================
echo ""
echo "########################################"
echo "# 3/5: Gemma 4 E2B - BFCL"
echo "########################################"
setup_vllm19_env

python3 -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-E2B-it \
    --host 0.0.0.0 --port $PORT \
    --dtype auto --trust-remote-code \
    --enforce-eager \
    --gpu-memory-utilization 0.70 \
    --max-model-len 4096 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048 \
    --served-model-name google/gemma-4-E2B-it \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 &
SERVER_PID=$!

if wait_for_server $PORT 300; then
    # Reset env for benchmark scripts
    export PYTHONUSERBASE=/projappl/project_2013932/vtoivone/sglang-env
    export PATH=/projappl/project_2013932/vtoivone/sglang-env/bin:$PATH
    run_bfcl "gemma-4-e2b-it" $PORT
fi
kill_server

# ==========================================================================
# 4. WebSearch: Qwen 3 0.6B (SGLang)
# ==========================================================================
echo ""
echo "########################################"
echo "# 4/5: Qwen 3 0.6B - WebSearch QA"
echo "########################################"
setup_sglang_env

python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --host 0.0.0.0 \
    --port $PORT \
    --dtype auto \
    --trust-remote-code \
    --served-model-name Qwen/Qwen3-0.6B \
    --tool-call-parser qwen &
SERVER_PID=$!

if wait_for_server $PORT 300; then
    run_websearch "qwen-3-0.6b" $PORT
fi
kill_server

# ==========================================================================
# 5. WebSearch: Llama 3.2 3B (SGLang)
# ==========================================================================
echo ""
echo "########################################"
echo "# 5/5: Llama 3.2 3B - WebSearch QA"
echo "########################################"
setup_sglang_env

python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.2-3B-Instruct \
    --host 0.0.0.0 \
    --port $PORT \
    --dtype auto \
    --trust-remote-code \
    --served-model-name meta-llama/Llama-3.2-3B-Instruct \
    --tool-call-parser llama3 &
SERVER_PID=$!

if wait_for_server $PORT 300; then
    run_websearch "llama-3.2-3b-instruct" $PORT
fi
kill_server

# ==========================================================================
# Done
# ==========================================================================
echo ""
echo "============================================"
echo "ALL BENCHMARKS COMPLETE"
echo "  Time: $(date)"
echo "============================================"
echo ""
echo "Results in: $BENCH_DIR/data/results/"
ls -la $BENCH_DIR/data/results/bfcl/ 2>/dev/null
ls -la $BENCH_DIR/data/results/websearch_qa/ 2>/dev/null
