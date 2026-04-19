#!/bin/bash
#SBATCH --job-name=vllm-qwen34b-mbpp
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=vllm-qwen34b-mbpp-%j.out
#SBATCH --error=vllm-qwen34b-mbpp-%j.err

# ==========================================================================
# Cross-engine calibration: Qwen 3 4B on MBPP via vLLM 0.19
#
# Paired with the existing SGLang sweep at
#   data/results/full_cloud_sweep/sglang_20260402_162457/qwen-3-4b_base.jsonl
# to show engine-invariance of relative scaling (§5.6 calibration table).
#
# Base mode only: avoids the vLLM Qwen tool-call parser dependency, which
# would confound engine effects with parser effects. Base is the cleanest
# anchor for the absolute-numbers-differ / relative-scaling-holds claim.
#
# Usage:
#   sbatch scripts/mbpp_qwen34b_vllm_batch.sh
# ==========================================================================

source /appl/profile/zz-csc-env.sh 2>/dev/null || true
module load pytorch/2.9
set -eo pipefail

export HF_HOME=/scratch/project_2013932/vtoivone/hf_cache
export HF_TOKEN=${HF_TOKEN}
export TMPDIR=${LOCAL_SCRATCH:-/tmp}
mkdir -p $TMPDIR
export APPTAINERENV_LD_LIBRARY_PATH=/users/vtoivone/cudnn_override
export LD_LIBRARY_PATH=/users/vtoivone/cudnn_override:$LD_LIBRARY_PATH
export VLLM_CACHE_DIR=${LOCAL_SCRATCH:-/tmp}/vllm_cache
mkdir -p $VLLM_CACHE_DIR 2>/dev/null || true
export POCKET_AGENT_HOME=/scratch/project_2013932/vtoivone/pocket-agent-cli/.pocket_agent_home
export POCKET_AGENT_DATA_DIR=/scratch/project_2013932/vtoivone/pocket-agent-cli/data
mkdir -p $POCKET_AGENT_HOME 2>/dev/null || true

PORT=30010
BENCH_DIR=/scratch/project_2013932/vtoivone/pocket-agent-cli

echo "============================================"
echo "vLLM calibration: Qwen 3 4B on MBPP"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $(hostname)"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Time:   $(date)"
echo "============================================"

# -- vLLM env (same as Gemma 4 BFCL run) --
export PYTHONUSERBASE=/scratch/project_2013932/vtoivone/vllm19-env
export PATH=/scratch/project_2013932/vtoivone/vllm19-env/bin:$PATH

python3 -c 'import vllm; print("vLLM:", vllm.__version__)' 2>/dev/null || echo "vLLM: version check failed"

# -- Launch vLLM server --
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --host 0.0.0.0 --port $PORT \
    --dtype auto --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --served-model-name Qwen/Qwen3-4B &
SERVER_PID=$!

# -- Wait for server --
for i in $(seq 1 60); do
    if curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q "id"; then
        echo "Server ready after $((i*5))s"
        break
    fi
    kill -0 $SERVER_PID 2>/dev/null || { echo "ERROR: Server died"; exit 1; }
    sleep 5
done

# -- Switch env for benchmark client (needs httpx + repo deps) --
export PYTHONUSERBASE=/projappl/project_2013932/vtoivone/sglang-env
export PATH=/projappl/project_2013932/vtoivone/sglang-env/bin:$PATH

cd $BENCH_DIR

echo ""
echo "--- MBPP sweep (base mode, 500 problems) ---"
python3 scripts/run_benchmarks_sglang.py \
    --dataset mbpp \
    --problems 500 \
    --models qwen-3-4b \
    --modes base \
    --port $PORT \
    --concurrency 50 \
    --output-dir data/results/full_vllm_sweep 2>&1

echo ""
echo "--- Stopping server ---"
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "============================================"
echo "vLLM CALIBRATION COMPLETE"
echo "  Time: $(date)"
echo "============================================"
ls -la $BENCH_DIR/data/results/full_vllm_sweep/ 2>/dev/null
