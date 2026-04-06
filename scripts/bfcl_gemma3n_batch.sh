#!/bin/bash
#SBATCH --job-name=bfcl-gemma3n
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --time=00:25:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=bfcl-gemma3n-%j.out
#SBATCH --error=bfcl-gemma3n-%j.err

# BFCL benchmark for Gemma 3n E2B (vLLM 0.16)
source /appl/profile/zz-csc-env.sh 2>/dev/null || true
module load pytorch/2.9
set -eo pipefail

export PYTHONUSERBASE=/projappl/project_2013932/vtoivone/vllm-env
export PATH=/projappl/project_2013932/vtoivone/vllm-env/bin:$PATH
export APPTAINERENV_LD_LIBRARY_PATH=/users/vtoivone/cudnn_override
export LD_LIBRARY_PATH=/users/vtoivone/cudnn_override:$LD_LIBRARY_PATH
export HF_HOME=/scratch/project_2013932/vtoivone/hf_cache
export HF_TOKEN=${HF_TOKEN}
export TMPDIR=${LOCAL_SCRATCH:-/tmp}
export FLASHINFER_WORKSPACE_BASE=${LOCAL_SCRATCH:-/tmp}
export XDG_CACHE_HOME=${LOCAL_SCRATCH:-/tmp}/cache
export VLLM_CACHE_DIR=${LOCAL_SCRATCH:-/tmp}/vllm_cache
mkdir -p $XDG_CACHE_HOME $VLLM_CACHE_DIR 2>/dev/null || true
export POCKET_AGENT_HOME=/scratch/project_2013932/vtoivone/pocket-agent-cli/.pocket_agent_home
export POCKET_AGENT_DATA_DIR=/scratch/project_2013932/vtoivone/pocket-agent-cli/data
mkdir -p $POCKET_AGENT_HOME 2>/dev/null || true

PORT=30010
BENCH_DIR=/scratch/project_2013932/vtoivone/pocket-agent-cli

echo "=== BFCL: Gemma 3n E2B (Job $SLURM_JOB_ID on $(hostname)) ==="

# Start vLLM server
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

# Wait for server
for i in $(seq 1 60); do
    if curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q "id"; then
        echo "Server ready after $((i*5))s"
        break
    fi
    kill -0 $SERVER_PID 2>/dev/null || { echo "Server died"; exit 1; }
    sleep 5
done

# Switch to sglang-env for benchmark deps
export PYTHONUSERBASE=/projappl/project_2013932/vtoivone/sglang-env
export PATH=/projappl/project_2013932/vtoivone/sglang-env/bin:$PATH

cd $BENCH_DIR

# Validate
echo "Validation (5 examples)..."
python3 scripts/run_bfcl_benchmark.py --models gemma-3n-e2b-it --port $PORT --limit 5 --concurrency 5 2>&1 || true

# Full run
echo "Full BFCL (45 examples)..."
python3 scripts/run_bfcl_benchmark.py --models gemma-3n-e2b-it --port $PORT --concurrency 30 2>&1

kill $SERVER_PID 2>/dev/null
echo "=== DONE ==="
