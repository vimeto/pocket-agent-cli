#!/bin/bash
#SBATCH --job-name=bfcl-qwen35
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=bfcl-qwen35-%j.out
#SBATCH --error=bfcl-qwen35-%j.err

# BFCL benchmark for Qwen 3.5 4B (SGLang)
source /appl/profile/zz-csc-env.sh 2>/dev/null || true
module load pytorch/2.9
set -eo pipefail

export PYTHONUSERBASE=/projappl/project_2013932/vtoivone/sglang-env
export PATH=/projappl/project_2013932/vtoivone/sglang-env/bin:$PATH
export HF_HOME=/scratch/project_2013932/vtoivone/hf_cache
export HF_TOKEN=${HF_TOKEN}
export SGLANG_DISABLE_CUDNN_CHECK=1
export LD_LIBRARY_PATH=/projappl/project_2013932/vtoivone/sglang-env/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export FLASHINFER_WORKSPACE_BASE=/scratch/project_2013932/vtoivone/flashinfer_cache
mkdir -p $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer 2>/dev/null || true
export TMPDIR=${LOCAL_SCRATCH:-/tmp}
mkdir -p $TMPDIR
export APPTAINERENV_LD_LIBRARY_PATH=/users/vtoivone/cudnn_override
export POCKET_AGENT_HOME=/scratch/project_2013932/vtoivone/pocket-agent-cli/.pocket_agent_home
export POCKET_AGENT_DATA_DIR=/scratch/project_2013932/vtoivone/pocket-agent-cli/data
mkdir -p $POCKET_AGENT_HOME 2>/dev/null || true

PORT=30010
BENCH_DIR=/scratch/project_2013932/vtoivone/pocket-agent-cli

echo "=== BFCL: Qwen 3.5 4B (Job $SLURM_JOB_ID on $(hostname)) ==="

python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-4B \
    --host 0.0.0.0 --port $PORT \
    --dtype auto --trust-remote-code \
    --served-model-name Qwen/Qwen3.5-4B \
    --tool-call-parser qwen3_coder &
SERVER_PID=$!

for i in $(seq 1 60); do
    if curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q "id"; then
        echo "Server ready after $((i*5))s"
        break
    fi
    kill -0 $SERVER_PID 2>/dev/null || { echo "Server died"; exit 1; }
    sleep 5
done

cd $BENCH_DIR

echo "Validation (5 examples)..."
python3 scripts/run_bfcl_benchmark.py --models qwen-3.5-4b --port $PORT --limit 5 --concurrency 5 2>&1 || true

echo "Full BFCL (45 examples)..."
python3 scripts/run_bfcl_benchmark.py --models qwen-3.5-4b --port $PORT --concurrency 30 2>&1

kill $SERVER_PID 2>/dev/null
echo "=== DONE ==="
