#!/bin/bash
# Monitor SLURM queue and run websearch benchmark on any available model.
# Works alongside the BFCL agent by piggybacking on its servers.
#
# Usage: bash scripts/ws_bench_monitor.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLI_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CLI_DIR"

PYTHON=".venv/bin/python"

# Models we need results for (Qwen 3 4B already done, DeepSeek R1 can't do tools)
declare -A MODEL_PORTS
MODEL_PORTS["Qwen/Qwen3-0.6B"]=30002
MODEL_PORTS["meta-llama/Llama-3.2-3B-Instruct"]=30003
MODEL_PORTS["Qwen/Qwen3.5-4B"]=30006
MODEL_PORTS["google/gemma-3n-E2B-it"]=30005
MODEL_PORTS["google/gemma-4-E2B-it"]=30007

declare -A MODEL_IDS
MODEL_IDS["Qwen/Qwen3-0.6B"]="qwen-3-0.6b"
MODEL_IDS["meta-llama/Llama-3.2-3B-Instruct"]="llama-3.2-3b-instruct"
MODEL_IDS["Qwen/Qwen3.5-4B"]="qwen-3.5-4b"
MODEL_IDS["google/gemma-3n-E2B-it"]="gemma-3n-e2b-it"
MODEL_IDS["google/gemma-4-E2B-it"]="gemma-4-e2b-it"

# Track which models we've already benchmarked
declare -A DONE

echo "=== Web Search QA Benchmark Monitor ==="
echo "Waiting for servers launched by other agents..."

while true; do
    # Check running jobs
    jobs=$(ssh mahti "squeue -u vtoivone -o '%i %j %N %T' --noheader 2>/dev/null" 2>/dev/null)

    running=$(echo "$jobs" | grep "RUNNING" || true)
    if [ -z "$running" ]; then
        sleep 10
        continue
    fi

    # Get the running job ID
    job_id=$(echo "$running" | awk '{print $1}' | head -1)
    node=$(echo "$running" | awk '{print $3}' | head -1)

    # Check which model it is
    model=$(ssh mahti "grep 'Model:' /users/vtoivone/*-${job_id}.out 2>/dev/null | head -1" 2>/dev/null | sed 's/.*Model: *//')

    if [ -z "$model" ]; then
        sleep 10
        continue
    fi

    model_id="${MODEL_IDS[$model]}"
    port="${MODEL_PORTS[$model]}"

    if [ -z "$model_id" ] || [ -z "$port" ]; then
        echo "Model $model not in our target list, skipping"
        sleep 10
        continue
    fi

    if [ -n "${DONE[$model_id]}" ]; then
        echo "Already benchmarked $model_id, skipping"
        sleep 10
        continue
    fi

    echo ""
    echo "=== Found $model on $node:$port (job $job_id) ==="

    # Set up tunnel
    kill $(lsof -t -i :$port 2>/dev/null) 2>/dev/null
    sleep 1
    ssh -f -N -L ${port}:${node}:${port} mahti 2>/dev/null

    # Wait for server to be ready
    echo "Waiting for server to be ready..."
    ready=0
    for i in $(seq 1 30); do
        result=$(curl -s http://localhost:${port}/v1/models 2>/dev/null)
        if echo "$result" | grep -q "model"; then
            echo "Server ready!"
            ready=1
            break
        fi
        sleep 5
    done

    if [ "$ready" = "0" ]; then
        echo "Server didn't become ready in time"
        continue
    fi

    # First validate with 5 problems
    echo "Validating $model_id..."
    $PYTHON scripts/validate_websearch_models.py --models "$model_id" --problems 5 2>&1

    # Then run full benchmark
    echo "Running full benchmark for $model_id..."
    $PYTHON scripts/run_websearch_benchmark.py \
        --problems 100 \
        --models "$model_id" \
        --network-conditions wifi 4g poor_cellular \
        --port "$port" \
        --concurrency 5 \
        --timeout 180 2>&1

    DONE[$model_id]=1
    echo "=== Completed $model_id ==="

    # Check if we have all models
    done_count=0
    for mid in "${MODEL_IDS[@]}"; do
        if [ -n "${DONE[$mid]}" ]; then
            done_count=$((done_count + 1))
        fi
    done

    if [ "$done_count" -ge 5 ]; then
        echo "All target models benchmarked!"
        break
    fi

    echo "Done: $done_count/5 models"
    sleep 10
done

echo "=== Monitor complete ==="
