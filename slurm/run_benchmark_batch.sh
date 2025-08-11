#!/bin/bash
#SBATCH --job-name=pocket-bench
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gres=gpu:a100:1,nvme:200
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

# Parse command line arguments
MODEL_NAME=${1:-"llama-3.2-3b-instruct"}
MODE=${2:-"all"}  # base, tool_submission, full_tool, or all
START_INDEX=${3:-0}
TOTAL_PROBLEMS=${4:-509}
BATCH_SIZE=${5:-10}
NUM_SAMPLES=${6:-10}

echo "================================="
echo "Pocket Agent Benchmark Job"
echo "================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Model: $MODEL_NAME"
echo "Mode: $MODE"
echo "Problems: $START_INDEX to $((START_INDEX + TOTAL_PROBLEMS - 1))"
echo "Batch size: $BATCH_SIZE"
echo "Samples per problem: $NUM_SAMPLES"
echo "================================="

# Set up environment
export PROJECT=2013932
PROJECT_DIR=/projappl/project_$PROJECT/$USER/pocket-agent-cli

# Activate environment
source $PROJECT_DIR/slurm/activate_env.sh

# Set work directory
cd $PROJECT_DIR

# Check if LOCAL_SCRATCH is available and use it
if [ -d "$LOCAL_SCRATCH" ]; then
    echo "Using LOCAL_SCRATCH for temporary files: $LOCAL_SCRATCH"
    export TMPDIR=$LOCAL_SCRATCH
    export TEMP=$LOCAL_SCRATCH
    
    # Copy code to local scratch for faster I/O
    echo "Copying code to local scratch..."
    cp -r pocket_agent_cli $LOCAL_SCRATCH/
    cd $LOCAL_SCRATCH
fi

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
export CUDA_VISIBLE_DEVICES=0

# Start GPU monitoring in background
echo ""
echo "Starting GPU monitoring..."
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw --format=csv -l 5 > $PROJECT_DIR/data/logs/gpu_monitor_${SLURM_JOB_ID}.csv &
GPU_MONITOR_PID=$!

# Start CPU/Memory monitoring
echo "Starting system monitoring..."
(while true; do
    echo "$(date +%s),$(top -bn1 | grep "Cpu(s)" | awk '{print $2}'),$(free -m | awk 'NR==2{printf "%.1f", $3*100/$2}')" >> $PROJECT_DIR/data/logs/cpu_monitor_${SLURM_JOB_ID}.csv
    sleep 5
done) &
CPU_MONITOR_PID=$!

# Docker check and setup (if available)
DOCKER_AVAILABLE=false
if command -v docker &> /dev/null; then
    echo ""
    echo "Checking Docker availability..."
    if docker info &> /dev/null; then
        DOCKER_AVAILABLE=true
        echo "✓ Docker is available"
        # Pull Python image if needed
        docker pull python:3.11-slim &> /dev/null || echo "Could not pull Docker image"
    else
        echo "⚠ Docker daemon not accessible"
    fi
else
    echo "⚠ Docker not installed"
fi

# Disable Docker if not available
if [ "$DOCKER_AVAILABLE" = false ]; then
    export DISABLE_DOCKER=1
    echo "Running without Docker sandboxing"
fi

# Function to run benchmark for a batch
run_benchmark_batch() {
    local start=$1
    local end=$2
    local mode=$3
    
    # Build problem IDs list
    local problem_ids=""
    for ((i=start; i<=end && i<START_INDEX+TOTAL_PROBLEMS; i++)); do
        if [ -z "$problem_ids" ]; then
            problem_ids="$i"
        else
            problem_ids="${problem_ids},$i"
        fi
    done
    
    echo ""
    echo "Running $mode benchmark for problems: $problem_ids"
    
    # Create output filename with timestamp
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_file="$PROJECT_DIR/data/results/bench_${MODEL_NAME}_${mode}_${timestamp}_job${SLURM_JOB_ID}.json"
    
    # Run benchmark with error handling
    if timeout 3600 pocket-agent benchmark \
        --model "$MODEL_NAME" \
        --mode "$mode" \
        --problems "$problem_ids" \
        --num-samples "$NUM_SAMPLES" \
        --output "$output_file"; then
        echo "✓ Batch completed successfully"
    else
        echo "⚠ Batch failed or timed out"
        # Log the failure
        echo "$(date),${mode},${problem_ids},FAILED" >> $PROJECT_DIR/data/logs/failures_${SLURM_JOB_ID}.log
    fi
    
    # Clean up any Docker containers if they exist
    if [ "$DOCKER_AVAILABLE" = true ]; then
        docker ps -a --filter "label=pocket_agent_cli_sandbox" -q | xargs -r docker rm -f &> /dev/null || true
    fi
}

# Main benchmark loop
echo ""
echo "Starting benchmark loop..."
echo "================================="

current_index=$START_INDEX
end_index=$((START_INDEX + TOTAL_PROBLEMS))

while [ $current_index -lt $end_index ]; do
    batch_end=$((current_index + BATCH_SIZE - 1))
    if [ $batch_end -ge $end_index ]; then
        batch_end=$((end_index - 1))
    fi
    
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing batch: $current_index to $batch_end"
    
    # Run benchmarks based on mode
    if [ "$MODE" = "all" ]; then
        run_benchmark_batch $current_index $batch_end "base"
        run_benchmark_batch $current_index $batch_end "tool_submission"
        run_benchmark_batch $current_index $batch_end "full_tool"
    else
        run_benchmark_batch $current_index $batch_end "$MODE"
    fi
    
    # Update progress
    current_index=$((batch_end + 1))
    completed=$((current_index - START_INDEX))
    progress=$((completed * 100 / TOTAL_PROBLEMS))
    echo "Progress: $completed/$TOTAL_PROBLEMS ($progress%)"
    
    # Check if we're running out of time (leave 10 minutes for cleanup)
    if [ -n "$SLURM_JOB_END_TIME" ]; then
        time_left=$((SLURM_JOB_END_TIME - $(date +%s)))
        if [ $time_left -lt 600 ]; then
            echo "Less than 10 minutes remaining, stopping benchmark"
            break
        fi
    fi
done

# Stop monitoring
echo ""
echo "Stopping monitors..."
kill $GPU_MONITOR_PID 2>/dev/null || true
kill $CPU_MONITOR_PID 2>/dev/null || true

# Copy results back if using LOCAL_SCRATCH
if [ "$PWD" = "$LOCAL_SCRATCH" ]; then
    echo "Copying results back to project directory..."
    cp -r *.json $PROJECT_DIR/data/results/ 2>/dev/null || true
fi

# Final summary
echo ""
echo "================================="
echo "Benchmark Job Complete"
echo "================================="
echo "End time: $(date)"
echo "Results saved to: $PROJECT_DIR/data/results/"
echo "GPU metrics: $PROJECT_DIR/data/logs/gpu_monitor_${SLURM_JOB_ID}.csv"
echo "CPU metrics: $PROJECT_DIR/data/logs/cpu_monitor_${SLURM_JOB_ID}.csv"

# Print any failures
if [ -f "$PROJECT_DIR/data/logs/failures_${SLURM_JOB_ID}.log" ]; then
    echo ""
    echo "Failed batches:"
    cat $PROJECT_DIR/data/logs/failures_${SLURM_JOB_ID}.log
fi