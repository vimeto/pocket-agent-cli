#!/bin/bash
#SBATCH --job-name=pocket-agent-benchmark
#SBATCH --account=project_<YOUR_PROJECT_ID>
#SBATCH --partition=gpusmall
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

# Load environment
source /projappl/$PROJECT/$USER/pocket-agent-cli/slurm/setup_environment.sh

# Set work directory
cd /projappl/$PROJECT/$USER/pocket-agent-cli

# Copy code to local fast storage
cp -r pocket_agent_cli $LOCAL_SCRATCH/
cd $LOCAL_SCRATCH

# Download model if needed (adjust model name as needed)
MODEL_NAME="llama-3.2-3b-instruct"
MODEL_PATH="$PROJECT_DIR/models"

# Check GPU availability
nvidia-smi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run benchmark with GPU acceleration
echo "Starting benchmark with model: $MODEL_NAME"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Run different benchmark modes
echo "Running base benchmark..."
pocket-agent benchmark \
    --model $MODEL_NAME \
    --mode base \
    --problems 1-10 \
    --output $PROJECT_DIR/results/benchmark_base_${SLURM_JOB_ID}.json

echo "Running tool submission benchmark..."
pocket-agent benchmark \
    --model $MODEL_NAME \
    --mode tool_submission \
    --problems 1-10 \
    --output $PROJECT_DIR/results/benchmark_tool_${SLURM_JOB_ID}.json

echo "Running full tool benchmark..."
pocket-agent benchmark \
    --model $MODEL_NAME \
    --mode full_tool \
    --problems 1-10 \
    --output $PROJECT_DIR/results/benchmark_full_${SLURM_JOB_ID}.json

# Monitor GPU usage during execution
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw --format=csv -l 5 > $PROJECT_DIR/logs/gpu_monitor_${SLURM_JOB_ID}.csv &
MONITOR_PID=$!

# Wait for benchmarks to complete
wait

# Stop GPU monitoring
kill $MONITOR_PID

echo "Benchmark completed!"
echo "Results saved to $PROJECT_DIR/results/"