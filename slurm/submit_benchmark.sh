#!/bin/bash
# Helper script to submit benchmark jobs with custom parameters

# Default values
MODEL="llama-3.2-3b-instruct"
MODE="all"
START=0
TOTAL=509
BATCH=10
SAMPLES=10
TIME="24:00:00"
PARTITION="gpusmall"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --start)
            START="$2"
            shift 2
            ;;
        --total)
            TOTAL="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model MODEL       Model name (default: llama-3.2-3b-instruct)"
            echo "  --mode MODE         Benchmark mode: base, tool_submission, full_tool, all (default: all)"
            echo "  --start INDEX       Starting problem index (default: 0)"
            echo "  --total COUNT       Total number of problems (default: 509)"
            echo "  --batch SIZE        Batch size (default: 10)"
            echo "  --samples NUM       Number of samples per problem (default: 10)"
            echo "  --time TIME         Wall time limit (default: 24:00:00)"
            echo "  --partition PART    SLURM partition (default: gpusmall)"
            echo ""
            echo "Examples:"
            echo "  # Run first 100 problems"
            echo "  $0 --total 100"
            echo ""
            echo "  # Run problems 100-200 with batch size 5"
            echo "  $0 --start 100 --total 100 --batch 5"
            echo ""
            echo "  # Quick test run"
            echo "  $0 --total 10 --samples 3 --time 1:00:00 --partition gputest"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if model exists
export PROJECT=2013932
MODEL_PATH="/projappl/project_$PROJECT/$USER/pocket-agent-cli/data/models"
if [ ! -d "$MODEL_PATH" ]; then
    echo "Models directory not found. Have you downloaded any models?"
    echo "Run: pocket-agent model download $MODEL"
    exit 1
fi

# Create logs directory if it doesn't exist
LOGS_DIR="/projappl/project_$PROJECT/$USER/pocket-agent-cli/data/logs"
mkdir -p "$LOGS_DIR"

# Generate job name
JOB_NAME="bench_${MODEL}_${MODE}_s${START}_t${TOTAL}"

# Submit the job
echo "================================="
echo "Submitting Benchmark Job"
echo "================================="
echo "Model: $MODEL"
echo "Mode: $MODE"
echo "Problems: $START to $((START + TOTAL - 1))"
echo "Batch size: $BATCH"
echo "Samples: $SAMPLES"
echo "Time limit: $TIME"
echo "Partition: $PARTITION"
echo "================================="

# Create a temporary submission script with modified parameters
TEMP_SCRIPT=$(mktemp)
cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --account=project_2013932
#SBATCH --partition=$PARTITION
#SBATCH --time=$TIME
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gres=gpu:a100:1,nvme:200
#SBATCH --output=$LOGS_DIR/benchmark_%j.out
#SBATCH --error=$LOGS_DIR/benchmark_%j.err

# Run the benchmark script with parameters
bash /projappl/project_2013932/\$USER/pocket-agent-cli/slurm/run_benchmark_batch.sh "$MODEL" "$MODE" "$START" "$TOTAL" "$BATCH" "$SAMPLES"
EOF

# Submit the job
sbatch "$TEMP_SCRIPT"
SUBMIT_STATUS=$?

# Clean up temp file
rm -f "$TEMP_SCRIPT"

if [ $SUBMIT_STATUS -eq 0 ]; then
    echo ""
    echo "Job submitted successfully!"
    echo "Check status with: squeue -u $USER"
    echo "View output with: tail -f $LOGS_DIR/benchmark_*.out"
else
    echo "Job submission failed!"
    exit 1
fi