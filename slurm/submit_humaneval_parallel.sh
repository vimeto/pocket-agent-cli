#!/bin/bash
# Submit parallel HumanEval benchmark using SLURM job arrays
# This is the fastest way to run HumanEval benchmarks on CSC Mahti

# Default values
MODEL="llama-3.2-3b-instruct"
MODE="base"
VERSION="Q4_K_M"
CONTEXT=8192
SAMPLES=10
TIME="02:00:00"
PARTITION="gpusmall"
PROBLEMS_PER_JOB=20  # Each job processes 20 problems

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
        --version)
            VERSION="$2"
            shift 2
            ;;
        --context)
            CONTEXT="$2"
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
        --problems-per-job)
            PROBLEMS_PER_JOB="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Submit parallel HumanEval benchmarks using SLURM job arrays."
            echo "This runs all 164 HumanEval problems in parallel across multiple nodes."
            echo ""
            echo "Options:"
            echo "  --model MODEL           Model name (default: llama-3.2-3b-instruct)"
            echo "  --mode MODE             Benchmark mode: base, tool_submission, full_tool (default: base)"
            echo "  --version VERSION       Model version: Q4_K_M, F16, BF16, etc. (default: Q4_K_M)"
            echo "  --context LENGTH        Context length for model (default: 8192)"
            echo "  --samples NUM           Number of samples per problem (default: 10)"
            echo "  --time TIME             Wall time limit per job (default: 02:00:00)"
            echo "  --partition PART        SLURM partition (default: gpusmall)"
            echo "  --problems-per-job N    Problems per array job (default: 20)"
            echo ""
            echo "Examples:"
            echo "  # Run all HumanEval with default settings (9 parallel jobs)"
            echo "  $0"
            echo ""
            echo "  # Run with F16 model version"
            echo "  $0 --model gemma-3n-e2b-it --version F16"
            echo ""
            echo "  # Run with more samples per problem"
            echo "  $0 --samples 20 --time 04:00:00"
            echo ""
            echo "  # Quick test with fewer problems per job"
            echo "  $0 --problems-per-job 10 --partition gputest --time 00:30:00"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# HumanEval has 164 problems
TOTAL_PROBLEMS=164

# Calculate number of array jobs needed
NUM_JOBS=$(( (TOTAL_PROBLEMS + PROBLEMS_PER_JOB - 1) / PROBLEMS_PER_JOB ))

# Set up paths
export PROJECT=2013932
PROJECT_DIR="/projappl/project_$PROJECT/$USER/pocket-agent-cli"
LOGS_DIR="$PROJECT_DIR/data/logs"
mkdir -p "$LOGS_DIR"

# Generate unique run ID
RUN_ID=$(date +%Y%m%d_%H%M%S)
JOB_NAME="humaneval_${MODEL}_${VERSION}_${MODE}_${RUN_ID}"

echo "================================="
echo "Submitting Parallel HumanEval Benchmark"
echo "================================="
echo "Model: $MODEL"
echo "Version: $VERSION"
echo "Mode: $MODE"
echo "Context Length: $CONTEXT"
echo "Samples per problem: $SAMPLES"
echo "Total problems: $TOTAL_PROBLEMS"
echo "Problems per job: $PROBLEMS_PER_JOB"
echo "Number of parallel jobs: $NUM_JOBS"
echo "Time limit per job: $TIME"
echo "Partition: $PARTITION"
echo "Run ID: $RUN_ID"
echo "================================="

# Create the array job script
TEMP_SCRIPT=$(mktemp)
cat > "$TEMP_SCRIPT" << 'OUTER_EOF'
#!/bin/bash
#SBATCH --job-name=JOB_NAME_PLACEHOLDER
#SBATCH --account=project_2013932
#SBATCH --partition=PARTITION_PLACEHOLDER
#SBATCH --time=TIME_PLACEHOLDER
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gres=gpu:a100:1,nvme:200
#SBATCH --array=0-ARRAY_MAX_PLACEHOLDER
#SBATCH --output=LOGS_DIR_PLACEHOLDER/humaneval_%A_%a.out
#SBATCH --error=LOGS_DIR_PLACEHOLDER/humaneval_%A_%a.err

# Configuration from submission
MODEL="MODEL_PLACEHOLDER"
MODE="MODE_PLACEHOLDER"
VERSION="VERSION_PLACEHOLDER"
CONTEXT="CONTEXT_PLACEHOLDER"
SAMPLES="SAMPLES_PLACEHOLDER"
PROBLEMS_PER_JOB="PROBLEMS_PER_JOB_PLACEHOLDER"
RUN_ID="RUN_ID_PLACEHOLDER"

# Calculate problem range for this array task
START_INDEX=$((SLURM_ARRAY_TASK_ID * PROBLEMS_PER_JOB))
END_INDEX=$((START_INDEX + PROBLEMS_PER_JOB - 1))

# Cap at 163 (HumanEval has problems 0-163)
if [ $END_INDEX -gt 163 ]; then
    END_INDEX=163
fi

TOTAL=$((END_INDEX - START_INDEX + 1))

echo "================================="
echo "HumanEval Array Job Task"
echo "================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Model: $MODEL"
echo "Version: $VERSION"
echo "Mode: $MODE"
echo "Problems: $START_INDEX to $END_INDEX ($TOTAL problems)"
echo "================================="

# Set up environment
export PROJECT=2013932
PROJECT_DIR="/projappl/project_$PROJECT/$USER/pocket-agent-cli"

# Activate environment
source $PROJECT_DIR/slurm/activate_env.sh

# Set work directory
cd $PROJECT_DIR

# Use LOCAL_SCRATCH if available
if [ -d "$LOCAL_SCRATCH" ]; then
    echo "Using LOCAL_SCRATCH: $LOCAL_SCRATCH"
    export TMPDIR=$LOCAL_SCRATCH
    export TEMP=$LOCAL_SCRATCH
    cp -r pocket_agent_cli $LOCAL_SCRATCH/
    cd $LOCAL_SCRATCH
fi

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
export CUDA_VISIBLE_DEVICES=0

# Disable Docker (not available on Mahti)
export DISABLE_DOCKER=1

# Build problem IDs list
problem_ids=""
for ((i=START_INDEX; i<=END_INDEX; i++)); do
    if [ -z "$problem_ids" ]; then
        problem_ids="$i"
    else
        problem_ids="${problem_ids},$i"
    fi
done

# Output directory
OUTPUT_DIR="$PROJECT_DIR/data/results/humaneval_${MODEL}_${VERSION}_${MODE}_${RUN_ID}/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Running benchmark for problems: $problem_ids"
echo "Output: $OUTPUT_DIR"

# Run benchmark
if pocket-agent benchmark \
    --model "$MODEL" \
    --model-version "$VERSION" \
    --dataset humaneval \
    --mode "$MODE" \
    --context-length "$CONTEXT" \
    --problems "$problem_ids" \
    --num-samples "$SAMPLES" \
    --output-dir "$OUTPUT_DIR"; then
    echo "✓ Task completed successfully"
else
    echo "⚠ Task failed"
    echo "$(date),humaneval,${MODE},${problem_ids},FAILED" >> $PROJECT_DIR/data/logs/failures_${RUN_ID}.log
fi

echo ""
echo "Task complete at $(date)"
OUTER_EOF

# Replace placeholders in the script
sed -i "s/JOB_NAME_PLACEHOLDER/$JOB_NAME/g" "$TEMP_SCRIPT"
sed -i "s/PARTITION_PLACEHOLDER/$PARTITION/g" "$TEMP_SCRIPT"
sed -i "s/TIME_PLACEHOLDER/$TIME/g" "$TEMP_SCRIPT"
sed -i "s/ARRAY_MAX_PLACEHOLDER/$((NUM_JOBS - 1))/g" "$TEMP_SCRIPT"
sed -i "s|LOGS_DIR_PLACEHOLDER|$LOGS_DIR|g" "$TEMP_SCRIPT"
sed -i "s/MODEL_PLACEHOLDER/$MODEL/g" "$TEMP_SCRIPT"
sed -i "s/MODE_PLACEHOLDER/$MODE/g" "$TEMP_SCRIPT"
sed -i "s/VERSION_PLACEHOLDER/$VERSION/g" "$TEMP_SCRIPT"
sed -i "s/CONTEXT_PLACEHOLDER/$CONTEXT/g" "$TEMP_SCRIPT"
sed -i "s/SAMPLES_PLACEHOLDER/$SAMPLES/g" "$TEMP_SCRIPT"
sed -i "s/PROBLEMS_PER_JOB_PLACEHOLDER/$PROBLEMS_PER_JOB/g" "$TEMP_SCRIPT"
sed -i "s/RUN_ID_PLACEHOLDER/$RUN_ID/g" "$TEMP_SCRIPT"

# Submit the array job
echo ""
echo "Submitting array job..."
sbatch "$TEMP_SCRIPT"
SUBMIT_STATUS=$?

# Clean up
rm -f "$TEMP_SCRIPT"

if [ $SUBMIT_STATUS -eq 0 ]; then
    echo ""
    echo "================================="
    echo "Job Array Submitted Successfully!"
    echo "================================="
    echo "Run ID: $RUN_ID"
    echo "Jobs submitted: $NUM_JOBS"
    echo "Results will be in: $PROJECT_DIR/data/results/humaneval_${MODEL}_${VERSION}_${MODE}_${RUN_ID}/"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u $USER"
    echo ""
    echo "View logs:"
    echo "  tail -f $LOGS_DIR/humaneval_*.out"
    echo ""
    echo "After completion, aggregate results with:"
    echo "  ls $PROJECT_DIR/data/results/humaneval_${MODEL}_${VERSION}_${MODE}_${RUN_ID}/*/benchmark_summary.json"
else
    echo "Job submission failed!"
    exit 1
fi
