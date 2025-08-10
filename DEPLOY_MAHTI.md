# Deployment Guide for Mahti Supercomputer

This guide provides step-by-step instructions for deploying and running the Pocket Agent CLI on the Mahti supercomputer with GPU support.

## Prerequisites

1. **Mahti Account**: Active CSC account with access to Mahti
2. **Project Allocation**: GPU billing units allocated to your project
3. **Storage**: Sufficient quota in `/projappl` and `/scratch`

## Initial Setup

### 1. Connect to Mahti

```bash
ssh <username>@mahti.csc.fi
```

### 2. Set Project Environment

```bash
export PROJECT=XXX  # Project number
```

### 3. Clone Repository

```bash
cd /projappl/project_$PROJECT/$USER
git clone <repository-url> pocket-agent-cli
cd pocket-agent-cli
```

### 4. Start Interactive Session (REQUIRED)

**Important:** You MUST use an interactive session for the initial setup. The login node has limited resources and some modules are not available.

```bash
# Request an interactive session for setup
srun --account=project_$PROJECT --partition=test --time=1:00:00 --mem=12000 --pty bash

# You should see your prompt change from login node to compute node:
# [username@mahti-login11 ~]$  -->  [username@c1101 ~]$

# Verify you're on compute node (should show c* or g* node):
hostname -s
```

### 5. Setup Environment

Once on a compute node (prompt shows c* or g* node like c1101), run the setup script:

```bash
# Verify you're on compute node (should show c* or g* like c1101):
hostname -s

# Make sure PROJECT is still set
export PROJECT=2013932  # Your project number

# Run setup
chmod +x slurm/setup_environment.sh
source slurm/setup_environment.sh
```

**Note:** The first run will take 5-10 minutes to create the Tykky environment.

**Common Issues:**
- If Tykky environment creation fails, the script will automatically clean up and retry
- llama-cpp-python is installed separately after the base environment
- CPU version is installed on regular nodes, GPU version on GPU nodes (g* nodes)
- If you see version errors, the script handles this by excluding problematic packages

This script will:
- Create necessary directories
- Load required modules (will auto-detect versions)
- Set up a Tykky containerized Python environment (required on Mahti)
- Install dependencies with CUDA support (when on compute nodes)

## Running Tests

### Quick GPU Test

First, update the SLURM script with your project ID:

```bash
# Edit the test script
nano slurm/test_gpu.sh

# Replace this line:
#SBATCH --account=project_<YOUR_PROJECT_ID>
# With your project number:
#SBATCH --account=project_2013932  # Or your project number
```

Then submit the test job:

```bash
# Submit test job
sbatch slurm/test_gpu.sh

# Monitor job status
squeue -u $USER

# Check output
tail -f logs/gpu_test_*.out
```

### Running Python GPU Tests

```bash
# Interactive session for testing (15 minutes)
sinteractive --partition gputest --gres=gpu:a100:1 --time 00:15:00

# Load environment
source slurm/setup_environment.sh

# Run GPU tests
python tests/test_gpu_inference.py
```

## Running Benchmarks

### 1. Download Models

First, download the required models:

```bash
# Set HuggingFace token if needed (for gated models)
export HF_TOKEN=your_token_here

# Download model
pocket-agent model download llama-3.2-3b-instruct
```

### 2. Submit Benchmark Job

Edit the SLURM script with your project ID:

```bash
# Edit the script
nano slurm/run_benchmark.sh

# Replace project_<YOUR_PROJECT_ID> with your actual project
# Adjust model name if using a different model
```

Submit the job:

```bash
sbatch slurm/run_benchmark.sh

# Monitor progress
squeue -u $USER
tail -f logs/benchmark_*.out
```

### 3. Monitor GPU Usage

The benchmark script automatically logs GPU metrics to:
```
$PROJECT_DIR/logs/gpu_monitor_${SLURM_JOB_ID}.csv
```

## SLURM Job Configuration

### GPU Partitions

| Partition | GPUs | Max Time | Use Case |
|-----------|------|----------|----------|
| gputest | 1 A100 | 15 min | Testing and debugging |
| gpusmall | 1-2 A100 | 72 hours | Single GPU jobs |
| gpumedium | 3+ A100 | 72 hours | Multi-GPU jobs |

### Resource Recommendations

For optimal performance:

```bash
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32  # 32 CPUs per GPU
#SBATCH --mem=120G          # 120GB RAM per GPU
```

### GPU Slices (Development)

For development with smaller resource requirements:

```bash
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100_1g.5gb:1  # 1/7th of A100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
```

## Environment Variables

Key environment variables for GPU execution:

```bash
# CUDA configuration
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/appl/spack/v20/install-tree/gcc-11.3.0/cuda-12.2.0

# llama-cpp-python CUDA compilation
export CMAKE_ARGS="-DLLAMA_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80"
export FORCE_CMAKE=1

# Performance tuning
export OMP_NUM_THREADS=32  # Match CPU count
```

## Troubleshooting

### 1. Module Not Found Errors

If you see errors like "The following module(s) are unknown: gcc/11.3.0":

```bash
# Check available versions
module spider gcc
module spider cuda

# Load available versions (example)
module load gcc/10.4.0  # Or whatever version is shown
module load cuda  # Load default version
```

### 2. pip-containerize: command not found

This means Tykky module is not loaded:

```bash
# Load Tykky module
module load tykky

# Verify it's available
pip-containerize --help
```

**Note:** This MUST be done on a compute node, not the login node.

### 3. CUDA Not Detected

```bash
# CUDA is only available on GPU compute nodes
# Request a GPU node
sinteractive --partition gputest --gres=gpu:a100:1 --time 00:15:00

# Then check CUDA
module list
nvcc --version
nvidia-smi
```

### 4. Running on Login Node

If you accidentally run setup on the login node:
- You'll see warnings about modules and commands not found
- Solution: Use `sinteractive` to get a compute node first
- Login nodes are only for light tasks, not installations

### 5. Model Loading Issues

```bash
# Check model path
ls -la /projappl/project_$PROJECT/$USER/pocket-agent-cli/models/

# Test with smaller model if memory issues
pocket-agent model download tinyllama-1.1b
```

### 6. Out of Memory

- Use GPU slices for testing
- Reduce batch size in configuration
- Use quantized models (Q4_K_M or smaller)

### 4. Slow Performance

```bash
# Check GPU utilization
nvidia-smi -l 1

# Ensure model is using GPU layers
export LLAMA_CUDA_FORCE_LAYERS=-1  # Use all layers on GPU
```

## Performance Monitoring

### Real-time GPU Monitoring

```bash
# In interactive session
watch -n 1 nvidia-smi

# Or log to file
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw --format=csv -l 1 > gpu_log.csv
```

### Analyzing Results

Benchmark results are saved in JSON format:

```bash
# View results
cat $PROJECT_DIR/results/benchmark_*.json | jq '.'

# Extract key metrics
jq '.summary.performance_metrics' $PROJECT_DIR/results/benchmark_*.json
```

## Best Practices

1. **Use Fast Local Storage**: Copy models to `$LOCAL_SCRATCH` for better I/O
2. **Batch Jobs**: Submit multiple experiments as array jobs
3. **Resource Efficiency**: Use GPU slices for development
4. **Module Management**: Always purge before loading modules
5. **Tykky Containers**: Use for complex Python environments

## Local Docker Testing

For local testing with CUDA (requires NVIDIA Docker):

```bash
# Build CUDA image
docker build -f Dockerfile.cuda -t pocket-agent-cuda .

# Run with GPU support
docker run --gpus all pocket-agent-cuda chat --model llama-3.2-3b-instruct

# Or with docker-compose
docker-compose -f docker-compose.cuda.yml up
```

## Support

- CSC Service Desk: servicedesk@csc.fi
- Mahti Documentation: https://docs.csc.fi/computing/systems-mahti/
- GPU Guide: https://docs.csc.fi/support/tutorials/gpu-ml/

## Key Files

- `slurm/setup_environment.sh` - Environment setup script
- `slurm/run_benchmark.sh` - Benchmark SLURM job
- `slurm/test_gpu.sh` - GPU testing job
- `tests/test_gpu_inference.py` - Python GPU tests
- `requirements-cuda.txt` - CUDA-specific dependencies
- `Dockerfile.cuda` - Docker image for CUDA testing
