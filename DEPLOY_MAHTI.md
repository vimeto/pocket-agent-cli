# Deployment Guide for Mahti Supercomputer

This guide provides complete instructions for deploying the Pocket Agent CLI on CSC's Mahti supercomputer with GPU support.

## Prerequisites

1. **Mahti Account**: Active CSC account with access to Mahti
2. **Project Allocation**: GPU billing units allocated to your project
3. **Storage**: At least 10GB quota in `/projappl`

## Complete Setup Workflow

### Step 1: Connect to Mahti

```bash
ssh <username>@mahti.csc.fi
```

### Step 2: Set Project Environment

```bash
# Set your project number (use your actual project number)
export PROJECT=2013932
```

### Step 3: Clone Repository

```bash
# Navigate to project directory
cd /projappl/project_$PROJECT/$USER

# Clone the repository
git clone <repository-url> pocket-agent-cli
cd pocket-agent-cli
```

### Step 4: Get Interactive Compute Node

**IMPORTANT**: Setup must be done on a compute node, not the login node.

```bash
# Request interactive session with guaranteed local storage

# Option 1: Request node with local NVMe storage (RECOMMENDED)
srun --account=project_$PROJECT --partition=small --time=2:00:00 --mem=16000 --gres=nvme:100 --pty bash

# Option 2: Use GPU test partition WITH explicit NVMe request
srun --account=project_$PROJECT --partition=gputest --gres=gpu:a100:1,nvme:100 --time=0:15:00 --mem=32000 --pty bash

# Option 3: Use GPU small partition with NVMe for longer sessions
srun --account=project_$PROJECT --partition=gpusmall --gres=gpu:a100:1,nvme:100 --time=2:00:00 --mem=32000 --pty bash

# Verify you're on compute node and have local storage
hostname -s
df -h $LOCAL_SCRATCH  # Should show available space

# Re-export PROJECT variable (needed after srun)
export PROJECT=2013932
```

### Step 5: Run Setup Script

```bash
# Navigate to the repository
cd /projappl/project_$PROJECT/$USER/pocket-agent-cli

# Make script executable
chmod +x slurm/setup_environment.sh

# Run setup (first run takes 5-10 minutes)
source slurm/setup_environment.sh
```

The setup script will:
1. Load required modules (gcc, tykky)
2. Create a containerized Python environment (required by CSC)
3. Install all dependencies
4. Install llama-cpp-python (CPU or GPU version based on node type)
5. Install pocket-agent-cli

### Step 6: Verify Installation

After setup completes, verify everything works:

```bash
# Check pocket-agent command
pocket-agent --version

# List available models
pocket-agent model list

# Check Python packages
python -c "import pocket_agent_cli; print('✓ Package works')"
python -c "import llama_cpp; print('✓ llama-cpp works')"
```

## Using the Environment

### In Future Sessions

When you log in again, just activate the environment:

```bash
# Set project and activate
export PROJECT=2013932
source /projappl/project_$PROJECT/$USER/pocket-agent-cli/slurm/activate_env.sh

# Now pocket-agent is available
pocket-agent --help
```

### Download Models

```bash
# Activate environment
source /projappl/project_$PROJECT/$USER/pocket-agent-cli/slurm/activate_env.sh

# Set HuggingFace token if needed (for gated models)
export HF_TOKEN=your_token_here

# Download a model (models stored in data/models/)
pocket-agent model download llama-3.2-3b-instruct
```

### Running Jobs

#### Interactive Testing

```bash
# Test chat interface
pocket-agent chat --model llama-3.2-3b-instruct

# Run a quick benchmark
pocket-agent benchmark --model llama-3.2-3b-instruct --problems 1-5
```

#### Batch Jobs

**Quick submission with helper script:**

```bash
# Activate environment first
source /projappl/project_2013932/$USER/pocket-agent-cli/slurm/activate_env.sh

# Submit with defaults (all 509 problems, batch size 10)
./slurm/submit_benchmark.sh

# Run first 100 problems only
./slurm/submit_benchmark.sh --total 100

# Run problems 100-200 with custom batch size
./slurm/submit_benchmark.sh --start 100 --total 100 --batch 5

# Quick test (10 problems, 3 samples each, 1 hour)
./slurm/submit_benchmark.sh --total 10 --samples 3 --time 1:00:00 --partition gputest

# Run specific mode only
./slurm/submit_benchmark.sh --mode base --total 50

# Check job status
squeue -u $USER

# View output
tail -f data/logs/benchmark_*.out
```

**Manual submission for full control:**

```bash
# Edit and submit the batch script directly
sbatch slurm/run_benchmark_batch.sh [model] [mode] [start] [total] [batch] [samples]

# Example: Run problems 0-99 with gemma model
sbatch slurm/run_benchmark_batch.sh gemma-2b base 0 100 10 5
```

## GPU Support

### For GPU Nodes

To use GPU acceleration, request a GPU node:

```bash
# Interactive GPU session (15 minutes for testing)
srun --account=project_$PROJECT --partition=gputest --gres=gpu:a100:1 --time=0:15:00 --mem=32000 --pty bash

# Re-run setup to install GPU version of llama-cpp-python
export PROJECT=2013932
cd /projappl/project_$PROJECT/$USER/pocket-agent-cli
source slurm/setup_environment.sh
```

### SLURM Partitions

| Partition | GPUs | Max Time | Use Case |
|-----------|------|----------|----------|
| test | 0 | 1 hour | Setup & testing |
| gputest | 1 A100 | 15 min | GPU testing |
| gpusmall | 1-2 A100 | 72 hours | Production runs |

## Troubleshooting

### Bus error (Väylävirhe) when running benchmarks

**Problem**: Getting bus errors when running benchmarks through containerized Python.

**Solution 1: Use the safe benchmark runner**
```bash
# On compute node with environment activated
bash /projappl/project_$PROJECT/$USER/pocket-agent-cli/slurm/run_benchmark_safe.sh gemma-3n-e2b-it base 1-10 5
```

**Solution 2: Use direct Python runner**
```bash
# Set up environment
export DISABLE_DOCKER=1
source /projappl/project_$PROJECT/$USER/pocket-agent-cli/slurm/activate_env.sh

# Use the direct Python script
python /projappl/project_$PROJECT/$USER/pocket-agent-cli/slurm/pocket_agent_direct.py benchmark \
    --model gemma-3n-e2b-it \
    --mode base \
    --problems 1-10 \
    --num-samples 5 \
    --output-dir ./results
```

**Solution 3: Run Python module directly**
```bash
# Bypass all wrappers
python -m pocket_agent_cli.cli benchmark \
    --model gemma-3n-e2b-it \
    --mode base \
    --problems 1-10 \
    --num-samples 5 \
    --output-dir ./results
```

**Note**: Bus errors are often caused by Singularity containerization conflicts. The above methods bypass the containerized wrapper.

### "Module not found" errors

Check available modules:
```bash
module spider gcc
module spider cuda
module spider tykky
```

### Tykky environment creation fails ("No space left on device")

**Root cause**: Not all nodes have local storage. You need a node with NVMe.

**Solutions**:
1. Request node with local NVMe storage:
   ```bash
   srun --account=project_$PROJECT --partition=small --time=2:00:00 --mem=16000 --gres=nvme:100 --pty bash
   ```
   Note: Use `--gres=nvme:100` (100GB) not `--tmp` on Mahti

2. Use GPU node (always has local storage):
   ```bash
   srun --account=project_$PROJECT --partition=gputest --gres=gpu:a100:1 --time=0:15:00 --pty bash
   # OR for longer sessions:
   srun --account=project_$PROJECT --partition=gpusmall --gres=gpu:a100:1 --time=2:00:00 --mem=32000 --pty bash
   ```

3. Check if LOCAL_SCRATCH exists:
   ```bash
   df -h $LOCAL_SCRATCH
   ```
   If it shows "No such file", you're on a node without local storage.

4. Check project quota:
   ```bash
   lfs quota -hg project_$PROJECT /projappl
   ```

### Python packages not found

The environment uses containerization. Always ensure PATH is set:
```bash
export PATH=/projappl/project_$PROJECT/$USER/pocket-agent-cli/tykky-env/bin:$PATH
```

### CUDA not available

**Problem**: Getting "installing CPU version" even on GPU nodes because nvcc not found.

**Solution 1: Run the CUDA fix script**
```bash
# On a GPU node
source /projappl/project_$PROJECT/$USER/pocket-agent-cli/slurm/fix_cuda.sh
```

**Solution 2: Manual fix**
```bash
# Find CUDA installations
find /appl /opt -name nvcc 2>/dev/null

# Try loading different CUDA module versions
module load cuda/11.7.0  # or cuda/11.8.0, cuda/12.0.0, cuda/12.1.0

# Verify nvcc is available
which nvcc

# Reinstall llama-cpp-python with CUDA
pip uninstall -y llama-cpp-python
CMAKE_ARGS="-DLLAMA_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80" pip install llama-cpp-python --no-cache-dir
```

**Notes**:
- CUDA is only available on GPU nodes (g* nodes)
- Request GPU node with `--gres=gpu:a100:1`
- Different Mahti nodes may have different CUDA module names

### Docker tool calling not working

**Problem**: Tool calling features fail because Docker is not available on compute nodes.

**Solution**: The benchmark automatically detects Docker availability and falls back to local execution.

```bash
# To explicitly disable Docker (if causing issues)
export DISABLE_DOCKER=1

# Run benchmark without Docker sandboxing
pocket-agent benchmark --model llama-3.2-3b-instruct --problems 1-10
```

**Note**: Mahti compute nodes don't have Docker. Tool calling will use subprocess isolation instead.

## Why Containerization?

CSC requires containerized Python environments because:
- Python packages create thousands of small files
- This severely degrades parallel filesystem performance
- Containerization packages everything into a single image
- **Direct pip/conda installation is deprecated and will be disabled**

## Quick Reference

```bash
# Set up environment (every login)
export PROJECT=2013932
export PATH=/projappl/project_$PROJECT/$USER/pocket-agent-cli/tykky-env/bin:$PATH

# Get compute node with local storage
srun --account=project_$PROJECT --partition=small --time=1:00:00 --mem=12000 --gres=nvme:100 --pty bash

# Get GPU node
srun --account=project_$PROJECT --partition=gputest --gres=gpu:a100:1 --time=0:15:00 --mem=32000 --pty bash

# Submit batch job
sbatch slurm/run_benchmark.sh

# Check jobs
squeue -u $USER

# Check project resources
csc-projects
```

## Support

- CSC Service Desk: servicedesk@csc.fi
- Mahti Docs: https://docs.csc.fi/computing/systems-mahti/
- GPU Guide: https://docs.csc.fi/support/tutorials/gpu-ml/