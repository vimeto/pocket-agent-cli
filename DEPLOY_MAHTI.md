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
# Request interactive session on compute node
srun --account=project_$PROJECT --partition=test --time=1:00:00 --mem=12000 --pty bash

# Verify you're on compute node (should show c* or g* like c1101)
hostname -s

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

When you log in again, you don't need to recreate the environment:

```bash
# Set project
export PROJECT=2013932

# Add environment to PATH
export PATH=/projappl/project_$PROJECT/$USER/pocket-agent-cli/tykky-env/bin:$PATH

# Now pocket-agent is available
pocket-agent --help
```

### Download Models

```bash
# Set HuggingFace token if needed (for gated models)
export HF_TOKEN=your_token_here

# Download a model
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

First, update SLURM scripts with your project ID:

```bash
cd /projappl/project_$PROJECT/$USER/pocket-agent-cli
sed -i "s/project_<YOUR_PROJECT_ID>/project_$PROJECT/g" slurm/*.sh
```

Submit jobs:

```bash
# GPU test (15 minutes)
sbatch slurm/test_gpu.sh

# Full benchmark (2 hours)
sbatch slurm/run_benchmark.sh

# Check job status
squeue -u $USER

# View output
tail -f logs/*.out
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

### "Module not found" errors

Check available modules:
```bash
module spider gcc
module spider cuda
module spider tykky
```

### Tykky environment creation fails

1. Ensure you're on compute node: `hostname -s`
2. Check disk quota: `lfs quota -hg project_$PROJECT /projappl`
3. Try with debug output:
```bash
CW_LOG_LEVEL=3 pip-containerize new --prefix tykky-env requirements-container.txt
```

### Python packages not found

The environment uses containerization. Always ensure PATH is set:
```bash
export PATH=/projappl/project_$PROJECT/$USER/pocket-agent-cli/tykky-env/bin:$PATH
```

### CUDA not available

- CUDA is only available on GPU nodes (g* nodes)
- Request GPU node with `--gres=gpu:a100:1`
- Module will be loaded automatically on GPU nodes

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

# Get compute node
srun --account=project_$PROJECT --partition=test --time=1:00:00 --mem=12000 --pty bash

# Get GPU node
srun --account=project_$PROJECT --partition=gputest --gres=gpu:a100:1 --time=0:15:00 --pty bash

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