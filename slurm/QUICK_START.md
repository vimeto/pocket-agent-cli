# Quick Start Commands for Mahti

## Initial Setup (One Time)

```bash
# 1. Connect to Mahti
ssh username@mahti.csc.fi

# 2. Set project
export PROJECT=2013932

# 3. Get GPU node with local storage
srun --account=project_$PROJECT --partition=gputest --gres=gpu:a100:1,nvme:100 --time=0:15:00 --pty bash

# 4. Re-export project (needed after srun)
export PROJECT=2013932

# 5. Navigate to project directory
cd /projappl/project_$PROJECT/$USER/pocket-agent-cli

# 6. Run setup
source slurm/setup_environment.sh

# 7. Download a model
pocket-agent model download gemma-3n-e2b-it
```

## Daily Use

```bash
# 1. Connect and set project
ssh username@mahti.csc.fi
export PROJECT=2013932

# 2. Get compute node (choose one):

# Option A: GPU node for benchmarks
srun --account=project_$PROJECT --partition=gputest --gres=gpu:a100:1 --time=0:15:00 --pty bash

# Option B: CPU node for setup/testing
srun --account=project_$PROJECT --partition=test --time=1:00:00 --mem=16000 --pty bash

# 3. Activate environment
export PROJECT=2013932
export DISABLE_DOCKER=1
source /projappl/project_$PROJECT/$USER/pocket-agent-cli/slurm/activate_env.sh

# 4. Run benchmarks (choose one):

# Safe runner (recommended if bus errors occur):
bash $PROJECT_DIR/slurm/run_benchmark_safe.sh gemma-3n-e2b-it base 1-10 5

# Direct Python module (bypasses wrappers):
python -m pocket_agent_cli.cli benchmark \
    --model gemma-3n-e2b-it \
    --mode base \
    --problems 1-10 \
    --num-samples 5 \
    --output-dir ./results

# Standard command (if no errors):
pocket-agent benchmark \
    --model gemma-3n-e2b-it \
    --mode base \
    --problems 1-10 \
    --num-samples 5 \
    --output-dir ./results
```

## Batch Job Submission

```bash
# Submit benchmark job
cd /projappl/project_$PROJECT/$USER/pocket-agent-cli
sbatch slurm/run_benchmark_batch.sh gemma-3n-e2b-it base 0 100 10 5

# Or use the helper script
./slurm/submit_benchmark.sh --model gemma-3n-e2b-it --total 100 --batch 10

# Check job status
squeue -u $USER

# View output
tail -f data/logs/benchmark_*.out
```

## Troubleshooting

```bash
# Test installation
bash slurm/test_installation.sh

# Diagnose CUDA issues
bash slurm/diagnose_cuda.sh

# Fix CUDA on GPU node
source slurm/fix_cuda.sh

# If bus errors occur, always use:
export DISABLE_DOCKER=1
python -m pocket_agent_cli.cli benchmark ...
```

## Important Notes

1. **Always set `DISABLE_DOCKER=1`** - Docker is not available on Mahti
2. **Use `--output-dir` not `--output`** for benchmark command
3. **Request nvme storage** for GPU nodes: `--gres=gpu:a100:1,nvme:100`
4. **If bus errors occur**, use direct Python module: `python -m pocket_agent_cli.cli`
5. **CUDA may need manual loading** on some nodes: `module load cuda/12.0.0`