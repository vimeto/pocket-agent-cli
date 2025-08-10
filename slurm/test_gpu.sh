#!/bin/bash
#SBATCH --job-name=pocket-agent-gpu-test
#SBATCH --account=project_<YOUR_PROJECT_ID>
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/gpu_test_%j.out
#SBATCH --error=logs/gpu_test_%j.err

# Quick GPU test script for Mahti

# Load environment
source /projappl/$PROJECT/$USER/pocket-agent-cli/slurm/setup_environment.sh

# Set work directory
cd /projappl/$PROJECT/$USER/pocket-agent-cli

# Test GPU availability
echo "=== GPU Information ==="
nvidia-smi

echo -e "\n=== CUDA Information ==="
nvcc --version

echo -e "\n=== Python Environment ==="
which python
python --version

echo -e "\n=== Testing llama-cpp-python CUDA support ==="
python -c "
import llama_cpp
import os

print('llama-cpp-python version:', llama_cpp.__version__)
print('CUDA available in environment:', 'CUDA_HOME' in os.environ)
print('CUDA_HOME:', os.environ.get('CUDA_HOME', 'Not set'))

# Try to load a small test
try:
    # This will fail if no model is available, but shows CUDA init
    from llama_cpp import Llama
    print('Llama class imported successfully')
except Exception as e:
    print(f'Import test: {e}')
"

echo -e "\n=== Testing GPU memory allocation ==="
python -c "
import subprocess
import json

# Get GPU memory info
result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free,memory.used', '--format=csv,nounits,noheader'], 
                       capture_output=True, text=True)
if result.returncode == 0:
    values = result.stdout.strip().split(', ')
    print(f'GPU Memory - Total: {values[0]} MB, Free: {values[1]} MB, Used: {values[2]} MB')
"

echo -e "\n=== Quick inference test (if model available) ==="
# This will only work if a model is already downloaded
MODEL_PATH="/projappl/$PROJECT/$USER/pocket-agent-cli/models"
if [ -d "$MODEL_PATH" ] && [ "$(ls -A $MODEL_PATH)" ]; then
    echo "Found models in $MODEL_PATH"
    ls -la $MODEL_PATH
    
    # Try a quick inference
    python -c "
from pocket_agent_cli.services.inference_service import InferenceService
from pocket_agent_cli.config import Model, InferenceConfig
from pathlib import Path

# This is just a test - adjust model path as needed
print('Attempting to initialize inference service...')
service = InferenceService()
print('Inference service initialized')
"
else
    echo "No models found in $MODEL_PATH - skipping inference test"
fi

echo -e "\n=== Test completed successfully ==="