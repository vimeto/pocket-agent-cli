#!/bin/bash
# Setup script for Mahti environment

# Set project directory (update with your actual project)
export PROJECT_DIR=/projappl/$PROJECT/$USER/pocket-agent-cli

# Create necessary directories
mkdir -p $PROJECT_DIR
mkdir -p $PROJECT_DIR/models
mkdir -p $PROJECT_DIR/results
mkdir -p $PROJECT_DIR/logs

# Load necessary modules
module purge
module load gcc/11.3.0
module load cuda/12.2.0

# Setup Python environment using Tykky (recommended for CSC)
export TYKKY_ENV=$PROJECT_DIR/tykky-env

# Check if Tykky environment exists
if [ ! -d "$TYKKY_ENV" ]; then
    echo "Creating Tykky environment..."
    
    # Create requirements file for Tykky
    cat > $PROJECT_DIR/requirements.txt << EOF
llama-cpp-python>=0.2.90
click>=8.1.7
rich>=13.7.1
httpx>=0.27.0
psutil>=6.0.0
docker>=7.1.0
jinja2>=3.1.4
pydantic>=2.8.2
tqdm>=4.66.5
aiofiles>=24.1.0
huggingface-hub>=0.24.5
line-profiler>=5.0.0
EOF
    
    # Create Tykky containerized environment
    pip-containerize new \
        --prefix $TYKKY_ENV \
        $PROJECT_DIR/requirements.txt
fi

# Add Tykky environment to PATH
export PATH="$TYKKY_ENV/bin:$PATH"

# Install llama-cpp-python with CUDA support
export CMAKE_ARGS="-DLLAMA_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80"
export FORCE_CMAKE=1

# Reinstall llama-cpp-python with CUDA if not already done
if ! python -c "import llama_cpp" 2>/dev/null; then
    pip install --force-reinstall llama-cpp-python
fi

echo "Environment setup complete!"
echo "Python: $(which python)"
echo "CUDA: $(nvcc --version | grep release)"