# Pocket Agent CLI

A command-line tool for local LLM inference and benchmarking with comprehensive performance monitoring. This tool provides the desktop equivalent of the mobile chat app, featuring the same models, prompts, and evaluation metrics.

## Features

- **Local LLM Inference**: Run quantized GGUF models using llama.cpp
- **Tool Calling**: LLMs can execute Python code and manage files in a sandboxed environment
- **Performance Monitoring**: Real-time tracking of CPU, memory, and system metrics
- **Benchmark Evaluation**: MBPP dataset testing with multiple evaluation modes
- **Model Management**: Download, list, and manage multiple models
- **Docker Sandboxing**: Secure code execution in isolated containers

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd pocket-agent-cli

# Install with uv
uv pip install -e .

# Or install with pip
pip install -e .
```

## Quick Start

### 1. List Available Models

```bash
pocket-agent model list
```

### 2. Download a Model

```bash
pocket-agent model download llama-3.2-3b-instruct

# For gated models, provide HF token
export HF_TOKEN=your_token_here
pocket-agent model download llama-3.2-3b-instruct
```

### 3. Interactive Chat

```bash
# Basic chat
pocket-agent chat --model llama-3.2-3b-instruct

# Chat with tool usage
pocket-agent chat --model llama-3.2-3b-instruct --tools
```

### 4. Run Benchmarks

```bash
# Run base benchmark (simple code generation)
pocket-agent benchmark --model llama-3.2-3b-instruct --mode base

# Run with tools
pocket-agent benchmark --model llama-3.2-3b-instruct --mode full_tool

# Run specific problems
pocket-agent benchmark --model llama-3.2-3b-instruct --problems 1,2,3

# Save results
pocket-agent benchmark --model llama-3.2-3b-instruct --output results.json
```

## Benchmark Modes

1. **Base Mode**: Simple code generation without tools
2. **Tool Submission Mode**: Reasoning with code submission tool
3. **Full Tool Mode**: Complete development environment with all tools

## Available Tools

- `run_python_code`: Execute Python code directly
- `run_python_file`: Execute a Python file
- `upsert_file`: Create or update files
- `delete_file`: Remove files
- `list_files`: List all files in sandbox
- `read_file`: Read file contents

## Performance Metrics

The tool tracks and reports:
- **TTFT** (Time to First Token): Latency before generation starts
- **TPS** (Tokens Per Second): Generation speed
- **CPU Usage**: Processor utilization
- **Memory Usage**: RAM consumption
- **Temperature**: System temperature (when available)
- **Power Consumption**: Estimated power usage

## Output Formats

Benchmark results can be exported in multiple formats:
- **JSON**: Complete structured data
- **CSV**: Tabular problem results
- **Markdown**: Human-readable report

## Docker Setup

For secure code execution, install Docker:

```bash
# macOS
brew install docker

# Linux
sudo apt-get install docker.io
sudo usermod -aG docker $USER
```

## Configuration

Models and data are stored in `~/.pocket-agent-cli/`:
- `models/`: Downloaded GGUF files
- `data/`: Benchmark datasets
- `sandbox/`: Temporary execution environments
- `results/`: Saved benchmark results

## Requirements

- Python 3.11+
- Docker (optional, for sandboxed execution)
- 4-8GB RAM (depending on model size)

## License

MIT