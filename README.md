# Pocket Agent CLI

A cross-platform measurement framework for quantifying end-to-end on-device LLM agent costs. Supports multiple inference engines, deployment architectures, and benchmarks.

## Inference Engines

| Engine | Platform | Models | Batching |
|--------|----------|--------|----------|
| **MLX** (`mlx-lm`) | macOS Apple Silicon | HuggingFace safetensors | Native batch via `batch_generate` |
| **SGLang** | Linux + NVIDIA GPU | HuggingFace, OpenAI API | Continuous batching + PagedAttention |
| **llama.cpp** | All (macOS, Linux, mobile) | GGUF | Single-request |

## Benchmarks

| Benchmark | Problems | Task Type | Tools Used |
|-----------|----------|-----------|------------|
| MBPP | 500 | Code generation | Python REPL, file ops |
| HumanEval | 164 | Code generation | Python REPL, file ops |
| GSM8K | 1,319 | Math reasoning | Python REPL (calculator) |

## Experiment Strategy

### Measurement Modes

We run experiments in two modes to balance accuracy and throughput:

**Mode 1: Accurate metrics (batch=1)**
- Single-problem execution for precise per-problem measurements
- Captures: TTFT, TPS, inter-token latency, energy per token, power draw
- Used for: latency characterization, energy analysis, thermal profiling

**Mode 2: Bulk evaluation (max batch/concurrency)**
- Maximum parallel execution for pass@k and accuracy measurements
- MLX: batch=100 (sweet spot on M2 Max 64GB, ~1.9 problems/s, 5.5 GB)
- SGLang: concurrency=150 (~27 problems/s on A100)
- Used for: pass@k computation, cross-model/cross-task comparisons

### Measured Throughput (Qwen3-4B)

**MLX on MacBook M2 Max (64GB):**

| Batch | Gen TPS | Prefill TPS | Memory | Problems/s |
|------:|--------:|------------:|-------:|-----------:|
| 1 | 102 | 157 | 2.4 GB | 0.25 |
| 20 | 219 | 378 | 3.9 GB | 1.4 |
| 50 | 267 | 409 | 4.9 GB | 1.8 |
| 100 | 287 | 388 | 5.5 GB | 1.9 |
| 150 | 276 | 348 | 5.5 GB | 1.9 |

**SGLang on A100 (40GB HBM2):**

| Concurrency | Wall Time (5 prob) | Problems/s |
|------------:|-----------:|-----------:|
| 1 | 43.1s | 0.12 |
| 10 | 3.5s | 2.8 |
| 50 | 3.3s | 15.1 |
| 100 | 6.5s | 15.5 |
| 150 | 5.5s | 27.2 |

### Estimated Experiment Time

| Experiment | Runs | MLX (batch=100) | SGLang (conc=150) |
|-----------|-----:|-----------------:|------------------:|
| Section 6: Deployment architecture | 37,800 | 5.5h | ~23 min |
| Section 5: On-device characterization | 15,000 | 2.2h | — |
| Mobile evaluation | 7,500 | — | — (~80h/device) |
| **Total (parallel)** | | **~7.7h** | |

## Deployment Architectures

The framework supports three deployment modes with simulated network conditions:

```
                    Inference Location
                 On-Device         Cloud (SGLang)
             ┌────────────────┬──────────────────────┐
 Tools       │ A: FULLY LOCAL │ B: HYBRID            │
 On-Device   │ (no network)   │ (cloud brain,        │
             │                │  local hands)        │
             ├────────────────┼──────────────────────┤
 Tools       │                │ C: FULLY CLOUD       │
 On-Cloud    │ (N/A)          │ (baseline)           │
             └────────────────┴──────────────────────┘
```

### Network Simulation Presets

| Preset | RTT | Jitter | Loss | Bandwidth |
|--------|----:|-------:|-----:|----------:|
| local | 0ms | 0ms | 0% | 10 Gbps |
| lan | 1ms | 0.5ms | 0% | 1 Gbps |
| wifi | 20ms | 5ms | 0.1% | 50 Mbps |
| 5g | 40ms | 15ms | 0.1% | 100 Mbps |
| 4g | 80ms | 30ms | 0.5% | 20 Mbps |
| poor_cellular | 200ms | 100ms | 2% | 2 Mbps |
| edge_case | 500ms | 200ms | 5% | 0.5 Mbps |

## Installation

```bash
cd cli

# Install base
pip install -e .

# Install with MLX support (macOS only)
pip install -e ".[mlx]"

# Install test dependencies
pip install -e ".[test]"
```

## Quick Start

### MLX Inference (macOS)

```python
from pocket_agent_cli.services.mlx_inference_service import MLXInferenceService
from pocket_agent_cli.config import InferenceConfig, Model

service = MLXInferenceService()
model = Model(id="qwen-3-4b", name="Qwen 3 4B", architecture="qwen",
              downloaded=True, default_version="Q4_K_M")
config = InferenceConfig(temperature=0.7, max_tokens=512)

service.load_model(model, config)

# Single streaming generation
for chunk in service.generate([{"role": "user", "content": "Hello!"}]):
    print(chunk["token"], end="")

# Batch generation (100 prompts)
results = service.batch_generate(prompts_list, max_tokens=256)
```

### SGLang Inference (Mahti A100)

```bash
# Submit SGLang server job
ssh mahti "sbatch ~/sglang-server.sh Qwen/Qwen3-4B qwen 30000"

# Query from login node
curl http://<node>.mahti.csc.fi:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"Hello!"}]}'
```

### Network Simulation

```python
from pocket_agent_cli.network import NetworkSimulator, NETWORK_PRESETS, HybridArchitectureWrapper

sim = NetworkSimulator(NETWORK_PRESETS["4g"], seed=42)
wrapper = HybridArchitectureWrapper(inference_service, sim)

# generate_with_tools now includes simulated network delays
text, tools, metrics = wrapper.generate_with_tools(messages, tools)
print(metrics["network"])  # transfer log, radio energy, total delay
```

## Models

| Model | Params | MLX (INT4) | SGLang (FP16) |
|-------|-------:|-----------|--------------|
| Qwen 3 4B | 4.0B | `mlx-community/Qwen3-4B-4bit` | `Qwen/Qwen3-4B` |
| Qwen 3 0.6B | 0.6B | `mlx-community/Qwen3-0.6B-4bit` | `Qwen/Qwen3-0.6B` |
| Llama 3.2 3B Instruct | 3.2B | `mlx-community/Llama-3.2-3B-Instruct-4bit` | `meta-llama/Llama-3.2-3B-Instruct` |
| DeepSeek R1 Distill Qwen 1.5B | 1.5B | `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Gemma 3n E2B IT | 2.0B | `google/gemma-3n-E2B-it` | `google/gemma-3n-E2B-it` |

## Directory Structure

```
cli/
├── pocket_agent_cli/
│   ├── services/
│   │   ├── inference_service.py      # llama.cpp backend
│   │   └── mlx_inference_service.py  # MLX backend
│   ├── datasets/
│   │   ├── mbpp.py                   # MBPP benchmark
│   │   ├── humaneval.py              # HumanEval benchmark
│   │   └── gsm8k.py                  # GSM8K math benchmark
│   ├── network/
│   │   ├── network_simulator.py      # Network condition simulation
│   │   ├── radio_model.py            # Cellular RRC state model
│   │   ├── deployment_architectures.py # Local/Hybrid/Cloud modes
│   │   └── transfer_event.py         # Transfer event logging
│   ├── tools/                        # Tool executor (Python REPL, file ops)
│   ├── monitoring/                   # Power/energy/thermal monitoring
│   └── config.py                     # Model and inference configuration
├── scripts/
│   └── e2e_engine_test.py           # Engine throughput benchmarking
├── tests/
│   ├── test_mlx_inference.py         # 15 tests
│   ├── test_gsm8k.py                 # 56 tests
│   └── test_network_simulator.py     # 52 tests
└── research/
    ├── paper.tex                     # MobiHoc 2026 paper
    └── MOBIHOC_REVISION_PLAN.md      # Revision plan
```

## License

MIT
