"""Configuration and constants for pocket-agent-cli."""

from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import os


# Directories
HOME_DIR = Path.home()
APP_DIR = HOME_DIR / ".pocket-agent-cli"
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"
SANDBOX_DIR = APP_DIR / "sandbox"
RESULTS_DIR = APP_DIR / "results"

# Create directories if they don't exist
for dir_path in [APP_DIR, MODELS_DIR, DATA_DIR, SANDBOX_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# Model configurations
DEFAULT_MODELS = [
    {
        "id": "llama-3.2-3b-instruct",
        "name": "Llama 3.2 3B Instruct",
        "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size": 2018916352,  # ~2GB
        "quantization": "Q4_K_M",
        "architecture": "llama",
        "requiresAuth": True,
    },
    {
        "id": "gemma-3n-e2b-it",
        "name": "Gemma 3n E2B IT",
        "url": "https://huggingface.co/bartowski/google_gemma-3n-E2B-it-GGUF/resolve/main/google_gemma-3n-E2B-it-Q4_K_M.gguf",
        "size": 1611466752,  # ~1.5GB
        "quantization": "Q4_K_M",
        "architecture": "gemma",
        "requiresAuth": False,
    },
    {
        "id": "deepseek-r1-distill-qwen-1.5b",
        "name": "DeepSeek R1 Distill Qwen 1.5B",
        "url": "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        "size": 1181908992,  # ~1.1GB
        "quantization": "Q4_K_M",
        "architecture": "qwen",
        "requiresAuth": False,
    },
    {
        "id": "qwen-3-4b",
        "name": "Qwen 3 4B",
        "url": "https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/main/Qwen_Qwen3-4B-Q4_K_M.gguf",
        "size": 2471854080,  # ~2.3GB
        "quantization": "Q4_K_M",
        "architecture": "qwen",
        "requiresAuth": False,
    },
    {
        "id": "qwen-3-0.6b",
        "name": "Qwen 3 0.6B",
        "url": "https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q4_K_M.gguf",
        "size": 515396608,  # ~0.48GB
        "quantization": "Q4_K_M",
        "architecture": "qwen",
        "requiresAuth": False,
    },
]


class InferenceConfig(BaseModel):
    """Configuration for LLM inference."""

    temperature: float = Field(default=0.1, ge=0.0, le=2.0)  # Lower for code generation
    max_tokens: int = Field(default=100, ge=1, le=8192)  # Temporarily reduced for profiling test
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)  # More focused sampling
    top_k: int = Field(default=40, ge=0)
    repeat_penalty: float = Field(default=1.1, ge=0.0, le=2.0)
    context_length: int = Field(default=4096, ge=512, le=32768)  # Reduced for better performance
    stop_tokens: Optional[List[str]] = Field(default_factory=lambda: [
        "<|im_end|>", "<|eot_id|>", "</s>", "<end_of_turn>",
        "\n\n\n",  # Multiple newlines
        "# Example",  # Stop before examples
        "# Test",  # Stop before test cases
        "# Usage",  # Stop before usage examples
    ])
    jinja: bool = Field(default=True)
    tool_choice: str = Field(default="auto", pattern="^(auto|required|none)$")
    tools: Optional[List[Dict[str, Any]]] = None

    # llama.cpp specific (optimized for M3 Max)
    n_threads: int = Field(default=12, ge=-1)  # M3 Max has 12 performance cores
    n_batch: int = Field(default=512, ge=1)  # Optimized for single-stream inference
    use_mlock: bool = Field(default=True)
    use_mmap: bool = Field(default=True)


class Model(BaseModel):
    """Model metadata."""

    id: str
    name: str
    size: int
    url: Optional[str] = None
    quantization: Optional[str] = None
    architecture: str
    downloaded: bool = False
    path: Optional[Path] = None
    requiresAuth: bool = False


class BenchmarkMode(BaseModel):
    """Benchmark evaluation mode."""

    name: str
    description: str
    system_prompt: str
    user_prompt_template: str
    requires_tools: bool = False
    max_iterations: int = 1


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark runs."""

    model_config = {"protected_namespaces": ()}  # Allow model_ prefix

    model_name: str = Field(description="Model ID or 'all' for all models")
    mode: str = Field(default="base", description="Benchmark mode or 'all' for all modes")
    problems_limit: Optional[int] = Field(default=None, description="Number of problems to run")
    problem_ids: Optional[List[int]] = Field(default=None, description="Specific problem IDs to run")
    num_samples: int = Field(default=10, description="Number of samples per problem for pass@k")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    enable_tools: bool = Field(default=True)
    system_monitoring: bool = Field(default=True)
    output_dir: Path = Field(default=RESULTS_DIR / "benchmarks")
    save_individual_runs: bool = Field(default=True, description="Save each run separately")
    compute_pass_at_k: List[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    parallel_runs: int = Field(default=1, description="Number of parallel runs for pass@k sampling")
    exhaustive_passes: bool = Field(default=False, description="Run all samples even if problem already passes")


# Benchmark modes matching the mobile app
BENCHMARK_MODES = {
    "base": BenchmarkMode(
        name="base",
        description="Simple code generation without tools",
        system_prompt="Output ONLY Python function code. No explanations, comments, or text.",
        user_prompt_template="{problem_description}\n\nONLY code:",
        requires_tools=False,
        max_iterations=1,
    ),
    "tool_submission": BenchmarkMode(
        name="tool_submission",
        description="Reasoning with code submission tool",
        system_prompt="Use submit_python_solution tool. Code only, no text.",
        user_prompt_template="{problem_description}\n\nSubmit solution:",
        requires_tools=True,
        max_iterations=1,
    ),
    "full_tool": BenchmarkMode(
        name="full_tool",
        description="Full development environment with all tools",
        system_prompt="""You are coding in Python, with a Python env available. You must return JSON-formatted tool calls. You should return anything except tool calls. Tools: run_python_code (code/file), upsert_file, read_file, submit_python_solution.
MUST submit final solution with submit_python_solution. Tool calls must be made in ```tool_call``` blocks, e.g. ```tool_call
{
    "name": "run_python_code",
    "parameters": {
        "code": "print('Hello, world!')"
    }
}
```""",
        user_prompt_template="{problem_description}",
        requires_tools=True,
        max_iterations=20,
    ),
}


# Tool definitions
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "Run Python code or file",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code",
                    },
                    "filename": {
                        "type": "string",
                        "description": "File path",
                    }
                },
                "oneOf": [
                    {"required": ["code"]},
                    {"required": ["filename"]}
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "upsert_file",
            "description": "Create/update file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Path",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Path",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_submission_tests",
            "description": "Test solution against problem test cases",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Path to solution file",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_python_solution",
            "description": "Submit final solution",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Code",
                    },
                    "filename": {
                        "type": "string",
                        "description": "File",
                    }
                },
                "oneOf": [
                    {"required": ["code"]},
                    {"required": ["filename"]}
                ],
            },
        },
    },
]


# Special tool for benchmark submission mode
SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_python_solution",
        "description": "Submit final solution",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Code",
                },
                "filename": {
                    "type": "string",
                    "description": "File",
                }
            },
            "oneOf": [
                {"required": ["code"]},
                {"required": ["filename"]}
            ],
        },
    },
}
