"""Configuration and constants for pocket-agent-cli."""

from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import os


# Directories with environment variable override support
HOME_DIR = Path.home()

# Allow environment variables to override default paths
if os.environ.get("POCKET_AGENT_HOME"):
    APP_DIR = Path(os.environ["POCKET_AGENT_HOME"])
else:
    APP_DIR = HOME_DIR / ".pocket-agent-cli"

# Individual directory overrides or defaults
MODELS_DIR = Path(os.environ.get("POCKET_AGENT_MODELS_DIR", APP_DIR / "models"))
DATA_DIR = Path(os.environ.get("POCKET_AGENT_DATA_DIR", APP_DIR / "data"))
SANDBOX_DIR = Path(os.environ.get("POCKET_AGENT_SANDBOX_DIR", APP_DIR / "sandbox"))
RESULTS_DIR = Path(os.environ.get("POCKET_AGENT_RESULTS_DIR", APP_DIR / "results"))

# Create directories if they don't exist
for dir_path in [APP_DIR, MODELS_DIR, DATA_DIR, SANDBOX_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# Model configurations with versioning support
DEFAULT_MODELS = [
    {
        "id": "llama-3.2-3b-instruct",
        "name": "Llama 3.2 3B Instruct",
        "architecture": "llama",
        "requiresAuth": True,
        "versions": {
            "Q4_K_M": {
                "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                "size": 2018916352,  # ~2GB
                "quantization": "Q4_K_M",
            },
            "F16": {
                "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf",
                "size": 6896974848,  # ~6.43GB
                "quantization": "F16",
            }
        },
        "default_version": "Q4_K_M"
    },
    {
        "id": "gemma-3n-e2b-it",
        "name": "Gemma 3n E2B IT",
        "architecture": "gemma",
        "requiresAuth": False,
        "versions": {
            "Q4_K_M": {
                "url": "https://huggingface.co/bartowski/google_gemma-3n-E2B-it-GGUF/resolve/main/google_gemma-3n-E2B-it-Q4_K_M.gguf",
                "size": 1611466752,  # ~1.5GB
                "quantization": "Q4_K_M",
            },
            "F16": {
                "url": "https://huggingface.co/ggml-org/gemma-3n-E2B-it-GGUF/resolve/main/gemma-3n-E2B-it-f16.gguf",
                "size": 9580102656,  # ~8.92GB
                "quantization": "F16",
            }
        },
        "default_version": "Q4_K_M"
    },
    {
        "id": "deepseek-r1-distill-qwen-1.5b",
        "name": "DeepSeek R1 Distill Qwen 1.5B",
        "architecture": "qwen",
        "requiresAuth": False,
        "versions": {
            "Q4_K_M": {
                "url": "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
                "size": 1181908992,  # ~1.1GB
                "quantization": "Q4_K_M",
            },
            "F16": {
                "url": "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-f16.gguf",
                "size": 3821789184,  # ~3.56GB
                "quantization": "F16",
            }
        },
        "default_version": "Q4_K_M"
    },
    {
        "id": "qwen-3-4b",
        "name": "Qwen 3 4B",
        "architecture": "qwen",
        "requiresAuth": False,
        "versions": {
            "Q4_K_M": {
                "url": "https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/main/Qwen_Qwen3-4B-Q4_K_M.gguf",
                "size": 2471854080,  # ~2.3GB
                "quantization": "Q4_K_M",
            },
            "F16": {
                "url": "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-BF16.gguf",
                "size": 8641699840,  # ~8.05GB
                "quantization": "F16",
            }
        },
        "default_version": "Q4_K_M"
    },
    {
        "id": "qwen-3-0.6b",
        "name": "Qwen 3 0.6B",
        "architecture": "qwen",
        "requiresAuth": False,
        "versions": {
            "Q4_K_M": {
                "url": "https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q4_K_M.gguf",
                "size": 515396608,  # ~0.48GB
                "quantization": "Q4_K_M",
            },
            "F16": {
                "url": "https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-f16.gguf",
                "size": 1621491712,  # ~1.51GB
                "quantization": "F16",
            }
        },
        "default_version": "Q4_K_M"
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


class ModelVersion(BaseModel):
    """Model version metadata."""

    url: str
    size: int
    quantization: str
    downloaded: bool = False
    path: Optional[Path] = None


class Model(BaseModel):
    """Model metadata with versioning support."""

    id: str
    name: str
    architecture: str
    requiresAuth: bool = False
    versions: Dict[str, ModelVersion] = {}
    default_version: str = "Q4_K_M"
    current_version: Optional[str] = None  # Track which version is currently loaded

    # Legacy fields for backward compatibility
    size: Optional[int] = None
    url: Optional[str] = None
    quantization: Optional[str] = None
    downloaded: bool = False
    path: Optional[Path] = None

    def get_version(self, version: Optional[str] = None) -> ModelVersion:
        """Get a specific version or the default."""
        version = version or self.default_version
        if version not in self.versions:
            raise ValueError(f"Version {version} not found for model {self.id}")
        return self.versions[version]

    def is_downloaded(self, version: Optional[str] = None) -> bool:
        """Check if a specific version is downloaded."""
        if version:
            return self.versions.get(version, ModelVersion(url="", size=0, quantization="")).downloaded
        # Check if any version is downloaded
        return any(v.downloaded for v in self.versions.values())


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
    model_version: Optional[str] = Field(default=None, description="Model version (Q4_K_M, F16, etc)")
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
        user_prompt_template="{problem}",
        requires_tools=False,
    ),
    "tool_submission": BenchmarkMode(
        name="tool_submission",
        description="Reasoning and code submission via tool",
        system_prompt="You are a Python programming expert. Solve the given problem step by step, then submit your solution using the submit_python_solution tool.",
        user_prompt_template="{problem}",
        requires_tools=True,
        max_iterations=1,
    ),
    "full_tool": BenchmarkMode(
        name="full_tool",
        description="Complete development environment with all tools",
        system_prompt="You are a Python programming expert. You have access to multiple tools to develop, test, and submit solutions. Use the tools effectively to solve the problem.",
        user_prompt_template="{problem}",
        requires_tools=True,
        max_iterations=5,
    ),
}


# Tool definitions
SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_python_solution",
        "description": "Submit a Python solution for evaluation",
        "parameters": {
            "type": "object",
            "properties": {
                "solution": {
                    "type": "string",
                    "description": "The complete Python solution code",
                }
            },
            "required": ["solution"],
        },
    },
}

# All available tools
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "Execute Python code and return the output",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "upsert_file",
            "description": "Create or update a file with the given content",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to create or update",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
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
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to read",
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
            "description": "Run test cases against the submitted solution",
            "parameters": {
                "type": "object",
                "properties": {
                    "solution": {
                        "type": "string",
                        "description": "The Python solution to test",
                    }
                },
                "required": ["solution"],
            },
        },
    },
    SUBMIT_TOOL,
]
