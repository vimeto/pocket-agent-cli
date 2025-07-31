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
]


class InferenceConfig(BaseModel):
    """Configuration for LLM inference."""
    
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0)
    repeat_penalty: float = Field(default=1.1, ge=0.0, le=2.0)
    context_length: int = Field(default=4096, ge=512, le=32768)
    stop_tokens: Optional[List[str]] = Field(default_factory=lambda: ["<|im_end|>", "<|eot_id|>", "</s>", "<end_of_turn>"])
    jinja: bool = Field(default=True)
    tool_choice: str = Field(default="auto", pattern="^(auto|required|none)$")
    tools: Optional[List[Dict[str, Any]]] = None
    
    # llama.cpp specific
    n_threads: int = Field(default=4, ge=1)
    n_batch: int = Field(default=512, ge=1)
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


# Benchmark modes matching the mobile app
BENCHMARK_MODES = {
    "base": BenchmarkMode(
        name="base",
        description="Simple code generation without tools",
        system_prompt="You are a helpful assistant that writes Python code.",
        user_prompt_template="{problem_description}",
        requires_tools=False,
        max_iterations=1,
    ),
    "tool_submission": BenchmarkMode(
        name="tool_submission",
        description="Reasoning with code submission tool",
        system_prompt="""You are a helpful assistant that solves programming problems step by step.
You have access to a special tool called submit_python_solution to submit your final solution.
Think through the problem carefully before submitting your solution.""",
        user_prompt_template="{problem_description}",
        requires_tools=True,
        max_iterations=1,
    ),
    "full_tool": BenchmarkMode(
        name="full_tool",
        description="Full development environment with all tools",
        system_prompt="""You are a helpful assistant with access to a Python development environment.
You can create, run, and debug code using the available tools.
Iterate on your solution until all test cases pass.""",
        user_prompt_template="{problem_description}",
        requires_tools=True,
        max_iterations=10,
    ),
}


# Tool definitions
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
            "name": "run_python_file",
            "description": "Execute a Python file by name",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to execute",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "upsert_file",
            "description": "Create or update a file with content",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file",
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
            "name": "delete_file",
            "description": "Delete a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to delete",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in the current directory",
            "parameters": {
                "type": "object",
                "properties": {},
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
]


# Special tool for benchmark submission mode
SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_python_solution",
        "description": "Submit your final Python solution for evaluation",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The complete Python code solution (optional if filename provided)",
                },
                "filename": {
                    "type": "string", 
                    "description": "Path to the Python file containing the solution (optional if code provided)",
                }
            },
            "oneOf": [
                {"required": ["code"]},
                {"required": ["filename"]}
            ],
        },
    },
}