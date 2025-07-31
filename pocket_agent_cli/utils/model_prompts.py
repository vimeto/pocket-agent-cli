"""Model-specific prompt configurations."""

from typing import Dict, Any


# Model-specific system prompts for different modes
MODEL_PROMPTS = {
    "gemma": {
        "base": {
            "system_prompt": "Output ONLY code. No text.",
            "user_suffix": "\n\nONLY Python function:"
        },
        "tool_submission": {
            "system_prompt": "Output ONLY: [submit_python_solution(code=\"...\")]",
            "user_suffix": "\n\nSubmit using [submit_python_solution(code=\"...\")]:"
        },
        "full_tool": {
            "system_prompt": """CRITICAL: You MUST output ONLY function calls in these EXACT formats:

[function_name(param1="value1", param2="value2")]
OR
{"name": "function_name", "parameters": {"param1": "value1"}}

EXAMPLES:

1. Simple submission:
Q: Write a function to add two numbers
A: [submit_python_solution(code="def add_two_numbers(a, b):\\n    return a + b")]

2. Full agentic workflow with testing:
Q: Write a function to find the factorial of a number
A: [upsert_file(filename="factorial.py", content="def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)")]
User: File created successfully
A: [run_submission_tests(filename="factorial.py")]
User: Test Results: 3/3 passed
----------------------------------------
Test 1: PASSED
Test 2: PASSED  
Test 3: PASSED
A: [submit_python_solution(filename="factorial.py")]

DO NOT:
- Use ```python blocks
- Add any explanations
- Output plain code
- Use any other format

ALWAYS use [function_name(...)] or {"name": "...", "parameters": {...}}""",
            "tool_format_example": """Example function calls:
[run_python_code(code="print('Hello')")]
[submit_python_solution(code="def solution():\\n    return 42")]
{"name": "read_file", "parameters": {"filename": "test.py"}}"""
        }
    },
    "llama": {
        "base": {
            "system_prompt": "Generate Python function. Code only.",
            "user_suffix": ""
        },
        "tool_submission": {
            "system_prompt": "Use submit_python_solution tool.",
            "user_suffix": ""
        },
        "full_tool": {
            "system_prompt": """Python environment available. Tools: run_python_code (code/file), upsert_file, read_file, submit_python_solution.
MUST submit final solution with submit_python_solution.""",
            "tool_format_example": """```tool_call
{"name": "submit_python_solution", "parameters": {"code": "solution code here"}}
```"""
        }
    },
    "qwen": {
        "base": {
            "system_prompt": "Generate Python code only.",
            "user_suffix": ""
        },
        "tool_submission": {
            "system_prompt": "Use submit_python_solution in ```tool_call block.",
            "user_suffix": ""
        },
        "full_tool": {
            "system_prompt": """Python environment. Tools: run_python_code, upsert_file, read_file, submit_python_solution.
Use JSON in ```tool_call blocks. MUST submit with submit_python_solution.""",
            "tool_format_example": """```tool_call
{"name": "submit_python_solution", "parameters": {"code": "solution"}}
```"""
        }
    },
    "default": {
        "base": {
            "system_prompt": "Generate Python function code only.",
            "user_suffix": ""
        },
        "tool_submission": {
            "system_prompt": "Use submit_python_solution tool to submit code.",
            "user_suffix": ""
        },
        "full_tool": {
            "system_prompt": """Python env. Tools: run_python_code (code/file), upsert_file, read_file, submit_python_solution.
MUST submit final solution with submit_python_solution.""",
            "tool_format_example": """```tool_call
{"name": "tool_name", "parameters": {}}
```"""
        }
    }
}


def get_model_prompt(architecture: str, mode: str) -> Dict[str, str]:
    """Get model-specific prompt configuration for a given mode.

    Args:
        architecture: Model architecture (gemma, llama, qwen, etc.)
        mode: Benchmark mode (base, tool_submission, full_tool)

    Returns:
        Dict with system_prompt and optional tool_format_example and user_suffix
    """
    model_config = MODEL_PROMPTS.get(architecture, MODEL_PROMPTS["default"])
    mode_config = model_config.get(mode, model_config.get("full_tool", {}))
    
    # Ensure we return a dict with at least system_prompt
    if isinstance(mode_config, dict):
        return mode_config
    else:
        # Fallback for old format
        return {"system_prompt": mode_config}

# Keep backward compatibility
def get_model_tool_prompt(architecture: str) -> Dict[str, str]:
    """Get model-specific tool prompt configuration (backward compatibility).

    Args:
        architecture: Model architecture (gemma, llama, qwen, etc.)

    Returns:
        Dict with system_prompt and tool_format_example
    """
    return get_model_prompt(architecture, "full_tool")
