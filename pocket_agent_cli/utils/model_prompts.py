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
            "system_prompt": """CRITICAL: You are an AI coding assistant with access to these EXACT tools:

1. run_python_code - Execute Python code directly or from a file
   Parameters: code (string) OR filename (string)
   Example: [run_python_code(code="print(2+2)")]

2. upsert_file - Create or update a file with content
   Parameters: filename (string), content (string)
   Example: [upsert_file(filename="solution.py", content="def func():\\n    return 42")]

3. read_file - Read contents of an existing file
   Parameters: filename (string)
   Example: [read_file(filename="test.py")]

4. run_submission_tests - Test your solution against problem test cases
   Parameters: filename (string)
   Example: [run_submission_tests(filename="solution.py")]

5. submit_python_solution - Submit your final solution (REQUIRED)
   Parameters: code (string) OR filename (string)
   Example: [submit_python_solution(code="def solution(x):\\n    return x * 2")]

OUTPUT FORMAT: Use ONLY [function_name(param="value")] or {"name": "function_name", "parameters": {...}}

WORKFLOW EXAMPLE:
Q: Write a function to calculate factorial
A: [upsert_file(filename="factorial.py", content="def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)")]
System: File created
A: [run_submission_tests(filename="factorial.py")]
System: All tests passed
A: [submit_python_solution(filename="factorial.py")]

RULES:
- MUST call submit_python_solution at the end
- NO explanations or plain text
- NO ```python blocks
- ONLY tool calls in the specified format""",
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
            "system_prompt": """You have a Python development environment with these tools:

AVAILABLE TOOLS:
1. run_python_code(code="..." OR filename="...") - Execute Python code
2. upsert_file(filename="...", content="...") - Create/update files
3. read_file(filename="...") - Read file contents
4. run_submission_tests(filename="...") - Test against problem test cases
5. submit_python_solution(code="..." OR filename="...") - Submit final solution (REQUIRED)

Use tools in ```tool_call blocks with JSON format:
```tool_call
{"name": "tool_name", "parameters": {"param": "value"}}
```

You MUST use submit_python_solution to submit your final solution.""",
            "tool_format_example": """```tool_call
{"name": "submit_python_solution", "parameters": {"code": "solution code here"}}
```"""
        }
    },
    "qwen": {
        "base": {
            "system_prompt": "You are a Python programmer. Complete functions by providing the implementation.",
            "user_suffix": "\n\nComplete this function:"
        },
        "tool_submission": {
            "system_prompt": """Submit Python solutions using this exact format:

```tool_call
{"name": "submit_python_solution", "parameters": {"code": "solution"}}
```

Always wrap tool calls in ```tool_call blocks.""",
            "user_suffix": "\n\nSubmit the complete function:"
        },
        "full_tool": {
            "system_prompt": """Example input: def add(a, b): '''Add two numbers'''
Example output:
```tool_call
{"name": "submit_python_solution", "parameters": {"code": "def add(a, b):\\n    return a + b"}}
```

Follow this exact pattern. Available tools:
1. run_python_code - Execute code
2. upsert_file - Create/update files
3. read_file - Read files
4. run_submission_tests - Test solution
5. submit_python_solution - Submit final solution (REQUIRED)

Always use ```tool_call format.""",
            "user_suffix": "\n\nYour output:",
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


def get_model_prompt(architecture: str, mode: str, model_id: str = None) -> Dict[str, str]:
    """Get model-specific prompt configuration for a given mode.

    Args:
        architecture: Model architecture (gemma, llama, qwen, etc.)
        mode: Benchmark mode (base, tool_submission, full_tool)
        model_id: Optional specific model ID for fine-tuned prompts

    Returns:
        Dict with system_prompt and optional tool_format_example and user_suffix
    """
    # Special handling for specific models in full_tool mode
    if mode == "full_tool" and model_id:
        if model_id == "qwen-3-0.6b":
            # Small model needs example-driven prompt
            return {
                "system_prompt": """Example input: def add(a, b): '''Add two numbers'''
Example output:
```tool_call
{"name": "submit_python_solution", "parameters": {"code": "def add(a, b):\\n    return a + b"}}
```

Follow this exact pattern. Available tools:
1. run_python_code - Execute code
2. upsert_file - Create/update files
3. read_file - Read files
4. run_submission_tests - Test solution
5. submit_python_solution - Submit final solution (REQUIRED)

Always use ```tool_call format.""",
                "user_suffix": "\n\nYour output:"
            }
        elif model_id in ["deepseek-r1-distill-qwen-1.5b", "qwen-3-4b"]:
            # Larger Qwen-based models need clearer instructions
            return {
                "system_prompt": """You are a Python coding assistant. Use these tools:

1. run_python_code(code="...") - Execute Python code
2. upsert_file(filename="...", content="...") - Create/update files
3. read_file(filename="...") - Read file contents
4. run_submission_tests(filename="...") - Test your solution
5. submit_python_solution(code="...") - Submit final solution (REQUIRED)

Output tool calls in ```tool_call blocks:
```tool_call
{"name": "submit_python_solution", "parameters": {"code": "def solution():\\n    return result"}}
```

You MUST call submit_python_solution with your complete solution.""",
                "user_suffix": "\n\nSolve and submit:"
            }
    
    # Default behavior
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
