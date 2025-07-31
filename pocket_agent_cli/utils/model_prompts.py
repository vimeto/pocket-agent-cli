"""Model-specific prompt configurations."""

from typing import Dict, Any


# Model-specific system prompts for tool calling
MODEL_TOOL_PROMPTS = {
    "gemma": {
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
    },
    "llama": {
        "system_prompt": """Python environment available. Tools: run_python_code (code/file), upsert_file, read_file, submit_python_solution.
MUST submit final solution with submit_python_solution.""",
        "tool_format_example": """```tool_call
{"name": "submit_python_solution", "parameters": {"code": "solution code here"}}
```"""
    },
    "qwen": {
        "system_prompt": """Python environment. Tools: run_python_code, upsert_file, read_file, submit_python_solution.
Use JSON in ```tool_call blocks. MUST submit with submit_python_solution.""",
        "tool_format_example": """```tool_call
{"name": "submit_python_solution", "parameters": {"code": "solution"}}
```"""
    },
    "default": {
        "system_prompt": """Python env. Tools: run_python_code (code/file), upsert_file, read_file, submit_python_solution.
MUST submit final solution with submit_python_solution.""",
        "tool_format_example": """```tool_call
{"name": "tool_name", "parameters": {}}
```"""
    }
}


def get_model_tool_prompt(architecture: str) -> Dict[str, str]:
    """Get model-specific tool prompt configuration.

    Args:
        architecture: Model architecture (gemma, llama, qwen, etc.)

    Returns:
        Dict with system_prompt and tool_format_example
    """
    return MODEL_TOOL_PROMPTS.get(architecture, MODEL_TOOL_PROMPTS["default"])
