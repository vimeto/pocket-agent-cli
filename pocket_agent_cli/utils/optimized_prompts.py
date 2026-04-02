"""Optimized prompt configurations per model × mode.

These prompts were tuned through 4 rounds of optimization on MBPP and GSM8K
subsets. Each model uses its native optimal tool-calling format.

Round 4 results on 50 MBPP problems:
  qwen-3-4b:            50% / 50% / 50%  (base / tool_sub / full_tool)
  qwen-3-0.6b:          16% / 10% /  8%
  llama-3.2-3b:         52% / 24% / 22%
  deepseek-r1-1.5b:     26% / 22% / 26%
  gemma-3n-e2b:         56% / 20% /  0%  (tool modes limited by model capability)
"""

from typing import Dict, Any


OPTIMIZED_PROMPTS: Dict[str, Dict[str, Dict[str, str]]] = {
    # ── Qwen 3 4B ─────────────────────────────────────────────────────
    # Best model for agentic tasks. Uses <tool_call> XML format natively.
    # Thinking tokens enabled — critical for performance.
    "qwen-3-4b": {
        "base": {
            "system": (
                "You are a Python programmer. Write the complete function "
                "implementation. After thinking, output the final code in a "
                "```python block."
            ),
            "user_prefix": "Write this Python function:\n\n",
            "user_suffix": "\n\nPut your final function in a ```python block.",
        },
        "tool_submission": {
            "system": (
                "You are a Python programmer. Write the function, then submit it.\n\n"
                "After thinking, submit using:\n"
                "<tool_call>\n"
                '{"name": "submit_python_solution", "arguments": '
                '{"code": "def func(args):\\n    ...your code..."}}\n'
                "</tool_call>"
            ),
            "user_prefix": "Solve and submit:\n\n",
            "user_suffix": "",
        },
        "full_tool": {
            "system": (
                "You are a Python coding assistant. Available tools:\n"
                "1. run_python_code(code) - Execute Python code\n"
                "2. submit_python_solution(code) - Submit final solution (REQUIRED)\n\n"
                "Call tools using:\n"
                "<tool_call>\n"
                '{"name": "<tool>", "arguments": {<args>}}\n'
                "</tool_call>\n\n"
                "Workflow: Write solution → test → submit. "
                "You MUST call submit_python_solution."
            ),
            "user_prefix": "Solve and submit:\n\n",
            "user_suffix": "",
        },
    },

    # ── Qwen 3 0.6B ──────────────────────────────────────────────────
    # Smallest model. Needs concise prompts. Same <tool_call> format.
    "qwen-3-0.6b": {
        "base": {
            "system": (
                "You are a Python programmer. Write the function. "
                "Put final code in ```python block."
            ),
            "user_prefix": "",
            "user_suffix": "\n\nWrite the function in a ```python block:",
        },
        "tool_submission": {
            "system": (
                "Write the function, then submit:\n"
                "<tool_call>\n"
                '{"name": "submit_python_solution", "arguments": '
                '{"code": "def func(...):\\n    ..."}}\n'
                "</tool_call>"
            ),
            "user_prefix": "",
            "user_suffix": "\n\nSubmit your solution:",
        },
        "full_tool": {
            "system": (
                "Tools: run_python_code(code), submit_python_solution(code).\n"
                "Format: <tool_call>"
                '{"name":"...","arguments":{...}}'
                "</tool_call>\n"
                "MUST submit solution."
            ),
            "user_prefix": "",
            "user_suffix": "\n\nSolve and submit:",
        },
    },

    # ── Llama 3.2 3B Instruct ────────────────────────────────────────
    # Outputs raw JSON tool calls. No thinking tokens.
    "llama-3.2-3b-instruct": {
        "base": {
            "system": (
                "You are a Python programmer. Write the complete function. "
                "Output the code in a ```python block."
            ),
            "user_prefix": "Write this function:\n\n",
            "user_suffix": "",
        },
        "tool_submission": {
            "system": (
                "You are a Python programmer. Write the complete function, "
                "then submit it.\n\n"
                "To submit, output this JSON:\n"
                '{"name": "submit_python_solution", "arguments": '
                '{"code": "def func_name(args):\\n    ...your implementation..."}}\n\n'
                "IMPORTANT: The 'code' field must contain the FULL Python "
                "function definition (source code), NOT function call arguments."
            ),
            "user_prefix": "Write and submit this function:\n\n",
            "user_suffix": "",
        },
        "full_tool": {
            "system": (
                "You have Python tools:\n"
                '1. run_python_code: {"name": "run_python_code", '
                '"arguments": {"code": "..."}}\n'
                '2. submit_python_solution: {"name": "submit_python_solution", '
                '"arguments": {"code": "..."}}\n\n'
                "Write the function, test it, then submit with "
                "submit_python_solution."
            ),
            "user_prefix": "Solve:\n\n",
            "user_suffix": "",
        },
    },

    # ── DeepSeek R1 Distill Qwen 1.5B ────────────────────────────────
    # Reasoning model. Long thinking chains. Uses <tool_call> or ```python.
    "deepseek-r1-distill-qwen-1.5b": {
        "base": {
            "system": (
                "You are a Python programmer. Think through the problem, "
                "then write the function. Put your final code in a "
                "```python block."
            ),
            "user_prefix": "Write this function:\n\n",
            "user_suffix": "",
        },
        "tool_submission": {
            "system": (
                "Write the Python function. After reasoning, submit:\n"
                "<tool_call>\n"
                '{"name": "submit_python_solution", "arguments": '
                '{"code": "def func(args):\\n    ...code..."}}\n'
                "</tool_call>"
            ),
            "user_prefix": "Solve and submit:\n\n",
            "user_suffix": "",
        },
        "full_tool": {
            "system": (
                "You are a Python assistant with tools:\n"
                "1. run_python_code(code) - Run code\n"
                "2. submit_python_solution(code) - Submit (REQUIRED)\n\n"
                "After reasoning, call tools:\n"
                "<tool_call>\n"
                '{"name": "tool", "arguments": {"code": "..."}}\n'
                "</tool_call>"
            ),
            "user_prefix": "Solve:\n\n",
            "user_suffix": "",
        },
    },

    # ── Gemma 3n E2B IT ──────────────────────────────────────────────
    # No thinking tokens. Uses Python-style [func(args)] format.
    # Tool modes are limited by model capability, not prompts.
    "gemma-3n-e2b-it": {
        "base": {
            "system": "Output ONLY the Python function. No text.",
            "user_prefix": "",
            "user_suffix": "\n\nONLY Python function:",
        },
        "tool_submission": {
            "system": (
                'Output ONLY: [submit_python_solution(code="def func(...):\\n    ...")]'
            ),
            "user_prefix": "",
            "user_suffix": '\n\nSubmit using [submit_python_solution(code="...")]:'
        },
        "full_tool": {
            "system": (
                "CRITICAL: You are an AI coding assistant with tools:\n"
                '1. run_python_code(code="...") - Execute code\n'
                '2. upsert_file(filename="...", content="...") - Write file\n'
                '3. run_submission_tests(filename="...") - Test\n'
                '4. submit_python_solution(code="...") - Submit (REQUIRED)\n\n'
                "Use: [tool_name(param=\"value\")]\n"
                "Workflow: Write → Test → Submit.\n"
                "NO explanations. ONLY tool calls."
            ),
            "user_prefix": "",
            "user_suffix": "",
        },
    },
}


def get_optimized_prompt(model_id: str, mode: str) -> Dict[str, str]:
    """Get the optimized prompt config for a model × mode combination.

    Args:
        model_id: Model identifier (e.g., 'qwen-3-4b')
        mode: Benchmark mode ('base', 'tool_submission', 'full_tool')

    Returns:
        Dict with 'system', 'user_prefix', 'user_suffix' keys.

    Raises:
        KeyError: If model_id or mode not found.
    """
    if model_id not in OPTIMIZED_PROMPTS:
        raise KeyError(
            f"No optimized prompts for model '{model_id}'. "
            f"Available: {list(OPTIMIZED_PROMPTS.keys())}"
        )
    prompts = OPTIMIZED_PROMPTS[model_id]
    if mode not in prompts:
        raise KeyError(
            f"No optimized prompt for mode '{mode}' on model '{model_id}'. "
            f"Available: {list(prompts.keys())}"
        )
    return prompts[mode]
