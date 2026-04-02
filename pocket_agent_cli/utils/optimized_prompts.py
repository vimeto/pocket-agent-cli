"""Prompt configurations per model × mode.

Closely mirrors the original study prompts from model_prompts.py but adapted
for each engine's native tool-calling format (MLX uses tokenizer's
apply_chat_template; SGLang uses OpenAI-compatible API).

Paper reference results (FP16, A100, MBPP Pass@1):
  Qwen 3 4B:   base=40.5%, full_tool=76.4%  (+36pp)
  Qwen 3 0.6B: base=37%,   full_tool=41%    (+4pp)
  Gemma 3n:    base=~42%,  full_tool=~34%   (-8pp)
  Llama 3.2:   base=~42%,  full_tool=~34%   (-8pp)
  DeepSeek R1: base=36.5%, full_tool=~37%   (~0pp)
"""

from typing import Dict


# ── Qwen 3 (4B and 0.6B) ─────────────────────────────────────────────
# Native format: <tool_call>{"name":...,"arguments":{...}}</tool_call>
# Thinking tokens enabled (critical for performance).

_QWEN_PROMPTS = {
    "base": {
        "system": "You are a Python programmer. Complete functions by providing the implementation.",
        "user_suffix": "\n\nComplete this function:",
    },
    "tool_submission": {
        "system": (
            "Submit Python solutions using this format:\n\n"
            "<tool_call>\n"
            '{"name": "submit_python_solution", "arguments": {"code": "solution"}}\n'
            "</tool_call>"
        ),
        "user_suffix": "\n\nSubmit the complete function:",
    },
    "full_tool": {
        "system": (
            "Available tools:\n"
            "1. run_python_code - Execute code\n"
            "2. upsert_file - Create/update files\n"
            "3. read_file - Read files\n"
            "4. run_submission_tests - Test solution\n"
            "5. submit_python_solution - Submit final solution (REQUIRED)\n\n"
            "Use <tool_call> format. MUST call submit_python_solution."
        ),
        "user_suffix": "\n\nYour output:",
    },
}

# ── Llama 3.2 3B Instruct ────────────────────────────────────────────
# Native format: raw JSON {"name":...,"arguments":{...}}
# No thinking tokens.

_LLAMA_PROMPTS = {
    "base": {
        "system": "You are a helpful coding assistant. Write complete Python functions.",
        "user_suffix": "",
    },
    "tool_submission": {
        "system": (
            "Write the Python function, then submit your SOURCE CODE using "
            "submit_python_solution.\n"
            "The code parameter must contain the complete function definition."
        ),
        "user_suffix": "",
    },
    "full_tool": {
        "system": (
            "You have these tools:\n"
            "1. run_python_code(code) - Execute Python code\n"
            "2. submit_python_solution(code) - Submit final solution (REQUIRED)\n\n"
            "Write the function, test it with run_python_code, then submit "
            "the complete function SOURCE CODE with submit_python_solution."
        ),
        "user_suffix": "",
    },
}

# ── DeepSeek R1 Distill Qwen 1.5B ────────────────────────────────────
# Reasoning model (long thinking chains). Qwen distill, same <tool_call> format.

_DEEPSEEK_PROMPTS = {
    "base": {
        "system": "You are a Python programmer. Complete functions by providing the implementation.",
        "user_suffix": "\n\nComplete this function:",
    },
    "tool_submission": {
        "system": (
            "Submit Python solutions using this format:\n\n"
            "<tool_call>\n"
            '{"name": "submit_python_solution", "arguments": {"code": "solution"}}\n'
            "</tool_call>"
        ),
        "user_suffix": "\n\nSubmit the complete function:",
    },
    "full_tool": {
        "system": (
            "Available tools:\n"
            "1. run_python_code - Execute code\n"
            "2. submit_python_solution - Submit final solution (REQUIRED)\n\n"
            "Use <tool_call> format. MUST call submit_python_solution."
        ),
        "user_suffix": "\n\nSolve and submit:",
    },
}

# ── Gemma 3n E2B IT ──────────────────────────────────────────────────
# Native format: [func_name(param="value")]
# No thinking tokens.

_GEMMA_PROMPTS = {
    "base": {
        "system": "Output ONLY code. No text.",
        "user_suffix": "\n\nONLY Python function:",
    },
    "tool_submission": {
        "system": 'Output ONLY: [submit_python_solution(code="...")]',
        "user_suffix": '\n\nSubmit using [submit_python_solution(code="...")]:',
    },
    "full_tool": {
        "system": (
            "CRITICAL: You are an AI coding assistant with these tools:\n\n"
            '1. run_python_code(code="...") - Execute code\n'
            '2. upsert_file(filename="...", content="...") - Create/update files\n'
            '3. run_submission_tests(filename="...") - Test solution\n'
            '4. submit_python_solution(code="...") - Submit final solution (REQUIRED)\n\n'
            'Use ONLY [function_name(param="value")] format.\n'
            "MUST call submit_python_solution at the end."
        ),
        "user_suffix": "",
    },
}


OPTIMIZED_PROMPTS: Dict[str, Dict[str, Dict[str, str]]] = {
    "qwen-3-4b": _QWEN_PROMPTS,
    "qwen-3-0.6b": _QWEN_PROMPTS,
    "llama-3.2-3b-instruct": _LLAMA_PROMPTS,
    "deepseek-r1-distill-qwen-1.5b": _DEEPSEEK_PROMPTS,
    "gemma-3n-e2b-it": _GEMMA_PROMPTS,
}


def get_optimized_prompt(model_id: str, mode: str) -> Dict[str, str]:
    """Get prompt config for a model × mode combination.

    Returns dict with 'system' and optionally 'user_suffix' keys.
    """
    if model_id not in OPTIMIZED_PROMPTS:
        raise KeyError(f"No prompts for '{model_id}'. Available: {list(OPTIMIZED_PROMPTS.keys())}")
    prompts = OPTIMIZED_PROMPTS[model_id]
    if mode not in prompts:
        raise KeyError(f"No prompt for mode '{mode}'. Available: {list(prompts.keys())}")
    return prompts[mode]
