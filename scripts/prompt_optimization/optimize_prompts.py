#!/usr/bin/env python3
"""Prompt optimization for all models × modes.

Iteratively tests and improves prompts for each model×mode combination
on small subsets, then validates on larger sets.

Usage:
    cd /Users/vilhelmtoivonen/code/phd/pocket-agent/cli
    HF_TOKEN=<your_token> .venv/bin/python scripts/prompt_optimization/optimize_prompts.py --round 1
"""

import argparse
import json
import os
import sys
import time
import gc
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pocket_agent_cli.services.mlx_inference_service import MLXInferenceService
from pocket_agent_cli.config import InferenceConfig, Model
from pocket_agent_cli.utils.tool_extractor import ToolExtractor
from pocket_agent_cli.datasets.registry import DatasetRegistry
# Import datasets to register them
from pocket_agent_cli.datasets import mbpp, humaneval, gsm8k

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

MODELS = [
    {"id": "qwen-3-4b", "name": "Qwen 3 4B", "arch": "qwen"},
    {"id": "qwen-3-0.6b", "name": "Qwen 3 0.6B", "arch": "qwen"},
    {"id": "llama-3.2-3b-instruct", "name": "Llama 3.2 3B", "arch": "llama"},
    {"id": "deepseek-r1-distill-qwen-1.5b", "name": "DeepSeek R1 1.5B", "arch": "qwen"},
    {"id": "gemma-3n-e2b-it", "name": "Gemma 3n E2B", "arch": "gemma"},
]

MODES = ["base", "tool_submission", "full_tool"]

# ── Prompt Templates (evolved per round) ──────────────────────────────────

def get_prompts(round_num: int = None) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Return prompt templates from optimized_prompts.py."""
    from pocket_agent_cli.utils.optimized_prompts import OPTIMIZED_PROMPTS
    return OPTIMIZED_PROMPTS


def _base_prompts():
    """Base mode prompts (no tools, just code generation)."""
    return {
        "qwen": {
            "system": "You are a Python programmer. Write the complete function implementation. Output ONLY the Python function code, nothing else.",
            "user_prefix": "Complete the following Python function:\n\n",
            "user_suffix": "",
        },
        "llama": {
            "system": "You are a Python programmer. Write the complete function implementation. Output ONLY the Python function code, nothing else.",
            "user_prefix": "Complete the following Python function:\n\n",
            "user_suffix": "",
        },
        "gemma": {
            "system": "Output ONLY the Python function code. No explanations.",
            "user_prefix": "",
            "user_suffix": "\n\nONLY Python function:",
        },
    }


def _tool_submission_prompts():
    """Tool-submission mode prompts (structured output via submit tool)."""
    return {
        "qwen": {
            "system": (
                "You are a Python programmer. Submit your solution using the submit_python_solution tool.\n\n"
                "When ready, call the tool like this:\n"
                "<tool_call>\n"
                '{"name": "submit_python_solution", "arguments": {"code": "<your complete function>"}}\n'
                "</tool_call>"
            ),
            "user_prefix": "Solve this and submit using the tool:\n\n",
            "user_suffix": "",
        },
        "llama": {
            "system": (
                "You are a Python programmer. Submit your solution using the submit_python_solution tool.\n\n"
                "When ready, output:\n"
                '{"name": "submit_python_solution", "arguments": {"code": "<your complete function>"}}'
            ),
            "user_prefix": "Solve this and submit using the tool:\n\n",
            "user_suffix": "",
        },
        "gemma": {
            "system": (
                "Submit solution using: [submit_python_solution(code=\"...\")]"
            ),
            "user_prefix": "",
            "user_suffix": "\n\nSubmit using [submit_python_solution(code=\"...\")]:",
        },
    }


def _full_tool_prompts():
    """Full-tool mode prompts (complete agentic workflow)."""
    return {
        "qwen": {
            "system": (
                "You are a Python coding assistant with tools. Available tools:\n\n"
                "1. run_python_code(code) - Execute Python code and see output\n"
                "2. upsert_file(filename, content) - Create/update a file\n"
                "3. read_file(filename) - Read a file\n"
                "4. run_submission_tests(filename) - Test your solution\n"
                "5. submit_python_solution(code) - Submit final solution (REQUIRED)\n\n"
                "Call tools using:\n"
                "<tool_call>\n"
                '{"name": "<tool>", "arguments": {<args>}}\n'
                "</tool_call>\n\n"
                "Workflow: Write code → Test it → Fix errors → Submit solution.\n"
                "You MUST call submit_python_solution with your final code."
            ),
            "user_prefix": "Solve this problem:\n\n",
            "user_suffix": "",
        },
        "llama": {
            "system": (
                "You are a Python coding assistant. You have these tools:\n\n"
                "1. run_python_code(code) - Execute Python code\n"
                "2. upsert_file(filename, content) - Create/update files\n"
                "3. read_file(filename) - Read files\n"
                "4. run_submission_tests(filename) - Test solution\n"
                "5. submit_python_solution(code) - Submit final solution (REQUIRED)\n\n"
                "Output tool calls as JSON:\n"
                '{"name": "<tool>", "arguments": {<args>}}\n\n'
                "You MUST submit your final solution with submit_python_solution."
            ),
            "user_prefix": "Solve this:\n\n",
            "user_suffix": "",
        },
        "gemma": {
            "system": (
                "You are a coding assistant with tools:\n"
                "1. run_python_code(code=\"...\") - Execute code\n"
                "2. upsert_file(filename=\"...\", content=\"...\") - Write file\n"
                "3. read_file(filename=\"...\") - Read file\n"
                "4. run_submission_tests(filename=\"...\") - Test solution\n"
                "5. submit_python_solution(code=\"...\") - Submit (REQUIRED)\n\n"
                "Use format: [tool_name(param=\"value\")]\n"
                "Workflow: Write → Test → Submit.\n"
                "NO explanations. ONLY tool calls."
            ),
            "user_prefix": "",
            "user_suffix": "",
        },
    }


def _round1_prompts():
    """Round 1: Initial prompts based on model_prompts.py patterns."""
    base = _base_prompts()
    tool_sub = _tool_submission_prompts()
    full_tool = _full_tool_prompts()

    result = {}
    for m in MODELS:
        arch = m["arch"]
        result[m["id"]] = {
            "base": base.get(arch, base["qwen"]),
            "tool_submission": tool_sub.get(arch, tool_sub["qwen"]),
            "full_tool": full_tool.get(arch, full_tool["qwen"]),
        }
    return result


def _round2_prompts():
    """Round 2: Improved based on round 1 failure analysis. Loaded from file if exists."""
    path = OUT_DIR / "round2_prompts.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Fallback to round 1 if not yet created
    return _round1_prompts()


def _round3_prompts():
    """Round 3: Cross-benchmark validated. Loaded from file if exists."""
    path = OUT_DIR / "round3_prompts.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return _round2_prompts()


def _round4_prompts():
    """Round 4: Final prompts. Loaded from file if exists."""
    path = OUT_DIR / "round4_prompts.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return _round3_prompts()


# ── Evaluation Logic ──────────────────────────────────────────────────────

def evaluate_code_solution(problem_prompt: str, model_response: str, test_cases: List[str]) -> Dict[str, Any]:
    """Evaluate a code solution against test cases.

    Extracts code from the response, runs it, then runs test cases.
    """
    import subprocess
    import tempfile

    # Extract code from response
    code = extract_code_from_response(model_response)
    if not code:
        return {"passed": False, "error": "No code extracted", "code": None}

    # Run test cases
    test_code = code + "\n\n"
    for tc in test_cases:
        test_code += tc + "\n"

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=10
            )
            os.unlink(f.name)

        if result.returncode == 0:
            return {"passed": True, "code": code, "output": result.stdout}
        else:
            return {"passed": False, "code": code, "error": result.stderr[:500]}
    except subprocess.TimeoutExpired:
        return {"passed": False, "code": code, "error": "Timeout"}
    except Exception as e:
        return {"passed": False, "code": code, "error": str(e)}


def extract_code_from_response(response: str) -> Optional[str]:
    """Extract Python code from a model response."""
    # Step 0: Strip <think>...</think> blocks (Qwen/DeepSeek reasoning)
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    # Also strip incomplete thinking blocks (model hit max_tokens mid-think)
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL).strip()

    # Use cleaned text for extraction, fall back to raw if cleaned is empty
    texts_to_try = [cleaned, response] if cleaned else [response]

    for text in texts_to_try:
        # Try tool call extraction first (highest priority)
        te = ToolExtractor()
        tool_calls, _ = te.extract_tools(text)
        if tool_calls:
            for tc in tool_calls:
                params = tc.get("parameters", tc.get("arguments", {}))
                code = params.get("code", "")
                if code and len(code) > 10:
                    return code

        # Try ```python blocks
        matches = re.findall(r'```python\s*(.*?)```', text, re.DOTALL)
        if matches:
            return matches[-1].strip()

        # Try to find function definition
        match = re.search(r'(def \w+\([^)]*\):.*?)(?=\n(?:def |\n\n[A-Z]|\Z))', text, re.DOTALL)
        if match:
            return match.group(1).strip()

    return None


def evaluate_gsm8k_solution(problem_prompt: str, model_response: str,
                             ground_truth: float) -> Dict[str, Any]:
    """Evaluate a GSM8K math solution."""
    from pocket_agent_cli.datasets.gsm8k import extract_numeric_answer, numeric_answers_match

    extracted = extract_numeric_answer(model_response)
    if extracted is None:
        return {"passed": False, "error": "No numeric answer found", "extracted": None}

    passed = numeric_answers_match(extracted, ground_truth)
    return {"passed": passed, "extracted": extracted, "expected": ground_truth}


# ── Runner ────────────────────────────────────────────────────────────────

def format_problem_prompt(problem, prompt_config: Dict[str, str]) -> str:
    """Format a problem into a user message, including test cases for function name hints."""
    text = problem.prompt

    # Include test cases so the model knows expected function names
    if hasattr(problem, "test_cases") and problem.test_cases:
        # Filter out GSM8K-style test cases (EXPECTED_ANSWER:)
        code_tests = [t for t in problem.test_cases if not t.startswith("EXPECTED_ANSWER")]
        if code_tests:
            test_str = "\n".join(code_tests[:3])
            text += f"\n\nTest cases:\n{test_str}"

    return prompt_config.get("user_prefix", "") + text + prompt_config.get("user_suffix", "")


def run_single_problem(
    service: MLXInferenceService,
    prompt_config: Dict[str, str],
    problem,
    mode: str,
    tools: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Run a single problem through the model and return the response."""
    messages = []

    # System message
    if prompt_config.get("system"):
        messages.append({"role": "system", "content": prompt_config["system"]})

    # User message with test cases included
    user_content = format_problem_prompt(problem, prompt_config)
    messages.append({"role": "user", "content": user_content})

    t0 = time.time()

    if mode == "base":
        # No tools, just generate
        full_text = ""
        raw_text = ""
        metrics = {}
        for chunk in service.generate(messages, stream=True):
            full_text += chunk["token"]
            raw_text += chunk.get("raw_token", chunk["token"])
            metrics = chunk["metrics"]
        elapsed = time.time() - t0
        # Use raw_text if filtered text is empty (thinking models)
        response = full_text.strip() if full_text.strip() else raw_text.strip()
        return {
            "response": response,
            "tool_calls": None,
            "elapsed_s": elapsed,
            "tokens": metrics.get("tokens", 0),
        }
    else:
        # With tools
        tool_defs = tools or _default_code_tools()
        response_text, tool_calls, metrics = service.generate_with_tools(
            messages, tool_defs
        )
        elapsed = time.time() - t0
        return {
            "response": response_text or "",
            "tool_calls": tool_calls,
            "elapsed_s": elapsed,
            "tokens": metrics.get("tokens", 0),
        }


def _default_code_tools():
    """Default tool definitions for code benchmarks."""
    return [
        {"type": "function", "function": {
            "name": "run_python_code",
            "description": "Execute Python code and return output",
            "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code"}}, "required": ["code"]},
        }},
        {"type": "function", "function": {
            "name": "submit_python_solution",
            "description": "Submit your final Python solution",
            "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Complete Python code"}}, "required": ["code"]},
        }},
    ]


def run_evaluation(
    round_num: int,
    n_problems: int = 20,
    dataset_name: str = "mbpp",
    models_filter: Optional[List[str]] = None,
    modes_filter: Optional[List[str]] = None,
):
    """Run prompt evaluation for a given round."""
    prompts = get_prompts(round_num)
    results = {
        "round": round_num,
        "n_problems": n_problems,
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "model_results": {},
    }

    # Load dataset
    ds_cls = DatasetRegistry.get(dataset_name)
    if ds_cls is None:
        print(f"Dataset '{dataset_name}' not found!")
        return results

    from pocket_agent_cli.config import DATA_DIR
    ds = ds_cls(DATA_DIR)
    if not ds.is_downloaded():
        print(f"Downloading {dataset_name}...")
        ds.download()

    problems = ds.load(split="test", limit=n_problems)
    print(f"Loaded {len(problems)} {dataset_name} problems")

    service = MLXInferenceService()

    for model_def in MODELS:
        model_id = model_def["id"]
        if models_filter and model_id not in models_filter:
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_def['name']}")
        print(f"{'='*60}")

        model = Model(
            id=model_id,
            name=model_def["name"],
            architecture=model_def["arch"],
            downloaded=True,
            default_version="Q4_K_M",
            current_version="Q4_K_M",
        )
        # Thinking models (Qwen, DeepSeek) need 8k+ tokens for reasoning chains
        is_thinking = model_def["arch"] == "qwen"
        config = InferenceConfig(
            temperature=0.7,
            max_tokens=8192 if is_thinking else 2048,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            context_length=16384,
            jinja=True,
        )

        print(f"Loading {model_id}...")
        service.load_model(model, config)

        model_results = {}

        for mode in MODES:
            if modes_filter and mode not in modes_filter:
                continue

            prompt_config = prompts.get(model_id, {}).get(mode, {})
            if not prompt_config:
                print(f"  [{mode}] No prompt config, skipping")
                continue

            print(f"\n  Mode: {mode}")
            passed = 0
            failures = []
            mode_results = []

            for i, problem in enumerate(problems):
                try:
                    result = run_single_problem(
                        service, prompt_config, problem, mode
                    )

                    # Evaluate
                    if dataset_name == "gsm8k":
                        gt = problem.metadata.get("ground_truth_answer", 0)
                        eval_result = evaluate_gsm8k_solution(
                            problem.prompt, result["response"], gt
                        )
                    else:
                        eval_result = evaluate_code_solution(
                            problem.prompt, result["response"], problem.test_cases
                        )

                    result["evaluation"] = eval_result
                    result["problem_id"] = problem.task_id
                    mode_results.append(result)

                    if eval_result["passed"]:
                        passed += 1
                    else:
                        failures.append({
                            "problem_id": problem.task_id,
                            "error": eval_result.get("error", "wrong answer"),
                            "response_preview": result["response"][:200],
                        })

                    status = "PASS" if eval_result["passed"] else "FAIL"
                    print(f"    [{i+1}/{len(problems)}] {problem.task_id}: {status} ({result['elapsed_s']:.1f}s, {result['tokens']} tok)")

                except Exception as e:
                    print(f"    [{i+1}/{len(problems)}] {problem.task_id}: ERROR - {e}")
                    mode_results.append({
                        "problem_id": problem.task_id,
                        "error": str(e),
                        "evaluation": {"passed": False, "error": str(e)},
                    })

            pass_rate = passed / len(problems) if problems else 0
            print(f"  {mode} Pass@1: {passed}/{len(problems)} = {pass_rate:.1%}")

            model_results[mode] = {
                "pass_rate": pass_rate,
                "passed": passed,
                "total": len(problems),
                "failures": failures[:10],  # Keep top 10 failures for analysis
                "results": mode_results,
            }

        results["model_results"][model_id] = model_results

        service.unload_model()
        gc.collect()

        # Cooling break between models (15 min for thermals)
        remaining_models = [m for m in MODELS if m["id"] not in results["model_results"]]
        if remaining_models:
            cool_mins = int(os.environ.get("COOL_MINUTES", "15"))
            if cool_mins > 0:
                print(f"\n  Cooling break: {cool_mins} min before next model...")
                time.sleep(cool_mins * 60)

    # Save results
    out_file = OUT_DIR / f"round{round_num}_{dataset_name}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"ROUND {round_num} SUMMARY ({dataset_name}, {n_problems} problems)")
    print(f"{'='*60}")
    print(f"{'Model':<35} {'Base':>6} {'ToolSub':>8} {'FullTool':>9}")
    print("-" * 60)
    for model_def in MODELS:
        mid = model_def["id"]
        if mid not in results["model_results"]:
            continue
        mr = results["model_results"][mid]
        base_r = mr.get("base", {}).get("pass_rate", 0)
        ts_r = mr.get("tool_submission", {}).get("pass_rate", 0)
        ft_r = mr.get("full_tool", {}).get("pass_rate", 0)
        print(f"{mid:<35} {base_r:>5.0%} {ts_r:>7.0%} {ft_r:>8.0%}")

    return results


def analyze_failures(results: Dict) -> Dict[str, Any]:
    """Analyze failure patterns across models and modes."""
    analysis = {}
    for model_id, model_results in results.get("model_results", {}).items():
        model_analysis = {}
        for mode, mode_data in model_results.items():
            failures = mode_data.get("failures", [])
            patterns = {
                "no_code": 0,
                "syntax_error": 0,
                "wrong_answer": 0,
                "no_tool_call": 0,
                "timeout": 0,
                "other": 0,
            }
            for f in failures:
                err = f.get("error", "")
                if "No code" in err or "No tool" in err:
                    patterns["no_code"] += 1
                elif "SyntaxError" in err or "IndentationError" in err:
                    patterns["syntax_error"] += 1
                elif "assert" in err.lower():
                    patterns["wrong_answer"] += 1
                elif "Timeout" in err:
                    patterns["timeout"] += 1
                elif "tool" in err.lower():
                    patterns["no_tool_call"] += 1
                else:
                    patterns["other"] += 1
            model_analysis[mode] = {
                "total_failures": len(failures),
                "patterns": patterns,
            }
        analysis[model_id] = model_analysis
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt optimization")
    parser.add_argument("--round", type=int, default=1, help="Optimization round (1-4)")
    parser.add_argument("--problems", type=int, default=20, help="Number of problems")
    parser.add_argument("--dataset", default="mbpp", help="Dataset (mbpp, humaneval, gsm8k)")
    parser.add_argument("--models", nargs="*", help="Filter to specific model IDs")
    parser.add_argument("--modes", nargs="*", help="Filter to specific modes")
    args = parser.parse_args()

    results = run_evaluation(
        round_num=args.round,
        n_problems=args.problems,
        dataset_name=args.dataset,
        models_filter=args.models,
        modes_filter=args.modes,
    )
