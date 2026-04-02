#!/usr/bin/env python3
"""Batched benchmark runner for MLX.

Uses MLX's batch_generate for maximum throughput on Apple Silicon,
with battery guard, cooling breaks, and support for all 3 modes
(base, tool_submission, full_tool).

Usage:
    cd /path/to/pocket-agent/cli
    HF_TOKEN=<token> python scripts/run_benchmarks.py \
        --dataset mbpp --problems 30 \
        --models qwen-3-4b llama-3.2-3b-instruct \
        --modes base tool_submission full_tool \
        --batch-size 100 --concurrency 10
"""

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make pocket_agent_cli importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from pocket_agent_cli.services.mlx_inference_service import MLXInferenceService
from pocket_agent_cli.config import InferenceConfig, Model
from pocket_agent_cli.utils.tool_extractor import ToolExtractor
from pocket_agent_cli.utils.battery_guard import wait_for_battery
from pocket_agent_cli.utils.optimized_prompts import get_optimized_prompt
from pocket_agent_cli.datasets.registry import DatasetRegistry
# Register datasets
from pocket_agent_cli.datasets import mbpp, humaneval, gsm8k

# ── Constants ────────────────────────────────────────────────────────────────

MODELS = [
    {"id": "qwen-3-4b", "name": "Qwen 3 4B", "arch": "qwen"},
    {"id": "qwen-3-0.6b", "name": "Qwen 3 0.6B", "arch": "qwen"},
    {"id": "llama-3.2-3b-instruct", "name": "Llama 3.2 3B", "arch": "llama"},
    {"id": "deepseek-r1-distill-qwen-1.5b", "name": "DeepSeek R1 1.5B", "arch": "qwen"},
    {"id": "gemma-3n-e2b-it", "name": "Gemma 3n E2B", "arch": "gemma"},
]

MODEL_MAP = {m["id"]: m for m in MODELS}

MODES = ["base", "tool_submission", "full_tool"]


# ── Prompt Formatting ────────────────────────────────────────────────────────

def format_problem_prompt(problem, prompt_config: Dict[str, str]) -> str:
    """Format a problem into a user message, including test cases for hints."""
    text = problem.prompt

    # Include test cases so the model knows expected function names
    if hasattr(problem, "test_cases") and problem.test_cases:
        code_tests = [t for t in problem.test_cases if not t.startswith("EXPECTED_ANSWER")]
        if code_tests:
            test_str = "\n".join(code_tests[:3])
            text += f"\n\nTest cases:\n{test_str}"

    return prompt_config.get("user_prefix", "") + text + prompt_config.get("user_suffix", "")


def build_messages(problem, prompt_config: Dict[str, str]) -> List[Dict[str, str]]:
    """Build chat messages for a problem."""
    messages = []
    if prompt_config.get("system"):
        messages.append({"role": "system", "content": prompt_config["system"]})
    user_content = format_problem_prompt(problem, prompt_config)
    messages.append({"role": "user", "content": user_content})
    return messages


# ── Code Extraction & Evaluation ─────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    """Strip <think>...</think> blocks from text, including incomplete ones."""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Also handle incomplete thinking blocks (model hit max_tokens mid-think)
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL).strip()
    return cleaned


def extract_code_from_response(response: str) -> Optional[str]:
    """Extract Python code from a model response."""
    cleaned = strip_thinking(response)
    texts_to_try = [cleaned, response] if cleaned else [response]

    for text in texts_to_try:
        # Try tool call extraction first
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
        match = re.search(
            r'(def \w+\([^)]*\):.*?)(?=\n(?:def |\n\n[A-Z]|\Z))',
            text, re.DOTALL,
        )
        if match:
            return match.group(1).strip()

    return None


def extract_code_from_tool_calls(tool_calls: List[Dict]) -> Optional[str]:
    """Extract code from parsed tool calls (submit_python_solution preferred)."""
    if not tool_calls:
        return None

    # Prefer submit_python_solution
    for tc in tool_calls:
        if tc.get("name") == "submit_python_solution":
            params = tc.get("parameters", tc.get("arguments", {}))
            code = params.get("code", "")
            if code and len(code) > 5:
                return code

    # Fall back to any tool with a code param
    for tc in tool_calls:
        params = tc.get("parameters", tc.get("arguments", {}))
        code = params.get("code", "")
        if code and len(code) > 5:
            return code

    return None


def evaluate_code_solution(code: str, test_cases: List[str]) -> Dict[str, Any]:
    """Run code + test cases in a subprocess."""
    if not code:
        return {"passed": False, "error": "No code extracted", "code": None}

    test_code = code + "\n\n"
    for tc in test_cases:
        test_code += tc + "\n"

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=10,
            )
            os.unlink(f.name)

        if result.returncode == 0:
            return {"passed": True, "code": code, "output": result.stdout[:500]}
        else:
            return {"passed": False, "code": code, "error": result.stderr[:500]}
    except subprocess.TimeoutExpired:
        return {"passed": False, "code": code, "error": "Timeout (10s)"}
    except Exception as e:
        return {"passed": False, "code": code, "error": str(e)}


def evaluate_gsm8k_solution(response: str, ground_truth: float) -> Dict[str, Any]:
    """Evaluate a GSM8K math solution."""
    from pocket_agent_cli.datasets.gsm8k import extract_numeric_answer, numeric_answers_match

    cleaned = strip_thinking(response)
    extracted = extract_numeric_answer(cleaned)
    if extracted is None:
        # Try the raw response too
        extracted = extract_numeric_answer(response)
    if extracted is None:
        return {"passed": False, "error": "No numeric answer found", "extracted": None}

    passed = numeric_answers_match(extracted, ground_truth)
    return {"passed": passed, "extracted": extracted, "expected": ground_truth}


def evaluate_problem(problem, response: str, tool_calls, dataset_name: str) -> Dict[str, Any]:
    """Evaluate a single problem result."""
    if dataset_name == "gsm8k":
        gt = problem.metadata.get("ground_truth_answer", 0)
        return evaluate_gsm8k_solution(response, gt)

    # Code problem — try tool calls first, then raw response
    code = None
    if tool_calls:
        code = extract_code_from_tool_calls(tool_calls)
    if not code:
        code = extract_code_from_response(response)

    return evaluate_code_solution(code, problem.test_cases)


# ── Tool execution for full_tool mode ────────────────────────────────────────

def execute_tool(tool_call: Dict, problem) -> str:
    """Execute a tool call and return the observation string."""
    name = tool_call.get("name", "")
    params = tool_call.get("parameters", tool_call.get("arguments", {}))

    if name == "run_python_code":
        code = params.get("code", "")
        if not code:
            return "Error: no code provided"
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                f.flush()
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True, text=True, timeout=10,
                )
                os.unlink(f.name)
            output = result.stdout[:1000]
            if result.returncode != 0:
                output += "\nError:\n" + result.stderr[:500]
            return output if output.strip() else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: execution timed out (10s)"
        except Exception as e:
            return f"Error: {e}"

    elif name == "submit_python_solution":
        code = params.get("code", "")
        if not code:
            return "Error: no code provided to submit"
        return f"SUBMITTED: solution accepted ({len(code)} chars)"

    elif name == "upsert_file":
        return f"File '{params.get('filename', '?')}' written."

    elif name == "read_file":
        return f"File not available in benchmark sandbox."

    elif name == "run_submission_tests":
        # Actually run the tests for the problem
        code = params.get("solution", params.get("code", ""))
        if not code:
            return "Error: no solution provided"
        result = evaluate_code_solution(code, problem.test_cases)
        if result["passed"]:
            return "All tests passed!"
        else:
            return f"Tests failed: {result.get('error', 'unknown')}"

    else:
        return f"Unknown tool: {name}"


# ── Default tool definitions ─────────────────────────────────────────────────

BENCHMARK_TOOLS = [
    {"type": "function", "function": {
        "name": "run_python_code",
        "description": "Execute Python code and return output",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code"}},
            "required": ["code"],
        },
    }},
    {"type": "function", "function": {
        "name": "submit_python_solution",
        "description": "Submit your final Python solution",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Complete Python code"}},
            "required": ["code"],
        },
    }},
]


# ── Execution Strategies ─────────────────────────────────────────────────────

def run_base_batched(
    service: MLXInferenceService,
    problems: List,
    prompt_config: Dict[str, str],
    dataset_name: str,
    max_tokens: int,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Base mode: pure batch generation, extract code, run tests."""
    results = []
    total = len(problems)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_problems = problems[batch_start:batch_end]
        print(f"    Batch {batch_start//batch_size + 1}: problems {batch_start+1}-{batch_end}")

        # Build message lists for the batch
        prompts = [build_messages(p, prompt_config) for p in batch_problems]

        t0 = time.time()
        batch_results = service.batch_generate(prompts, max_tokens=max_tokens)
        batch_time = time.time() - t0

        for i, (problem, br) in enumerate(zip(batch_problems, batch_results)):
            response = br["text"]
            raw_response = br.get("raw_text", response)

            evaluation = evaluate_problem(problem, response, br.get("tool_calls"), dataset_name)

            status = "PASS" if evaluation["passed"] else "FAIL"
            idx = batch_start + i + 1
            print(f"      [{idx}/{total}] {problem.task_id}: {status}")

            results.append({
                "problem_id": problem.task_id,
                "response": response,
                "raw_response": raw_response,
                "tool_calls": br.get("tool_calls"),
                "evaluation": evaluation,
                "metrics": {
                    "batch_time_s": batch_time,
                    "tokens": br["metrics"].get("tokens", 0),
                    "generation_tps": br["metrics"].get("generation_tps", 0),
                    "prompt_tps": br["metrics"].get("prompt_tps", 0),
                    "peak_memory_gb": br["metrics"].get("peak_memory_gb", 0),
                },
            })

    return results


def run_tool_submission_batched(
    service: MLXInferenceService,
    problems: List,
    prompt_config: Dict[str, str],
    dataset_name: str,
    max_tokens: int,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Tool submission mode: batch generate, parse tool calls from responses."""
    results = []
    total = len(problems)
    tool_extractor = ToolExtractor()

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_problems = problems[batch_start:batch_end]
        print(f"    Batch {batch_start//batch_size + 1}: problems {batch_start+1}-{batch_end}")

        prompts = [build_messages(p, prompt_config) for p in batch_problems]

        t0 = time.time()
        batch_results = service.batch_generate(prompts, max_tokens=max_tokens)
        batch_time = time.time() - t0

        model_arch = service.current_model.architecture if service.current_model else None

        for i, (problem, br) in enumerate(zip(batch_problems, batch_results)):
            response = br["text"]
            raw_response = br.get("raw_text", response)

            # Parse tool calls explicitly from the response
            tool_calls = br.get("tool_calls")
            if not tool_calls:
                cleaned = strip_thinking(response)
                tool_calls, _ = tool_extractor.extract_tools(cleaned, model_arch)
                # Normalize arguments -> parameters
                for tc in (tool_calls or []):
                    if "arguments" in tc and "parameters" not in tc:
                        tc["parameters"] = tc.pop("arguments")

            evaluation = evaluate_problem(problem, response, tool_calls, dataset_name)

            status = "PASS" if evaluation["passed"] else "FAIL"
            idx = batch_start + i + 1
            print(f"      [{idx}/{total}] {problem.task_id}: {status}")

            results.append({
                "problem_id": problem.task_id,
                "response": response,
                "raw_response": raw_response,
                "tool_calls": tool_calls if tool_calls else None,
                "evaluation": evaluation,
                "metrics": {
                    "batch_time_s": batch_time,
                    "tokens": br["metrics"].get("tokens", 0),
                    "generation_tps": br["metrics"].get("generation_tps", 0),
                    "prompt_tps": br["metrics"].get("prompt_tps", 0),
                    "peak_memory_gb": br["metrics"].get("peak_memory_gb", 0),
                },
            })

    return results


def run_full_tool_parallel(
    service: MLXInferenceService,
    problems: List,
    prompt_config: Dict[str, str],
    dataset_name: str,
    max_tokens: int,
    concurrency: int,
    max_iterations: int = 5,
    timeout_s: int = 120,
) -> List[Dict[str, Any]]:
    """Full tool / agentic mode: parallel sequential loops per problem.

    Each problem runs its own generate -> parse tool -> execute -> feedback loop.
    Multiple problems run concurrently via ThreadPoolExecutor.
    """
    # We need a lock for the MLX service since it's not thread-safe
    service_lock = threading.Lock()
    total = len(problems)
    results_map: Dict[int, Dict] = {}

    def run_agentic_problem(idx: int, problem) -> Dict[str, Any]:
        """Run one problem through the agentic loop."""
        messages = build_messages(problem, prompt_config)
        tool_extractor = ToolExtractor()
        model_arch = service.current_model.architecture if service.current_model else None

        all_tool_calls = []
        submitted_code = None
        problem_start = time.time()
        total_tokens = 0
        iterations_run = 0

        for iteration in range(max_iterations):
            iterations_run = iteration + 1
            elapsed = time.time() - problem_start
            if elapsed > timeout_s:
                break

            # Generate — must lock since MLX model is shared
            with service_lock:
                try:
                    response_text, tool_calls, metrics = service.generate_with_tools(
                        messages, BENCHMARK_TOOLS, max_tokens=max_tokens,
                    )
                except Exception as e:
                    response_text = f"Error: {e}"
                    tool_calls = None
                    metrics = {}

            total_tokens += metrics.get("tokens", 0)

            # Normalize tool calls
            if tool_calls:
                for tc in tool_calls:
                    if "arguments" in tc and "parameters" not in tc:
                        tc["parameters"] = tc.pop("arguments")
                all_tool_calls.extend(tool_calls)
            else:
                # Try extracting from text
                cleaned = strip_thinking(response_text)
                parsed, _ = tool_extractor.extract_tools(cleaned, model_arch)
                if parsed:
                    for tc in parsed:
                        if "arguments" in tc and "parameters" not in tc:
                            tc["parameters"] = tc.pop("arguments")
                    tool_calls = parsed
                    all_tool_calls.extend(parsed)

            # Check if submit was called
            if tool_calls:
                for tc in tool_calls:
                    if tc.get("name") == "submit_python_solution":
                        params = tc.get("parameters", {})
                        submitted_code = params.get("code", "")
                        break

            if submitted_code:
                break

            # Execute tools and feed observations back
            if tool_calls:
                # Add assistant response
                messages.append({"role": "assistant", "content": response_text})

                observations = []
                for tc in tool_calls:
                    if tc.get("name") == "submit_python_solution":
                        continue  # Already handled above
                    obs = execute_tool(tc, problem)
                    observations.append(f"[{tc['name']}] {obs}")

                if observations:
                    obs_text = "\n".join(observations)
                    messages.append({"role": "user", "content": f"Tool output:\n{obs_text}"})
            else:
                # No tool calls — model just gave text. Try one more iteration.
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": "Please call submit_python_solution with your final code.",
                })

        total_time = time.time() - problem_start

        # Evaluate
        full_response = response_text
        evaluation = evaluate_problem(
            problem, full_response,
            all_tool_calls if all_tool_calls else None,
            dataset_name,
        )

        status = "PASS" if evaluation["passed"] else "FAIL"
        print(f"      [{idx+1}/{total}] {problem.task_id}: {status} ({len(all_tool_calls)} tools, {total_time:.1f}s)")

        return {
            "problem_id": problem.task_id,
            "response": full_response,
            "raw_response": full_response,
            "tool_calls": all_tool_calls if all_tool_calls else None,
            "evaluation": evaluation,
            "metrics": {
                "total_time_s": total_time,
                "tokens": total_tokens,
                "iterations": iterations_run,
            },
        }

    # Run problems concurrently
    results = [None] * total
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(run_agentic_problem, i, p): i
            for i, p in enumerate(problems)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"      [{idx+1}/{total}] ERROR: {e}")
                results[idx] = {
                    "problem_id": problems[idx].task_id,
                    "response": "",
                    "raw_response": "",
                    "tool_calls": None,
                    "evaluation": {"passed": False, "error": str(e)},
                    "metrics": {},
                }

    return [r for r in results if r is not None]


# ── Main Runner ──────────────────────────────────────────────────────────────

def run_benchmarks(args) -> Dict[str, Any]:
    """Main benchmark runner."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Benchmark run: {run_id}")
    print(f"Output dir: {output_dir}")
    print(f"Dataset: {args.dataset}, problems: {args.problems}")
    print(f"Models: {args.models}")
    print(f"Modes: {args.modes}")
    print()

    # Load dataset
    ds_cls = DatasetRegistry.get(args.dataset)
    if ds_cls is None:
        print(f"ERROR: Dataset '{args.dataset}' not found!")
        print(f"Available: {DatasetRegistry.list_names()}")
        sys.exit(1)

    from pocket_agent_cli.config import DATA_DIR
    ds = ds_cls(DATA_DIR)
    if not ds.is_downloaded():
        print(f"Downloading {args.dataset}...")
        ds.download()

    problems = ds.load(split="test", limit=args.problems)
    print(f"Loaded {len(problems)} problems\n")

    # Resolve models
    selected_models = []
    for model_id in args.models:
        if model_id in MODEL_MAP:
            selected_models.append(MODEL_MAP[model_id])
        else:
            print(f"WARNING: Unknown model '{model_id}', skipping")
    if not selected_models:
        print("ERROR: No valid models selected")
        sys.exit(1)

    service = MLXInferenceService()
    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "n_problems": len(problems),
        "args": vars(args),
        "model_results": {},
    }

    for model_idx, model_def in enumerate(selected_models):
        model_id = model_def["id"]
        model_arch = model_def["arch"]

        print(f"\n{'='*70}")
        print(f"Model: {model_def['name']} ({model_id})")
        print(f"{'='*70}")

        # Battery guard before each model
        wait_for_battery(args.min_battery)

        # Determine max_tokens and context_length based on architecture
        is_thinking = model_arch == "qwen"
        max_tokens = args.max_tokens_thinking if is_thinking else args.max_tokens_normal
        context_length = 16384 if is_thinking else 8192

        # Create model config
        model = Model(
            id=model_id,
            name=model_def["name"],
            architecture=model_arch,
            downloaded=True,
            default_version=args.version,
            current_version=args.version,
        )
        config = InferenceConfig(
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            context_length=context_length,
            jinja=True,
        )

        print(f"Loading model (version={args.version}, max_tokens={max_tokens}, ctx={context_length})...")
        service.load_model(model, config)

        model_summary = {}

        for mode in args.modes:
            if mode not in MODES:
                print(f"  WARNING: Unknown mode '{mode}', skipping")
                continue

            print(f"\n  Mode: {mode}")
            print(f"  {'-'*50}")

            # Get optimized prompt config
            try:
                prompt_config = get_optimized_prompt(model_id, mode)
            except KeyError as e:
                print(f"  WARNING: {e}, skipping")
                continue

            t0 = time.time()

            if mode == "base":
                results = run_base_batched(
                    service, problems, prompt_config, args.dataset,
                    max_tokens, args.batch_size,
                )
            elif mode == "tool_submission":
                results = run_tool_submission_batched(
                    service, problems, prompt_config, args.dataset,
                    max_tokens, args.batch_size,
                )
            elif mode == "full_tool":
                results = run_full_tool_parallel(
                    service, problems, prompt_config, args.dataset,
                    max_tokens, args.concurrency,
                )
            else:
                continue

            mode_time = time.time() - t0
            passed = sum(1 for r in results if r["evaluation"]["passed"])
            pass_rate = passed / len(results) if results else 0

            print(f"\n  {mode} — {passed}/{len(results)} = {pass_rate:.1%} ({mode_time:.1f}s)")

            model_summary[mode] = {
                "pass_rate": pass_rate,
                "passed": passed,
                "total": len(results),
                "time_s": mode_time,
            }

            # Save per-model-mode JSONL
            jsonl_path = output_dir / f"{model_id}_{mode}.jsonl"
            with open(jsonl_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, default=str) + "\n")
            print(f"  Saved: {jsonl_path}")

        summary["model_results"][model_id] = model_summary

        # Unload model and clean up
        service.unload_model()
        gc.collect()

        # Cooling break between models (skip after last model)
        if model_idx < len(selected_models) - 1 and args.cool_minutes > 0:
            print(f"\n  Cooling break: {args.cool_minutes} min...")
            time.sleep(args.cool_minutes * 60)

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_path}")

    # Print final summary table
    print(f"\n{'='*70}")
    print(f"BENCHMARK SUMMARY — {args.dataset} ({len(problems)} problems)")
    print(f"{'='*70}")
    header = f"{'Model':<35}"
    for mode in args.modes:
        header += f" {mode:>14}"
    print(header)
    print("-" * 70)

    for model_def in selected_models:
        mid = model_def["id"]
        mr = summary["model_results"].get(mid, {})
        row = f"{mid:<35}"
        for mode in args.modes:
            md = mr.get(mode, {})
            rate = md.get("pass_rate", 0)
            passed = md.get("passed", 0)
            total = md.get("total", 0)
            row += f" {passed:>3}/{total:<3} {rate:>5.0%}"
        print(row)

    print(f"\nResults in: {output_dir}")
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batched benchmark runner for MLX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", default="mbpp",
        choices=["mbpp", "humaneval", "gsm8k"],
        help="Dataset to benchmark on",
    )
    parser.add_argument(
        "--problems", type=int, default=30,
        help="Number of problems to run",
    )
    parser.add_argument(
        "--models", nargs="*",
        default=["qwen-3-4b"],
        help="Model IDs to benchmark",
    )
    parser.add_argument(
        "--modes", nargs="*",
        default=["base", "tool_submission", "full_tool"],
        help="Execution modes to run",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Batch size for base/tool_submission modes",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Concurrent problems for full_tool mode",
    )
    parser.add_argument(
        "--cool-minutes", type=int, default=15,
        help="Cooling break between models (minutes)",
    )
    parser.add_argument(
        "--min-battery", type=int, default=90,
        help="Minimum battery percent before starting a model",
    )
    parser.add_argument(
        "--max-tokens-thinking", type=int, default=8192,
        help="Max tokens for thinking models (qwen arch)",
    )
    parser.add_argument(
        "--max-tokens-normal", type=int, default=2048,
        help="Max tokens for non-thinking models",
    )
    parser.add_argument(
        "--version", default="Q4_K_M",
        help="Model quantization version",
    )
    parser.add_argument(
        "--output-dir", default="data/results",
        help="Output directory for results",
    )

    args = parser.parse_args()
    run_benchmarks(args)


if __name__ == "__main__":
    main()
