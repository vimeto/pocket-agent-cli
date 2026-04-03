#!/usr/bin/env python3
"""MLX benchmark sweep with full per-problem metrics.

Two strategies:
1. batch=N for throughput (accuracy measurement)
2. batch=1 streaming for per-problem metrics (TTFT, TPS, energy, token breakdown)

Both run for each model×mode. Uses battery guard and cooling breaks.

Usage:
    HF_TOKEN=<token> python scripts/run_mlx_sweep.py \
        --dataset mbpp --problems 150 --batch-size 50
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from pocket_agent_cli.services.mlx_inference_service import MLXInferenceService
from pocket_agent_cli.config import InferenceConfig, Model
from pocket_agent_cli.utils.tool_extractor import ToolExtractor
from pocket_agent_cli.utils.optimized_prompts import get_optimized_prompt
from pocket_agent_cli.utils.battery_guard import wait_for_battery
from pocket_agent_cli.datasets.registry import DatasetRegistry
from pocket_agent_cli.datasets import mbpp, humaneval, gsm8k

MODELS = [
    {"id": "qwen-3-4b", "name": "Qwen 3 4B", "arch": "qwen"},
    {"id": "qwen-3-0.6b", "name": "Qwen 3 0.6B", "arch": "qwen"},
    {"id": "llama-3.2-3b-instruct", "name": "Llama 3.2 3B", "arch": "llama"},
    {"id": "deepseek-r1-distill-qwen-1.5b", "name": "DeepSeek R1 1.5B", "arch": "qwen"},
    {"id": "gemma-3n-e2b-it", "name": "Gemma 3n E2B", "arch": "gemma"},
]


def build_messages(problem, prompt_config):
    messages = []
    if prompt_config.get("system"):
        messages.append({"role": "system", "content": prompt_config["system"]})
    text = problem.prompt
    if hasattr(problem, "test_cases") and problem.test_cases:
        code_tests = [t for t in problem.test_cases if not t.startswith("EXPECTED_ANSWER")]
        if code_tests:
            text += "\n\nTest cases:\n" + "\n".join(code_tests[:3])
    suffix = prompt_config.get("user_suffix", "")
    messages.append({"role": "user", "content": text + suffix})
    return messages


def strip_thinking(text):
    c = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return re.sub(r'<think>.*', '', c, flags=re.DOTALL).strip()


def extract_code(response, tool_calls=None):
    if tool_calls:
        for tc in (tool_calls if isinstance(tool_calls, list) else []):
            params = tc.get("parameters", tc.get("arguments", {}))
            code = params.get("code", "")
            if code and len(code) > 10:
                return code

    text = strip_thinking(response) or response
    te = ToolExtractor()
    tcs, _ = te.extract_tools(text)
    if tcs:
        for tc in tcs:
            params = tc.get("parameters", tc.get("arguments", {}))
            code = params.get("code", "")
            if code and len(code) > 10:
                return code

    matches = re.findall(r'```python\s*(.*?)```', text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    match = re.search(r'(def \w+\([^)]*\):.*?)(?=\n(?:def |\n\n[A-Z]|\Z))', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def evaluate_code(code, test_cases):
    if not code:
        return {"passed": False, "error": "No code extracted", "test_details": []}
    test_code = code + "\n\n" + "\n".join(test_cases)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            result = subprocess.run(
                [sys.executable, f.name], capture_output=True, text=True, timeout=10
            )
            os.unlink(f.name)
        if result.returncode == 0:
            return {"passed": True, "test_details": [{"test": t, "passed": True} for t in test_cases]}
        return {"passed": False, "error": result.stderr[:300],
                "test_details": [{"test": t, "passed": False, "error": result.stderr[:100]} for t in test_cases]}
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "Timeout", "test_details": []}
    except Exception as e:
        return {"passed": False, "error": str(e)[:200], "test_details": []}


def evaluate_gsm8k(response, ground_truth):
    from pocket_agent_cli.datasets.gsm8k import extract_numeric_answer, numeric_answers_match
    extracted = extract_numeric_answer(response)
    if extracted is None:
        return {"passed": False, "error": "No number found"}
    return {"passed": numeric_answers_match(extracted, ground_truth),
            "extracted": extracted, "expected": ground_truth}


def run_single_streaming(service, messages, max_tokens):
    """Run a single problem with streaming, capturing full metrics."""
    t0 = time.time()
    ttft = None
    tokens = 0
    thinking_tokens = 0
    regular_tokens = 0
    full_text = ""
    raw_text = ""
    metrics = {}

    for chunk in service.generate(messages, stream=True, max_tokens=max_tokens):
        if ttft is None:
            ttft = (time.time() - t0) * 1000  # ms

        full_text += chunk["token"]
        raw_text += chunk.get("raw_token", chunk["token"])
        tokens += 1

        if chunk.get("is_thinking"):
            thinking_tokens += 1
        else:
            regular_tokens += 1

        metrics = chunk.get("metrics", metrics)

    elapsed = time.time() - t0
    tps = tokens / elapsed if elapsed > 0 else 0

    return {
        "response": full_text.strip() or strip_thinking(raw_text).strip(),
        "raw_response": raw_text,
        "ttft_ms": round(ttft, 1) if ttft else None,
        "tps": round(tps, 1),
        "total_tokens": tokens,
        "thinking_tokens": thinking_tokens,
        "regular_tokens": regular_tokens,
        "thinking_ratio": round(thinking_tokens / tokens, 3) if tokens > 0 else 0,
        "elapsed_s": round(elapsed, 2),
        "generation_tps": metrics.get("generation_tps", tps),
        "prompt_tps": metrics.get("prompt_tps"),
        "prompt_tokens": metrics.get("prompt_tokens"),
        "peak_memory_gb": metrics.get("peak_memory_gb"),
        "energy_summary": metrics.get("energy_summary"),
        "current_power_watts": metrics.get("current_power_watts"),
    }


def run_sweep(args):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_cls = DatasetRegistry.get(args.dataset)
    from pocket_agent_cli.config import DATA_DIR
    ds = ds_cls(DATA_DIR)
    if not ds.is_downloaded():
        ds.download()
    problems = ds.load(split="test", limit=args.problems)
    print(f"Loaded {len(problems)} {args.dataset} problems")
    print(f"Output: {out_dir}")
    print(f"Batch size: {args.batch_size}, Streaming metrics: yes")

    selected_models = [m for m in MODELS if not args.models or m["id"] in args.models]
    selected_modes = args.modes or ["base", "tool_submission", "full_tool"]
    all_results = {}

    service = MLXInferenceService()

    for model_idx, model_def in enumerate(selected_models):
        model_id = model_def["id"]
        is_thinking = model_def["arch"] == "qwen"
        max_tokens = args.max_tokens_thinking if is_thinking else args.max_tokens_normal

        wait_for_battery(args.min_battery)

        print(f"\n{'='*60}")
        print(f"Model {model_idx+1}/{len(selected_models)}: {model_def['name']}")
        print(f"  max_tokens={max_tokens}, thinking={is_thinking}")
        print(f"{'='*60}")

        model = Model(id=model_id, name=model_def["name"], architecture=model_def["arch"],
                      downloaded=True, default_version="Q4_K_M", current_version="Q4_K_M")
        config = InferenceConfig(
            temperature=0.7, max_tokens=max_tokens, top_p=0.9, top_k=40,
            repeat_penalty=1.1, context_length=16384 if is_thinking else 8192, jinja=True,
        )

        service.load_model(model, config)
        model_results = {}

        for mode in selected_modes:
            prompt_config = get_optimized_prompt(model_id, mode)
            print(f"\n  Mode: {mode}")
            print(f"  {'-'*50}")

            results = []
            passed = 0
            t_mode_start = time.time()

            for i, problem in enumerate(problems):
                messages = build_messages(problem, prompt_config)

                t0 = time.time()

                if mode == "full_tool":
                    # Agentic loop (sequential per problem)
                    result = run_agentic_problem(service, messages, problem, model_def,
                                                 prompt_config, args.dataset, max_tokens)
                else:
                    # Single-turn streaming
                    gen = run_single_streaming(service, messages, max_tokens)
                    code = extract_code(gen["response"], None)

                    # Also try tool extraction
                    te = ToolExtractor()
                    tcs, _ = te.extract_tools(gen.get("raw_response", gen["response"]))

                    if tcs:
                        for tc in tcs:
                            params = tc.get("parameters", tc.get("arguments", {}))
                            c = params.get("code", "")
                            if c and len(c) > 10:
                                code = c
                                break

                    if args.dataset == "gsm8k":
                        gt = problem.metadata.get("ground_truth_answer", 0)
                        evaluation = evaluate_gsm8k(gen["response"], gt)
                    else:
                        evaluation = evaluate_code(code, problem.test_cases)

                    result = {
                        "problem_id": problem.task_id,
                        "passed": evaluation["passed"],
                        "error": evaluation.get("error"),
                        "response": gen["response"][:500],
                        "tool_calls": [{"name": tc.get("name"), "args_preview": str(tc.get("parameters", {}))[:100]} for tc in (tcs or [])],
                        "evaluation": evaluation,
                        "metrics": {
                            "ttft_ms": gen["ttft_ms"],
                            "tps": gen["tps"],
                            "total_tokens": gen["total_tokens"],
                            "thinking_tokens": gen["thinking_tokens"],
                            "regular_tokens": gen["regular_tokens"],
                            "thinking_ratio": gen["thinking_ratio"],
                            "elapsed_s": gen["elapsed_s"],
                            "generation_tps": gen.get("generation_tps"),
                            "prompt_tps": gen.get("prompt_tps"),
                            "prompt_tokens": gen.get("prompt_tokens"),
                            "peak_memory_gb": gen.get("peak_memory_gb"),
                            "energy_summary": gen.get("energy_summary"),
                            "power_watts": gen.get("current_power_watts"),
                        },
                        "iterations": 1,
                        "tool_call_count": len(tcs or []),
                    }

                results.append(result)
                if result["passed"]:
                    passed += 1

                status = "PASS" if result["passed"] else "FAIL"
                ttft = result.get("metrics", {}).get("ttft_ms", "?")
                tps = result.get("metrics", {}).get("tps", "?")
                tok = result.get("metrics", {}).get("total_tokens", "?")
                think = result.get("metrics", {}).get("thinking_tokens", 0)
                iters = result.get("iterations", 1)
                tc_count = result.get("tool_call_count", 0)
                elapsed = result.get("metrics", {}).get("elapsed_s", time.time() - t0)

                if (i + 1) % 5 == 0 or (i + 1) == len(problems):
                    print(f"    [{i+1}/{len(problems)}] {passed}/{i+1} passed "
                          f"(last: {status} ttft={ttft}ms tps={tps} tok={tok} "
                          f"think={think} iters={iters} tools={tc_count} {elapsed:.1f}s)")

            mode_time = time.time() - t_mode_start
            pass_rate = passed / len(problems) if problems else 0
            print(f"  {mode}: {passed}/{len(problems)} = {pass_rate:.0%} ({mode_time:.1f}s)")

            model_results[mode] = {
                "pass_rate": pass_rate, "passed": passed,
                "total": len(problems), "elapsed_s": round(mode_time, 1),
            }

            # Save per-problem JSONL
            out_file = out_dir / f"{model_id}_{mode}.jsonl"
            with open(out_file, "w") as f:
                for r in results:
                    f.write(json.dumps(r, default=str) + "\n")
            print(f"  Saved: {out_file}")

        all_results[model_id] = model_results
        service.unload_model()
        gc.collect()

        # Cooling break
        if model_idx < len(selected_models) - 1:
            cool = args.cool_minutes
            if cool > 0:
                print(f"\n  Cooling: {cool} min...")
                time.sleep(cool * 60)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ({args.dataset}, {args.problems} problems, MLX)")
    print(f"{'='*60}")
    print(f"\n{'Model':<35} {'Base':>6} {'ToolSub':>8} {'FullTool':>9}")
    print("-" * 60)
    for mid, mr in all_results.items():
        b = mr.get("base", {}).get("pass_rate", 0)
        t = mr.get("tool_submission", {}).get("pass_rate", 0)
        f = mr.get("full_tool", {}).get("pass_rate", 0)
        print(f"{mid:<35} {b:>5.0%} {t:>7.0%} {f:>8.0%}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"run_id": run_id, "dataset": args.dataset,
                    "n_problems": args.problems, "results": all_results,
                    "timestamp": datetime.now().isoformat()}, f, indent=2)
    print(f"\nResults: {out_dir}")


def run_agentic_problem(service, messages, problem, model_def, prompt_config,
                        dataset_name, max_tokens, max_iterations=5, timeout_s=300):
    """Multi-turn agentic loop with full metrics per iteration."""
    t0 = time.time()
    submitted_code = None
    iterations = 0
    total_tokens = 0
    total_thinking = 0
    total_regular = 0
    tool_calls_total = 0
    ttft_first = None
    all_tps = []

    for iteration in range(max_iterations):
        if time.time() - t0 > timeout_s:
            break
        iterations += 1

        gen = run_single_streaming(service, messages, max_tokens)
        total_tokens += gen["total_tokens"]
        total_thinking += gen["thinking_tokens"]
        total_regular += gen["regular_tokens"]
        all_tps.append(gen["tps"])
        if ttft_first is None:
            ttft_first = gen["ttft_ms"]

        content = gen.get("raw_response", gen["response"])
        messages.append({"role": "assistant", "content": content})

        # Parse tool calls from text
        te = ToolExtractor()
        cleaned = strip_thinking(content) or content
        tcs, _ = te.extract_tools(cleaned)

        if not tcs:
            code = extract_code(content)
            if code:
                submitted_code = code
            break

        for tc in (tcs or []):
            tool_calls_total += 1
            name = tc.get("name", "")
            params = tc.get("parameters", tc.get("arguments", {}))

            if name == "submit_python_solution":
                submitted_code = params.get("code", "")
                break

            # Execute run_python_code locally
            if name == "run_python_code":
                code = params.get("code", "")
                try:
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                        f.write(code)
                        f.flush()
                        r = subprocess.run([sys.executable, f.name],
                                           capture_output=True, text=True, timeout=10)
                        os.unlink(f.name)
                    obs = (r.stdout[:300] or "(no output)") if r.returncode == 0 else f"Error:\n{r.stderr[:300]}"
                except Exception as e:
                    obs = f"Error: {e}"
                messages.append({"role": "user", "content": f"Tool result ({name}):\n{obs}"})

        if submitted_code:
            break

    elapsed = time.time() - t0

    if not submitted_code:
        for msg in reversed(messages):
            c = msg.get("content", "")
            if c:
                submitted_code = extract_code(c)
                if submitted_code:
                    break

    if dataset_name == "gsm8k":
        gt = problem.metadata.get("ground_truth_answer", 0)
        evaluation = evaluate_gsm8k(messages[-1].get("content", ""), gt)
    else:
        evaluation = evaluate_code(submitted_code, problem.test_cases)

    return {
        "problem_id": problem.task_id,
        "passed": evaluation["passed"],
        "error": evaluation.get("error"),
        "evaluation": evaluation,
        "metrics": {
            "ttft_ms": ttft_first,
            "tps": round(sum(all_tps) / len(all_tps), 1) if all_tps else 0,
            "total_tokens": total_tokens,
            "thinking_tokens": total_thinking,
            "regular_tokens": total_regular,
            "thinking_ratio": round(total_thinking / total_tokens, 3) if total_tokens > 0 else 0,
            "elapsed_s": round(elapsed, 2),
        },
        "iterations": iterations,
        "tool_call_count": tool_calls_total,
    }


def main():
    parser = argparse.ArgumentParser(description="MLX benchmark sweep with full metrics")
    parser.add_argument("--dataset", default="mbpp")
    parser.add_argument("--problems", type=int, default=150)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--modes", nargs="*", default=["base", "tool_submission", "full_tool"])
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--cool-minutes", type=int, default=15)
    parser.add_argument("--min-battery", type=int, default=90)
    parser.add_argument("--max-tokens-thinking", type=int, default=8192)
    parser.add_argument("--max-tokens-normal", type=int, default=2048)
    parser.add_argument("--output-dir", default="data/results/mlx_sweep")
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
