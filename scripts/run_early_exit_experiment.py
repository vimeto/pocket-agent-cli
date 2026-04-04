#!/usr/bin/env python3
"""Early-exit thinking-budget experiment for MLX inference.

Tests the hypothesis that capping thinking tokens improves or maintains
accuracy while reducing latency and energy.  Uses the two-phase
generation approach in MLXInferenceService.generate_with_thinking_budget().

Sweep matrix:
    models   x  thinking budgets  x  MBPP problems
    3 models    7 budgets            150 problems

Budgets: 0, 256, 512, 1024, 2048, 4096, None (unlimited ~8192)

Usage:
    python scripts/run_early_exit_experiment.py
    python scripts/run_early_exit_experiment.py --models qwen-3-4b --budgets 0 512 2048
    python scripts/run_early_exit_experiment.py --problems 50 --min-battery 80
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

# ── Models to test ───────────────────────────────────────────────────────
THINKING_MODELS = [
    {"id": "qwen-3-4b",                    "name": "Qwen 3 4B",       "arch": "qwen"},
    {"id": "qwen-3-0.6b",                  "name": "Qwen 3 0.6B",     "arch": "qwen"},
    {"id": "deepseek-r1-distill-qwen-1.5b", "name": "DeepSeek R1 1.5B", "arch": "qwen"},
]

# None = unlimited (up to max_tokens)
DEFAULT_BUDGETS = [0, 256, 512, 1024, 2048, 4096, None]


# ── Helpers (reused from run_mlx_sweep.py) ───────────────────────────────
def strip_thinking(text: str) -> str:
    c = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return re.sub(r'<think>.*', '', c, flags=re.DOTALL).strip()


def extract_code(response: str, tool_calls=None) -> Optional[str]:
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

    match = re.search(
        r'(def \w+\([^)]*\):.*?)(?=\n(?:def |\n\n[A-Z]|\Z))', text, re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return None


def evaluate_code(code: Optional[str], test_cases: List[str]) -> Dict[str, Any]:
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
            return {
                "passed": True,
                "test_details": [{"test": t, "passed": True} for t in test_cases],
            }
        return {
            "passed": False,
            "error": result.stderr[:300],
            "test_details": [
                {"test": t, "passed": False, "error": result.stderr[:100]}
                for t in test_cases
            ],
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "Timeout", "test_details": []}
    except Exception as e:
        return {"passed": False, "error": str(e)[:200], "test_details": []}


def build_messages(problem, prompt_config: Dict[str, str]) -> List[Dict[str, str]]:
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


# ── Single-problem runner with thinking budget ───────────────────────────
def run_single_with_budget(
    service: MLXInferenceService,
    messages: List[Dict[str, str]],
    max_tokens: int,
    thinking_budget: Optional[int],
) -> Dict[str, Any]:
    """Run one problem with a thinking budget, returning full metrics."""
    t0 = time.time()
    ttft = None
    tokens = 0
    thinking_tokens = 0
    regular_tokens = 0
    full_text = ""
    raw_text = ""
    metrics: Dict[str, Any] = {}
    was_truncated = False

    for chunk in service.generate_with_thinking_budget(
        messages, stream=True, thinking_budget=thinking_budget, max_tokens=max_tokens
    ):
        if ttft is None and chunk.get("token"):
            ttft = (time.time() - t0) * 1000

        full_text += chunk.get("token", "")
        raw_text += chunk.get("raw_token", chunk.get("token", ""))
        tokens += 1

        if chunk.get("is_thinking"):
            thinking_tokens += 1
        else:
            regular_tokens += 1

        if chunk.get("thinking_was_truncated"):
            was_truncated = True

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
        "thinking_budget": thinking_budget,
        "thinking_was_truncated": was_truncated,
        "generation_tps": metrics.get("generation_tps", tps),
        "prompt_tps": metrics.get("prompt_tps"),
        "prompt_tokens": metrics.get("prompt_tokens"),
        "peak_memory_gb": metrics.get("peak_memory_gb"),
        "energy_summary": metrics.get("energy_summary"),
        "current_power_watts": metrics.get("current_power_watts"),
    }


# ── Main sweep ───────────────────────────────────────────────────────────
def run_experiment(args):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"early_exit_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds_cls = DatasetRegistry.get(args.dataset)
    from pocket_agent_cli.config import DATA_DIR
    ds = ds_cls(DATA_DIR)
    if not ds.is_downloaded():
        ds.download()
    problems = ds.load(split="test", limit=args.problems)
    print(f"Loaded {len(problems)} {args.dataset} problems")
    print(f"Output: {out_dir}")

    # Resolve models & budgets
    selected_models = [
        m for m in THINKING_MODELS
        if not args.models or m["id"] in args.models
    ]
    budgets: List[Optional[int]] = []
    for b in args.budgets:
        budgets.append(None if b == "none" else int(b))

    print(f"Models: {[m['id'] for m in selected_models]}")
    print(f"Budgets: {budgets}")

    mode = "base"  # single-turn, code-only
    service = MLXInferenceService()
    all_summaries: List[Dict[str, Any]] = []

    for model_idx, model_def in enumerate(selected_models):
        model_id = model_def["id"]
        max_tokens = args.max_tokens

        wait_for_battery(args.min_battery)

        print(f"\n{'=' * 70}")
        print(f"Model {model_idx + 1}/{len(selected_models)}: {model_def['name']}")
        print(f"{'=' * 70}")

        model = Model(
            id=model_id, name=model_def["name"], architecture=model_def["arch"],
            downloaded=True, default_version="Q4_K_M", current_version="Q4_K_M",
        )
        config = InferenceConfig(
            temperature=0.7, max_tokens=max_tokens, top_p=0.9, top_k=40,
            repeat_penalty=1.1, context_length=16384, jinja=True,
        )
        service.load_model(model, config)

        prompt_config = get_optimized_prompt(model_id, mode)

        for budget in budgets:
            budget_label = "unlimited" if budget is None else str(budget)
            wait_for_battery(args.min_battery)

            print(f"\n  Budget: {budget_label}")
            print(f"  {'-' * 60}")

            results: List[Dict[str, Any]] = []
            passed = 0
            total_thinking = 0
            total_tokens_all = 0
            total_elapsed = 0.0
            truncated_count = 0

            for i, problem in enumerate(problems):
                messages = build_messages(problem, prompt_config)

                gen = run_single_with_budget(service, messages, max_tokens, budget)

                # Extract code and evaluate
                te = ToolExtractor()
                tcs, _ = te.extract_tools(gen.get("raw_response", gen["response"]))
                code = extract_code(gen["response"], tcs)

                evaluation = evaluate_code(code, problem.test_cases)

                result = {
                    "problem_id": problem.task_id,
                    "model_id": model_id,
                    "thinking_budget": budget,
                    "thinking_budget_label": budget_label,
                    "passed": evaluation["passed"],
                    "error": evaluation.get("error"),
                    "response_preview": gen["response"][:500],
                    "evaluation": evaluation,
                    "metrics": {
                        "ttft_ms": gen["ttft_ms"],
                        "tps": gen["tps"],
                        "total_tokens": gen["total_tokens"],
                        "thinking_tokens": gen["thinking_tokens"],
                        "regular_tokens": gen["regular_tokens"],
                        "thinking_ratio": gen["thinking_ratio"],
                        "elapsed_s": gen["elapsed_s"],
                        "thinking_was_truncated": gen["thinking_was_truncated"],
                        "generation_tps": gen.get("generation_tps"),
                        "prompt_tps": gen.get("prompt_tps"),
                        "prompt_tokens": gen.get("prompt_tokens"),
                        "peak_memory_gb": gen.get("peak_memory_gb"),
                        "energy_summary": gen.get("energy_summary"),
                        "power_watts": gen.get("current_power_watts"),
                    },
                }
                results.append(result)
                if result["passed"]:
                    passed += 1
                total_thinking += gen["thinking_tokens"]
                total_tokens_all += gen["total_tokens"]
                total_elapsed += gen["elapsed_s"]
                if gen["thinking_was_truncated"]:
                    truncated_count += 1

                status = "PASS" if result["passed"] else "FAIL"
                if (i + 1) % 10 == 0 or (i + 1) == len(problems):
                    print(
                        f"    [{i+1}/{len(problems)}] {passed}/{i+1} passed  "
                        f"(last: {status}  think_tok={gen['thinking_tokens']}  "
                        f"total_tok={gen['total_tokens']}  trunc={gen['thinking_was_truncated']}  "
                        f"{gen['elapsed_s']:.1f}s)"
                    )

            n = len(problems)
            pass_rate = passed / n if n else 0
            avg_thinking = total_thinking / n if n else 0
            avg_tokens = total_tokens_all / n if n else 0
            avg_elapsed = total_elapsed / n if n else 0

            # Compute avg energy
            energy_values = [
                r["metrics"]["energy_summary"]["total_energy_joules"]
                for r in results
                if r["metrics"].get("energy_summary")
                and r["metrics"]["energy_summary"].get("total_energy_joules")
            ]
            avg_energy = sum(energy_values) / len(energy_values) if energy_values else 0

            summary = {
                "model_id": model_id,
                "model_name": model_def["name"],
                "thinking_budget": budget,
                "thinking_budget_label": budget_label,
                "pass_rate": round(pass_rate, 4),
                "passed": passed,
                "total": n,
                "avg_thinking_tokens": round(avg_thinking, 1),
                "avg_total_tokens": round(avg_tokens, 1),
                "avg_elapsed_s": round(avg_elapsed, 2),
                "avg_energy_joules": round(avg_energy, 2),
                "truncated_count": truncated_count,
                "truncated_pct": round(truncated_count / n, 3) if n else 0,
            }
            all_summaries.append(summary)

            print(
                f"\n  >> {model_def['name']} budget={budget_label}: "
                f"{passed}/{n} = {pass_rate:.0%}  "
                f"avg_think={avg_thinking:.0f}  avg_tok={avg_tokens:.0f}  "
                f"avg_time={avg_elapsed:.1f}s  avg_energy={avg_energy:.1f}J  "
                f"truncated={truncated_count}/{n}"
            )

            # Save per-problem JSONL
            out_file = out_dir / f"{model_id}_budget_{budget_label}.jsonl"
            with open(out_file, "w") as f:
                for r in results:
                    f.write(json.dumps(r, default=str) + "\n")
            print(f"  Saved: {out_file}")

        # Unload model, cool down
        service.unload_model()
        gc.collect()
        if model_idx < len(selected_models) - 1 and args.cool_minutes > 0:
            print(f"\n  Cooling: {args.cool_minutes} min...")
            time.sleep(args.cool_minutes * 60)

    # ── Print summary table ──────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print(f"EARLY-EXIT EXPERIMENT SUMMARY ({args.dataset}, {args.problems} problems)")
    print(f"{'=' * 90}")

    header = (
        f"{'Model':<30} {'Budget':>8} {'Pass@1':>7} "
        f"{'AvgThink':>9} {'AvgTok':>8} {'AvgTime':>8} {'AvgEnergy':>10} {'Trunc%':>7}"
    )
    print(header)
    print("-" * 90)

    for s in all_summaries:
        print(
            f"{s['model_name']:<30} {s['thinking_budget_label']:>8} "
            f"{s['pass_rate']:>6.0%} "
            f"{s['avg_thinking_tokens']:>9.0f} {s['avg_total_tokens']:>8.0f} "
            f"{s['avg_elapsed_s']:>7.1f}s {s['avg_energy_joules']:>9.1f}J "
            f"{s['truncated_pct']:>6.0%}"
        )

    # ── Identify pareto frontier ─────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("PARETO FRONTIER (non-dominated: higher pass@1 AND lower energy)")
    print(f"{'=' * 90}")

    for model_def in selected_models:
        mid = model_def["id"]
        model_sums = [s for s in all_summaries if s["model_id"] == mid]
        if not model_sums:
            continue

        # A point is pareto-optimal if no other point has both
        # higher pass_rate AND lower avg_energy_joules
        pareto = []
        for s in model_sums:
            dominated = False
            for other in model_sums:
                if other is s:
                    continue
                if (other["pass_rate"] >= s["pass_rate"]
                        and other["avg_energy_joules"] <= s["avg_energy_joules"]
                        and (other["pass_rate"] > s["pass_rate"]
                             or other["avg_energy_joules"] < s["avg_energy_joules"])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(s)

        pareto.sort(key=lambda x: x["avg_energy_joules"])
        print(f"\n  {model_def['name']}:")
        for s in pareto:
            print(
                f"    budget={s['thinking_budget_label']:>8}  "
                f"pass@1={s['pass_rate']:.0%}  "
                f"energy={s['avg_energy_joules']:.1f}J  "
                f"time={s['avg_elapsed_s']:.1f}s  "
                f"think_tok={s['avg_thinking_tokens']:.0f}"
            )

    # Save summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "dataset": args.dataset,
                "n_problems": args.problems,
                "budgets": [b if b is not None else "unlimited" for b in budgets],
                "models": [m["id"] for m in selected_models],
                "results": all_summaries,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    print(f"\nFull results: {out_dir}")
    print(f"Summary:      {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Early-exit thinking-budget experiment (MLX)"
    )
    parser.add_argument("--dataset", default="mbpp")
    parser.add_argument("--problems", type=int, default=150)
    parser.add_argument(
        "--models", nargs="*",
        help="Model IDs to test (default: all thinking models)",
    )
    parser.add_argument(
        "--budgets", nargs="*", default=["0", "256", "512", "1024", "2048", "4096", "none"],
        help="Thinking budgets to sweep. Use 'none' for unlimited.",
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--cool-minutes", type=int, default=10)
    parser.add_argument("--min-battery", type=int, default=90)
    parser.add_argument(
        "--output-dir", default="data/results/early_exit",
    )
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
