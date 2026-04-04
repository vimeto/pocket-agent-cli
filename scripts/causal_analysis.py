#!/usr/bin/env python3
"""Causal analysis of model differences and capability-gated agency.

Analyzes the MLX sweep results to understand:
1. Why some models benefit from tools while others don't
2. Tool call parse rates and error recovery per model
3. Thinking token patterns and their correlation with success
4. Formalize the capability threshold

Uses the pre-computed MLX sweep JSONL files with full per-problem metrics.
"""

import json
import sys
import statistics
from pathlib import Path
from collections import defaultdict

MLX_DIR = Path("data/results/mlx_sweep/20260403_091508")

MODELS = ["qwen-3-4b", "qwen-3-0.6b", "llama-3.2-3b-instruct",
          "deepseek-r1-distill-qwen-1.5b", "gemma-3n-e2b-it"]
MODES = ["base", "tool_submission", "full_tool"]


def load_results():
    """Load all MLX sweep results."""
    data = {}
    for model in MODELS:
        data[model] = {}
        for mode in MODES:
            f = MLX_DIR / f"{model}_{mode}.jsonl"
            if f.exists():
                with open(f) as fh:
                    data[model][mode] = [json.loads(l) for l in fh]
            else:
                data[model][mode] = []
    return data


def analyze_pass_rates(data):
    """Pass rates per model × mode with tool benefit."""
    print("=" * 70)
    print("1. PASS RATES & TOOL BENEFIT")
    print("=" * 70)
    print(f"\n{'Model':<35} {'Base':>6} {'ToolSub':>8} {'FullTool':>9} {'Delta':>7}")
    print("-" * 67)

    for model in MODELS:
        rates = {}
        for mode in MODES:
            results = data[model][mode]
            if results:
                passed = sum(1 for r in results if r.get("passed"))
                rates[mode] = passed / len(results)
            else:
                rates[mode] = 0

        delta = rates.get("full_tool", 0) - rates.get("base", 0)
        sign = "+" if delta >= 0 else ""
        print(f"{model:<35} {rates.get('base',0):>5.0%} {rates.get('tool_submission',0):>7.0%} "
              f"{rates.get('full_tool',0):>8.0%} {sign}{delta:>5.0%}")


def analyze_tool_calls(data):
    """Tool call patterns per model."""
    print(f"\n{'='*70}")
    print("2. TOOL CALL PATTERNS")
    print("=" * 70)
    print(f"\n{'Model':<35} {'AvgTools':>9} {'AvgIter':>8} {'0-tool%':>8} {'Multi%':>7}")
    print("-" * 70)

    for model in MODELS:
        results = data[model].get("full_tool", [])
        if not results:
            continue

        tool_counts = [r.get("tool_call_count", 0) for r in results]
        iterations = [r.get("iterations", 1) for r in results]
        zero_tools = sum(1 for t in tool_counts if t == 0)
        multi_iter = sum(1 for i in iterations if i > 1)

        avg_tools = statistics.mean(tool_counts) if tool_counts else 0
        avg_iter = statistics.mean(iterations) if iterations else 0
        zero_pct = zero_tools / len(results) if results else 0
        multi_pct = multi_iter / len(results) if results else 0

        print(f"{model:<35} {avg_tools:>8.1f} {avg_iter:>7.1f} {zero_pct:>7.0%} {multi_pct:>6.0%}")


def analyze_thinking_patterns(data):
    """Thinking token analysis — correlation with success."""
    print(f"\n{'='*70}")
    print("3. THINKING TOKEN PATTERNS")
    print("=" * 70)

    for model in MODELS:
        results = data[model].get("base", [])
        if not results:
            continue

        thinking = [r["metrics"].get("thinking_tokens", 0) for r in results]
        regular = [r["metrics"].get("regular_tokens", 0) for r in results]
        total = [r["metrics"].get("total_tokens", 0) for r in results]
        passed = [r.get("passed", False) for r in results]

        if not any(t > 0 for t in thinking):
            # Non-thinking model
            avg_tok = statistics.mean(total) if total else 0
            pass_rate = sum(passed) / len(passed) if passed else 0
            print(f"\n{model} (non-thinking):")
            print(f"  Avg tokens: {avg_tok:.0f}, Pass rate: {pass_rate:.0%}")
            continue

        # Thinking model analysis
        think_ratio = [t / tot if tot > 0 else 0 for t, tot in zip(thinking, total)]
        avg_think = statistics.mean(thinking) if thinking else 0
        avg_regular = statistics.mean(regular) if regular else 0
        avg_ratio = statistics.mean(think_ratio) if think_ratio else 0

        # Correlation: thinking tokens vs success
        pass_think = [t for t, p in zip(thinking, passed) if p]
        fail_think = [t for t, p in zip(thinking, passed) if not p]

        avg_pass_think = statistics.mean(pass_think) if pass_think else 0
        avg_fail_think = statistics.mean(fail_think) if fail_think else 0

        # Pearson correlation
        if len(set(passed)) > 1 and len(thinking) > 2:
            n = len(thinking)
            mean_t = statistics.mean(thinking)
            mean_p = statistics.mean([1 if p else 0 for p in passed])
            cov = sum((t - mean_t) * ((1 if p else 0) - mean_p)
                      for t, p in zip(thinking, passed)) / (n - 1)
            std_t = statistics.stdev(thinking) if len(thinking) > 1 else 1
            std_p = statistics.stdev([1 if p else 0 for p in passed]) if len(set(passed)) > 1 else 1
            r = cov / (std_t * std_p) if std_t * std_p > 0 else 0
        else:
            r = 0

        print(f"\n{model} (thinking model):")
        print(f"  Avg thinking tokens: {avg_think:.0f}")
        print(f"  Avg regular tokens: {avg_regular:.0f}")
        print(f"  Thinking ratio: {avg_ratio:.1%}")
        print(f"  Pass thinking avg: {avg_pass_think:.0f}")
        print(f"  Fail thinking avg: {avg_fail_think:.0f}")
        print(f"  Correlation (thinking vs success): r={r:.3f}")
        if r < -0.3:
            print(f"  → Verbosity negatively correlated with success")
        elif r > 0.3:
            print(f"  → More thinking helps")
        else:
            print(f"  → Weak correlation")


def analyze_error_patterns(data):
    """Error classification per model."""
    print(f"\n{'='*70}")
    print("4. ERROR TAXONOMY")
    print("=" * 70)

    for model in MODELS:
        print(f"\n{model}:")
        for mode in MODES:
            results = data[model].get(mode, [])
            failed = [r for r in results if not r.get("passed")]
            if not failed:
                continue

            categories = defaultdict(int)
            for r in failed:
                err = r.get("error") or r.get("evaluation", {}).get("error", "")
                if "No code" in err:
                    categories["no_code"] += 1
                elif "NameError" in err or "name" in err.lower() and "not defined" in err.lower():
                    categories["name_error"] += 1
                elif "SyntaxError" in err or "IndentationError" in err:
                    categories["syntax"] += 1
                elif "assert" in err.lower():
                    categories["wrong_answer"] += 1
                elif "Timeout" in err:
                    categories["timeout"] += 1
                elif "TypeError" in err or "AttributeError" in err:
                    categories["type_error"] += 1
                else:
                    categories["other"] += 1

            total_fail = len(failed)
            cats_str = ", ".join(f"{k}={v}" for k, v in sorted(categories.items(), key=lambda x: -x[1]))
            print(f"  {mode}: {total_fail} failures — {cats_str}")


def analyze_latency_patterns(data):
    """TTFT, TPS, and energy patterns across models."""
    print(f"\n{'='*70}")
    print("5. LATENCY & ENERGY PATTERNS")
    print("=" * 70)

    print(f"\n{'Model':<35} {'TTFT':>8} {'TPS':>6} {'Energy':>8} {'Power':>7} {'GPUUtil':>8}")
    print("-" * 75)

    for model in MODELS:
        results = data[model].get("base", [])
        if not results:
            continue

        ttfts = [r["metrics"].get("ttft_ms") for r in results if r["metrics"].get("ttft_ms")]
        tpss = [r["metrics"].get("tps") for r in results if r["metrics"].get("tps")]
        energies = [r["metrics"].get("energy_summary", {}).get("total_energy_joules", 0) for r in results]
        powers = [r["metrics"].get("energy_summary", {}).get("avg_power_watts", 0) for r in results]
        gpus = [r["metrics"].get("energy_summary", {}).get("gpu_utilization_avg_percent", 0) for r in results]

        avg_ttft = statistics.mean(ttfts) if ttfts else 0
        avg_tps = statistics.mean(tpss) if tpss else 0
        avg_energy = statistics.mean(energies) if energies else 0
        avg_power = statistics.mean(powers) if powers else 0
        avg_gpu = statistics.mean(gpus) if gpus else 0

        print(f"{model:<35} {avg_ttft:>6.0f}ms {avg_tps:>5.1f} {avg_energy:>6.0f}J {avg_power:>5.1f}W {avg_gpu:>6.1f}%")


def capability_threshold(data):
    """Formalize the capability-gated agency threshold."""
    print(f"\n{'='*70}")
    print("6. CAPABILITY-GATED AGENCY THRESHOLD")
    print("=" * 70)

    print(f"\n{'Model':<35} {'Base%':>6} {'Full%':>6} {'Delta':>7} {'Benefit':>8}")
    print("-" * 65)

    for model in MODELS:
        base_results = data[model].get("base", [])
        full_results = data[model].get("full_tool", [])
        if not base_results or not full_results:
            continue

        base_rate = sum(1 for r in base_results if r.get("passed")) / len(base_results)
        full_rate = sum(1 for r in full_results if r.get("passed")) / len(full_results)
        delta = full_rate - base_rate
        benefit = "YES" if delta > 0.02 else ("NO" if delta < -0.02 else "NEUTRAL")

        print(f"{model:<35} {base_rate:>5.0%} {full_rate:>5.0%} {delta:>+6.0%} {benefit:>8}")

    print(f"\nCapability-gated agency rule:")
    print(f"  Enable Full-Tool mode when:")
    print(f"  1. Base-mode Pass@1 > 40% (sufficient coding capability)")
    print(f"  2. Tool call parse rate > 80% (can produce valid tool calls)")
    print(f"  3. Full-Tool Pass@1 > Base Pass@1 (tools actually help)")


def main():
    data = load_results()
    analyze_pass_rates(data)
    analyze_tool_calls(data)
    analyze_thinking_patterns(data)
    analyze_error_patterns(data)
    analyze_latency_patterns(data)
    capability_threshold(data)


if __name__ == "__main__":
    main()
