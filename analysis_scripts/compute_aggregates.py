#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

KS = (1, 3, 5, 10)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n <= 0:
        return 0.0
    k = min(k, n)
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    try:
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)
    except ValueError:
        return 0.0


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float | None, float | None]:
    if total <= 0:
        return (None, None)
    p = successes / total
    denom = 1 + z**2 / total
    centre = p + z**2 / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    lower = max(0.0, (centre - margin) / denom)
    upper = min(1.0, (centre + margin) / denom)
    return (lower, upper)


def bootstrap_mean(values: Sequence[float], n_boot: int = 1000, seed: int = 1729) -> Tuple[float | None, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return (None, None)
    rng = random.Random(seed)
    samples = []
    n = len(clean)
    for _ in range(n_boot):
        draw = [clean[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(draw) / n)
    samples.sort()
    lower_idx = int(0.025 * (n_boot - 1))
    upper_idx = int(0.975 * (n_boot - 1))
    return (samples[lower_idx], samples[upper_idx])


def summarise_group(df: pd.DataFrame) -> Dict[str, object]:
    attempts = len(df)
    successes = int(df["success"].sum())
    per_attempt_rate = successes / attempts if attempts else None
    ci_low, ci_high = wilson_interval(successes, attempts)

    problem_entries: List[Tuple[int, int]] = []
    for _, problem_df in df.groupby("problem_id"):
        n = len(problem_df)
        c = int(problem_df["success"].sum())
        problem_entries.append((n, c))

    pass_scores: Dict[int, List[float]] = {k: [] for k in KS}
    for n, c in problem_entries:
        for k in KS:
            pass_scores[k].append(pass_at_k(n, c, k))

    pass_summary = {}
    for k, values in pass_scores.items():
        mean_value = float(np.mean(values)) if values else None
        low, high = bootstrap_mean(values) if values else (None, None)
        pass_summary[f"pass@{k}"] = mean_value
        pass_summary[f"pass@{k}_low"] = low
        pass_summary[f"pass@{k}_high"] = high

    tool_counts = df["explicit_tool_calls"].fillna(df["num_tool_calls"])
    tool_counts = tool_counts.fillna(0)

    summary = {
        "attempts": attempts,
        "successes": successes,
        "per_attempt_pass_rate": per_attempt_rate,
        "per_attempt_pass_rate_low": ci_low,
        "per_attempt_pass_rate_high": ci_high,
        "problems": len(problem_entries),
        "avg_samples_per_problem": float(np.mean([n for n, _ in problem_entries])) if problem_entries else None,
        "median_samples_per_problem": float(np.median([n for n, _ in problem_entries])) if problem_entries else None,
        "duration_mean_s": df["duration_seconds"].mean(),
        "duration_median_s": df["duration_seconds"].median(),
        "duration_p95_s": df["duration_seconds"].quantile(0.95),
        "ttft_mean_ms": df["ttft_ms"].mean(),
        "ttft_median_ms": df["ttft_ms"].median(),
        "ttft_p95_ms": df["ttft_ms"].quantile(0.95),
        "tokens_mean": df["total_tokens"].mean(),
        "tps_mean": df["tps"].mean(),
        "overall_tps_mean": df["overall_tps"].mean(),
        "tool_calls_mean": tool_counts.mean(),
        "tool_calls_median": tool_counts.median(),
        "tool_latency_mean_ms": df["avg_tool_call_ms"].mean(),
        "tool_latency_total_ms": df["total_tool_time_ms"].mean(),
        "energy_mean_j": df["energy_joules"].mean(),
        "energy_median_j": df["energy_joules"].median(),
        "energy_total_j": df["energy_joules"].sum(),
        "thinking_ratio_mean": df["thinking_ratio"].mean(),
        "thinking_tokens_mean": df["thinking_tokens"].mean(),
    }
    summary.update(pass_summary)
    return summary


def compute_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (model, quant, mode), group in df.groupby(["model", "quantization", "mode"], sort=False):
        summary = summarise_group(group)
        summary.update({"model": model, "quantization": quant, "mode": mode})
        rows.append(summary)
    columns = [
        "model",
        "quantization",
        "mode",
        "attempts",
        "successes",
        "problems",
        "avg_samples_per_problem",
        "median_samples_per_problem",
        "per_attempt_pass_rate",
        "per_attempt_pass_rate_low",
        "per_attempt_pass_rate_high",
    ]
    for k in KS:
        columns.extend([f"pass@{k}", f"pass@{k}_low", f"pass@{k}_high"])
    columns.extend(
        [
            "duration_mean_s",
            "duration_median_s",
            "duration_p95_s",
            "ttft_mean_ms",
            "ttft_median_ms",
            "ttft_p95_ms",
            "tokens_mean",
            "tps_mean",
            "overall_tps_mean",
            "tool_calls_mean",
            "tool_calls_median",
            "tool_latency_mean_ms",
            "tool_latency_total_ms",
            "energy_mean_j",
            "energy_median_j",
            "energy_total_j",
            "thinking_ratio_mean",
            "thinking_tokens_mean",
        ]
    )
    return pd.DataFrame(rows)[columns]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute aggregated metrics for paper figures")
    parser.add_argument("--problems", type=Path, default=Path("analysis_scripts/output_full/problem_metrics.csv"), help="Path to problem metrics CSV")
    parser.add_argument("--output", type=Path, default=Path("analysis_scripts/output_full/model_mode_summary.csv"), help="Where to write the summary table")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.problems)
    summary = compute_summary_table(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Wrote summary for {len(summary)} configurations to {args.output}.")


if __name__ == "__main__":
    main()
