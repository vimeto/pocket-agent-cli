#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Sequence

import pandas as pd

from analysis_scripts.benchmark_loader import iter_problem_records
from analysis_scripts.run_finder import filter_runs, find_runs


@dataclass
class ProblemRow:
    job_id: int
    model: str
    quantization: str
    mode: str
    session_id: str
    problem_id: int
    run_id: int
    success: bool
    duration_seconds: float | None
    context_length: int | None
    cold_start: bool | None
    tests_passed: int
    tests_total: int
    test_pass_rate: float | None
    iteration_count: int | None
    submission_via_tool: bool | None
    submission_found: bool | None
    energy_joules: float | None
    energy_per_token: float | None
    avg_power_watts: float | None
    max_power_watts: float | None
    min_power_watts: float | None
    cpu_avg_percent: float | None
    cpu_max_percent: float | None
    memory_avg_percent: float | None
    memory_max_percent: float | None
    ttft_ms: float | None
    total_tokens: int | None
    tps: float | None
    overall_tps: float | None
    itl_mean_ms: float | None
    itl_median_ms: float | None
    itl_p95_ms: float | None
    itl_p99_ms: float | None
    num_tool_calls: int | None
    avg_tool_call_ms: float | None
    total_tool_time_ms: float | None
    explicit_tool_calls: int | None
    response_length: int | None
    thinking_tokens: int | None
    regular_tokens: int | None
    thinking_ratio: float | None
    has_thinking: bool | None
    test_results_json: str | None


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalise_latencies(raw: Iterable[float]) -> List[float]:
    latencies: List[float] = []
    for item in raw:
        if item is None:
            continue
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if value < 0:
            continue
        if value < 1.0:
            value *= 1000.0
        latencies.append(value)
    return latencies


def _percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    d = k - f
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * d


def collect_problem_rows(data_root: Path, min_job_id: int | None) -> List[ProblemRow]:
    runs = find_runs(data_root)
    runs = filter_runs(runs, min_job_id=min_job_id, max_age_hours=None)
    rows: List[ProblemRow] = []

    for run in runs:
        for record in iter_problem_records(run):
            metrics = record.get("metrics", {}) or {}
            energy = metrics.get("energy_summary", {}) or {}
            thinking = metrics.get("thinking_stats", {}) or {}
            latencies_raw = record.get("inter_token_latencies") or metrics.get("inter_token_latencies") or []
            latencies_ms = _normalise_latencies(latencies_raw)

            ttft_ms = latencies_ms[0] if latencies_ms else None
            total_tokens = len(latencies_ms) if latencies_ms else None

            normal_latencies = [value for value in latencies_ms[1:] if value < 500.0 and value > 0.0]
            tps = (1000.0 / mean(normal_latencies)) if normal_latencies else None
            overall_tps = (total_tokens / record["duration_seconds"]) if (total_tokens and record.get("duration_seconds")) else None

            itl_mean_ms = mean(latencies_ms) if latencies_ms else None
            itl_median_ms = _percentile(latencies_ms, 50.0) if latencies_ms else None
            itl_p95_ms = _percentile(latencies_ms, 95.0) if latencies_ms else None
            itl_p99_ms = _percentile(latencies_ms, 99.0) if latencies_ms else None

            tool_latencies = [value for value in latencies_ms if value >= 500.0]
            num_tool_calls = len(tool_latencies) if latencies_ms else None
            avg_tool_call_ms = mean(tool_latencies) if tool_latencies else None
            total_tool_time_ms = sum(tool_latencies) if tool_latencies else None

            explicit_tool_calls = len(record.get("tool_calls") or []) if record.get("tool_calls") is not None else None
            response_length = len(record.get("response") or "")

            tests = record.get("test_results") or []
            tests_total = len(tests)
            tests_passed = sum(1 for test in tests if test.get("passed")) if tests_total else 0
            test_pass_rate = (tests_passed / tests_total) if tests_total else None

            row = ProblemRow(
                job_id=int(record["job_id"]),
                model=record["model"],
                quantization=record["quantization"],
                mode=record["mode"],
                session_id=record.get("session_id", ""),
                problem_id=_to_int(record.get("problem_id")) or 0,
                run_id=_to_int(record.get("run_id")) or 0,
                success=bool(record.get("success")),
                duration_seconds=_to_float(record.get("duration_seconds")),
                context_length=_to_int(record.get("context_length_used")),
                cold_start=record.get("cold_start"),
                tests_passed=tests_passed,
                tests_total=tests_total,
                test_pass_rate=test_pass_rate,
                iteration_count=_to_int(metrics.get("iteration_count")),
                submission_via_tool=metrics.get("submission_via_tool"),
                submission_found=metrics.get("submission_found"),
                energy_joules=_to_float(energy.get("total_energy_joules")),
                energy_per_token=_to_float(metrics.get("energy_per_token_joules")),
                avg_power_watts=_to_float(energy.get("avg_power_watts")),
                max_power_watts=_to_float(energy.get("max_power_watts")),
                min_power_watts=_to_float(energy.get("min_power_watts")),
                cpu_avg_percent=_to_float(energy.get("cpu_avg_percent")),
                cpu_max_percent=_to_float(energy.get("cpu_max_percent")),
                memory_avg_percent=_to_float(energy.get("memory_avg_percent")),
                memory_max_percent=_to_float(energy.get("memory_max_percent")),
                ttft_ms=ttft_ms,
                total_tokens=total_tokens,
                tps=tps,
                overall_tps=overall_tps,
                itl_mean_ms=itl_mean_ms,
                itl_median_ms=itl_median_ms,
                itl_p95_ms=itl_p95_ms,
                itl_p99_ms=itl_p99_ms,
                num_tool_calls=num_tool_calls,
                avg_tool_call_ms=avg_tool_call_ms,
                total_tool_time_ms=total_tool_time_ms,
                explicit_tool_calls=explicit_tool_calls,
                response_length=response_length,
                thinking_tokens=_to_int(thinking.get("thinking_tokens")),
                regular_tokens=_to_int(thinking.get("regular_tokens")),
                thinking_ratio=_to_float(thinking.get("thinking_ratio")),
                has_thinking=thinking.get("has_thinking"),
                test_results_json=json.dumps(tests),
            )
            rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export per-problem metrics from benchmark runs")
    parser.add_argument("--data-root", type=Path, default=Path("data/results"), help="Benchmark results root directory")
    parser.add_argument("--output", type=Path, default=Path("analysis_scripts/output_full/problem_metrics.csv"), help="Where to write the CSV output")
    parser.add_argument("--min-job-id", type=int, default=0, help="Ignore runs with job ids below this threshold")
    args = parser.parse_args(argv)

    rows = collect_problem_rows(args.data_root, args.min_job_id)
    if not rows:
        print("No problem records found.")
        return

    df = pd.DataFrame([asdict(row) for row in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    parquet_path = args.output.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        parquet_msg = f" and {parquet_path}"
    except (ImportError, ValueError):
        parquet_msg = ""
    print(f"Exported {len(df)} problem records to {args.output}{parquet_msg}.")


if __name__ == "__main__":
    main()
