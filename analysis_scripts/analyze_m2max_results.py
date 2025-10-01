#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PLOT_STYLE = {
    "F16": "#f9c8d2",
    "Q4_K_M": "#cfe2cf",
}
MODE_MARKERS = {
    "base": "s",
    "tool_submission": "o",
    "full_tool": "^",
}


@dataclass
class AttemptRow:
    model: str
    quant: str
    mode: str
    problem_id: int
    run_id: int
    duration_s: float
    tokens: int
    success: bool
    tool_calls: int
    context_len: Optional[int]
    inter_token_avg_ms: Optional[float]
    inter_token_p95_ms: Optional[float]
    ttft_ms: Optional[float]
    energy_joules: Optional[float]
    energy_per_token_j: Optional[float]

    @property
    def tps(self) -> Optional[float]:
        if self.duration_s and self.tokens:
            return self.tokens / self.duration_s
        return None


def _load_attempt(file_path: Path, model: str, quant: str, mode: str) -> Optional[AttemptRow]:
    data = json.loads(file_path.read_text())
    metrics = data.get("metrics", {})
    thinking = metrics.get("thinking_stats", {})
    tokens = thinking.get("total_tokens")
    duration = data.get("duration_seconds")
    if tokens is None or not duration:
        return None

    inter_latencies = metrics.get("inter_token_latencies") or data.get("inter_token_latencies") or []
    if inter_latencies:
        inter_latencies = np.asarray(inter_latencies, dtype=float)
        inter_avg = float(inter_latencies.mean())
        inter_p95 = float(np.percentile(inter_latencies, 95))
        ttft = float(inter_latencies[0])
    else:
        inter_avg = inter_p95 = ttft = None

    energy_summary = metrics.get("energy_summary", {})
    energy_j = energy_summary.get("total_energy_joules")
    energy_per_token = metrics.get("energy_per_token_joules")
    if energy_per_token is None and energy_j is not None and tokens:
        energy_per_token = energy_j / tokens

    return AttemptRow(
        model=model,
        quant=quant,
        mode=mode,
        problem_id=int(data.get("problem_id", -1)),
        run_id=int(data.get("run_id", -1)),
        duration_s=float(duration),
        tokens=int(tokens),
        success=bool(data.get("success")),
        tool_calls=len(data.get("tool_calls") or []),
        context_len=data.get("context_length_used"),
        inter_token_avg_ms=inter_avg,
        inter_token_p95_ms=inter_p95,
        ttft_ms=ttft,
        energy_joules=energy_j,
        energy_per_token_j=energy_per_token,
    )


def load_attempts(root: Path) -> pd.DataFrame:
    rows: List[AttemptRow] = []
    for bench_file in sorted(root.glob("**/bench_*.json")):
        summary = json.loads(bench_file.read_text())
        config = summary.get("config", {})
        model = config.get("model_name") or summary.get("model_id")
        quant = config.get("model_version", "unknown").upper()
        mode = summary.get("mode") or config.get("mode")
        run_dir = bench_file.parent / "runs" / bench_file.stem
        if not run_dir.exists():
            continue
        for run_file in run_dir.glob("problem_*_run_*.json"):
            attempt = _load_attempt(run_file, model, quant, mode)
            if attempt:
                rows.append(attempt)
    if not rows:
        raise RuntimeError("No attempts parsed from dataset")
    df = pd.DataFrame([row.__dict__ for row in rows])
    df["tps"] = df.apply(lambda r: r["tokens"] / r["duration_s"] if r["duration_s"] else np.nan, axis=1)
    return df


def summarise_attempts(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["model", "quant", "mode"], sort=True)
    records: List[Dict[str, object]] = []
    for (model, quant, mode), group in grouped:
        duration_sum = group["duration_s"].sum()
        tokens_sum = group["tokens"].sum()
        tps_weighted = tokens_sum / duration_sum if duration_sum else np.nan
        record = {
            "model": model,
            "quant": quant,
            "mode": mode,
            "attempts": len(group),
            "problems": group["problem_id"].nunique(),
            "success_rate": group["success"].mean(),
            "tokens_mean": group["tokens"].mean(),
            "tokens_median": group["tokens"].median(),
            "duration_mean_s": group["duration_s"].mean(),
            "duration_median_s": group["duration_s"].median(),
            "tps_mean": group["tps"].mean(),
            "tps_median": group["tps"].median(),
            "tps_weighted": tps_weighted,
            "inter_token_avg_ms": group["inter_token_avg_ms"].mean(),
            "inter_token_p95_ms": group["inter_token_p95_ms"].mean(),
            "ttft_mean_ms": group["ttft_ms"].mean(),
            "ttft_median_ms": group["ttft_ms"].median(),
            "energy_mean_joules": group["energy_joules"].mean(),
            "energy_per_token_mean_j": group["energy_per_token_j"].mean(),
            "tool_call_rate": group["tool_calls"].mean(),
        }
        records.append(record)
    return pd.DataFrame(records)


def plot_tps(summary: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(10, 4.2))
    order = sorted(summary["model"].unique())
    palette = [PLOT_STYLE.get(q, "#4e79a7") for q in summary["quant"].unique()]
    sns.barplot(
        data=summary,
        x="model",
        y="tps_weighted",
        hue="quant",
        palette=palette,
        errorbar="sd"
    )
    plt.ylabel("Weighted TPS (tokens/s)")
    plt.xlabel("Model")
    plt.title("Throughput on MacBook M2 Max (MBPP subset)")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "m2max_tps_by_model.pdf")
    plt.close()


def plot_duration(summary: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(10, 4.2))
    sns.barplot(
        data=summary,
        x="model",
        y="duration_mean_s",
        hue="quant",
        palette=[PLOT_STYLE.get(q, "#4e79a7") for q in summary["quant"].unique()],
        errorbar="sd"
    )
    plt.ylabel("Mean attempt duration (s)")
    plt.xlabel("Model")
    plt.title("Attempt duration on MacBook M2 Max")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "m2max_duration_by_model.pdf")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse MacBook M2 Max benchmark runs")
    parser.add_argument("root", type=Path, help="Root directory containing copied benchmark files")
    parser.add_argument("--out", type=Path, default=Path("analysis_scripts/output_full/m2max"), help="Output directory for tables/figures")
    args = parser.parse_args()

    attempts_df = load_attempts(args.root)
    summary_df = summarise_attempts(attempts_df)

    args.out.mkdir(parents=True, exist_ok=True)
    attempts_df.to_csv(args.out / "attempt_metrics.csv", index=False)
    summary_df.to_csv(args.out / "summary_metrics.csv", index=False)

    plot_tps(summary_df, Path("research/figures"))
    plot_duration(summary_df, Path("research/figures"))

    # Console summary
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("=== Summary by model/quant/mode ===")
        print(summary_df.sort_values(["model", "quant", "mode"]))


if __name__ == "__main__":
    main()
