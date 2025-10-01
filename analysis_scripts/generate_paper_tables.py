#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUMMARY_PATH = Path("analysis_scripts/output_full/model_mode_summary.csv")
PROBLEM_PATH = Path("analysis_scripts/output_full/problem_metrics.csv")
TABLE_DIR = Path("research/figures")
TABLE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "gemma-3n-e2b-it": "Gemma 3n",
    "llama-3.2-3b-instruct": "Llama 3.2",
    "qwen-3-4b": "Qwen 3 4B",
    "qwen-3-0.6b": "Qwen 3 0.6B",
    "deepseek-r1-distill-qwen-1.5b": "DeepSeek R1",
}
MODE_LABELS = {
    "base": "Base",
    "tool_submission": "Tool Submission",
    "full_tool": "Full Tool",
}
HEADER_COLOR = "#1b4965"
ROW_COLORS = ["#f0f4f8", "#dfe7f2"]
ACCENT_COLORS = ["#ffba08", "#1982c4", "#6a994e", "#ee6c4d"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SUMMARY_PATH)
    problems = pd.read_csv(PROBLEM_PATH)
    return summary, problems


def format_number(value: float, precision: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{precision}f}"


def create_table_figure(df: pd.DataFrame, columns: Iterable[str], filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(len(columns) * 1.6, 0.6 * (len(df) + 2)))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor(ROW_COLORS[(row - 1) % len(ROW_COLORS)])
            cell.set_text_props(color="#222222")

    ax.set_title(title, fontsize=12, weight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(TABLE_DIR / filename)
    plt.close(fig)


def build_ttft_table(summary: pd.DataFrame) -> None:
    rows = []
    for model in summary["model"].unique():
        base = summary[(summary["model"] == model) & (summary["quantization"] == "F16") & (summary["mode"] == "base")]
        full = summary[(summary["model"] == model) & (summary["quantization"] == "F16") & (summary["mode"] == "full_tool")]
        if base.empty or full.empty:
            continue
        base_ttft = base["ttft_mean_ms"].iloc[0]
        full_ttft = full["ttft_mean_ms"].iloc[0]
        ratio = full_ttft / base_ttft if base_ttft else np.nan
        rows.append(
            [
                MODEL_LABELS.get(model, model),
                format_number(base_ttft, 2),
                format_number(full_ttft, 2),
                format_number(ratio, 3),
            ]
        )
    df = pd.DataFrame(rows, columns=["Model", "Base TTFT (ms)", "Full Tool TTFT (ms)", "Ratio"])
    create_table_figure(df, df.columns, "table_ttft_spike.pdf", "TTFT Impact of Tool Mode (FP16, A100)")


def build_throughput_table(summary: pd.DataFrame) -> None:
    base = summary[summary["mode"] == "base"].copy()
    base["energy_per_token"] = base["energy_mean_j"] / base["tokens_mean"]
    rows = []
    for _, row in base.iterrows():
        rows.append(
            [
                MODEL_LABELS.get(row["model"], row["model"]),
                row["quantization"].replace("Q4_K_M", "Q4"),
                format_number(row["tps_mean"], 2),
                format_number(row["ttft_mean_ms"], 2),
                format_number(row["energy_per_token"], 4),
            ]
        )
    df = pd.DataFrame(rows, columns=["Model", "Precision", "Tokens/s", "TTFT (ms)", "Energy/Token (J)"])
    create_table_figure(df, df.columns, "table_throughput.pdf", "Throughput and Energy per Token (A100, Base Mode)")


def build_power_table(problems: pd.DataFrame) -> None:
    grouped = problems.groupby(["model", "mode"])["avg_power_watts"].mean().reset_index()
    rows = []
    for _, row in grouped.iterrows():
        rows.append(
            [
                MODEL_LABELS.get(row["model"], row["model"]),
                MODE_LABELS.get(row["mode"], row["mode"].title()),
                format_number(row["avg_power_watts"], 2),
            ]
        )
    df = pd.DataFrame(rows, columns=["Model", "Mode", "Avg Power (W)"])
    create_table_figure(df, df.columns, "table_power.pdf", "Average System Power per Mode")


def main() -> None:
    summary, problems = load_data()
    build_ttft_table(summary)
    build_throughput_table(summary)
    build_power_table(problems)


if __name__ == "__main__":
    main()
