#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from functools import lru_cache
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# Ensure Matplotlib can write caches inside the workspace
mpl_cache = Path(".matplotlib_cache")
mpl_cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache.resolve()))

sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

FONT_SCALE = 1.5


def _scale_font(size: float) -> float:
    """Return a font size scaled by the global FONT_SCALE."""
    return size * FONT_SCALE

MODE_ORDER = ["base", "tool_submission", "full_tool"]
MODE_LABELS = {
    "base": "Base",
    "tool_submission": "Tool Submission",
    "full_tool": "Full Tool",
}

MODE_COLORS = {
    "base": "#8fd3b6",
    "tool_submission": "#f3c6a5",
    "full_tool": "#b5c7eb",
}

PRECISION_COLORS = {"F16": "#f9c8d2", "Q4_K_M": "#cfe2cf"}
QUANT_LABELS = {"F16": "FP16", "Q4_K_M": "Q4"}
PLATFORM_HATCH = {"mac": "", "a100": "//"}
THINKING_COLOR = "#f5a3a9"
ACTIONABLE_COLOR = "#c4e7c4"
UNIQUE_TOOL_COLOR = "#d8e7f1"
DUP_TOOL_COLOR = "#95adc7"
MODEL_LABELS = {
    "gemma-3n-e2b-it": "Gemma 3n",
    "llama-3.2-3b-instruct": "Llama 3.2",
    "qwen-3-4b": "Qwen 3 4B",
    "qwen-3-0.6b": "Qwen 3 0.6B",
    "deepseek-r1-distill-qwen-1.5b": "DeepSeek R1",
}
FIG_DIR = Path("research/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = Path("analysis_scripts/output_full/model_mode_summary.csv")
PROBLEM_PATH = Path("analysis_scripts/output_full/problem_metrics.csv")
PREFILL_ROOT = Path(".pocket_agent_home/results/prefill_tests")
MOBILE_PREFILL_ROOT = Path("analysis_scripts/output_full/prefill_mobile")


def load_summary() -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY_PATH)
    for k in [1, 3, 5, 10]:
        summary[f"pass@{k}_pct"] = summary[f"pass@{k}"] * 100.0
        summary[f"pass@{k}_low_pct"] = summary[f"pass@{k}_low"].fillna(summary[f"pass@{k}"]) * 100.0
        summary[f"pass@{k}_high_pct"] = summary[f"pass@{k}_high"].fillna(summary[f"pass@{k}"]) * 100.0
    summary["per_attempt_pass_rate_pct"] = summary["per_attempt_pass_rate"].fillna(0) * 100.0
    return summary


def load_problem_dataframe() -> pd.DataFrame:
    return pd.read_csv(PROBLEM_PATH)


def model_order(df: pd.DataFrame) -> Iterable[str]:
    order = [m for m in MODEL_LABELS if m in df["model"].unique()]
    return order


def draw_accuracy(summary: pd.DataFrame) -> None:
    f16_data = summary[(summary["quantization"] == "F16") & (summary["mode"].isin(MODE_ORDER))]
    order = model_order(f16_data)
    if not len(order):
        return

    x = np.arange(len(order))
    width = 0.13
    metric_specs = [
        ("pass@1_pct", "Pass@1", None),
        ("pass@10_pct", "Pass@10", "//"),
    ]

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})

    for idx_mode, mode in enumerate(MODE_ORDER):
        mode_df = f16_data[f16_data["mode"] == mode]
        if mode_df.empty:
            continue
        base_shift = (idx_mode - (len(MODE_ORDER) - 1) / 2) * (len(metric_specs) * width + 0.01)
        for idx_metric, (metric, _, hatch) in enumerate(metric_specs):
            offsets = x + base_shift + (idx_metric - (len(metric_specs) - 1) / 2) * width
            stat_prefix = metric.replace("_pct", "")
            low_col = f"{stat_prefix}_low_pct"
            high_col = f"{stat_prefix}_high_pct"
            for offset, model in zip(offsets, order):
                subset = mode_df[mode_df["model"] == model]
                if subset.empty:
                    continue
                value = subset[metric].iloc[0]
                lower = subset[low_col].iloc[0] if low_col in subset.columns else value
                upper = subset[high_col].iloc[0] if high_col in subset.columns else value
                ax_top.bar(
                    offset,
                    value,
                    width=width,
                    color=MODE_COLORS[mode],
                    hatch=hatch,
                    edgecolor="#222222",
                    linewidth=0.5,
                    zorder=3,
                )
                ax_top.errorbar(
                    offset,
                    value,
                    yerr=[[value - lower], [upper - value]],
                    fmt="none",
                    ecolor="#222222",
                    elinewidth=0.8,
                    capsize=3,
                    zorder=4,
                )

    ax_top.set_ylabel("Accuracy (%)")
    ax_top.set_ylim(0, 100)
    ax_top.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    mode_handles = [Patch(facecolor=MODE_COLORS[mode], edgecolor="#222222", label=MODE_LABELS[mode]) for mode in MODE_ORDER]
    metric_handles = [
        Patch(facecolor="#cccccc", edgecolor="#222222", label="Pass@1"),
        Patch(facecolor="#cccccc", edgecolor="#222222", hatch="//", label="Pass@10"),
    ]
    legend1 = ax_top.legend(handles=mode_handles, loc="upper left", ncol=3)
    ax_top.add_artist(legend1)
    ax_top.legend(handles=metric_handles, loc="upper right")

    diff_records = []
    for model in order:
        for mode in MODE_ORDER:
            fp16_row = summary[(summary["model"] == model) & (summary["quantization"] == "F16") & (summary["mode"] == mode)]
            q4_row = summary[(summary["model"] == model) & (summary["quantization"] == "Q4_K_M") & (summary["mode"] == mode)]
            if fp16_row.empty or q4_row.empty:
                continue
            delta = fp16_row["pass@1_pct"].iloc[0] - q4_row["pass@1_pct"].iloc[0]
            diff_records.append((model, mode, delta))

    if diff_records:
        for idx_mode, mode in enumerate(MODE_ORDER):
            base_shift = (idx_mode - (len(MODE_ORDER) - 1) / 2) * (width)
            offsets = x + base_shift
            for offset, model in zip(offsets, order):
                delta_entry = next((delta for (m, md, delta) in diff_records if m == model and md == mode), None)
                if delta_entry is None:
                    continue
                ax_bottom.bar(
                    offset,
                    delta_entry,
                    width=width,
                    color=MODE_COLORS[mode],
                    edgecolor="#222222",
                    linewidth=0.5,
                    zorder=3,
                )

        ax_bottom.axhline(0, color="#222222", linewidth=0.8)
        ax_bottom.set_ylabel("FP16 – INT4 Pass@1 (pp)")
        ax_bottom.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels([MODEL_LABELS.get(model, model) for model in order])

    fig.suptitle("Accuracy Across Modes")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "accuracy_fp16.pdf")
    plt.close(fig)


def compute_tool_benefit(summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    for quant in ["F16", "Q4_K_M"]:
        base = summary[(summary["quantization"] == quant) & (summary["mode"] == "base")]
        full = summary[(summary["quantization"] == quant) & (summary["mode"] == "full_tool")]
        merged = pd.merge(base, full, on="model", suffixes=("_base", "_full"))
        if merged.empty:
            continue
        merged["pass_gain_pp"] = merged["pass@1_pct_full"] - merged["pass@1_pct_base"]
        merged["latency_delta_s"] = merged["duration_mean_s_full"] - merged["duration_mean_s_base"]
        merged["tool_benefit_pp_per_s"] = merged.apply(
            lambda row: row["pass_gain_pp"] / row["latency_delta_s"] if row["latency_delta_s"] else np.nan,
            axis=1,
        )
        merged["quantization"] = quant
        records.append(merged)
    return pd.concat(records) if records else pd.DataFrame()


def draw_tool_benefit(summary: pd.DataFrame) -> None:
    benefit = compute_tool_benefit(summary)
    if benefit.empty:
        return
    order = model_order(benefit)
    quant_order = ["F16", "Q4_K_M"]
    width = 0.18
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for idx, quant in enumerate(quant_order):
        offsets = x + (idx - (len(quant_order) - 1) / 2) * width
        values = []
        for model in order:
            match = benefit[(benefit["model"] == model) & (benefit["quantization"] == quant)]
            values.append(match["tool_benefit_pp_per_s"].iloc[0] if not match.empty else np.nan)
        ax.bar(
            offsets,
            values,
            width=width,
            color=PRECISION_COLORS[quant],
            hatch="" if quant == "F16" else "//",
            edgecolor="#222222",
            linewidth=0.5,
            zorder=3,
        )
    ax.axhline(0, color="#222222", linewidth=0.8, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(model, model) for model in order])
    ax.set_ylabel("Pass@1 gain (pp) per extra second")

    ymin, ymax = ax.get_ylim()          # talteen nykyiset rajat
    ax.axhspan(0, ymax, facecolor="green", alpha=0.08, zorder=0)  # y>0
    ax.axhspan(ymin, 0, facecolor="red",   alpha=0.05, zorder=0)  # y<=0
    ax.set_ylim(ymin, ymax)             # palautetaan rajat, ettei varjostus muuta niitä

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    quant_handles = [
        Patch(facecolor=PRECISION_COLORS[q], edgecolor="#222222",
            hatch="" if q == "F16" else "//", label=QUANT_LABELS[q])
        for q in quant_order
    ]
    meaning_handles = [
        Patch(facecolor="green", alpha=0.3, label="Worth using tools"),
        Patch(facecolor="red",   alpha=0.3, label="NOT worth using tools"),
    ]

    leg_left = ax.legend(
        handles=quant_handles,
        loc="upper left",
        frameon=True,
        title="Quantization",
    )

    leg_right = ax.legend(
        handles=meaning_handles,
        loc="upper right",
        frameon=True,
        title="Interpretation",
    )

    ax.add_artist(leg_left)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "tool_benefit.pdf")
    plt.close(fig)


def draw_overhead_breakdown(summary: pd.DataFrame, problems: pd.DataFrame) -> None:
    relevant_summary = summary[summary["mode"].isin(MODE_ORDER)]
    if relevant_summary.empty:
        return

    models = model_order(relevant_summary)
    if not models:
        return

    quant_order = ["F16", "Q4_K_M"]
    quant_hatch = {"F16": "", "Q4_K_M": "//"}

    fig, (ax_duration, ax_tools) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})

    width = 0.12
    total_bars = len(MODE_ORDER) * len(quant_order)
    base_positions = np.arange(len(models))

    # Top panel: durations per model/mode/quantization
    for mode_idx, mode in enumerate(MODE_ORDER):
        for quant_idx, quant in enumerate(quant_order):
            offsets = base_positions + ((mode_idx * len(quant_order) + quant_idx) - (total_bars - 1) / 2) * width
            values = []
            for model in models:
                match = relevant_summary[(relevant_summary["model"] == model) & (relevant_summary["quantization"] == quant) & (relevant_summary["mode"] == mode)]
                values.append(match["duration_mean_s"].iloc[0] if not match.empty else np.nan)
            hatch = quant_hatch[quant]
            ax_duration.bar(
                offsets,
                values,
                width=width,
                color=MODE_COLORS[mode],
                hatch=hatch,
                edgecolor="#222222",
                linewidth=0.4,
                zorder=3,
            )

    ax_duration.set_ylabel("Mean attempt duration (s)")
    ax_duration.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax_duration.set_title("(a) Duration by Mode and Precision")

    mode_handles = [Patch(facecolor=MODE_COLORS[mode], edgecolor="#222222", label=MODE_LABELS[mode]) for mode in MODE_ORDER]
    quant_handles = [Patch(facecolor="#ffffff", edgecolor="#222222", hatch=quant_hatch[quant], label=QUANT_LABELS.get(quant, quant)) for quant in quant_order]
    legend1 = ax_duration.legend(handles=mode_handles, loc="upper left", ncol=3, frameon=False)
    ax_duration.add_artist(legend1)
    ax_duration.legend(handles=quant_handles, loc="upper right", frameon=False)

    # Bottom panel: tool usage with duplicate breakdown
    usage_stats = _tool_usage_summary()
    if usage_stats.empty:
        return

    for mode_idx, mode in enumerate(MODE_ORDER):
        for quant_idx, quant in enumerate(quant_order):
            offsets = base_positions + ((mode_idx * len(quant_order) + quant_idx) - (total_bars - 1) / 2) * width
            total_vals = []
            dup_vals = []
            for model in models:
                match = usage_stats[(usage_stats["model"] == model) & (usage_stats["quantization"] == quant) & (usage_stats["mode"] == mode)]
                if match.empty:
                    total_vals.append(np.nan)
                    dup_vals.append(np.nan)
                else:
                    total = match["avg_total"].iloc[0]
                    dup = match["avg_duplicates"].iloc[0]
                    total_vals.append(total)
                    dup_vals.append(dup)
            unique_vals = [total - dup if not np.isnan(total) and not np.isnan(dup) else np.nan for total, dup in zip(total_vals, dup_vals)]

            hatch = quant_hatch[quant]
            ax_tools.bar(
                offsets,
                unique_vals,
                width=width,
                color=UNIQUE_TOOL_COLOR,
                hatch=hatch,
                edgecolor="#222222",
                linewidth=0.4,
                zorder=3,
            )
            ax_tools.bar(
                offsets,
                dup_vals,
                width=width,
                bottom=unique_vals,
                color=DUP_TOOL_COLOR,
                hatch=hatch,
                edgecolor="#222222",
                linewidth=0.4,
                zorder=4,
            )

    ax_tools.set_ylabel("Tool calls per attempt")
    ax_tools.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax_tools.set_title("(b) Tool usage with duplicate submissions")

    ax_tools.set_xticks(base_positions)
    ax_tools.set_xticklabels([MODEL_LABELS.get(model, model) for model in models])

    unique_handle = Patch(facecolor=UNIQUE_TOOL_COLOR, edgecolor="#222222", label="Unique tool calls")
    duplicate_handle = Patch(facecolor=DUP_TOOL_COLOR, edgecolor="#222222", label="Duplicate repeats")
    quant_handles_tools = [Patch(facecolor="#ffffff", edgecolor="#222222", hatch=quant_hatch[q], label=QUANT_LABELS[q]) for q in quant_order]
    ax_tools.legend(handles=[unique_handle, duplicate_handle] + quant_handles_tools, loc="upper right", frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "overhead_breakdown.pdf")
    plt.close(fig)


def _read_prefill_file(path: Path, default_device: str) -> Iterable[Dict[str, object]]:
    payload = json.loads(path.read_text())
    metadata = payload.get("metadata", {})
    device_info = metadata.get("device_info", {})
    rows: List[Dict[str, object]] = []

    device_name = (
        device_info.get("model")
        or metadata.get("device")
        or default_device
    )
    system_name = device_info.get("systemName") or metadata.get("platform")
    if system_name:
        device_label = f"{device_name} ({system_name})"
    else:
        device_label = device_name

    # Tidy verbose mobile labels copied from device metadata
    device_label = device_label.replace(" (iOS)", "").replace(" (ios)", "").strip()

    model_name = metadata.get("model_id") or payload.get("model")
    if isinstance(model_name, str):
        model_name = model_name.lower()

    quant = (
        metadata.get("model_version")
        or payload.get("model_version")
        or metadata.get("quantization")
        or "Q4_K_M"
    )
    if isinstance(quant, str):
        quant = quant.upper()

    for entry in payload.get("results", []):
        rows.append(
            {
                "device": device_label,
                "model": model_name,
                "quantization": quant,
                "requested_tokens": entry.get("requested_tokens"),
                "actual_tokens": entry.get("actual_tokens"),
                "ttft_ms": entry.get("ttft_ms"),
            }
        )
    return rows


def load_prefill_curves() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    if PREFILL_ROOT.exists():
        for model_dir in PREFILL_ROOT.iterdir():
            if not model_dir.is_dir():
                continue
            for path in model_dir.glob("*.json"):
                rows.extend(_read_prefill_file(path, default_device="MacBook M2 Max"))

    if MOBILE_PREFILL_ROOT.exists():
        for path in MOBILE_PREFILL_ROOT.glob("*.json"):
            rows.extend(_read_prefill_file(path, default_device="Mobile Device"))

    return pd.DataFrame(rows)


def _summarise_prefill(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["device", "model", "quantization", "requested_tokens"], dropna=False)
        .agg(
            ttft_mean=("ttft_ms", "mean"),
            ttft_p10=("ttft_ms", lambda s: s.quantile(0.1)),
            ttft_p90=("ttft_ms", lambda s: s.quantile(0.9)),
        )
        .reset_index()
    )


def draw_prefill(prefill_df: pd.DataFrame) -> None:
    if prefill_df.empty:
        return

    # Separate desktop vs mobile measurements
    desktop_label = "MacBook M2 Max"
    desktop_df = prefill_df[prefill_df["device"].str.contains("MacBook", na=False)]
    mobile_df = prefill_df[~prefill_df["device"].str.contains("MacBook", na=False)]

    if desktop_df.empty and mobile_df.empty:
        return

    panels = []
    if not desktop_df.empty:
        panels.append(("MacBook M2 Max", _summarise_prefill(desktop_df)))
    if not mobile_df.empty:
        mobile_devices = sorted(mobile_df["device"].unique())
        if len(mobile_devices) == 1:
            panel_title = mobile_devices[0]
        elif len(mobile_devices) == 2:
            panel_title = " & ".join(mobile_devices)
        else:
            panel_title = "Mobile Devices"
        panels.append((panel_title, _summarise_prefill(mobile_df)))

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(8.0, 4.0 * len(panels)),
        sharex=True,
    )
    if len(panels) == 1:
        axes = [axes]

    combo_keys = sorted({(row["device"], row["model"]) for _, row in prefill_df.iterrows()})
    combo_palette = sns.color_palette("deep", len(combo_keys) or 1)
    combo_colors = {key: color for key, color in zip(combo_keys, combo_palette)}

    max_tokens_axis = max(3500, int(prefill_df["requested_tokens"].max()))
    reference_lines = [(460, "Base prompt (≈460)"), (1550, "Full-tool prompt (≈1550)")]

    for ax, (panel_title, panel_df) in zip(axes, panels):
        is_mobile_panel = "Mac" not in panel_title
        scale = 1000.0 if is_mobile_panel else 1.0
        for (device_name, model, quant), subset in panel_df.groupby(
            ["device", "model", "quantization"], dropna=False
        ):
            label = (
                f"{device_name}: {MODEL_LABELS.get(model, model)} "
                f"{QUANT_LABELS.get(quant, quant)}"
            )
            color = combo_colors.get((device_name, model), "#4e79a7")
            ax.plot(
                subset["requested_tokens"],
                subset["ttft_mean"] / scale,
                label=label,
                color=color,
            )
            ax.fill_between(
                subset["requested_tokens"],
                subset["ttft_p10"] / scale,
                subset["ttft_p90"] / scale,
                color=color,
                alpha=0.2,
            )

        for idx_ref, (x_val, label) in enumerate(reference_lines):
            ax.axvline(x_val, color="#cccccc", linestyle="--", linewidth=0.9, zorder=0)
            y_min, y_max = ax.get_ylim()
            span = y_max - y_min if y_max > y_min else 1.0
            y_pos = y_max - 0.05 * span
            x_shift = -70 # if idx_ref == 0 else 70
            ha = "right" if x_shift < 0 else "left"
            ax.text(
                x_val + x_shift,
                y_pos,
                label,
                fontsize=_scale_font(8),
                color="#666666",
                rotation=90,
                ha=ha,
                va="top",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
            )

        ax.set_ylabel("TTFT (s)" if is_mobile_panel else "TTFT (ms)")
        ax.set_title(panel_title)
        ax.grid(alpha=0.35)
        ax.set_xlim(0, max_tokens_axis)
        ax.legend(frameon=False, fontsize=_scale_font(9), loc="upper right")

    axes[-1].set_xlabel("Prompt length (tokens)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "prefill_qwen.pdf")
    plt.close(fig)


def draw_quantization(summary: pd.DataFrame) -> None:
    subset = summary[summary["mode"] == "base"]
    models = model_order(subset)
    if not models:
        return

    mac_summary_path = Path("analysis_scripts/output_full/m2max/summary_metrics.csv")
    mac_summary = None
    if mac_summary_path.exists():
        mac_summary = pd.read_csv(mac_summary_path)
        mac_summary = mac_summary[mac_summary["mode"] == "base"]

    quant_order = ["F16", "Q4_K_M"]
    platforms = ["mac", "a100"]
    width = 0.18
    base_positions = np.arange(len(models))
    total_bars = len(quant_order) * len(platforms)

    fig, (ax_tps, ax_energy) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    energy_records: List[Dict[str, object]] = []

    for plat_idx, platform in enumerate(platforms):
        hatch = PLATFORM_HATCH[platform]
        for quant_idx, quant in enumerate(quant_order):
            offsets = base_positions + ((plat_idx * len(quant_order) + quant_idx) - (total_bars - 1) / 2) * width
            tps_values = []
            energy_values = []
            for model in models:
                if platform == "mac":
                    if mac_summary is None:
                        tps_values.append(np.nan)
                        energy_values.append(np.nan)
                    else:
                        match = mac_summary[(mac_summary["model"] == model) & (mac_summary["quant"] == quant)]
                        if match.empty:
                            tps_values.append(np.nan)
                            energy_values.append(np.nan)
                        else:
                            tps_values.append(match["tps_weighted"].iloc[0])
                            energy_values.append(match["energy_per_token_mean_j"].iloc[0])
                else:
                    match = subset[(subset["model"] == model) & (subset["quantization"] == quant)]
                    if match.empty:
                        tps_values.append(np.nan)
                        energy_values.append(np.nan)
                    else:
                        tps_values.append(match["tps_mean"].iloc[0])
                        tokens_mean = match["tokens_mean"].iloc[0]
                        energy_mean = match["energy_mean_j"].iloc[0]
                        energy_values.append(energy_mean / tokens_mean if tokens_mean else np.nan)

                energy_records.append({
                    "platform": platform,
                    "model": model,
                    "quant": quant,
                    "value": energy_values[-1],
                    "x": offsets[len(energy_values) - 1],
                    "width": width,
                })
            color = PRECISION_COLORS[quant]
            ax_tps.bar(
                offsets,
                tps_values,
                width=width,
                color=color,
                hatch=hatch,
                edgecolor="#222222",
                linewidth=0.5,
                zorder=3,
            )
            ax_energy.bar(
                offsets,
                energy_values,
                width=width,
                color=color,
                hatch=hatch,
                edgecolor="#222222",
                linewidth=0.5,
                zorder=3,
            )

    energy_vals = [rec["value"] for rec in energy_records if rec["value"] and np.isfinite(rec["value"])]
    energy_cap = None
    if energy_vals:
        sorted_vals = sorted(energy_vals)
        if len(sorted_vals) >= 2 and sorted_vals[-1] > sorted_vals[-2] * 2:
            energy_cap = sorted_vals[-2] * 1.2
        else:
            energy_cap = sorted_vals[-1]
        energy_cap = max(energy_cap, sorted_vals[-2]) if len(sorted_vals) >= 2 else energy_cap
        ax_energy.set_ylim(0, energy_cap * 1.05)

        for rec in energy_records:
            value = rec["value"]
            if not value or not np.isfinite(value):
                continue
            if energy_cap and value > energy_cap:
                ax_energy.bar(
                    rec["x"],
                    energy_cap,
                    width=rec["width"],
                    color=PRECISION_COLORS[rec["quant"]],
                    hatch=PLATFORM_HATCH[rec["platform"]],
                    edgecolor="#222222",
                    linewidth=0.5,
                    zorder=3,
                )
                ax_energy.annotate(
                    f"{value:.1f}",
                    xy=(rec["x"], energy_cap),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=_scale_font(8),
                    color="#444444",
                )

    xticklabels = [MODEL_LABELS.get(model, model) for model in models]
    ax_energy.set_xticks(base_positions)
    ax_energy.set_xticklabels(xticklabels, rotation=15, ha="right")

    ax_tps.set_ylabel("Tokens/s")
    ax_energy.set_ylabel("J per token")
    ax_tps.set_title("Throughput across platforms")
    ax_tps.set_ylim([0, 160])
    ax_energy.set_title("Energy per token across platforms")

    for ax in (ax_tps, ax_energy):
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    precision_handles = [Patch(facecolor=PRECISION_COLORS[q], edgecolor="#222222", label=QUANT_LABELS[q]) for q in quant_order]
    platform_handles = [Patch(facecolor="#ffffff", edgecolor="#222222", hatch=PLATFORM_HATCH[p], label="Mac" if p == "mac" else "A100") for p in platforms]
    ax_tps.legend(handles=precision_handles + platform_handles, loc="upper right", ncol=2, frameon=False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "quantization_a100.pdf")
    plt.close(fig)


def bootstrap_ci(values: Iterable[float], n_boot: int = 2000, seed: int = 2025) -> Tuple[float, float]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = [np.mean(rng.choice(clean, size=len(clean), replace=True)) for _ in range(n_boot)]
    return (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


def _energy_to_success(problems: pd.DataFrame, model: str, quant: str, mode: str) -> Tuple[pd.Series, pd.Series]:
    subset = problems[(problems["model"] == model) & (problems["quantization"] == quant) & (problems["mode"] == mode)]
    rows = []
    for _, group in subset.groupby("problem_id"):
        ordered = group.sort_values(["job_id", "session_id", "run_id"])
        cumulative = 0.0
        attempts = 0
        for _, row in ordered.iterrows():
            cumulative += row["energy_joules"]
            attempts += 1
            if row["success"]:
                rows.append({"energy": cumulative, "attempts": attempts})
                break
    if not rows:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    return df["energy"], df["attempts"]


def compute_energy_success_stats(problems: pd.DataFrame) -> pd.DataFrame:
    filtered = problems[problems["mode"].isin(["base", "full_tool"])]
    records: List[Dict[str, object]] = []

    for (model, quant, mode), group in filtered.groupby(["model", "quantization", "mode"]):
        energies: List[float] = []
        attempts: List[int] = []

        for _, problem_rows in group.groupby("problem_id"):
            ordered = problem_rows.sort_values(["job_id", "session_id", "run_id"])
            cumulative = 0.0
            attempt_count = 0
            for _, row in ordered.iterrows():
                energy = row.get("energy_joules")
                if not isinstance(energy, (int, float)) or not math.isfinite(float(energy)):
                    continue
                cumulative += float(energy)
                attempt_count += 1
                if bool(row.get("success")):
                    energies.append(cumulative)
                    attempts.append(attempt_count)
                    break

        if energies:
            energy_arr = np.asarray(energies, dtype=float)
            attempts_arr = np.asarray(attempts, dtype=float)
            energy_mean = energy_arr.mean()
            ci_low = np.percentile(energy_arr, 2.5)
            ci_high = np.percentile(energy_arr, 97.5)
            attempts_mean = attempts_arr.mean() if len(attempts_arr) else float("nan")
        else:
            energy_mean = float("nan")
            ci_low = float("nan")
            ci_high = float("nan")
            attempts_mean = float("nan")

        records.append(
            {
                "model": model,
                "quantization": quant,
                "mode": mode,
                "energy_mean_kJ": energy_mean / 1000.0,
                "energy_ci_low_kJ": ci_low / 1000.0,
                "energy_ci_high_kJ": ci_high / 1000.0,
                "attempts_mean": attempts_mean,
            }
        )

    return pd.DataFrame(records)


def refresh_energy_per_success_table(
    problems: pd.DataFrame,
    table_path: Optional[Path] = None,
) -> pd.DataFrame:
    path = table_path or (FIG_DIR / "energy_per_success_table.csv")
    summary = compute_energy_success_stats(problems)

    if path.exists():
        table = pd.read_csv(path)
    else:
        table = summary[["model", "quantization", "mode"]].copy()
        table["A100_kJ"] = float("nan")

    key_cols = ["model", "quantization", "mode"]
    table_idx = table.set_index(key_cols)
    summary_idx = summary.set_index(key_cols)

    if "A100_kJ" not in table_idx.columns:
        table_idx["A100_kJ"] = np.nan

    replacement = summary_idx[["energy_mean_kJ"]].rename(columns={"energy_mean_kJ": "A100_kJ"})
    replacement["A100_kJ"] = replacement["A100_kJ"].round(3)

    table_idx.update(replacement)

    updated = table_idx.reset_index()
    updated.to_csv(path, index=False, float_format="%.3f", na_rep="")
    return updated


def draw_energy_per_success(problems: pd.DataFrame) -> None:
    models = model_order(problems)
    quant_pairs = [("F16", "FP16"), ("Q4_K_M", "Q4_K_M")]
    modes = ["base", "full_tool"]

    plot_data = []
    for quant, quant_label in quant_pairs:
        for model in models:
            for mode in modes:
                energy, attempts = _energy_to_success(problems, model, quant, mode)
                if energy.empty:
                    continue
                low, high = bootstrap_ci(energy)
                plot_data.append(
                    {
                        "quant": quant,
                        "quant_label": quant_label,
                        "model": model,
                        "mode": mode,
                        "energy_mean": energy.mean(),
                        "energy_low": low,
                        "energy_high": high,
                        "attempts_mean": attempts.mean(),
                    }
                )

    if not plot_data:
        return

    df = pd.DataFrame(plot_data)
    fig, axes = plt.subplots(len(quant_pairs), 1, figsize=(10, 8), sharex=True)
    if len(quant_pairs) == 1:
        axes = [axes]

    width = 0.18
    base_positions = np.arange(len(models))

    mode_handles = [Patch(facecolor=MODE_COLORS[m], edgecolor="#222222", label=MODE_LABELS[m]) for m in modes]
    quant_handles = [Patch(facecolor="#ffffff", edgecolor="#222222", hatch="" if q == "F16" else "//", label=QUANT_LABELS[q]) for q, _ in quant_pairs]

    for ax, (quant, quant_label) in zip(axes, quant_pairs):
        ax.set_title(f"{quant_label}")
        quant_df = df[df["quant"] == quant]
        for idx_mode, mode in enumerate(modes):
            offsets = base_positions + (idx_mode - (len(modes) - 1) / 2) * (width + 0.1)
            mode_df = quant_df[quant_df["mode"] == mode]
            for offset, model in zip(offsets, models):
                row = mode_df[mode_df["model"] == model]
                if row.empty:
                    continue
                mean_val = row["energy_mean"].iloc[0]
                low = row["energy_low"].iloc[0]
                high = row["energy_high"].iloc[0]
                attempts = row["attempts_mean"].iloc[0]
                ax.bar(
                    offset,
                    mean_val,
                    width=width,
                    color=MODE_COLORS[mode],
                    hatch="" if quant == "F16" else "//",
                    edgecolor="#222222",
                    linewidth=0.5,
                    zorder=3,
                )
                ax.errorbar(
                    offset,
                    mean_val,
                    yerr=[[mean_val - low], [high - mean_val]],
                    fmt="none",
                    ecolor="#222222",
                    elinewidth=0.8,
                    capsize=3,
                    zorder=4,
                )
                ax.text(
                    offset,
                    mean_val + max(5.0, mean_val * 0.03),
                    f"μ={attempts:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=_scale_font(8),
                )
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_ylabel("Energy per solved problem (J)")

    axes[-1].set_xticks(base_positions)
    axes[-1].set_xticklabels([MODEL_LABELS.get(model, model) for model in models])
    axes[0].legend(handles=mode_handles + quant_handles, loc="upper right", frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "energy_per_success_qwen3_4b.pdf")
    plt.close(fig)


def load_energy_table(table_path: Path | None = None) -> pd.DataFrame:
    path = table_path or (FIG_DIR / "energy_per_success_table.csv")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['model'] = df['model'].str.lower()
    df['quantization'] = df['quantization'].str.upper()
    df['mode'] = df['mode'].str.lower()
    return df


def draw_energy_per_success_panels() -> None:
    df = load_energy_table()
    if df.empty:
        return

    devices = [
        ("A100_kJ", "A100"),
        ("MacBook_M2_Max_kJ", "MacBook M2 Max"),
        ("iPhone15_kJ", "iPhone 15"),
    ]
    models_order = [
        "deepseek-r1-distill-qwen-1.5b",
        "gemma-3n-e2b-it",
        "llama-3.2-3b-instruct",
        "qwen-3-0.6b",
        "qwen-3-4b",
    ]
    quant_order = ["Q4_K_M", "F16"]
    quant_labels = {"Q4_K_M": "Q4", "F16": "FP16"}
    mode_order = ["base", "full_tool"]

    bar_specs = [
        ("Q4_K_M", "base"),
        ("Q4_K_M", "full_tool"),
        ("F16", "base"),
        ("F16", "full_tool"),
    ]
    width = 0.22
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(bar_specs))

    x_idx = np.arange(len(models_order))
    fig, axes = plt.subplots(len(devices), 1, figsize=(11.5, 9.5), sharex=True)

    for ax, (value_column, title) in zip(axes, devices):
        ax.set_facecolor("#f7f7f7")
        ax.set_axisbelow(True)

        for idx, (quant, mode) in enumerate(bar_specs):
            values = []
            for model in models_order:
                match = df[(df['model'] == model) & (df['quantization'] == quant) & (df['mode'] == mode)]
                val = match[value_column].iloc[0] if not match.empty else np.nan
                values.append(val)
            color = MODE_COLORS[mode]
            hatch = "//" if quant == "F16" else ""
            bar_positions = x_idx + offsets[idx]
            ax.bar(
                bar_positions,
                values,
                width=width,
                color=color,
                edgecolor="#2f2f2f",
                linewidth=0.6,
                hatch=hatch,
                alpha=0.9,
                label=(quant_labels[quant] + " " + mode.title()) if idx == 0 else None,
            )

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        ax.set_title(title)
        ax.set_ylabel("Energy per success (kJ)")
        # ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        positive = df[value_column].dropna()
        positive = positive[positive > 0]
        if not positive.empty:
            min_positive = positive.min()
            if min_positive < 0.05:
                ymin = 0.0
            else:
                ymin = min_positive * 0.6
            ymax = positive.max() * 1.8
            if ymax <= ymin:
                ymax = ymin + 0.1
            ax.set_ylim(ymin, ymax)

        ax.set_xticks(x_idx)
        if ax is axes[-1]:
            labels = [MODEL_LABELS.get(model, model) for model in models_order]
            ax.set_xticklabels(labels, rotation=0, ha="center")
        else:
            ax.set_xticklabels([])

    legend_elements = [
        Patch(facecolor=MODE_COLORS[mode], edgecolor="#2f2f2f", linewidth=0.6, label=mode.title())
        for mode in mode_order
    ] + [
        Patch(facecolor="#dcdcdc", edgecolor="#2f2f2f", linewidth=0.6, hatch="//", label="FP16"),
        Patch(facecolor="#dcdcdc", edgecolor="#2f2f2f", linewidth=0.6, hatch="", label="Q4"),
    ]

    fig.legend(
        legend_elements,
        ["Base", "Full-Tool", "FP16", "Q4"],
        loc="upper center",
        frameon=False,
        ncol=4,
        bbox_to_anchor=(0.5, 1.00),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_panels = FIG_DIR / "energy_per_success_panels.pdf"
    fig.savefig(output_panels)
    fig.savefig(FIG_DIR / "energy_per_success_qwen3_4b.pdf")
    plt.close(fig)


def _lighten(color: str, amount: float = 0.5) -> Tuple[float, float, float]:
    base = mcolors.to_rgb(color)
    return tuple(1 - amount * (1 - comp) for comp in base)


@lru_cache(maxsize=1)
def _tool_usage_summary() -> pd.DataFrame:
    from collections import defaultdict
    from analysis_scripts.run_finder import find_runs
    from analysis_scripts.benchmark_loader import iter_problem_records

    aggregates: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(lambda: {"total": 0.0, "duplicates": 0.0, "count": 0.0})
    for run in find_runs(Path("data/results")):
        for record in iter_problem_records(run):
            calls = record.get("tool_calls") or []
            total = float(len(calls))
            unique = float(len({call.get("name") for call in calls})) if calls else 0.0
            duplicates = max(total - unique, 0.0)
            key = (record["model"], record["quantization"], record["mode"])
            aggregates[key]["total"] += total
            aggregates[key]["duplicates"] += duplicates
            aggregates[key]["count"] += 1.0

    rows = []
    for (model, quant, mode), values in aggregates.items():
        if values["count"] == 0:
            continue
        rows.append(
            {
                "model": model,
                "quantization": quant,
                "mode": mode,
                "avg_total": values["total"] / values["count"],
                "avg_duplicates": values["duplicates"] / values["count"],
            }
        )
    return pd.DataFrame(rows)


def draw_thinking_share(summary: pd.DataFrame, problems: pd.DataFrame) -> None:
    thinking_models = ["qwen-3-4b", "qwen-3-0.6b"]
    quant_order = ["F16", "Q4_K_M"]
    modes = MODE_ORDER

    subset = problems[problems["model"].isin(thinking_models)]
    if subset.empty:
        return

    # Single row figure with both models side-by-side (smaller width for larger text)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))

    width = 0.14
    # Create positions for both models and all mode combinations
    n_groups = len(modes) * len(thinking_models)
    base_positions = np.arange(n_groups)

    # Group positions: [Qwen 4B: base, tool_sub, full] [Qwen 0.6B: base, tool_sub, full]
    for model_idx, model in enumerate(thinking_models):
        display_name = MODEL_LABELS.get(model, model)
        model_subset = subset[subset["model"] == model]

        for quant_idx, quant in enumerate(quant_order):
            thinking_vals = []
            actionable_vals = []
            positions = []

            for mode_idx, mode in enumerate(modes):
                # Position calculation: model offset + mode offset + quant offset
                pos = model_idx * len(modes) + mode_idx + (quant_idx - (len(quant_order) - 1) / 2) * width
                positions.append(pos)

                rows = model_subset[(model_subset["quantization"] == quant) & (model_subset["mode"] == mode)]
                if rows.empty:
                    thinking_vals.append(0.0)
                    actionable_vals.append(0.0)
                    continue
                thinking_tokens = rows["thinking_tokens"].mean()
                actionable_tokens = rows["regular_tokens"].mean()
                thinking_tokens = 0.0 if math.isnan(thinking_tokens) else thinking_tokens
                actionable_tokens = 0.0 if math.isnan(actionable_tokens) else actionable_tokens
                thinking_vals.append(thinking_tokens)
                actionable_vals.append(actionable_tokens)

            hatch = "" if quant == "F16" else "//"
            ax.bar(
                positions,
                actionable_vals,
                width=width,
                color=ACTIONABLE_COLOR,
                hatch=hatch,
                edgecolor="#222222",
                linewidth=0.4,
                zorder=2,
            )
            ax.bar(
                positions,
                thinking_vals,
                width=width,
                bottom=actionable_vals,
                color=THINKING_COLOR,
                hatch=hatch,
                edgecolor="#222222",
                linewidth=0.4,
                zorder=3,
            )

            # Add actionable token counts on top
            for pos, think, action in zip(positions, thinking_vals, actionable_vals):
                total = think + action
                if total > 0:
                    ax.text(
                        pos,
                        total + max(20.0, total * 0.05),
                        f"{int(round(action))}",
                        ha="center",
                        va="bottom",
                        fontsize=_scale_font(7),
                    )

    # Set x-axis labels with model names and modes
    ax.set_ylabel("Tokens")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    # Create tick positions and labels (mode labels only)
    tick_positions = []
    tick_labels = []
    for model_idx, model in enumerate(thinking_models):
        for mode_idx, mode in enumerate(modes):
            tick_positions.append(model_idx * len(modes) + mode_idx)
            tick_labels.append(MODE_LABELS[mode])

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=_scale_font(10))

    # Add model titles above the x-axis, slightly separated
    for model_idx, model in enumerate(thinking_models):
        center_pos = model_idx * len(modes) + (len(modes) - 1) / 2
        ax.text(
            center_pos,
            -0.12,
            MODEL_LABELS.get(model, model),
            ha="center",
            va="top",
            fontsize=_scale_font(10),
            transform=ax.get_xaxis_transform(),
        )

    # Add vertical separator between models
    separator_x = len(modes) - 0.5
    ax.axvline(separator_x, color="#888888", linestyle="--", linewidth=1.0, alpha=0.5)

    # Legend
    activity_handles = [
        Patch(facecolor=ACTIONABLE_COLOR, edgecolor="#222222", label="Actionable tokens"),
        Patch(facecolor=THINKING_COLOR, edgecolor="#222222", label="Thinking tokens"),
    ]
    quant_handles = [Patch(facecolor="#ffffff", edgecolor="#222222", hatch="" if q == "F16" else "//", label=QUANT_LABELS[q]) for q in quant_order]
    handles = activity_handles + quant_handles
    labels = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=4)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "thinking_share_qwen3_4b.pdf")
    plt.close(fig)


def draw_thinking_success_correlation(problems: pd.DataFrame) -> None:
    target_models = ["qwen-3-4b", "qwen-3-0.6b", "deepseek-r1-distill-qwen-1.5b"]
    if problems.empty:
        return

    records: List[Dict[str, object]] = []
    for model in target_models:
        model_df = problems[problems["model"] == model]
        if model_df.empty:
            continue
        for quant in ["F16", "Q4_K_M"]:
            quant_df = model_df[model_df["quantization"] == quant].copy()
            if quant_df.empty:
                continue
            quant_df["thinking_tokens"] = quant_df["thinking_tokens"].fillna(0.0)
            quant_df["actionable_tokens"] = quant_df["regular_tokens"].fillna(0.0)
            quant_df["total_tokens"] = quant_df["thinking_tokens"] + quant_df["actionable_tokens"]

            for mode in MODE_ORDER:
                mode_df = quant_df[quant_df["mode"] == mode]
                if mode_df.empty:
                    continue
                successes = mode_df["success"].astype(int)
                thinking = mode_df["thinking_tokens"]
                if successes.nunique() <= 1:
                    corr = float("nan")
                else:
                    corr = float(np.corrcoef(thinking, successes)[0, 1])
                records.append(
                    {
                        "model": model,
                        "quantization": quant,
                        "mode": mode,
                        "corr": corr,
                        "thinking_success_mean": thinking[mode_df["success"] == True].mean(),
                        "thinking_failure_mean": thinking[mode_df["success"] == False].mean(),
                    }
                )

    if not records:
        return

    corr_df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, len(target_models), figsize=(12.5, 3.6), sharey=True)
    if len(target_models) == 1:
        axes = [axes]

    quant_palette = {"F16": PRECISION_COLORS.get("F16", "#f9c8d2"), "Q4_K_M": PRECISION_COLORS.get("Q4_K_M", "#cfe2cf")}
    mode_markers = {"base": "o", "tool_submission": "s", "full_tool": "^"}

    for ax, model in zip(axes, target_models):
        subset = corr_df[corr_df["model"] == model]
        if subset.empty:
            ax.set_visible(False)
            continue

        for (_, row) in subset.iterrows():
            quant = row["quantization"]
            mode = row["mode"]
            ax.scatter(
                row["thinking_failure_mean"],
                row["corr"],
                color=quant_palette.get(quant, "#4e79a7"),
                marker=mode_markers.get(mode, "o"),
                s=80,
                edgecolor="#222222",
                linewidth=0.5,
            )
        ax.axhline(0, color="#999999", linestyle=":", linewidth=0.8)
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("Avg thinking tokens (failures)")
        ax.grid(alpha=0.35)

    axes[0].set_ylabel("Corr(thinking tokens, success)")

    legend_handles = [
        plt.Line2D([], [], marker=marker, linestyle="", color="#555555", markersize=7, label=MODE_LABELS.get(mode, mode.title()))
        for mode, marker in mode_markers.items()
    ]
    legend_handles.extend(
        [
            plt.Line2D([], [], marker="o", linestyle="", color=quant_palette.get(quant, "#4e79a7"), markersize=7, label=QUANT_LABELS.get(quant, quant))
            for quant in ["F16", "Q4_K_M"]
        ]
    )
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper right", frameon=False, ncol=2)

    fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(FIG_DIR / "thinking_success_correlation.pdf")
    plt.close(fig)


def _wilson_interval(successes: np.ndarray, trials: np.ndarray, z: float = 1.96) -> Tuple[np.ndarray, np.ndarray]:
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    centre = successes + z**2 / 2
    denom = trials + z**2
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = successes / trials
        margin = z * np.sqrt((fraction * (1 - fraction) + z**2 / (4 * trials)) / trials)
        lower = (centre / denom) - (margin / denom)
        upper = (centre / denom) + (margin / denom)
    lower = np.clip(lower, 0.0, 1.0)
    upper = np.clip(upper, 0.0, 1.0)
    return lower, upper


def _binned_success(df: pd.DataFrame, value_col: str, bins: int = 40) -> pd.DataFrame | None:
    if df.empty or df[value_col].nunique() < 4:
        return None
    min_val = df[value_col].min()
    max_val = df[value_col].max()
    if not np.isfinite(min_val) or not np.isfinite(max_val) or min_val == max_val:
        return None
    bin_edges = np.linspace(min_val, max_val, bins)
    grouped = (
        df.groupby(pd.cut(df[value_col], bin_edges, include_lowest=True, duplicates="drop"))
        .agg(
            tokens_mean=(value_col, "mean"),
            success_mean=("success", "mean"),
            successes=("success", "sum"),
            trials=("success", "count"),
        )
        .dropna()
        .reset_index(drop=True)
    )
    if grouped.empty:
        return None
    low, high = _wilson_interval(grouped["successes"].to_numpy(), grouped["trials"].to_numpy())
    grouped["ci_low"] = low
    grouped["ci_high"] = high
    return grouped


def draw_thinking_success_curves(problems: pd.DataFrame) -> None:
    target_models = ["qwen-3-4b", "qwen-3-0.6b"]
    if problems.empty:
        return

    palette = {"F16": PRECISION_COLORS.get("F16", "#f9c8d2"), "Q4_K_M": PRECISION_COLORS.get("Q4_K_M", "#cfe2cf")}
    line_styles = {"base": "-", "tool_submission": "--", "full_tool": ":"}
    rng = np.random.default_rng(2025)

    fig, axes = plt.subplots(len(target_models), 1, figsize=(8.6, 6.0), sharex=True, sharey=True)
    if len(target_models) == 1:
        axes = [axes]

    for ax, model in zip(axes, target_models):
        model_df = problems[problems["model"] == model].copy()
        if model_df.empty:
            ax.set_visible(False)
            continue
        model_df["thinking_tokens"] = model_df["thinking_tokens"].fillna(0.0)
        model_df["success"] = model_df["success"].astype(int)

        # lightly show all attempts
        sample_idx = model_df.index
        y = model_df.loc[sample_idx, "success"].to_numpy().astype(float)
        jitter = rng.uniform(-0.02, 0.02, size=len(sample_idx))
        ax.scatter(
            model_df.loc[sample_idx, "thinking_tokens"],
            y + jitter,
            s=6,
            color="#4e79a7",
            alpha=0.05,
            linewidth=0,
        )

        handles = []
        labels = []
        for quant in ["F16", "Q4_K_M"]:
            quant_df = model_df[model_df["quantization"] == quant]
            if quant_df.empty:
                continue
            color = palette.get(quant, "#4e79a7")
            for mode in MODE_ORDER:
                mode_df = quant_df[quant_df["mode"] == mode]
                if mode_df.empty:
                    continue
                binned = _binned_success(mode_df[["thinking_tokens", "success"]], "thinking_tokens")
                if binned is None:
                    continue
                ax.plot(
                    binned["tokens_mean"],
                    binned["success_mean"],
                    color=color,
                    linestyle=line_styles.get(mode, "-"),
                    linewidth=2.0,
                    label=f"{MODE_LABELS.get(mode, mode.title())} {QUANT_LABELS.get(quant, quant)}",
                )
                ax.fill_between(
                    binned["tokens_mean"],
                    binned["ci_low"],
                    binned["ci_high"],
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                )

        ax.axhline(0, color="#bbbbbb", linestyle=":", linewidth=0.8)
        ax.axhline(1, color="#bbbbbb", linestyle=":", linewidth=0.8)
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_ylabel("Success rate")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.35)

    axes[-1].set_xlabel("Thinking tokens per attempt")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2)

    max_tokens = problems["thinking_tokens"].dropna().max()
    if np.isfinite(max_tokens):
        axes[-1].set_xlim(0, max_tokens * 1.02)

    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig(FIG_DIR / "thinking_tokens_success_curve_qwen.pdf")
    plt.close(fig)


def main() -> None:
    summary = load_summary()
    problems = load_problem_dataframe()
    prefill_df = load_prefill_curves()

    draw_accuracy(summary)
    draw_tool_benefit(summary)
    draw_overhead_breakdown(summary, problems)
    draw_prefill(prefill_df)
    draw_quantization(summary)
    draw_energy_per_success(problems)
    refresh_energy_per_success_table(problems)
    draw_energy_per_success_panels()
    draw_thinking_share(summary, problems)
    draw_thinking_success_correlation(problems)
    draw_thinking_success_curves(problems)


if __name__ == "__main__":
    main()
