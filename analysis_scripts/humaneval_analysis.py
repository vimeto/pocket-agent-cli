#!/usr/bin/env python3
"""
HumanEval Benchmark Analysis Module.

This module provides functions to load and analyze HumanEval benchmark results,
generating figures and tables with consistent styling matching the paper figures.
Uses the same color schemes as generate_paper_figures.py.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# Import shared styling from generate_paper_figures
from analysis_scripts.generate_paper_figures import (
    MODE_ORDER,
    MODE_LABELS,
    MODE_COLORS,
    PRECISION_COLORS,
    QUANT_LABELS,
    MODEL_LABELS,
    FIG_DIR,
    FONT_SCALE,
    _scale_font,
)

# HumanEval results directory
HUMANEVAL_RESULTS_DIR = Path("tmp/results")


def parse_dir_name(dir_name: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Parse result directory name to extract model, version, mode, timestamp."""
    if "Q4_K_M" in dir_name:
        version = "Q4_K_M"
        idx = dir_name.index("Q4_K_M")
    elif "F16" in dir_name:
        version = "F16"
        idx = dir_name.index("F16")
    else:
        return None, None, None, None

    model = dir_name[len("humaneval_"):idx - 1]
    rest = dir_name[idx + len(version) + 1:]

    # Handle modes that contain underscores
    if rest.startswith("tool_submission_"):
        mode = "tool_submission"
        timestamp = rest[len("tool_submission_"):]
    elif rest.startswith("full_tool_"):
        mode = "full_tool"
        timestamp = rest[len("full_tool_"):]
    elif rest.startswith("base_"):
        mode = "base"
        timestamp = rest[len("base_"):]
    else:
        parts = rest.split("_")
        mode = parts[0]
        timestamp = "_".join(parts[1:])

    return model, version, mode, timestamp


def load_all_benchmark_summaries(results_dir: Path = HUMANEVAL_RESULTS_DIR) -> Dict[str, Dict]:
    """Load all benchmark summaries grouped by configuration."""
    all_data = defaultdict(lambda: {
        "pass_at_1": [],
        "pass_at_3": [],
        "pass_at_5": [],
        "pass_at_10": [],
        "total_problems": 0,
        "passed_problems": 0,
        "total_duration": 0,
        "total_energy_joules": 0,
        "avg_context_length": [],
        "full_tool_iterations": [],
        "submission_rate": [],
        "tasks_completed": 0,
        "timestamps": [],
    })

    if not results_dir.exists():
        return dict(all_data)

    for result_dir in sorted(results_dir.iterdir()):
        if not result_dir.is_dir() or not result_dir.name.startswith("humaneval_"):
            continue

        model, version, mode, timestamp = parse_dir_name(result_dir.name)
        if model is None:
            continue

        key = f"{model}|{version}|{mode}"

        for task_dir in result_dir.iterdir():
            if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
                continue

            summary_file = task_dir / "benchmark_summary.json"
            if not summary_file.exists():
                continue

            try:
                with open(summary_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            for session in data.get("sessions", []):
                stats = session.get("aggregate_stats", {})
                pass_at_k = stats.get("pass_at_k", {})

                agg = all_data[key]
                agg["timestamps"].append(timestamp)

                if "overall_pass_at_1" in pass_at_k:
                    agg["pass_at_1"].append(pass_at_k["overall_pass_at_1"])
                if "overall_pass_at_3" in pass_at_k:
                    agg["pass_at_3"].append(pass_at_k["overall_pass_at_3"])
                if "overall_pass_at_5" in pass_at_k:
                    agg["pass_at_5"].append(pass_at_k["overall_pass_at_5"])
                if "overall_pass_at_10" in pass_at_k:
                    agg["pass_at_10"].append(pass_at_k["overall_pass_at_10"])

                agg["total_problems"] += stats.get("total_problems", 0)
                agg["passed_problems"] += stats.get("passed_problems", 0)
                agg["total_duration"] += stats.get("total_duration_seconds", 0)
                agg["tasks_completed"] += 1

                if "system_metrics" in stats:
                    agg["total_energy_joules"] += stats["system_metrics"].get("total_energy_joules", 0)

                if "context_length" in stats:
                    agg["avg_context_length"].append(stats["context_length"].get("avg", 0))

                if "full_tool_iterations" in stats:
                    agg["full_tool_iterations"].append(stats["full_tool_iterations"].get("avg", 0))
                    agg["submission_rate"].append(stats["full_tool_iterations"].get("submission_rate", 0))

    return dict(all_data)


def safe_mean(values: List) -> Optional[float]:
    """Calculate mean, handling empty lists."""
    filtered = [v for v in values if v is not None]
    return statistics.mean(filtered) if filtered else None


def get_humaneval_pass_at_1(model: str, version: str, mode: str) -> Optional[float]:
    """Get Pass@1 for a specific configuration."""
    data = load_all_benchmark_summaries()
    key = f"{model}|{version}|{mode}"
    if key in data:
        return safe_mean(data[key]["pass_at_1"])
    return None


def draw_humaneval_mode_comparison() -> None:
    """
    Draw HumanEval mode comparison figure with consistent paper styling.

    Uses the same color scheme as the main paper figures (MODE_COLORS).
    """
    data = load_all_benchmark_summaries()
    if not data:
        print("No HumanEval data available")
        return

    models = sorted(set(k.split("|")[0] for k in data.keys()))
    # Filter to models we care about
    models = [m for m in models if m in MODEL_LABELS]

    if not models:
        print("No recognized models in HumanEval data")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(models))
    width = 0.25

    for i, mode in enumerate(MODE_ORDER):
        values = []
        for model in models:
            # Use F16 values preferentially
            key = f"{model}|F16|{mode}"
            if key in data and data[key]["pass_at_1"]:
                values.append(safe_mean(data[key]["pass_at_1"]) * 100)
            else:
                # Try Q4_K_M
                key = f"{model}|Q4_K_M|{mode}"
                if key in data and data[key]["pass_at_1"]:
                    values.append(safe_mean(data[key]["pass_at_1"]) * 100)
                else:
                    values.append(0)

        ax.bar(
            x + i * width,
            values,
            width,
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
            edgecolor="#222222",
            linewidth=0.5,
        )

    ax.set_ylabel('Pass@1 Accuracy (%)')
    ax.set_title('HumanEval Performance: Mode Comparison (F16)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.legend()
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_ylim(0, 100)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "humaneval_mode_comparison.pdf")
    plt.close(fig)

    print(f"  Saved humaneval_mode_comparison.pdf")


def draw_humaneval_quantization_impact() -> None:
    """
    Draw HumanEval quantization impact figure with consistent paper styling.

    Uses PRECISION_COLORS for Q4_K_M vs F16 comparison.
    """
    data = load_all_benchmark_summaries()
    if not data:
        print("No HumanEval data available")
        return

    models = sorted(set(k.split("|")[0] for k in data.keys()))
    models = [m for m in models if m in MODEL_LABELS]

    if not models:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, mode in enumerate(MODE_ORDER):
        ax = axes[i]

        q4_vals = []
        f16_vals = []
        model_names = []

        for model in models:
            q4_key = f"{model}|Q4_K_M|{mode}"
            f16_key = f"{model}|F16|{mode}"

            q4_p1 = safe_mean(data.get(q4_key, {}).get("pass_at_1", [])) or 0
            f16_p1 = safe_mean(data.get(f16_key, {}).get("pass_at_1", [])) or 0

            if q4_p1 > 0 or f16_p1 > 0:
                q4_vals.append(q4_p1 * 100)
                f16_vals.append(f16_p1 * 100)
                model_names.append(MODEL_LABELS.get(model, model[:10]))

        if not model_names:
            ax.set_visible(False)
            continue

        x = np.arange(len(model_names))
        width = 0.35

        ax.bar(
            x - width / 2,
            q4_vals,
            width,
            label='Q4_K_M',
            color=PRECISION_COLORS["Q4_K_M"],
            edgecolor="#222222",
            linewidth=0.5,
        )
        ax.bar(
            x + width / 2,
            f16_vals,
            width,
            label='F16',
            color=PRECISION_COLORS["F16"],
            edgecolor="#222222",
            linewidth=0.5,
        )

        ax.set_ylabel('Pass@1 (%)')
        ax.set_title(f'{MODE_LABELS[mode]} Mode')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "humaneval_quantization_impact.pdf")
    plt.close(fig)

    print(f"  Saved humaneval_quantization_impact.pdf")


def draw_humaneval_energy_efficiency() -> None:
    """
    Draw HumanEval energy per successful solution figure with improved visual encoding.

    This uses the paper's key metric: "Energy per Successful Solution" (kJ/success).
    Lower values are better - they indicate more energy-efficient configurations.

    Visual encoding:
    - Bar height: Energy per successful solution (kJ) - lower is better
    - Colors: Modes (Base = green, Full Tool = blue)
    - Hatching: Quantization (F16 = solid, Q4 = hatched)
    - Annotations: Pass@1 accuracy (%) shown above each bar
    """
    data = load_all_benchmark_summaries()
    if not data:
        print("No HumanEval data available")
        return

    # Group data by model
    model_data = defaultdict(list)
    for key, d in data.items():
        if d["passed_problems"] > 0 and d["total_energy_joules"] > 0:
            model, version, mode = key.split("|")
            if model not in MODEL_LABELS:
                continue
            # Energy per successful solution (kJ/success) - LOWER IS BETTER
            energy_per_success = (d["total_energy_joules"] / 1000) / d["passed_problems"]
            p1 = safe_mean(d["pass_at_1"]) * 100 if d["pass_at_1"] else 0
            model_data[model].append({
                "version": version,
                "mode": mode,
                "energy_per_success": energy_per_success,
                "accuracy": p1,
            })

    if not model_data:
        print("No energy data available")
        return

    # Create figure with model-grouped bars
    fig, ax = plt.subplots(figsize=(14, 6.5))

    # Define hatching patterns for quantization
    quant_hatch = {"F16": "", "Q4_K_M": "///"}

    # Order models by minimum energy per success (best/lowest first)
    model_order = sorted(model_data.keys(),
                        key=lambda m: min(d["energy_per_success"] for d in model_data[m]))

    # Calculate bar positions
    n_models = len(model_order)
    modes_to_show = ["base", "full_tool"]  # Focus on base vs full_tool comparison
    quants_to_show = ["F16", "Q4_K_M"]

    n_bars_per_model = len(modes_to_show) * len(quants_to_show)
    bar_width = 0.18
    group_width = n_bars_per_model * bar_width + 0.15

    x_positions = np.arange(n_models) * group_width

    # Track max value for y-axis scaling
    max_energy = 0

    # Plot bars
    for model_idx, model in enumerate(model_order):
        bar_idx = 0
        for mode in modes_to_show:
            for quant in quants_to_show:
                # Find matching data
                matching = [d for d in model_data[model]
                           if d["mode"] == mode and d["version"] == quant]
                if not matching:
                    bar_idx += 1
                    continue

                d = matching[0]
                x = x_positions[model_idx] + (bar_idx - n_bars_per_model/2 + 0.5) * bar_width

                energy = d["energy_per_success"]
                max_energy = max(max_energy, energy)

                ax.bar(
                    x,
                    energy,
                    width=bar_width,
                    color=MODE_COLORS[mode],
                    hatch=quant_hatch[quant],
                    edgecolor="#222222",
                    linewidth=0.8,
                    zorder=3,
                )

                # Add Pass@1 accuracy annotation on top of bar
                ax.annotate(
                    f'{d["accuracy"]:.0f}%',
                    xy=(x, energy),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=_scale_font(7),
                    color='#444444',
                )

                bar_idx += 1

    # Set up axes
    ax.set_ylabel('Energy per Successful Solution (kJ)')
    ax.set_xlabel('Model')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in model_order])
    ax.set_title('HumanEval: Energy per Successful Solution by Model, Mode, and Quantization')
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max_energy * 1.15)  # Add headroom for annotations

    # Create legend with mode colors and quantization patterns
    from matplotlib.patches import Patch
    mode_handles = [
        Patch(facecolor=MODE_COLORS[mode], edgecolor="#222222", label=MODE_LABELS[mode])
        for mode in modes_to_show
    ]
    quant_handles = [
        Patch(facecolor="#cccccc", edgecolor="#222222", hatch=quant_hatch[q],
              label=QUANT_LABELS.get(q, q))
        for q in quants_to_show
    ]

    ax.legend(handles=mode_handles + quant_handles, loc='upper right', ncol=2, frameon=True)

    # Add explanatory subtitle below the figure
    subtitle = "Bar height: Energy per successful solution (lower is better)  |  Annotations: Pass@1 accuracy (%)"
    ax.text(0.5, -0.12, subtitle, transform=ax.transAxes, ha='center', va='top',
            fontsize=_scale_font(9), style='italic', color='#555555')

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)  # Make room for subtitle
    fig.savefig(FIG_DIR / "humaneval_energy_efficiency.pdf")
    plt.close(fig)

    print(f"  Saved humaneval_energy_efficiency.pdf")


def generate_humaneval_summary_table() -> pd.DataFrame:
    """
    Generate a summary DataFrame of HumanEval results.

    Returns:
        DataFrame with columns: model, quantization, mode, pass_at_1, pass_at_10,
        energy_kj, problems, passed
    """
    data = load_all_benchmark_summaries()

    records = []
    for key, d in data.items():
        model, version, mode = key.split("|")

        p1 = safe_mean(d["pass_at_1"])
        p10 = safe_mean(d["pass_at_10"])

        records.append({
            "model": model,
            "quantization": version,
            "mode": mode,
            "pass_at_1": p1 * 100 if p1 else None,
            "pass_at_10": p10 * 100 if p10 else None,
            "energy_kj": d["total_energy_joules"] / 1000 if d["total_energy_joules"] > 0 else None,
            "problems": d["total_problems"],
            "passed": d["passed_problems"],
        })

    return pd.DataFrame(records)


def generate_mode_impact_table() -> pd.DataFrame:
    """
    Generate a table showing mode impact (base vs tool_submission vs full_tool).

    Returns:
        DataFrame with delta columns showing improvement/degradation
    """
    data = load_all_benchmark_summaries()

    # Group by model/version
    grouped = defaultdict(dict)
    for key, d in data.items():
        model, version, mode = key.split("|")
        p1 = safe_mean(d["pass_at_1"])
        grouped[(model, version)][mode] = p1

    records = []
    for (model, version) in sorted(grouped.keys()):
        modes = grouped[(model, version)]
        base = modes.get("base")
        ts = modes.get("tool_submission")
        ft = modes.get("full_tool")

        delta_bf = (ft - base) * 100 if (base and ft) else None

        records.append({
            "model": model,
            "quantization": version,
            "base": base * 100 if base else None,
            "tool_submission": ts * 100 if ts else None,
            "full_tool": ft * 100 if ft else None,
            "delta_base_to_full": delta_bf,
        })

    return pd.DataFrame(records)


def print_key_insights() -> None:
    """Print key insights from HumanEval analysis."""
    data = load_all_benchmark_summaries()
    if not data:
        print("No HumanEval data available")
        return

    print("\n" + "=" * 60)
    print("KEY HUMANEVAL INSIGHTS")
    print("=" * 60)

    # Capability threshold analysis
    print("\n1. CAPABILITY THRESHOLD CONFIRMATION")
    print("-" * 40)

    model_modes = defaultdict(dict)
    for key, d in data.items():
        model, version, mode = key.split("|")
        if version == "F16":
            p1 = safe_mean(d["pass_at_1"])
            if p1:
                model_modes[model][mode] = p1

    for model in sorted(model_modes.keys()):
        modes = model_modes[model]
        base = modes.get("base", 0)
        ft = modes.get("full_tool", 0)

        if ft > base + 0.05:
            benefit = f"+{(ft - base) * 100:.0f}pp"
            status = "BENEFITS"
        elif ft < base - 0.05:
            benefit = f"{(ft - base) * 100:.0f}pp"
            status = "HURT"
        else:
            benefit = f"{(ft - base) * 100:+.0f}pp"
            status = "NEUTRAL"

        model_label = MODEL_LABELS.get(model, model)
        print(f"  {model_label:20} Base: {base * 100:5.1f}% → Full: {ft * 100:5.1f}% ({benefit}) [{status}]")

    # Best performers
    print("\n2. BEST CONFIGURATIONS")
    print("-" * 40)

    best_acc = max(data.items(), key=lambda x: safe_mean(x[1]["pass_at_1"]) or 0)
    model, version, mode = best_acc[0].split("|")
    p1 = safe_mean(best_acc[1]["pass_at_1"])
    print(f"  Highest Accuracy: {MODEL_LABELS.get(model, model)} {version} {mode} = {p1 * 100:.1f}%")

    # Energy efficiency
    efficiencies = []
    for key, d in data.items():
        if d["passed_problems"] > 0 and d["total_energy_joules"] > 0:
            eff = d["passed_problems"] / (d["total_energy_joules"] / 1000)
            efficiencies.append((eff, key, d))

    if efficiencies:
        best_eff = max(efficiencies, key=lambda x: x[0])
        model, version, mode = best_eff[1].split("|")
        print(f"  Best Energy Efficiency: {MODEL_LABELS.get(model, model)} {version} {mode} = {best_eff[0]:.2f} passed/kJ")

    print("\n" + "=" * 60)


def main() -> None:
    """Generate all HumanEval analysis outputs."""
    print("Generating HumanEval analysis...")

    # Ensure output directory exists
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Generate figures
    print("\nGenerating figures...")
    draw_humaneval_mode_comparison()
    draw_humaneval_quantization_impact()
    draw_humaneval_energy_efficiency()

    # Print insights
    print_key_insights()

    # Generate summary table
    print("\nGenerating summary tables...")
    summary_df = generate_humaneval_summary_table()
    if not summary_df.empty:
        output_path = FIG_DIR / "humaneval_summary.csv"
        summary_df.to_csv(output_path, index=False)
        print(f"  Saved {output_path}")

    mode_df = generate_mode_impact_table()
    if not mode_df.empty:
        output_path = FIG_DIR / "humaneval_mode_impact.csv"
        mode_df.to_csv(output_path, index=False)
        print(f"  Saved {output_path}")

    print("\nHumanEval analysis complete!")


if __name__ == "__main__":
    main()
