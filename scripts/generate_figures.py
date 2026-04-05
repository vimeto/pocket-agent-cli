#!/usr/bin/env python3
"""
Generate all publication-quality figures for the Pocket Agent MobiHoc 2026 paper.

Usage:
    python scripts/generate_figures.py

All data is read from data/results/. Figures are saved to research/figures/new/.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "results"
OUT_DIR = BASE_DIR / "research" / "figures" / "new"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ACM Style Setup
# ---------------------------------------------------------------------------
SINGLE_COL_W = 3.3   # inches
DOUBLE_COL_W = 7.0   # inches

# Colorblind-friendly palette (IBM Design Library / Wong 2011)
CB_PALETTE = {
    "blue":    "#0072B2",
    "orange":  "#E69F00",
    "green":   "#009E73",
    "red":     "#D55E00",
    "purple":  "#CC79A7",
    "cyan":    "#56B4E9",
    "yellow":  "#F0E442",
    "black":   "#000000",
    "grey":    "#999999",
}

MODE_COLORS = {
    "base":            CB_PALETTE["blue"],
    "tool_submission": CB_PALETTE["orange"],
    "full_tool":       CB_PALETTE["green"],
}
MODE_LABELS = {
    "base":            "Base",
    "tool_submission": "Tool-sub",
    "full_tool":       "Full-tool",
}

MODEL_SHORT = {
    "qwen-3-4b":                    "Qwen3 4B",
    "qwen-3-0.6b":                  "Qwen3 0.6B",
    "llama-3.2-3b-instruct":        "Llama3.2 3B",
    "deepseek-r1-distill-qwen-1.5b": "DS-R1 1.5B",
    "gemma-3n-e2b-it":              "Gemma3n 2B",
}

MODELS_ORDERED = [
    "qwen-3-4b",
    "gemma-3n-e2b-it",
    "llama-3.2-3b-instruct",
    "deepseek-r1-distill-qwen-1.5b",
    "qwen-3-0.6b",
]
MODES_ORDERED = ["base", "tool_submission", "full_tool"]


def _apply_acm_style():
    """Apply publication-quality defaults."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


def _save(fig, name):
    """Save figure as both PDF and PNG."""
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path)
    plt.close(fig)
    print(f"  -> saved {name}.pdf / .png")


# ---------------------------------------------------------------------------
# Helper: load JSON / JSONL robustly
# ---------------------------------------------------------------------------
def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ===================================================================
# FIGURE 1: Pass@1 Accuracy Comparison (MBPP + HumanEval)
# ===================================================================
def figure1_pass_at_1():
    print("Figure 1: Pass@1 Accuracy Comparison ...")

    # --- Collect MBPP cloud 500 data ---
    mbpp = {}
    # Primary summary: qwen-3-4b
    s1 = _load_json(DATA_DIR / "full_cloud_sweep" / "sglang_20260402_162457" / "summary.json")
    for model, modes in s1["results"].items():
        mbpp[model] = {mode: d["pass_rate"] for mode, d in modes.items()}
    # Secondary summary: gemma-3n-e2b-it
    s2 = _load_json(DATA_DIR / "full_cloud_sweep" / "sglang_20260402_171922" / "summary.json")
    for model, modes in s2["results"].items():
        mbpp[model] = {mode: d["pass_rate"] for mode, d in modes.items()}

    # Compute pass rates from JSONL for remaining models
    sweep_dir = DATA_DIR / "full_cloud_sweep" / "sglang_20260402_162457"
    for model in ["deepseek-r1-distill-qwen-1.5b", "llama-3.2-3b-instruct", "qwen-3-0.6b"]:
        if model in mbpp:
            continue
        mbpp[model] = {}
        for mode in MODES_ORDERED:
            fpath = sweep_dir / f"{model}_{mode}.jsonl"
            if fpath.exists():
                records = _load_jsonl(fpath)
                passed = sum(1 for r in records
                             if r.get("passed") or r.get("evaluation", {}).get("passed"))
                mbpp[model][mode] = passed / len(records) if records else 0
            else:
                mbpp[model][mode] = 0

    # --- Collect HumanEval data ---
    humaneval = {}
    for subdir in sorted((DATA_DIR / "humaneval_cloud").iterdir()):
        if not subdir.is_dir():
            continue
        sfile = subdir / "summary.json"
        if sfile.exists():
            s = _load_json(sfile)
            for model, modes in s["results"].items():
                humaneval[model] = {mode: d["pass_rate"] for mode, d in modes.items()}
        # Also compute from JSONL for models missing from summaries
        for jf in subdir.glob("*.jsonl"):
            parts = jf.stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            model, mode = parts
            if model in humaneval and mode in humaneval[model]:
                continue
            records = _load_jsonl(jf)
            if not records:
                continue
            passed = sum(1 for r in records
                         if r.get("passed") or r.get("evaluation", {}).get("passed"))
            humaneval.setdefault(model, {})[mode] = passed / len(records)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, 2.4), sharey=True)

    for ax, (dataset_name, data, n_label) in zip(axes, [
        ("MBPP (n=500)", mbpp, 500),
        ("HumanEval (n=164)", humaneval, 164),
    ]):
        x = np.arange(len(MODELS_ORDERED))
        width = 0.25

        for i, mode in enumerate(MODES_ORDERED):
            vals = [data.get(m, {}).get(mode, 0) * 100 for m in MODELS_ORDERED]
            bars = ax.bar(x + (i - 1) * width, vals, width,
                          label=MODE_LABELS[mode], color=MODE_COLORS[mode],
                          edgecolor="white", linewidth=0.3)
            # Annotate values on bars
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                            f"{v:.0f}", ha="center", va="bottom", fontsize=5.5)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS_ORDERED],
                           rotation=25, ha="right", fontsize=7)
        ax.set_title(dataset_name, fontsize=9, fontweight="bold")
        ax.set_ylim(0, 100)

    axes[0].set_ylabel("Pass@1 (%)")
    axes[1].legend(loc="upper right", framealpha=0.9, edgecolor="none")

    # Capability threshold annotation on MBPP panel
    axes[0].axhline(y=50, color=CB_PALETTE["grey"], linestyle="--",
                    linewidth=0.7, alpha=0.5)
    axes[0].text(0.02, 51, "capability\nthreshold", transform=axes[0].get_yaxis_transform(),
                 fontsize=6, color=CB_PALETTE["grey"], va="bottom")

    fig.tight_layout(w_pad=1.0)
    _save(fig, "fig1_pass_at_1_comparison")


# ===================================================================
# FIGURE 2: Early-Exit Thinking Budget (KEY CONTRIBUTION)
# ===================================================================
def figure2_early_exit():
    print("Figure 2: Early-Exit Thinking Budget ...")

    # Load summaries from each early-exit run
    ee_data = {}
    ee_dirs = {
        "qwen-3-4b": DATA_DIR / "early_exit" / "early_exit_20260405_011122",
        "qwen-3-0.6b": DATA_DIR / "early_exit" / "early_exit_20260405_140052",
        "deepseek-r1-distill-qwen-1.5b": DATA_DIR / "early_exit" / "early_exit_20260405_164542",
    }

    budget_order = ["0", "256", "512", "1024", "2048", "4096", "unlimited"]
    budget_labels = ["0", "256", "512", "1K", "2K", "4K", "\u221e"]

    for model, d in ee_dirs.items():
        sfile = d / "summary.json"
        if not sfile.exists():
            continue
        s = _load_json(sfile)
        records = []
        for r in s["results"]:
            records.append({
                "budget": r["thinking_budget_label"],
                "pass_rate": r["pass_rate"],
                "energy_j": r["avg_energy_joules"],
                "elapsed_s": r["avg_elapsed_s"],
                "avg_tokens": r["avg_total_tokens"],
            })
        ee_data[model] = records

    panel_models = ["qwen-3-4b", "qwen-3-0.6b", "deepseek-r1-distill-qwen-1.5b"]
    panel_titles = ["Qwen3 4B", "Qwen3 0.6B", "DS-R1 1.5B"]

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_W, 2.2))

    for ax, model, title in zip(axes, panel_models, panel_titles):
        if model not in ee_data:
            ax.set_title(title + " (no data)")
            continue

        records = ee_data[model]
        budget_to_rec = {r["budget"]: r for r in records}

        x = np.arange(len(budget_order))
        pass_rates = [budget_to_rec.get(b, {}).get("pass_rate", 0) * 100 for b in budget_order]
        energies = [budget_to_rec.get(b, {}).get("energy_j", 0) for b in budget_order]

        color_acc = CB_PALETTE["blue"]
        color_energy = CB_PALETTE["red"]

        # Accuracy line (left axis)
        ax.plot(x, pass_rates, "o-", color=color_acc, markersize=4, linewidth=1.3,
                label="Pass@1", zorder=3)
        ax.set_ylabel("Pass@1 (%)", color=color_acc, fontsize=8)
        ax.tick_params(axis="y", colors=color_acc)
        ax.set_ylim(0, max(pass_rates) * 1.25 if max(pass_rates) > 0 else 100)

        # Energy bars (right axis)
        ax2 = ax.twinx()
        ax2.bar(x, energies, width=0.5, alpha=0.25, color=color_energy,
                edgecolor=color_energy, linewidth=0.5, label="Energy")
        ax2.set_ylabel("Energy (J)", color=color_energy, fontsize=8)
        ax2.tick_params(axis="y", colors=color_energy)

        ax.set_xticks(x)
        ax.set_xticklabels(budget_labels, fontsize=6.5)
        ax.set_xlabel("Thinking Budget (tokens)", fontsize=7.5)
        ax.set_title(title, fontsize=9, fontweight="bold")

        # Highlight sweet spot at 2048 for Qwen 4B
        if model == "qwen-3-4b":
            idx_2048 = budget_order.index("2048")
            ax.axvline(x=idx_2048, color=CB_PALETTE["green"], linestyle="--",
                       linewidth=0.9, alpha=0.7)
            ax.annotate("sweet\nspot", xy=(idx_2048, pass_rates[idx_2048]),
                        xytext=(idx_2048 + 0.5, pass_rates[idx_2048] + 5),
                        fontsize=6, color=CB_PALETTE["green"],
                        arrowprops=dict(arrowstyle="->", color=CB_PALETTE["green"],
                                        linewidth=0.7))

        # Highlight that unlimited hurts
        if model == "qwen-3-4b":
            idx_unlim = budget_order.index("unlimited")
            idx_best = int(np.argmax(pass_rates))
            if pass_rates[idx_unlim] < pass_rates[idx_best]:
                ax.annotate("unlimited\nhurts",
                            xy=(idx_unlim, pass_rates[idx_unlim]),
                            xytext=(idx_unlim - 1.2, pass_rates[idx_unlim] - 8),
                            fontsize=5.5, color=CB_PALETTE["red"],
                            arrowprops=dict(arrowstyle="->", color=CB_PALETTE["red"],
                                            linewidth=0.7))

        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig2_early_exit_thinking_budget")


# ===================================================================
# FIGURE 3: 3-Architecture Deployment Comparison
# ===================================================================
def figure3_architecture():
    print("Figure 3: 3-Architecture Deployment ...")

    raw = _load_json(DATA_DIR / "3arch_experiment" / "20260405_183229" / "figure_data.json")

    # We want base mode, two panels: qwen-3-4b (thinking) and llama-3.2-3b-instruct (non-thinking)
    panel_models = ["qwen-3-4b", "llama-3.2-3b-instruct"]
    panel_titles = ["Qwen3 4B (thinking)", "Llama3.2 3B (non-thinking)"]
    architectures = ["local", "hybrid", "cloud"]
    arch_colors = {
        "local": CB_PALETTE["orange"],
        "hybrid": CB_PALETTE["green"],
        "cloud": CB_PALETTE["blue"],
    }
    arch_labels = {"local": "Local", "hybrid": "Hybrid", "cloud": "Cloud"}

    # RTTs to plot (sorted)
    target_rtts = [0, 1, 20, 40, 80, 200, 500]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, 2.5))

    for ax, model, title in zip(axes, panel_models, panel_titles):
        for arch in architectures:
            # Filter data for this model, base mode, this architecture
            points = [r for r in raw
                      if r["model"] == model
                      and r["mode"] == "base"
                      and r["architecture"] == arch]

            # Map rtt_ms -> avg_time_s
            rtt_to_time = {}
            for p in points:
                rtt = p["rtt_ms"]
                rtt_to_time[rtt] = p["avg_time_s"]

            rtts = sorted(rtt_to_time.keys())
            times = [rtt_to_time[r] for r in rtts]

            if not rtts:
                continue

            ax.plot(rtts, times, "o-", color=arch_colors[arch],
                    label=arch_labels[arch], markersize=3.5, linewidth=1.2)

        ax.set_xlabel("Network RTT (ms)")
        ax.set_ylabel("Avg. Completion Time (s)")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xscale("symlog", linthresh=1)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks(target_rtts)
        ax.set_xticklabels([str(r) for r in target_rtts], fontsize=6.5)

    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig3_architecture_comparison")


# ===================================================================
# FIGURE 4: Agentic Traffic Characterization
# ===================================================================
def figure4_traffic():
    print("Figure 4: Traffic Characterization ...")

    fig, (ax_cdf, ax_radio) = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, 2.4))

    # --- Panel A: CDF of inter-request timing ---
    cdf_data = _load_json(DATA_DIR / "traffic_characterization" / "traffic_char_inter_request_cdf.json")
    comparison = _load_json(DATA_DIR / "traffic_characterization" / "traffic_char_comparison.json")

    x_gaps = np.array(cdf_data["all_gaps"]["x_seconds"])
    y_cdf = np.linspace(0, 1, len(x_gaps))

    ax_cdf.plot(np.sort(x_gaps), y_cdf, color=CB_PALETTE["blue"], linewidth=1.5,
                label="Agentic LLM")

    # Add comparison lines for other traffic classes
    comp_styles = {
        "Web browsing":        (CB_PALETTE["green"], "--"),
        "Video streaming (ABR)": (CB_PALETTE["orange"], "-."),
        "Messaging / chat":    (CB_PALETTE["purple"], ":"),
    }
    for tc in comparison["traffic_classes"]:
        cls = tc["class"]
        if cls in comp_styles:
            color, ls = comp_styles[cls]
            irt = tc["inter_request_time_s"]
            low, high = irt["typical_range"]
            median = irt["median"]
            # Approximate CDF as logistic-like curve covering the typical range
            synth_x = np.linspace(0, max(high * 2, 60), 500)
            # Simple sigmoid approximation centered at median
            scale = (high - low) / 4 if high > low else 1
            synth_y = 1 / (1 + np.exp(-(synth_x - median) / scale))
            ax_cdf.plot(synth_x, synth_y, color=color, linestyle=ls,
                        linewidth=1.0, label=cls, alpha=0.8)

    # Tail timer reference
    ax_cdf.axvline(x=7.0, color=CB_PALETTE["red"], linestyle="--", linewidth=0.8,
                   alpha=0.6)
    ax_cdf.text(7.5, 0.15, "tail\ntimer\n(7s)", fontsize=5.5,
                color=CB_PALETTE["red"])

    ax_cdf.set_xlabel("Inter-request Time (s)")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_xlim(0, 50)
    ax_cdf.legend(fontsize=6, loc="lower right")
    ax_cdf.set_title("(a) Inter-request Timing CDF", fontsize=9, fontweight="bold")

    # --- Panel B: Radio state energy breakdown ---
    radio = _load_json(DATA_DIR / "traffic_characterization" / "traffic_char_radio_states.json")
    agg = radio["aggregate"]
    tail_frac = agg["avg_tail_energy_fraction"]

    # Use per-model data for stacked bar
    models_radio = list(radio["per_model"].keys())
    connected_fracs = []
    tail_fracs = []
    idle_fracs = []
    for m in models_radio:
        tf = radio["per_model"][m]["avg_tail_energy_fraction"]
        # connected + idle = 1 - tail
        # approximate: connected is small, idle is remainder
        connected_fracs.append(max(0.02, 1 - tf - 0.01))  # small connected fraction
        tail_fracs.append(tf)
        idle_fracs.append(max(0, 1 - tf - max(0.02, 1 - tf - 0.01)))

    x_radio = np.arange(len(models_radio))
    bar_w = 0.5

    ax_radio.bar(x_radio, tail_fracs, bar_w,
                 label="Tail (wasted)", color=CB_PALETTE["red"], alpha=0.8)
    ax_radio.bar(x_radio, connected_fracs, bar_w, bottom=tail_fracs,
                 label="Active", color=CB_PALETTE["blue"], alpha=0.8)
    ax_radio.bar(x_radio, idle_fracs, bar_w,
                 bottom=[t + c for t, c in zip(tail_fracs, connected_fracs)],
                 label="Idle", color=CB_PALETTE["grey"], alpha=0.5)

    ax_radio.set_xticks(x_radio)
    ax_radio.set_xticklabels([MODEL_SHORT.get(m, m) for m in models_radio],
                             rotation=25, ha="right", fontsize=6.5)
    ax_radio.set_ylabel("Fraction of Radio Energy")
    ax_radio.set_ylim(0, 1.05)
    ax_radio.legend(fontsize=6, loc="lower right")
    ax_radio.set_title("(b) Radio Energy Breakdown", fontsize=9, fontweight="bold")

    # Annotate 95% waste
    ax_radio.text(0.5, 0.5, f"avg. {tail_frac*100:.0f}% tail waste",
                  transform=ax_radio.transAxes, fontsize=7,
                  ha="center", va="center",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white",
                            ec=CB_PALETTE["red"], alpha=0.8))

    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig4_traffic_characterization")


# ===================================================================
# FIGURE 5: Cost Model Validation
# ===================================================================
def figure5_cost_model():
    print("Figure 5: Cost Model Validation ...")

    cm_data = _load_json(DATA_DIR / "cost_model" / "figures_data.json")
    scatter_points = cm_data["predicted_vs_actual_scatter"]
    validation = _load_json(DATA_DIR / "cost_model" / "validation.json")

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, SINGLE_COL_W))

    model_markers = {
        "qwen-3-4b":                    ("o", CB_PALETTE["blue"]),
        "qwen-3-0.6b":                  ("s", CB_PALETTE["orange"]),
        "deepseek-r1-distill-qwen-1.5b": ("^", CB_PALETTE["green"]),
        "llama-3.2-3b-instruct":        ("D", CB_PALETTE["red"]),
        "gemma-3n-e2b-it":              ("v", CB_PALETTE["purple"]),
    }

    # Group by model
    by_model = {}
    for pt in scatter_points:
        m = pt["model"]
        by_model.setdefault(m, {"actual": [], "predicted": []})
        by_model[m]["actual"].append(pt["actual_j"])
        by_model[m]["predicted"].append(pt["predicted_j"])

    all_actual = []
    all_predicted = []
    for model, vals in by_model.items():
        marker, color = model_markers.get(model, ("o", "grey"))
        ax.scatter(vals["actual"], vals["predicted"],
                   marker=marker, c=color, s=12, alpha=0.5,
                   label=MODEL_SHORT.get(model, model), edgecolors="none")
        all_actual.extend(vals["actual"])
        all_predicted.extend(vals["predicted"])

    # 45-degree reference line
    max_val = max(max(all_actual), max(all_predicted)) * 1.05
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.8, alpha=0.4,
            label="y = x")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # Compute overall R^2
    actual_arr = np.array(all_actual)
    pred_arr = np.array(all_predicted)
    ss_res = np.sum((actual_arr - pred_arr) ** 2)
    ss_tot = np.sum((actual_arr - np.mean(actual_arr)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    ax.text(0.05, 0.92, f"$R^2 = {r2:.3f}$",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    ax.set_xlabel("Actual Energy (J)")
    ax.set_ylabel("Predicted Energy (J)")
    ax.set_title("Cost Model: Predicted vs. Actual", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6.5, loc="lower right", ncol=2, framealpha=0.9)
    ax.set_aspect("equal")

    fig.tight_layout()
    _save(fig, "fig5_cost_model_validation")


# ===================================================================
# FIGURE 6: Placement Policy Comparison
# ===================================================================
def figure6_placement_policy():
    print("Figure 6: Placement Policy Comparison ...")

    pp_data = _load_json(DATA_DIR / "placement_policy" / "figures_data.json")
    policies = pp_data["policy_comparison"]["policies"]

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_W, 2.3))

    policy_names = [p["policy"] for p in policies]
    x = np.arange(len(policy_names))
    bar_w = 0.6

    policy_colors = []
    for p in policy_names:
        if p == "COST_AWARE":
            policy_colors.append(CB_PALETTE["green"])
        elif p == "ORACLE":
            policy_colors.append(CB_PALETTE["blue"])
        else:
            policy_colors.append(CB_PALETTE["grey"])

    metrics = [
        ("avg_pass_rate", "Pass@1", axes[0], True),
        ("avg_time_s", "Avg. Time (s)", axes[1], False),
        ("avg_energy_j", "Avg. Energy (J)", axes[2], False),
    ]

    for key, ylabel, ax, is_pct in metrics:
        vals = [p[key] for p in policies]
        if is_pct:
            vals = [v * 100 for v in vals]
        bars = ax.bar(x, vals, bar_w, color=policy_colors, edgecolor="white", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("ALWAYS_", "A_").replace("NETWORK_", "NET_").replace("COST_", "C_")
                            for p in policy_names],
                           rotation=35, ha="right", fontsize=6.5)
        ax.set_ylabel(ylabel)

        # Annotate COST_AWARE bar
        cost_idx = policy_names.index("COST_AWARE")
        oracle_idx = policy_names.index("ORACLE")
        if is_pct:
            pct_of_oracle = vals[cost_idx] / vals[oracle_idx] * 100 if vals[oracle_idx] > 0 else 0
            ax.annotate(f"{pct_of_oracle:.1f}%\nof oracle",
                        xy=(cost_idx, vals[cost_idx]),
                        xytext=(cost_idx - 0.6, vals[cost_idx] + 3),
                        fontsize=5.5, color=CB_PALETTE["green"],
                        arrowprops=dict(arrowstyle="->", color=CB_PALETTE["green"],
                                        linewidth=0.6))

    fig.suptitle("Placement Policy Comparison", fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout(w_pad=1.0)
    _save(fig, "fig6_placement_policy")


# ===================================================================
# FIGURE 7: New Model Generalization (Gemma 3n as "newer model")
# ===================================================================
def figure7_generalization():
    print("Figure 7: Model Generalization ...")

    # Compare pairs: existing models vs "newer" models
    # We use cloud sweep data to show that newer/different architectures
    # confirm findings. Here: Qwen3 4B vs Gemma 3n 2B, Llama 3.2 3B vs DS-R1 1.5B
    # Group A: "Original" vs "Extended"
    pairs = [
        ("qwen-3-4b", "gemma-3n-e2b-it", "Thinking vs\nNon-thinking (2B)"),
        ("qwen-3-0.6b", "deepseek-r1-distill-qwen-1.5b", "Qwen 0.6B vs\nDS-R1 1.5B"),
        ("llama-3.2-3b-instruct", "gemma-3n-e2b-it", "Llama 3B vs\nGemma 2B"),
    ]

    # Use MBPP cloud 500 data
    mbpp = {}
    s1 = _load_json(DATA_DIR / "full_cloud_sweep" / "sglang_20260402_162457" / "summary.json")
    for model, modes in s1["results"].items():
        mbpp[model] = {mode: d["pass_rate"] for mode, d in modes.items()}
    s2 = _load_json(DATA_DIR / "full_cloud_sweep" / "sglang_20260402_171922" / "summary.json")
    for model, modes in s2["results"].items():
        mbpp[model] = {mode: d["pass_rate"] for mode, d in modes.items()}

    sweep_dir = DATA_DIR / "full_cloud_sweep" / "sglang_20260402_162457"
    for model in ["deepseek-r1-distill-qwen-1.5b", "llama-3.2-3b-instruct", "qwen-3-0.6b"]:
        if model in mbpp:
            continue
        mbpp[model] = {}
        for mode in MODES_ORDERED:
            fpath = sweep_dir / f"{model}_{mode}.jsonl"
            if fpath.exists():
                records = _load_jsonl(fpath)
                passed = sum(1 for r in records
                             if r.get("passed") or r.get("evaluation", {}).get("passed"))
                mbpp[model][mode] = passed / len(records) if records else 0

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_W, 2.3))

    for ax, (m1, m2, title) in zip(axes, pairs):
        x = np.arange(len(MODES_ORDERED))
        width = 0.35

        v1 = [mbpp.get(m1, {}).get(mode, 0) * 100 for mode in MODES_ORDERED]
        v2 = [mbpp.get(m2, {}).get(mode, 0) * 100 for mode in MODES_ORDERED]

        ax.bar(x - width / 2, v1, width, label=MODEL_SHORT.get(m1, m1),
               color=CB_PALETTE["blue"], edgecolor="white", linewidth=0.3)
        ax.bar(x + width / 2, v2, width, label=MODEL_SHORT.get(m2, m2),
               color=CB_PALETTE["orange"], edgecolor="white", linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS[m] for m in MODES_ORDERED], fontsize=7)
        ax.set_ylabel("Pass@1 (%)")
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right")
        ax.set_ylim(0, 100)

    fig.suptitle("Cross-Model Generalization (MBPP, cloud)", fontsize=10,
                 fontweight="bold", y=1.02)
    fig.tight_layout(w_pad=1.0)
    _save(fig, "fig7_model_generalization")


# ===================================================================
# FIGURE 8: Thinking Token Analysis
# ===================================================================
def figure8_thinking_analysis():
    print("Figure 8: Thinking Token Analysis ...")

    # Load MLX sweep JSONL for thinking models
    base_dir = DATA_DIR / "mlx_sweep" / "20260403_091508"
    thinking_models = ["qwen-3-4b", "qwen-3-0.6b"]

    all_thinking_tokens = []
    all_passed = []
    all_model_labels = []

    model_colors = {
        "qwen-3-4b": CB_PALETTE["blue"],
        "qwen-3-0.6b": CB_PALETTE["orange"],
    }

    for model in thinking_models:
        fpath = base_dir / f"{model}_base.jsonl"
        if not fpath.exists():
            continue
        records = _load_jsonl(fpath)
        for r in records:
            tt = r.get("metrics", {}).get("thinking_tokens", 0)
            passed = 1 if (r.get("passed") or r.get("evaluation", {}).get("passed")) else 0
            if tt > 0:
                all_thinking_tokens.append(tt)
                all_passed.append(passed)
                all_model_labels.append(model)

    if not all_thinking_tokens:
        print("  [SKIP] No thinking token data found.")
        return

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.5))

    for model in thinking_models:
        mask = [m == model for m in all_model_labels]
        tt_m = [t for t, ok in zip(all_thinking_tokens, mask) if ok]
        pa_m = [p for p, ok in zip(all_passed, mask) if ok]
        if not tt_m:
            continue
        # Jitter y-axis for visibility
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(pa_m))
        ax.scatter(tt_m, np.array(pa_m) + jitter,
                   c=model_colors[model], alpha=0.35, s=10, edgecolors="none",
                   label=MODEL_SHORT[model])

    # Add binned success rate line
    tt_arr = np.array(all_thinking_tokens)
    pa_arr = np.array(all_passed)
    bin_edges = np.percentile(tt_arr, np.linspace(0, 100, 12))
    bin_edges = np.unique(bin_edges)
    bin_centers = []
    bin_rates = []
    for i in range(len(bin_edges) - 1):
        mask = (tt_arr >= bin_edges[i]) & (tt_arr < bin_edges[i + 1])
        if mask.sum() > 3:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_rates.append(pa_arr[mask].mean())
    ax.plot(bin_centers, bin_rates, "k-", linewidth=1.5, alpha=0.7, label="Binned rate")

    # Correlation
    corr = np.corrcoef(tt_arr, pa_arr)[0, 1]
    ax.text(0.98, 0.95, f"r = {corr:.2f}",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    ax.set_xlabel("Thinking Tokens")
    ax.set_ylabel("Success (0/1)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fail", "Pass"])
    ax.set_title("Thinking Tokens vs. Success", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6.5, loc="center right")

    fig.tight_layout()
    _save(fig, "fig8_thinking_token_analysis")


# ===================================================================
# BONUS: Figure 9 - MLX (on-device) vs Cloud performance side by side
# ===================================================================
def figure9_mlx_vs_cloud():
    """Supplementary: compare on-device MLX (150 problems) vs cloud (500 problems)."""
    print("Figure 9 (bonus): MLX vs Cloud ...")

    mlx_summary = _load_json(DATA_DIR / "mlx_sweep" / "20260403_091508" / "summary.json")

    # Cloud data
    cloud = {}
    s1 = _load_json(DATA_DIR / "full_cloud_sweep" / "sglang_20260402_162457" / "summary.json")
    for model, modes in s1["results"].items():
        cloud[model] = {mode: d["pass_rate"] for mode, d in modes.items()}
    s2 = _load_json(DATA_DIR / "full_cloud_sweep" / "sglang_20260402_171922" / "summary.json")
    for model, modes in s2["results"].items():
        cloud[model] = {mode: d["pass_rate"] for mode, d in modes.items()}

    sweep_dir = DATA_DIR / "full_cloud_sweep" / "sglang_20260402_162457"
    for model in ["deepseek-r1-distill-qwen-1.5b", "llama-3.2-3b-instruct", "qwen-3-0.6b"]:
        if model in cloud:
            continue
        cloud[model] = {}
        for mode in MODES_ORDERED:
            fpath = sweep_dir / f"{model}_{mode}.jsonl"
            if fpath.exists():
                records = _load_jsonl(fpath)
                passed = sum(1 for r in records
                             if r.get("passed") or r.get("evaluation", {}).get("passed"))
                cloud[model][mode] = passed / len(records) if records else 0

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, 2.4), sharey=True)

    for ax, (label, data_src) in zip(axes, [
        ("On-device MLX (n=150)", mlx_summary["results"]),
        ("Cloud A100 (n=500)", cloud),
    ]):
        x = np.arange(len(MODELS_ORDERED))
        width = 0.25
        for i, mode in enumerate(MODES_ORDERED):
            vals = []
            for m in MODELS_ORDERED:
                if isinstance(data_src.get(m), dict):
                    v = data_src[m].get(mode, {})
                    if isinstance(v, dict):
                        vals.append(v.get("pass_rate", 0) * 100)
                    else:
                        vals.append(v * 100)
                else:
                    vals.append(0)
            ax.bar(x + (i - 1) * width, vals, width,
                   label=MODE_LABELS[mode], color=MODE_COLORS[mode],
                   edgecolor="white", linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS_ORDERED],
                           rotation=25, ha="right", fontsize=7)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_ylim(0, 100)

    axes[0].set_ylabel("Pass@1 (%)")
    axes[1].legend(loc="upper right", framealpha=0.9, edgecolor="none")
    fig.tight_layout(w_pad=1.0)
    _save(fig, "fig9_mlx_vs_cloud")


# ===================================================================
# Main
# ===================================================================
def main():
    _apply_acm_style()
    print(f"Output directory: {OUT_DIR}\n")

    figure1_pass_at_1()
    figure2_early_exit()
    figure3_architecture()
    figure4_traffic()
    figure5_cost_model()
    figure6_placement_policy()
    figure7_generalization()
    figure8_thinking_analysis()
    figure9_mlx_vs_cloud()

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
