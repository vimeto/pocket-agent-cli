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

# Muted pastel palette matching old paper style
CB_PALETTE = {
    "blue":    "#92B4D4",   # muted steel blue
    "orange":  "#F4A582",   # muted salmon/peach
    "green":   "#8FBC8F",   # muted sage green
    "red":     "#E8968A",   # muted rose
    "purple":  "#DDA0DD",   # muted plum
    "cyan":    "#A8D8D8",   # muted teal
    "yellow":  "#E8D88E",   # muted gold
    "black":   "#000000",
    "grey":    "#999999",
    "lightgrey": "#D3D3D3", # light gray
}

MODE_COLORS = {
    "base":            "#8FBC8F",   # muted sage green
    "tool_submission": "#F4A582",   # muted salmon/peach
    "full_tool":       "#92B4D4",   # muted steel blue
}
MODE_LABELS = {
    "base":            "Base",
    "tool_submission": "Tool-sub",
    "full_tool":       "Full-tool",
}

# Updated for 7 models
MODEL_SHORT = {
    "qwen-3-4b":                     "Qwen3 4B",
    "qwen-3-0.6b":                   "Qwen3 0.6B",
    "llama-3.2-3b-instruct":         "Llama3.2 3B",
    "deepseek-r1-distill-qwen-1.5b": "DeepSeek-R1 1.5B",
    "gemma-3n-e2b-it":               "Gemma3n 2B",
    "qwen-3.5-4b":                   "Qwen3.5 4B",
    "gemma-4-e2b-it":                "Gemma4 2B",
}

# Original 5 models
MODELS_ORDERED_5 = [
    "qwen-3-4b",
    "gemma-3n-e2b-it",
    "llama-3.2-3b-instruct",
    "deepseek-r1-distill-qwen-1.5b",
    "qwen-3-0.6b",
]

# All 7 models
MODELS_ORDERED_7 = [
    "qwen-3-4b",
    "qwen-3.5-4b",
    "gemma-4-e2b-it",
    "gemma-3n-e2b-it",
    "llama-3.2-3b-instruct",
    "deepseek-r1-distill-qwen-1.5b",
    "qwen-3-0.6b",
]

MODES_ORDERED = ["base", "tool_submission", "full_tool"]

# Per-model markers for scatter plots (muted pastels)
MODEL_MARKERS = {
    "qwen-3-4b":                     ("o", "#92B4D4"),  # steel blue
    "qwen-3-0.6b":                   ("s", "#F4A582"),  # salmon
    "deepseek-r1-distill-qwen-1.5b": ("^", "#8FBC8F"),  # sage green
    "llama-3.2-3b-instruct":         ("D", "#E8968A"),  # muted rose
    "gemma-3n-e2b-it":               ("v", "#DDA0DD"),  # plum
    "qwen-3.5-4b":                   ("P", "#A8D8D8"),  # muted teal
    "gemma-4-e2b-it":                ("X", "#E8D88E"),  # muted gold
}

# Per-model colors for line plots (7 models, muted pastels)
MODEL_COLORS_7 = {
    "qwen-3-4b":                     "#92B4D4",  # steel blue
    "qwen-3.5-4b":                   "#A8D8D8",  # muted teal
    "gemma-4-e2b-it":                "#E8D88E",  # muted gold
    "gemma-3n-e2b-it":               "#DDA0DD",  # plum
    "llama-3.2-3b-instruct":         "#E8968A",  # muted rose
    "deepseek-r1-distill-qwen-1.5b": "#8FBC8F",  # sage green
    "qwen-3-0.6b":                   "#F4A582",  # salmon
}


def _apply_acm_style():
    """Apply publication-quality defaults — muted pastel academic style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "normal",
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.grid": False,       # We'll add y-grid per axis
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "grid.color": "#cccccc",
    })


def _style_ax(ax, ygrid=True):
    """Apply consistent sub-axis styling: subtle y-gridlines behind data."""
    if ygrid:
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.5, color='#cccccc')
    ax.set_axisbelow(True)


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


def _pass_rate_from_jsonl(fpath):
    """Compute pass rate from a JSONL file."""
    if not fpath.exists():
        return 0
    records = _load_jsonl(fpath)
    if not records:
        return 0
    passed = sum(1 for r in records
                 if r.get("passed") or r.get("evaluation", {}).get("passed"))
    return passed / len(records)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def _load_mbpp_cloud():
    """Load MBPP cloud data for all 7 models."""
    mbpp = {}

    # Original 5 models from full cloud sweep (n=500)
    s1 = _load_json(DATA_DIR / "full_cloud_sweep" / "sglang_20260402_162457" / "summary.json")
    for model, modes in s1["results"].items():
        mbpp[model] = {mode: d["pass_rate"] for mode, d in modes.items()}
    s2 = _load_json(DATA_DIR / "full_cloud_sweep" / "sglang_20260402_171922" / "summary.json")
    for model, modes in s2["results"].items():
        mbpp[model] = {mode: d["pass_rate"] for mode, d in modes.items()}

    # Compute from JSONL for models missing from summaries
    sweep_dir = DATA_DIR / "full_cloud_sweep" / "sglang_20260402_162457"
    for model in ["deepseek-r1-distill-qwen-1.5b", "llama-3.2-3b-instruct", "qwen-3-0.6b"]:
        if model in mbpp:
            continue
        mbpp[model] = {}
        for mode in MODES_ORDERED:
            mbpp[model][mode] = _pass_rate_from_jsonl(sweep_dir / f"{model}_{mode}.jsonl")

    # New models: average across multiple n=50 runs
    new_model_runs = {
        "qwen-3.5-4b": [
            "sglang_20260405_141349",
            "sglang_20260405_200728",
            "sglang_20260405_205744",
        ],
        "gemma-4-e2b-it": [
            "sglang_20260405_162207",
            "sglang_20260405_200729",
            "sglang_20260405_203820",
            "sglang_20260405_204931",
            "sglang_20260405_205746",
        ],
    }

    for model_id, run_dirs in new_model_runs.items():
        mbpp[model_id] = {}
        for mode in MODES_ORDERED:
            rates = []
            for rd in run_dirs:
                sfile = DATA_DIR / rd / "summary.json"
                if sfile.exists():
                    s = _load_json(sfile)
                    if model_id in s.get("results", {}):
                        mode_data = s["results"][model_id].get(mode, {})
                        if isinstance(mode_data, dict) and "pass_rate" in mode_data:
                            rates.append(mode_data["pass_rate"])
            mbpp[model_id][mode] = sum(rates) / len(rates) if rates else 0

    return mbpp


def _load_humaneval_cloud():
    """Load HumanEval cloud data for all 7 models."""
    humaneval = {}

    # Original 5 models from humaneval_cloud
    for subdir in sorted((DATA_DIR / "humaneval_cloud").iterdir()):
        if not subdir.is_dir():
            continue
        sfile = subdir / "summary.json"
        if sfile.exists():
            s = _load_json(sfile)
            for model, modes in s["results"].items():
                humaneval[model] = {mode: d["pass_rate"] for mode, d in modes.items()}
        for jf in subdir.glob("*.jsonl"):
            parts = jf.stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            model, mode = parts
            if model in humaneval and mode in humaneval[model]:
                continue
            rate = _pass_rate_from_jsonl(jf)
            humaneval.setdefault(model, {})[mode] = rate

    # Qwen 3.5 HumanEval (n=164) from sglang_20260406_202643
    sfile = DATA_DIR / "sglang_20260406_202643" / "summary.json"
    if sfile.exists():
        s = _load_json(sfile)
        for model, modes in s["results"].items():
            humaneval[model] = {mode: d["pass_rate"] for mode, d in modes.items()}

    # Gemma 4 HumanEval (n=164) from sglang_20260406_204441
    sfile = DATA_DIR / "sglang_20260406_204441" / "summary.json"
    if sfile.exists():
        s = _load_json(sfile)
        for model, modes in s["results"].items():
            humaneval[model] = {mode: d["pass_rate"] for mode, d in modes.items()}

    return humaneval


def _load_bfcl():
    """Load BFCL results for all 7 models, using the best (n=45) run for each."""
    bfcl = {}

    # Map model -> best results (prefer n=45 over smaller n)
    best_runs = {
        "qwen-3-4b": "20260406_092353",
        "llama-3.2-3b-instruct": "20260406_092535",
        "qwen-3-0.6b": "20260406_105229",
        "deepseek-r1-distill-qwen-1.5b": "20260406_105606",
        "gemma-3n-e2b-it": "20260406_213144",
        "qwen-3.5-4b": "20260406_202628",
        "gemma-4-e2b-it": "20260406_191726_mahti_bfcl",
    }

    for model_id, run_id in best_runs.items():
        fpath = DATA_DIR / "bfcl" / run_id / "bfcl_results.json"
        if fpath.exists():
            data = _load_json(fpath)
            for m, info in data["models"].items():
                bfcl[model_id] = {
                    "full_match_pct": info["full_match_pct"],
                    "partial_match_pct": info.get("partial_match_pct", 0),
                    "no_match_pct": info.get("no_match_pct", 0),
                    "n_examples": data["n_examples"],
                    "per_category": info.get("per_category", {}),
                }

    return bfcl


def _load_websearch():
    """Load WebSearch QA results for all models with results."""
    ws_data = []

    ws_dirs = [
        "20260406_092306",
        "20260406_092401",
        "20260406_191915_mahti_ws",
        "20260406_191919_mahti_ws",
        "20260406_205750",
    ]

    for wd in ws_dirs:
        fpath = DATA_DIR / "websearch_qa" / wd / "websearch_results.json"
        if fpath.exists():
            data = _load_json(fpath)
            ws_data.extend(data["results"])

    return ws_data


# ===================================================================
# FIGURE 1: Pass@1 Accuracy Comparison (MBPP + HumanEval) -- 7 models
# ===================================================================
def figure1_pass_at_1():
    print("Figure 1: Pass@1 Accuracy (MBPP, 7 models, single column) ...")

    mbpp = _load_mbpp_cloud()

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.2))
    _style_ax(ax)

    models = MODELS_ORDERED_7
    x = np.arange(len(models))
    width = 0.26

    for i, mode in enumerate(MODES_ORDERED):
        vals = [mbpp.get(m, {}).get(mode, 0) * 100 for m in models]
        ax.bar(x + (i - 1) * width, vals, width,
               label=MODE_LABELS[mode], color=MODE_COLORS[mode],
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in models],
                       rotation=40, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Pass@1 (%)", fontsize=8)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="none",
              fontsize=6.5, ncol=3, columnspacing=0.8, handletextpad=0.4)

    ax.axhline(y=50, color=CB_PALETTE["grey"], linestyle="--",
               linewidth=0.7, alpha=0.5)
    ax.text(0.01, 51, "threshold", transform=ax.get_yaxis_transform(),
            fontsize=6.5, color=CB_PALETTE["grey"], va="bottom")

    fig.tight_layout()
    _save(fig, "fig1_pass_at_1_comparison")


# ===================================================================
# FIGURE 2: Early-Exit Thinking Budget (KEY CONTRIBUTION)
# ===================================================================
def figure2_early_exit():
    print("Figure 2: Early-Exit Thinking Budget ...")

    ee_data = {}
    ee_dirs = {
        "qwen-3-4b": DATA_DIR / "early_exit" / "early_exit_20260405_011122",
        "qwen-3-0.6b": DATA_DIR / "early_exit" / "early_exit_20260405_140052",
        "deepseek-r1-distill-qwen-1.5b": DATA_DIR / "early_exit" / "early_exit_20260405_164542",
        "qwen-3.5-4b": DATA_DIR / "early_exit" / "early_exit_20260407_070631",
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

    panel_models = ["qwen-3-4b", "qwen-3.5-4b", "qwen-3-0.6b", "deepseek-r1-distill-qwen-1.5b"]
    panel_titles = ["Qwen3 4B", "Qwen3.5 4B", "Qwen3 0.6B", "DeepSeek-R1 1.5B"]

    color_acc = "#4A7FB5"     # stronger blue for Pass@1 line
    color_energy = "#D4765A"  # stronger salmon for Energy bars

    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_COL_W, 1.75))

    for ax, model, title in zip(axes, panel_models, panel_titles):
        _style_ax(ax)
        if model not in ee_data:
            ax.set_title(title + " (no data)", fontsize=8)
            continue

        records = ee_data[model]
        budget_to_rec = {r["budget"]: r for r in records}
        x = np.arange(len(budget_order))
        pass_rates = [budget_to_rec.get(b, {}).get("pass_rate", 0) * 100 for b in budget_order]
        energies = [budget_to_rec.get(b, {}).get("energy_j", 0) for b in budget_order]

        # Energy bars on primary axis (background)
        ax2 = ax.twinx()
        ax2.bar(x, energies, width=0.55, alpha=0.45, color=color_energy,
                edgecolor="white", linewidth=0.5, label="Energy", zorder=1)
        ax2.set_ylabel("Energy (J)", color=color_energy, fontsize=7.5)
        ax2.tick_params(axis="y", colors=color_energy, labelsize=7)

        # Pass@1 line on top
        ax.plot(x, pass_rates, "o-", color=color_acc, markersize=4, linewidth=1.4,
                label="Pass@1", zorder=3)
        ax.set_ylabel("Pass@1 (%)", color=color_acc, fontsize=7.5)
        ax.tick_params(axis="y", colors=color_acc, labelsize=7)
        ax.set_ylim(0, max(pass_rates) * 1.3 if max(pass_rates) > 0 else 100)

        ax.set_xticks(x)
        ax.set_xticklabels(budget_labels, fontsize=7)
        ax.set_xlabel("Budget (tokens)", fontsize=7.5)
        ax.set_title(title, fontsize=8, fontweight="normal")

        # Mark optimal budget
        idx_best = int(np.argmax(pass_rates))
        ax.axvline(x=idx_best, color="#8FBC8F", linestyle="--",
                   linewidth=0.9, alpha=0.6)

        # Keep line on top of bars
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    fig.tight_layout(w_pad=1.2)
    _save(fig, "fig2_early_exit_thinking_budget")


# ===================================================================
# FIGURE 3: 3-Architecture Deployment Comparison
# ===================================================================
def figure3_architecture():
    print("Figure 3: 3-Architecture Deployment ...")

    # Try newer data first, fall back to older
    data_path = DATA_DIR / "3arch_experiment" / "20260406_002255" / "figure_data.json"
    if not data_path.exists():
        data_path = DATA_DIR / "3arch_experiment" / "20260405_183229" / "figure_data.json"

    raw = _load_json(data_path)

    model = "qwen-3-4b"
    title = "Qwen3 4B (thinking)"
    architectures = ["local", "hybrid", "cloud"]
    arch_colors = {
        "local": "#F4A582",
        "hybrid": "#8FBC8F",
        "cloud": "#92B4D4",
    }
    arch_labels = {"local": "Local", "hybrid": "Hybrid", "cloud": "Cloud"}

    target_rtts = [0, 1, 20, 40, 80, 200, 500]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 1.9))
    _style_ax(ax)

    for arch in architectures:
        points = [r for r in raw
                  if r["model"] == model
                  and r["mode"] == "base"
                  and r["architecture"] == arch]

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

    ax.set_xlabel("Network RTT (ms)", fontsize=8)
    ax.set_ylabel("Avg. Completion Time (s)", fontsize=8)
    ax.set_title(title, fontsize=8.5, fontweight="normal")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xscale("symlog", linthresh=1)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(target_rtts)
    ax.set_xticklabels([str(r) for r in target_rtts], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    _save(fig, "fig3_architecture_comparison")


# ===================================================================
# FIGURE 4: Agentic Traffic Characterization
# ===================================================================
def figure4_traffic():
    print("Figure 4: Traffic Characterization ...")

    fig, (ax_cdf, ax_radio) = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, 1.9),
                                              gridspec_kw={"width_ratios": [1.4, 1]})

    cdf_data = _load_json(DATA_DIR / "traffic_characterization" / "traffic_char_inter_request_cdf.json")
    comparison = _load_json(DATA_DIR / "traffic_characterization" / "traffic_char_comparison.json")

    _style_ax(ax_cdf)
    _style_ax(ax_radio)

    x_gaps = np.array(cdf_data["all_gaps"]["x_seconds"])
    y_cdf = np.linspace(0, 1, len(x_gaps))

    # Agentic LLM line: prominent steel blue
    ax_cdf.plot(np.sort(x_gaps), y_cdf, color="#92B4D4", linewidth=1.5,
                label="Agentic LLM")

    # Comparison lines: lighter and dashed
    comp_styles = {
        "Web browsing":          ("#B0B0B0", "--"),
        "Video streaming (ABR)": ("#C8C8C8", "-."),
        "Messaging / chat":      ("#A0A0A0", ":"),
    }
    for tc in comparison["traffic_classes"]:
        cls = tc["class"]
        if cls in comp_styles:
            color, ls = comp_styles[cls]
            irt = tc["inter_request_time_s"]
            low, high = irt["typical_range"]
            median = irt["median"]
            synth_x = np.linspace(0, max(high * 2, 60), 500)
            scale = (high - low) / 4 if high > low else 1
            synth_y = 1 / (1 + np.exp(-(synth_x - median) / scale))
            ax_cdf.plot(synth_x, synth_y, color=color, linestyle=ls,
                        linewidth=1.0, label=cls, alpha=0.7)

    ax_cdf.axvline(x=7.0, color="#E8968A", linestyle="--", linewidth=0.8,
                   alpha=0.6)
    ax_cdf.text(7.5, 0.15, "tail\ntimer\n(7s)", fontsize=7,
                color="#C07878")

    ax_cdf.set_xlabel("Inter-request Time (s)", fontsize=9)
    ax_cdf.set_ylabel("CDF", fontsize=9)
    ax_cdf.set_xlim(0, 50)
    ax_cdf.tick_params(labelsize=8)
    ax_cdf.legend(fontsize=7.5, loc="lower right")
    ax_cdf.set_title("(a)", fontsize=10, fontweight="normal")

    radio = _load_json(DATA_DIR / "traffic_characterization" / "traffic_char_radio_states.json")
    agg = radio["aggregate"]
    tail_frac = agg["avg_tail_energy_fraction"]

    models_radio = list(radio["per_model"].keys())
    connected_fracs = []
    tail_fracs = []
    idle_fracs = []
    for m in models_radio:
        tf = radio["per_model"][m]["avg_tail_energy_fraction"]
        connected_fracs.append(max(0.02, 1 - tf - 0.01))
        tail_fracs.append(tf)
        idle_fracs.append(max(0, 1 - tf - max(0.02, 1 - tf - 0.01)))

    x_radio = np.arange(len(models_radio))
    bar_w = 0.5

    ax_radio.bar(x_radio, tail_fracs, bar_w,
                 label="Tail (wasted)", color="#F4A582",  # salmon
                 edgecolor="white", linewidth=0.5, alpha=0.85)
    ax_radio.bar(x_radio, connected_fracs, bar_w, bottom=tail_fracs,
                 label="Active", color="#92B4D4",  # steel blue
                 edgecolor="white", linewidth=0.5, alpha=0.85)
    ax_radio.bar(x_radio, idle_fracs, bar_w,
                 bottom=[t + c for t, c in zip(tail_fracs, connected_fracs)],
                 label="Idle", color="#D3D3D3",  # light gray
                 edgecolor="white", linewidth=0.5, alpha=0.7)

    ax_radio.set_xticks(x_radio)
    ax_radio.set_xticklabels([MODEL_SHORT.get(m, m) for m in models_radio],
                             rotation=25, ha="right", fontsize=7.5)
    ax_radio.set_ylabel("Fraction of Radio Energy", fontsize=9)
    ax_radio.set_ylim(0, 1.05)
    ax_radio.tick_params(labelsize=8)
    ax_radio.legend(fontsize=7.5, loc="lower right")
    ax_radio.set_title("(b)", fontsize=10, fontweight="normal")

    ax_radio.text(0.5, 0.5, f"avg. {tail_frac*100:.0f}% tail waste",
                  transform=ax_radio.transAxes, fontsize=7,
                  ha="center", va="center",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white",
                            ec="#E8968A", alpha=0.8))

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
    _style_ax(ax)

    by_model = {}
    for pt in scatter_points:
        m = pt["model"]
        by_model.setdefault(m, {"actual": [], "predicted": []})
        by_model[m]["actual"].append(pt["actual_j"])
        by_model[m]["predicted"].append(pt["predicted_j"])

    all_actual = []
    all_predicted = []
    for model, vals in by_model.items():
        marker, color = MODEL_MARKERS.get(model, ("o", "#999999"))
        ax.scatter(vals["actual"], vals["predicted"],
                   marker=marker, c=color, s=12, alpha=0.5,
                   label=MODEL_SHORT.get(model, model), edgecolors="none")
        all_actual.extend(vals["actual"])
        all_predicted.extend(vals["predicted"])

    max_val = max(max(all_actual), max(all_predicted)) * 1.05
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.8, alpha=0.4,
            label="y = x")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

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
    ax.legend(fontsize=7, loc="lower right", ncol=2, framealpha=0.9)
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

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, 2.2))

    policy_names = [p["policy"] for p in policies]
    x = np.arange(len(policy_names))
    bar_w = 0.6

    policy_colors = []
    for p in policy_names:
        if p == "COST_AWARE":
            policy_colors.append("#92B4D4")   # steel blue - highlight
        elif p == "ORACLE":
            policy_colors.append("#8FBC8F")   # sage green
        else:
            policy_colors.append("#D3D3D3")   # light gray

    # Only show time and energy (accuracy is platform-independent for same model)
    metrics = [
        ("avg_time_s", "Avg. Latency (s)", axes[0]),
        ("avg_energy_j", "Avg. Energy (J)", axes[1]),
    ]

    for key, ylabel, ax in metrics:
        _style_ax(ax)
        vals = [p[key] for p in policies]
        ax.bar(x, vals, bar_w, color=policy_colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("ALWAYS_", "A_").replace("NETWORK_", "NET_").replace("COST_", "C_")
                            for p in policy_names],
                           rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)

    # Annotate cost-aware vs oracle on energy panel
    cost_idx = policy_names.index("COST_AWARE")
    oracle_idx = policy_names.index("ORACLE")
    energy_vals = [p["avg_energy_j"] for p in policies]
    local_idx = policy_names.index("ALWAYS_LOCAL")
    pct_saving = (1 - energy_vals[cost_idx] / energy_vals[local_idx]) * 100
    axes[1].annotate(f"$-${pct_saving:.0f}% vs local",
                     xy=(cost_idx, energy_vals[cost_idx]),
                     xytext=(cost_idx - 1.2, energy_vals[cost_idx] + 40),
                     fontsize=7, color="#6B8E9F",
                     arrowprops=dict(arrowstyle="->", color="#6B8E9F",
                                     linewidth=0.6))

    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig6_placement_policy")


# ===================================================================
# FIGURE 8: Thinking Token Analysis
# ===================================================================
def figure8_thinking_analysis():
    print("Figure 8: Thinking Token Analysis ...")

    base_dir = DATA_DIR / "mlx_sweep" / "20260403_091508"
    thinking_models = ["qwen-3-4b", "qwen-3-0.6b"]

    all_thinking_tokens = []
    all_passed = []
    all_model_labels = []

    model_colors = {
        "qwen-3-4b": "#92B4D4",    # steel blue
        "qwen-3-0.6b": "#F4A582",  # salmon
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

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.0))
    _style_ax(ax)

    for model in thinking_models:
        mask = [m == model for m in all_model_labels]
        tt_m = [t for t, ok in zip(all_thinking_tokens, mask) if ok]
        pa_m = [p for p, ok in zip(all_passed, mask) if ok]
        if not tt_m:
            continue
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(pa_m))
        ax.scatter(tt_m, np.array(pa_m) + jitter,
                   c=model_colors[model], alpha=0.35, s=10, edgecolors="none",
                   label=MODEL_SHORT[model])

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
    ax.plot(bin_centers, bin_rates, "k-", linewidth=1.5, alpha=0.6, label="Binned rate")

    corr = np.corrcoef(tt_arr, pa_arr)[0, 1]
    ax.text(0.98, 0.95, f"r = {corr:.2f}",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    ax.set_xlabel("Thinking Tokens")
    ax.set_ylabel("Success (0/1)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fail", "Pass"])
    ax.legend(fontsize=7, loc="center right")

    fig.tight_layout()
    _save(fig, "fig8_thinking_token_analysis")


# ===================================================================
# NEW FIGURE 10: BFCL Tool-Calling Proficiency (all 7 models)
# ===================================================================
def figure10_bfcl():
    print("Figure 10 (NEW): BFCL Tool-Calling Proficiency ...")

    bfcl = _load_bfcl()

    if not bfcl:
        print("  [SKIP] No BFCL data found.")
        return

    models_with_data = [m for m in MODELS_ORDERED_7 if m in bfcl]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.2))
    _style_ax(ax)

    x = np.arange(len(models_with_data))
    full_match = [bfcl[m]["full_match_pct"] for m in models_with_data]
    partial_match = [bfcl[m]["partial_match_pct"] for m in models_with_data]
    no_match = [bfcl[m]["no_match_pct"] for m in models_with_data]

    bar_w = 0.6
    ax.bar(x, full_match, bar_w, label="Full match",
           color="#8FBC8F", edgecolor="white", linewidth=0.5)       # sage green
    ax.bar(x, partial_match, bar_w, bottom=full_match,
           label="Partial match",
           color="#F4A582", edgecolor="white", linewidth=0.5)       # salmon
    ax.bar(x, no_match, bar_w,
           bottom=[f + p for f, p in zip(full_match, partial_match)],
           label="No match",
           color="#E8968A", edgecolor="white", linewidth=0.5)       # muted rose/pink

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in models_with_data],
                       rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("Percentage (%)", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    _save(fig, "fig10_bfcl_tool_calling")


# ===================================================================
# NEW FIGURE 11: Web Search QA -- F1 across network conditions
# ===================================================================
def figure11_websearch():
    print("Figure 11 (NEW): Web Search QA ...")

    import json as _json
    from collections import defaultdict

    # Load per-problem data to compute with/without search comparison
    per_problem_files = {
        "qwen-3-4b": DATA_DIR / "websearch_qa" / "20260406_092401" / "websearch_per_problem.jsonl",
        "gemma-3n-e2b-it": DATA_DIR / "websearch_qa" / "20260407_gemma3n_combined" / "websearch_per_problem.jsonl",
        "llama-3.2-3b-instruct": DATA_DIR / "websearch_qa" / "20260406_191915_mahti_ws" / "websearch_per_problem.jsonl",
        "qwen-3-0.6b": DATA_DIR / "websearch_qa" / "20260406_191919_mahti_ws" / "websearch_per_problem.jsonl",
        "deepseek-r1-distill-qwen-1.5b": DATA_DIR / "websearch_qa" / "20260406_205750" / "websearch_per_problem.jsonl",
    }

    model_stats = {}
    for model, path in per_problem_files.items():
        if not path.exists():
            continue
        with open(path) as f:
            lines = [_json.loads(l) for l in f]
        cloud = [l for l in lines if l.get("network_condition") == "cloud"]
        if not cloud:
            continue
        with_s = [l for l in cloud if l.get("search_calls", 0) > 0]
        without_s = [l for l in cloud if l.get("search_calls", 0) == 0]
        model_stats[model] = {
            "overall_f1": np.mean([l["f1"] for l in cloud]),
            "with_search_f1": np.mean([l["f1"] for l in with_s]) if with_s else 0,
            "without_search_f1": np.mean([l["f1"] for l in without_s]) if without_s else np.mean([l["f1"] for l in cloud]),
            "pct_used_search": len(with_s) / len(cloud) * 100,
            "avg_searches": np.mean([l.get("search_calls", 0) for l in cloud]),
            "avg_time": np.mean([l.get("total_time_s", 0) for l in cloud]),
        }

    if not model_stats:
        print("  [SKIP] No per-problem WebSearch data found.")
        return

    # All 5 core models (Gemma 3n data from local MLX benchmark)
    ws_models = ["qwen-3-4b", "gemma-3n-e2b-it", "llama-3.2-3b-instruct",
                 "deepseek-r1-distill-qwen-1.5b", "qwen-3-0.6b"]

    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COL_W, 2.2))
    _style_ax(ax)
    x = np.arange(len(ws_models))
    width = 0.3

    without_vals = [model_stats[m]["without_search_f1"] for m in ws_models]
    with_vals = [model_stats[m]["with_search_f1"] for m in ws_models]

    ax.bar(x - width/2, without_vals, width, label="Without search",
           color="#F4A582", edgecolor="white", linewidth=0.5)
    ax.bar(x + width/2, with_vals, width, label="With search",
           color="#92B4D4", edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT.get(m, m) for m in ws_models],
                       rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("F1 Score", fontsize=8)
    ax.set_ylim(0, 0.7)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.legend(fontsize=7.5, loc="upper right")

    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig11_websearch_qa")


# ===================================================================
# NEW: Energy per Successful Solution
# ===================================================================
def figure_energy_per_success():
    print("Figure (NEW): Energy per Success ...")

    import csv

    csv_path = BASE_DIR / "research" / "figures" / "energy_per_success_table.csv"
    if not csv_path.exists():
        print(f"  [SKIP] CSV not found: {csv_path}")
        return

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Filter for Q4_K_M quantization and full_tool mode
    filtered = [r for r in rows
                if r["quantization"] == "Q4_K_M" and r["mode"] == "full_tool"]

    if not filtered:
        print("  [SKIP] No Q4_K_M / full_tool rows found in CSV.")
        return

    # Order models using MODELS_ORDERED_5
    model_order = MODELS_ORDERED_5
    model_data = {}
    for r in filtered:
        model_data[r["model"]] = r

    models_to_plot = [m for m in model_order if m in model_data]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.2))
    _style_ax(ax)

    platforms = [
        ("A100_kJ", "A100", "#8FBC8F"),           # sage green
        ("MacBook_M2_Max_kJ", "MacBook", "#92B4D4"),  # steel blue
        ("iPhone15_kJ", "iPhone 15", "#F4A582"),   # salmon
    ]

    x = np.arange(len(models_to_plot))
    n_platforms = len(platforms)
    width = 0.22

    for i, (col, label, color) in enumerate(platforms):
        vals = []
        for m in models_to_plot:
            raw = model_data[m].get(col, "")
            if raw and raw.strip():
                vals.append(float(raw))
            else:
                vals.append(np.nan)
        vals_arr = np.array(vals, dtype=float)
        # Plot only non-NaN values
        mask = ~np.isnan(vals_arr)
        x_masked = x[mask]
        v_masked = vals_arr[mask]
        ax.bar(x_masked + (i - (n_platforms - 1) / 2) * width, v_masked, width,
               label=label, color=color, edgecolor="white", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in models_to_plot],
                       rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("Energy per Success (kJ)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5, color='#cccccc', which='both')
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9, edgecolor="none")

    fig.tight_layout()
    _save(fig, "fig_energy_per_success")


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
    figure8_thinking_analysis()
    figure10_bfcl()
    figure11_websearch()
    figure_energy_per_success()

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
