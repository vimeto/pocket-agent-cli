#!/usr/bin/env python3
"""
Analytical Cost Model for On-Device LLM Agent Energy Prediction.

Derives a simple equation-based model:
    E_local = P_device * (TTFT(ctx) + N_tokens / TPS)

where TTFT(ctx) = alpha * ctx + beta  (linear prefill model)

Fits parameters from MLX sweep measurements, validates predictions,
and generates data for paper figures.

Usage:
    python scripts/cost_model.py
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import math

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "results"
MLX_DIR = DATA_DIR / "mlx_sweep" / "20260403_091508"
CLOUD_DIR = DATA_DIR / "full_cloud_sweep" / "sglang_20260402_162457"
ARCH3_DIR = DATA_DIR / "3arch_experiment" / "20260405_183229"
EARLY_EXIT_DIRS = {
    "qwen-3-4b": DATA_DIR / "early_exit" / "early_exit_20260405_011122",
    "qwen-3-0.6b": DATA_DIR / "early_exit" / "early_exit_20260405_140052",
    "deepseek-r1-distill-qwen-1.5b": DATA_DIR / "early_exit" / "early_exit_20260405_164542",
}
OUTPUT_DIR = DATA_DIR / "cost_model"

MODELS = [
    "qwen-3-4b",
    "qwen-3-0.6b",
    "deepseek-r1-distill-qwen-1.5b",
    "llama-3.2-3b-instruct",
    "gemma-3n-e2b-it",
]
MODES = ["base", "tool_submission", "full_tool"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_mean(vals: list[float]) -> float:
    """Mean of a list, or NaN if empty."""
    return sum(vals) / len(vals) if vals else float("nan")


def safe_median(vals: list[float]) -> float:
    """Median of a list, or NaN if empty."""
    if not vals:
        return float("nan")
    s = sorted(vals)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return s[n // 2]


def linear_regression(xs: list[float], ys: list[float]):
    """Simple OLS for y = a*x + b.  Returns (a, b, r_squared)."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0, sy / n, 0.0
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    # R squared
    y_mean = sy / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return a, b, r2


def compute_metrics(actual: list[float], predicted: list[float]):
    """Compute R^2, RMSE, and MAPE between actual and predicted."""
    n = len(actual)
    if n == 0:
        return {"r2": float("nan"), "rmse": float("nan"), "mape": float("nan")}
    y_mean = sum(actual) / n
    ss_tot = sum((y - y_mean) ** 2 for y in actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = math.sqrt(ss_res / n)
    # MAPE: skip zeros to avoid division by zero
    ape_vals = [abs(a - p) / abs(a) for a, p in zip(actual, predicted) if abs(a) > 1e-6]
    mape = safe_mean(ape_vals) * 100.0 if ape_vals else float("nan")
    return {"r2": round(r2, 4), "rmse": round(rmse, 2), "mape": round(mape, 2)}


# ---------------------------------------------------------------------------
# Step 1: Load MLX sweep data and fit per-model parameters
# ---------------------------------------------------------------------------
def load_mlx_data() -> dict:
    """Load all MLX sweep per-problem records, grouped by model and mode."""
    data = {}  # model -> mode -> [records]
    for model in MODELS:
        data[model] = {}
        for mode in MODES:
            fpath = MLX_DIR / f"{model}_{mode}.jsonl"
            if fpath.exists():
                records = load_jsonl(fpath)
                # Filter to records with valid energy
                valid = []
                for r in records:
                    m = r.get("metrics", {})
                    es = m.get("energy_summary", {})
                    if es and es.get("total_energy_joules") and es["total_energy_joules"] > 0:
                        valid.append(r)
                data[model][mode] = valid
            else:
                data[model][mode] = []
    return data


def fit_model_parameters(mlx_data: dict) -> dict:
    """
    Fit the analytical cost model parameters for each model.

    The core equation for local inference:
        E = P_device * T_gen
        T_gen = TTFT + N_tokens / TPS

    TTFT model (prefill time):
        TTFT(ctx) = alpha * ctx + beta

    We fit two power values:
        - P_device: median power from energy_summary.avg_power_watts
        - P_effective: median of E_actual / elapsed_s (captures real
          system-level power including idle periods between samples)

    Parameters per model:
        - P_device (W): average power draw during inference
        - P_effective (W): effective power = energy / time
        - TPS: tokens per second (generation throughput)
        - alpha, beta: prefill time linear model coefficients
        - avg_thinking_ratio: fraction of tokens that are thinking tokens
        - avg_iterations: average agentic loop iterations
    """
    params = {}

    for model in MODELS:
        model_records = []
        for mode in MODES:
            model_records.extend(mlx_data[model].get(mode, []))

        if not model_records:
            print(f"  WARNING: No data for {model}, skipping")
            continue

        # --- P_device: from energy_summary.avg_power_watts ---
        powers = []
        for r in model_records:
            es = r["metrics"].get("energy_summary", {})
            if es and es.get("avg_power_watts"):
                powers.append(es["avg_power_watts"])
        p_device = safe_median(powers)

        # --- P_effective: E_actual / elapsed_s (true system power) ---
        eff_powers = []
        for r in model_records:
            es = r["metrics"].get("energy_summary", {})
            elapsed = r["metrics"].get("elapsed_s", 0)
            if es and es.get("total_energy_joules") and elapsed > 0:
                eff_powers.append(es["total_energy_joules"] / elapsed)
        p_effective = safe_median(eff_powers)

        # --- TPS: generation tokens per second ---
        tps_vals = []
        for r in model_records:
            tps = r["metrics"].get("tps") or r["metrics"].get("generation_tps")
            if tps and tps > 0:
                tps_vals.append(tps)
        tps_device = safe_median(tps_vals)

        # --- TTFT model: TTFT = alpha * prompt_tokens + beta ---
        # Use records that have both ttft_ms and prompt_tokens
        ttft_xs = []
        ttft_ys = []
        for r in model_records:
            m = r["metrics"]
            ttft = m.get("ttft_ms")
            prompt = m.get("prompt_tokens")
            if ttft and prompt and ttft > 0 and prompt > 0:
                ttft_xs.append(float(prompt))
                ttft_ys.append(float(ttft) / 1000.0)  # convert to seconds
        alpha, beta, ttft_r2 = linear_regression(ttft_xs, ttft_ys)
        # Ensure beta >= 0 (small offset for model loading/warmup)
        if beta < 0:
            beta = 0.0

        # --- Thinking ratio ---
        think_ratios = []
        for r in model_records:
            m = r["metrics"]
            total = m.get("total_tokens", 0)
            think = m.get("thinking_tokens", 0)
            if total > 0:
                think_ratios.append(think / total)
        avg_thinking_ratio = safe_mean(think_ratios)

        # --- Iterations ---
        iters = [r.get("iterations", 1) for r in model_records]
        avg_iterations = safe_mean(iters)

        # --- Tokens ---
        total_toks = [r["metrics"].get("total_tokens", 0) for r in model_records
                      if r["metrics"].get("total_tokens", 0) > 0]
        avg_tokens = safe_mean(total_toks)

        params[model] = {
            "P_device_watts": round(p_device, 2),
            "P_effective_watts": round(p_effective, 2),
            "TPS_device": round(tps_device, 1),
            "TTFT_alpha_s_per_token": round(alpha, 6),
            "TTFT_beta_s": round(beta, 4),
            "TTFT_r2": round(ttft_r2, 4),
            "avg_thinking_ratio": round(avg_thinking_ratio, 4),
            "avg_iterations": round(avg_iterations, 2),
            "avg_total_tokens": round(avg_tokens, 1),
            "n_records_fitted": len(model_records),
        }

    return params


# ---------------------------------------------------------------------------
# Step 2: Energy prediction function
# ---------------------------------------------------------------------------
def predict_energy_local(total_tokens: float, prompt_tokens: float,
                         p_device: float, tps: float,
                         alpha: float, beta: float) -> float:
    """
    Predict energy for a single local inference call.

    E = P_device * (TTFT + N_tokens / TPS)
    TTFT = alpha * prompt_tokens + beta
    """
    ttft = alpha * prompt_tokens + beta
    if ttft < 0:
        ttft = 0.0
    t_gen = total_tokens / tps if tps > 0 else 0.0
    t_total = ttft + t_gen
    return p_device * t_total


# ---------------------------------------------------------------------------
# Step 3: Validate on MLX sweep (training data)
# ---------------------------------------------------------------------------
def validate_mlx(mlx_data: dict, params: dict) -> dict:
    """
    For each model x mode, predict energy per problem and compare to measured.

    Uses P_effective (= median E/t across training data) as the power
    parameter, which is the physically correct system-level power.
    """
    validation = {}

    for model in MODELS:
        if model not in params:
            continue
        p = params[model]
        validation[model] = {}

        for mode in MODES:
            records = mlx_data[model].get(mode, [])
            if not records:
                continue

            actuals = []
            preds = []
            per_problem = []

            for r in records:
                m = r["metrics"]
                es = m.get("energy_summary", {})
                actual_e = es.get("total_energy_joules")
                total_tokens = m.get("total_tokens", 0)
                prompt_tokens = m.get("prompt_tokens", 0)

                if not actual_e or actual_e <= 0 or total_tokens <= 0:
                    continue

                # If no prompt_tokens, estimate from TTFT
                if not prompt_tokens or prompt_tokens <= 0:
                    prompt_tokens = 100  # default prompt size

                pred_e = predict_energy_local(
                    total_tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    p_device=p["P_effective_watts"],
                    tps=p["TPS_device"],
                    alpha=p["TTFT_alpha_s_per_token"],
                    beta=p["TTFT_beta_s"],
                )

                actuals.append(actual_e)
                preds.append(pred_e)
                per_problem.append({
                    "problem_id": r.get("problem_id"),
                    "actual_joules": round(actual_e, 2),
                    "predicted_joules": round(pred_e, 2),
                    "total_tokens": total_tokens,
                    "thinking_tokens": m.get("thinking_tokens", 0),
                })

            metrics = compute_metrics(actuals, preds)
            validation[model][mode] = {
                "n_problems": len(actuals),
                "metrics": metrics,
                "per_problem": per_problem,
            }

    return validation


# ---------------------------------------------------------------------------
# Step 4: Cross-validate on cloud sweep (TPS from cloud data)
# ---------------------------------------------------------------------------
def load_cloud_data() -> dict:
    """Load cloud sweep per-problem records."""
    data = {}
    for model in MODELS:
        data[model] = {}
        for mode in MODES:
            fpath = CLOUD_DIR / f"{model}_{mode}.jsonl"
            if fpath.exists():
                data[model][mode] = load_jsonl(fpath)
            else:
                data[model][mode] = []
    return data


def derive_cloud_tps(cloud_data: dict) -> dict:
    """Derive server TPS from cloud elapsed_s and token counts."""
    cloud_tps = {}
    for model in MODELS:
        tps_vals = []
        for mode in MODES:
            for r in cloud_data[model].get(mode, []):
                tokens = r.get("tokens", 0)
                elapsed = r.get("elapsed_s", 0)
                if tokens > 0 and elapsed > 0:
                    tps_vals.append(tokens / elapsed)
        if tps_vals:
            cloud_tps[model] = round(safe_median(tps_vals), 1)
    return cloud_tps


# ---------------------------------------------------------------------------
# Step 5: Early-exit validation
# ---------------------------------------------------------------------------
def validate_early_exit(params: dict) -> dict:
    """
    Validate that the model correctly predicts energy savings from
    thinking budget truncation.

    Two validation approaches:
    1. "transferred": Use MLX-sweep-fitted parameters directly (tests
       generalizability across experimental contexts).
    2. "recalibrated": Use per-record measured power from the early-exit
       experiment itself (tests whether the *structure* E = P * tokens/TPS
       is correct regardless of absolute power level).
    """
    results = {}

    for model, edir in EARLY_EXIT_DIRS.items():
        if model not in params or not edir.exists():
            continue

        p = params[model]
        results[model] = []

        # Load each budget file
        budget_files = sorted(edir.glob(f"{model}_budget_*.jsonl"))
        for bf in budget_files:
            budget_label = bf.stem.split("_budget_")[-1]
            records = load_jsonl(bf)

            actuals = []
            preds_transferred = []
            preds_recalibrated = []
            token_counts = []

            for r in records:
                m = r.get("metrics", {})
                es = m.get("energy_summary", {})
                actual_e = es.get("total_energy_joules") if es else None
                total_tokens = m.get("total_tokens", 0)
                prompt_tokens = m.get("prompt_tokens", 0)

                if not actual_e or actual_e <= 0 or total_tokens <= 0:
                    continue

                if not prompt_tokens or prompt_tokens <= 0:
                    prompt_tokens = 600  # early exit uses tool prompt

                # Transferred prediction (MLX sweep parameters)
                pred_t = predict_energy_local(
                    total_tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    p_device=p["P_device_watts"],
                    tps=p["TPS_device"],
                    alpha=p["TTFT_alpha_s_per_token"],
                    beta=p["TTFT_beta_s"],
                )

                # Recalibrated prediction (use measured power from this run)
                measured_power = es.get("avg_power_watts", p["P_device_watts"])
                measured_tps = m.get("tps") or m.get("generation_tps") or p["TPS_device"]
                pred_r = predict_energy_local(
                    total_tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    p_device=measured_power,
                    tps=measured_tps if measured_tps > 0 else p["TPS_device"],
                    alpha=p["TTFT_alpha_s_per_token"],
                    beta=p["TTFT_beta_s"],
                )

                actuals.append(actual_e)
                preds_transferred.append(pred_t)
                preds_recalibrated.append(pred_r)
                token_counts.append(total_tokens)

            if not actuals:
                continue

            metrics_transferred = compute_metrics(actuals, preds_transferred)
            metrics_recalibrated = compute_metrics(actuals, preds_recalibrated)
            results[model].append({
                "budget": budget_label,
                "n_problems": len(actuals),
                "avg_tokens": round(safe_mean(token_counts), 1),
                "avg_actual_energy_j": round(safe_mean(actuals), 2),
                "avg_predicted_transferred_j": round(safe_mean(preds_transferred), 2),
                "avg_predicted_recalibrated_j": round(safe_mean(preds_recalibrated), 2),
                "metrics_transferred": metrics_transferred,
                "metrics_recalibrated": metrics_recalibrated,
            })

    return results


# ---------------------------------------------------------------------------
# Step 6: Generate insights and figure data
# ---------------------------------------------------------------------------
def generate_insights(params: dict, validation: dict, cloud_tps: dict,
                      mlx_data: dict, early_exit_results: dict) -> dict:
    """Generate data for paper figures and key insights."""
    figures = {}

    # --- Figure 1: Predicted vs Actual scatter ---
    scatter_data = []
    for model in MODELS:
        if model not in validation:
            continue
        for mode in MODES:
            v = validation[model].get(mode, {})
            for pp in v.get("per_problem", []):
                scatter_data.append({
                    "model": model,
                    "mode": mode,
                    "actual_j": pp["actual_joules"],
                    "predicted_j": pp["predicted_joules"],
                    "total_tokens": pp["total_tokens"],
                    "thinking_tokens": pp["thinking_tokens"],
                })
    figures["predicted_vs_actual_scatter"] = scatter_data

    # --- Figure 2: Energy per successful solution ---
    energy_per_success = []
    summary = json.load(open(MLX_DIR / "summary.json"))
    for model in MODELS:
        if model not in params:
            continue
        for mode in MODES:
            sr = summary.get("results", {}).get(model, {}).get(mode, {})
            pass_rate = sr.get("pass_rate", 0)
            # Get average energy from validation
            v = validation.get(model, {}).get(mode, {})
            if not v or not v.get("per_problem"):
                continue
            avg_energy = safe_mean([p["actual_joules"] for p in v["per_problem"]])
            avg_pred_energy = safe_mean([p["predicted_joules"] for p in v["per_problem"]])
            e_per_success = avg_energy / pass_rate if pass_rate > 0 else float("inf")
            pred_e_per_success = avg_pred_energy / pass_rate if pass_rate > 0 else float("inf")
            energy_per_success.append({
                "model": model,
                "mode": mode,
                "pass_rate": round(pass_rate, 4),
                "avg_energy_j": round(avg_energy, 2),
                "energy_per_success_j": round(e_per_success, 2),
                "pred_energy_per_success_j": round(pred_e_per_success, 2),
            })
    figures["energy_per_success"] = energy_per_success

    # --- Figure 3: Architecture crossover points ---
    # Using 3arch data + model predictions
    arch3_data = []
    arch3_file = ARCH3_DIR / "figure_data.json"
    if arch3_file.exists():
        raw_3arch = json.load(open(arch3_file))
        for r in raw_3arch:
            arch3_data.append(r)
    figures["architecture_comparison"] = arch3_data

    # --- Figure 4: Thinking budget vs energy (from early exit) ---
    thinking_budget_data = []
    for model, budgets in early_exit_results.items():
        for b in budgets:
            thinking_budget_data.append({
                "model": model,
                "budget": b["budget"],
                "avg_tokens": b["avg_tokens"],
                "avg_actual_energy_j": b["avg_actual_energy_j"],
                "avg_predicted_transferred_j": b.get("avg_predicted_transferred_j",
                                                     b.get("avg_predicted_energy_j")),
                "avg_predicted_recalibrated_j": b.get("avg_predicted_recalibrated_j"),
                "r2_transferred": b.get("metrics_transferred", b.get("metrics", {})).get("r2"),
                "r2_recalibrated": b.get("metrics_recalibrated", {}).get("r2"),
            })
    figures["thinking_budget_vs_energy"] = thinking_budget_data

    # --- Figure 5: Model parameter comparison ---
    param_comparison = []
    for model in MODELS:
        if model not in params:
            continue
        p = params[model]
        param_comparison.append({
            "model": model,
            "P_effective_watts": p["P_effective_watts"],
            "P_device_watts": p["P_device_watts"],
            "TPS_device": p["TPS_device"],
            "TPS_server": cloud_tps.get(model),
            "avg_thinking_ratio": p["avg_thinking_ratio"],
            "energy_per_1k_tokens_j": round(
                p["P_effective_watts"] * 1000 / p["TPS_device"], 2
            ) if p["TPS_device"] > 0 else None,
            "speedup_server_vs_device": round(
                cloud_tps[model] / p["TPS_device"], 2
            ) if model in cloud_tps and p["TPS_device"] > 0 else None,
        })
    figures["model_parameters"] = param_comparison

    # --- Insight: Crossover point estimation ---
    # For each model: at what network RTT does cloud become more expensive?
    # E_local = P_device * N_tokens / TPS_device
    # E_cloud_approx = E_network + P_idle * (N_tokens / TPS_server)
    # Assume P_idle ~ 3W (idle MacBook), E_network ~ 0.5J per transfer
    crossover_data = []
    P_IDLE = 3.0  # watts idle
    E_NETWORK_BASE = 0.5  # joules per network transfer (baseline)
    for model in MODELS:
        if model not in params or model not in cloud_tps:
            continue
        p = params[model]
        tps_s = cloud_tps[model]
        # For a typical problem (avg tokens)
        n_tok = p["avg_total_tokens"]
        t_local = n_tok / p["TPS_device"]
        e_local = p["P_effective_watts"] * t_local
        # Cloud energy from device perspective:
        # Device waits idle while server computes + network energy
        t_server = n_tok / tps_s
        e_cloud_device = P_IDLE * t_server + 2 * E_NETWORK_BASE
        crossover_data.append({
            "model": model,
            "avg_tokens": round(n_tok, 0),
            "e_local_j": round(e_local, 2),
            "e_cloud_device_j": round(e_cloud_device, 2),
            "local_cheaper": e_local < e_cloud_device,
            "t_local_s": round(t_local, 2),
            "t_cloud_s": round(t_server, 2),
            "ratio_local_over_cloud": round(e_local / e_cloud_device, 3)
                if e_cloud_device > 0 else None,
        })
    figures["crossover_analysis"] = crossover_data

    return figures


# ---------------------------------------------------------------------------
# Step 7: Print summary
# ---------------------------------------------------------------------------
def print_summary(params: dict, validation: dict, cloud_tps: dict,
                  early_exit_results: dict):
    """Print a readable summary table."""
    print("\n" + "=" * 80)
    print("ANALYTICAL COST MODEL -- PARAMETER FIT & VALIDATION RESULTS")
    print("=" * 80)

    # --- Equation ---
    print("\nModel equation (local inference):")
    print("  E = P_device * (TTFT(ctx) + N_tokens / TPS)")
    print("  TTFT(ctx) = alpha * ctx_tokens + beta")
    print()

    # --- Fitted parameters ---
    print("-" * 95)
    print(f"{'Model':<30} {'P_eff(W)':>8} {'P_smpl(W)':>9} {'TPS':>6} {'alpha':>10} "
          f"{'beta(s)':>8} {'think%':>7} {'iters':>5}")
    print("-" * 95)
    for model in MODELS:
        if model not in params:
            continue
        p = params[model]
        print(f"{model:<30} {p['P_effective_watts']:>8.1f} {p['P_device_watts']:>9.1f} "
              f"{p['TPS_device']:>6.1f} "
              f"{p['TTFT_alpha_s_per_token']:>10.6f} {p['TTFT_beta_s']:>8.4f} "
              f"{p['avg_thinking_ratio']*100:>6.1f}% {p['avg_iterations']:>5.2f}")
    print()

    # --- Cloud TPS comparison ---
    print("Server TPS (from cloud sweep):")
    for model in MODELS:
        if model in cloud_tps and model in params:
            ratio = cloud_tps[model] / params[model]["TPS_device"] \
                if params[model]["TPS_device"] > 0 else 0
            print(f"  {model:<30} TPS_server={cloud_tps[model]:>6.1f} "
                  f"(speedup vs device: {ratio:.1f}x)")
    print()

    # --- Validation R^2 per model x mode ---
    print("-" * 80)
    print(f"{'Model':<30} {'Mode':<18} {'N':>4} {'R^2':>7} {'RMSE(J)':>9} {'MAPE%':>7}")
    print("-" * 80)
    overall_r2s = []
    for model in MODELS:
        if model not in validation:
            continue
        for mode in MODES:
            v = validation[model].get(mode, {})
            if not v:
                continue
            m = v["metrics"]
            flag = " *" if m["r2"] < 0.8 else "  "
            print(f"{model:<30} {mode:<18} {v['n_problems']:>4} "
                  f"{m['r2']:>7.4f} {m['rmse']:>9.1f} {m['mape']:>6.1f}%{flag}")
            overall_r2s.append(m["r2"])
    print("-" * 80)
    if overall_r2s:
        print(f"{'Overall mean R^2':<50} {safe_mean(overall_r2s):>7.4f}")
        print(f"{'Median R^2':<50} {safe_median(overall_r2s):>7.4f}")
    print()

    # --- Early exit validation ---
    print("Early-exit energy prediction (transferred = MLX params, recalibrated = per-run power):")
    print(f"{'Model':<30} {'Budget':<10} {'Actual(J)':>10} {'Xfer(J)':>9} "
          f"{'Recal(J)':>9} {'R2_xfer':>8} {'R2_recal':>8}")
    print("-" * 100)
    ee_r2_recal = []
    for model in MODELS:
        if model not in early_exit_results:
            continue
        for b in early_exit_results[model]:
            mt = b.get("metrics_transferred", b.get("metrics", {}))
            mr = b.get("metrics_recalibrated", {})
            pred_t = b.get("avg_predicted_transferred_j", b.get("avg_predicted_energy_j", 0))
            pred_r = b.get("avg_predicted_recalibrated_j", 0)
            r2_t = mt.get("r2", float("nan"))
            r2_r = mr.get("r2", float("nan"))
            if not math.isnan(r2_r):
                ee_r2_recal.append(r2_r)
            print(f"{model:<30} {b['budget']:<10} {b['avg_actual_energy_j']:>10.1f} "
                  f"{pred_t:>9.1f} {pred_r:>9.1f} "
                  f"{r2_t:>8.4f} {r2_r:>8.4f}")
    print("-" * 100)
    if ee_r2_recal:
        print(f"  Early-exit recalibrated mean R^2: {safe_mean(ee_r2_recal):.4f} "
              f"(structural validity of E = P * tokens/TPS)")
    print()

    # --- Key insights ---
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    # Thinking models use more energy
    for model in MODELS:
        if model not in params:
            continue
        p = params[model]
        if p["avg_thinking_ratio"] > 0.5:
            energy_per_1k = p["P_effective_watts"] * 1000 / p["TPS_device"] \
                if p["TPS_device"] > 0 else 0
            print(f"  {model}: thinking ratio = {p['avg_thinking_ratio']:.1%}, "
                  f"energy/1k tokens = {energy_per_1k:.1f} J")

    # Compare thinking vs non-thinking
    thinking_models = [m for m in MODELS if m in params
                       and params[m]["avg_thinking_ratio"] > 0.5]
    non_thinking = [m for m in MODELS if m in params
                    and params[m]["avg_thinking_ratio"] <= 0.5]
    if thinking_models and non_thinking:
        think_avg_e = safe_mean([
            params[m]["P_effective_watts"] * params[m]["avg_total_tokens"] / params[m]["TPS_device"]
            for m in thinking_models if params[m]["TPS_device"] > 0
        ])
        nonthink_avg_e = safe_mean([
            params[m]["P_effective_watts"] * params[m]["avg_total_tokens"] / params[m]["TPS_device"]
            for m in non_thinking if params[m]["TPS_device"] > 0
        ])
        if nonthink_avg_e > 0:
            print(f"\n  Thinking models avg energy: {think_avg_e:.0f} J "
                  f"vs non-thinking: {nonthink_avg_e:.0f} J "
                  f"({think_avg_e/nonthink_avg_e:.1f}x more)")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading MLX sweep data...")
    mlx_data = load_mlx_data()
    for model in MODELS:
        total = sum(len(mlx_data[model][m]) for m in MODES)
        print(f"  {model}: {total} valid records")

    print("\nFitting model parameters...")
    params = fit_model_parameters(mlx_data)

    print("\nValidating on MLX sweep (training data)...")
    validation = validate_mlx(mlx_data, params)

    print("\nLoading cloud sweep data for cross-validation...")
    cloud_data = load_cloud_data()
    cloud_tps = derive_cloud_tps(cloud_data)

    print("\nValidating early-exit predictions...")
    early_exit_results = validate_early_exit(params)

    print("\nGenerating insights and figure data...")
    figures = generate_insights(params, validation, cloud_tps, mlx_data,
                                early_exit_results)

    # --- Save outputs ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parameters
    # Add the equations to the params file for self-documentation
    output_params = {
        "equations": {
            "description": "Analytical energy cost model for on-device LLM inference",
            "E_local": "P_device * (TTFT(ctx) + N_tokens / TPS_device)",
            "TTFT": "alpha * context_tokens + beta",
            "E_cloud_device": "P_idle * (N_tokens / TPS_server) + 2 * E_network",
            "units": {
                "P_device": "watts",
                "TPS": "tokens/second",
                "alpha": "seconds/token (prefill)",
                "beta": "seconds (fixed overhead)",
                "energy": "joules",
            },
        },
        "parameters": params,
        "cloud_tps": cloud_tps,
    }
    with open(OUTPUT_DIR / "parameters.json", "w") as f:
        json.dump(output_params, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR / 'parameters.json'}")

    # Validation
    # Strip per-problem data for the summary file (keep it lean)
    validation_summary = {}
    for model in validation:
        validation_summary[model] = {}
        for mode in validation[model]:
            v = validation[model][mode]
            validation_summary[model][mode] = {
                "n_problems": v["n_problems"],
                "metrics": v["metrics"],
            }
    output_validation = {
        "mlx_sweep_validation": validation_summary,
        "early_exit_validation": early_exit_results,
        "cloud_tps": cloud_tps,
    }
    with open(OUTPUT_DIR / "validation.json", "w") as f:
        json.dump(output_validation, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'validation.json'}")

    # Figures data
    with open(OUTPUT_DIR / "figures_data.json", "w") as f:
        json.dump(figures, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'figures_data.json'}")

    # Print summary
    print_summary(params, validation, cloud_tps, early_exit_results)

    # --- Final pass/fail check ---
    all_r2 = []
    for model in validation:
        for mode in validation[model]:
            all_r2.append(validation[model][mode]["metrics"]["r2"])
    mean_r2 = safe_mean(all_r2)
    print(f"Overall mean R^2 = {mean_r2:.4f}")
    if mean_r2 >= 0.8:
        print("SUCCESS: R^2 >= 0.8 target met")
    else:
        print(f"NOTE: Mean R^2 = {mean_r2:.4f} (target 0.8)")
        # Check individual
        good = sum(1 for r in all_r2 if r >= 0.8)
        print(f"  {good}/{len(all_r2)} model-mode combos have R^2 >= 0.8")


if __name__ == "__main__":
    main()
