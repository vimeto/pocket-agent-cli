#!/usr/bin/env python3
"""Network-aware placement policy evaluation (COL-169).

For each agentic task, decides whether to run locally, hybrid, or cloud
based on network conditions, model capability, and estimated energy cost.

Evaluates 5 static/dynamic policies against an oracle upper bound using
the 3-architecture experiment data (168 entries: 4 models x 3 archs x
7 network conditions x 2 modes).

Usage:
    python scripts/placement_policy.py
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "results"
ARCH_DATA = DATA_DIR / "3arch_experiment" / "20260405_183229" / "figure_data.json"
COST_PARAMS = DATA_DIR / "cost_model" / "parameters.json"
RADIO_DATA = DATA_DIR / "traffic_characterization" / "traffic_char_radio_states.json"
OUTPUT_DIR = DATA_DIR / "placement_policy"


# ── Load data ────────────────────────────────────────────────────────────────

def load_data() -> Tuple[List[Dict], Dict, Dict]:
    """Load all required datasets."""
    with open(ARCH_DATA) as f:
        arch_data = json.load(f)
    with open(COST_PARAMS) as f:
        cost_params = json.load(f)
    with open(RADIO_DATA) as f:
        radio_data = json.load(f)
    return arch_data, cost_params, radio_data


# ── Energy estimation ────────────────────────────────────────────────────────

def estimate_energy(entry: Dict, cost_params: Dict, radio_data: Dict) -> float:
    """Estimate energy (joules) for an experiment entry using the analytical
    cost model.

    Local:  E = P_device * (tokens / TPS_device)
    Hybrid: E = P_server_side * (tokens / TPS_cloud) + radio_tail per iteration
    Cloud:  E = P_idle_device * (tokens / TPS_cloud) + radio_energy_once
    """
    model = entry["model"]
    arch = entry["architecture"]
    tokens = entry["avg_tokens"]
    iterations = entry.get("avg_iterations", 1)

    params = cost_params["parameters"].get(model, {})
    cloud_tps = cost_params.get("cloud_tps", {}).get(model)

    P_device = params.get("P_device_watts", 13.0)
    TPS_device = params.get("TPS_device", 50.0)
    P_idle = 1.5  # idle device power while waiting for cloud (WiFi radio + display off)

    # Radio tail energy from traffic characterization
    tail_energy_per_iter = radio_data.get("per_model", {}).get(model, {}).get(
        "avg_tail_energy_j", radio_data["aggregate"]["avg_tail_waste_j"]
    )

    if arch == "local":
        # All computation on device
        if tokens == 0:
            return 0.0
        time_s = tokens / TPS_device
        energy = P_device * time_s
    elif arch == "cloud":
        # All computation on cloud; device just sends/receives
        if cloud_tps is None or tokens == 0:
            return 0.0
        time_s = tokens / cloud_tps
        # Device is idle during cloud inference, plus one round of network
        energy = P_idle * time_s + entry.get("avg_radio_tail_energy_j", 0.5)
    elif arch == "hybrid":
        # Cloud inference + local tool execution
        if cloud_tps is None or tokens == 0:
            return 0.0
        inference_time = tokens / cloud_tps
        # Device runs tools locally (small energy) + radio tail per iteration
        radio_energy = tail_energy_per_iter * max(iterations, 1)
        # P_idle during cloud inference + local tool execution energy
        tool_energy = 2.0 * iterations  # ~2W for 1s of local tool exec per iter
        energy = P_idle * inference_time + radio_energy + tool_energy
    else:
        energy = 0.0

    return round(energy, 2)


# ── Policy definitions ───────────────────────────────────────────────────────

def policy_always_local(model: str, mode: str, network_condition: str,
                        rtt_ms: float, cost_params: Dict, radio_data: Dict) -> str:
    """Always run on device."""
    return "local"


def policy_always_cloud(model: str, mode: str, network_condition: str,
                        rtt_ms: float, cost_params: Dict, radio_data: Dict) -> str:
    """Always offload to cloud."""
    return "cloud"


def policy_always_hybrid(model: str, mode: str, network_condition: str,
                         rtt_ms: float, cost_params: Dict, radio_data: Dict) -> str:
    """Always use hybrid (cloud inference + local tools)."""
    return "hybrid"


def policy_network_aware(model: str, mode: str, network_condition: str,
                         rtt_ms: float, cost_params: Dict, radio_data: Dict) -> str:
    """Switch based on network quality.

    WiFi/LAN  -> cloud  (low latency, no radio tail)
    5G        -> hybrid (moderate latency)
    4G/poor   -> local  (high latency, radio tail dominates)
    """
    if network_condition in ("wifi", "lan"):
        return "cloud"
    elif network_condition in ("5g",):
        return "hybrid"
    elif network_condition in ("4g", "poor_cellular", "edge_case"):
        return "local"
    elif network_condition == "local":
        return "local"  # No network means local
    else:
        return "local"


def _extract_param_billions(model_name: str) -> float:
    """Extract parameter count in billions from model name."""
    # Match patterns like "4b", "0.6b", "1.5b", "3b"
    match = re.search(r'(\d+(?:\.\d+)?)[bB]', model_name)
    return float(match.group(1)) if match else 3.0


def policy_cost_aware(model: str, mode: str, network_condition: str,
                      rtt_ms: float, cost_params: Dict, radio_data: Dict,
                      energy_threshold_J: float = 2.0,
                      latency_threshold_s: float = 1.0) -> str:
    """Minimize energy-per-success using TPS ratio + model capability.

    Uses two observable signals (no prior benchmarks needed):
      1. TPS ratio (device_tps / cloud_tps): when device is much faster,
         offloading adds network overhead without speedup benefit
      2. Model parameter count: models below 1B lack capability to
         leverage local execution effectively

    Decision tree:
      - No network -> local
      - device TPS > (energy_threshold_J / latency_threshold_s) cloud TPS
        AND model >= 1B params:
          tool mode -> local (cloud/hybrid often fail for these models)
          base mode -> hybrid if network OK, else local
      - All other cases -> cloud (5-8x lower device energy)

    The two thresholds (energy_threshold_J, latency_threshold_s) parameterize
    the TPS-ratio cutoff (as energy/latency) and the RTT cutoff for hybrid
    offload. Defaults (2.0 J, 1.0 s) reproduce the paper's point estimate.
    """
    params = cost_params["parameters"].get(model, {})
    cloud_tps = cost_params.get("cloud_tps", {}).get(model)
    TPS_device = params.get("TPS_device", 50.0)

    if cloud_tps is None or cloud_tps == 0:
        return "local"

    # No network available -> must run locally
    if network_condition == "local":
        return "local"

    # Signal 1: TPS ratio (cutoff = energy_threshold_J / latency_threshold_s;
    # at default (2.0 J, 1.0 s) this is 2.0, matching the original rule)
    tps_ratio = TPS_device / cloud_tps
    tps_cutoff = energy_threshold_J / latency_threshold_s

    # Signal 2: Model capability (parameter count from name)
    param_b = _extract_param_billions(model)
    is_capable = param_b >= 1.0

    # RTT cutoff (ms): 200 ms by default; scales with latency_threshold_s
    rtt_cutoff_ms = 200.0 * latency_threshold_s

    # Device is substantially faster (>cutoff) AND model is capable
    # -> local/hybrid can compete on quality while avoiding network cost
    if tps_ratio > tps_cutoff and is_capable:
        if mode == "full_tool":
            # Local tool execution with a capable fast model.
            return "local"
        else:
            # Base mode: hybrid only if network is usable.
            if rtt_ms <= rtt_cutoff_ms:
                return "hybrid"
            else:
                return "local"

    # Default: cloud offloading
    return "cloud"


POLICIES = {
    "ALWAYS_LOCAL": policy_always_local,
    "ALWAYS_CLOUD": policy_always_cloud,
    "ALWAYS_HYBRID": policy_always_hybrid,
    "NETWORK_AWARE": policy_network_aware,
    "COST_AWARE": policy_cost_aware,
}


# ── Evaluation ───────────────────────────────────────────────────────────────

def build_lookup(arch_data: List[Dict]) -> Dict[Tuple, Dict]:
    """Build a lookup: (model, mode, architecture, network_condition) -> entry."""
    lookup = {}
    for entry in arch_data:
        key = (entry["model"], entry["mode"], entry["architecture"],
               entry["network_condition"])
        lookup[key] = entry
    return lookup


def evaluate_policies(arch_data: List[Dict], cost_params: Dict,
                      radio_data: Dict) -> Dict[str, Any]:
    """Evaluate all policies using the 3-arch experiment data.

    For each (model, mode, network_condition) combination, each policy
    picks an architecture, and we look up the actual experimental result.
    """
    lookup = build_lookup(arch_data)

    # Get unique evaluation points: (model, mode, network_condition)
    eval_points = set()
    for entry in arch_data:
        eval_points.add((entry["model"], entry["mode"],
                         entry["network_condition"]))
    eval_points = sorted(eval_points)

    results = {}

    for policy_name, policy_fn in POLICIES.items():
        policy_results = []

        for model, mode, net_cond in eval_points:
            # Get RTT for this network condition
            sample = lookup.get((model, mode, "cloud", net_cond))
            rtt_ms = sample["rtt_ms"] if sample else 0.0

            # Policy decides architecture
            chosen_arch = policy_fn(model, mode, net_cond, rtt_ms,
                                    cost_params, radio_data)

            # Look up actual result
            key = (model, mode, chosen_arch, net_cond)
            entry = lookup.get(key)

            if entry is None:
                continue

            energy = estimate_energy(entry, cost_params, radio_data)

            policy_results.append({
                "model": model,
                "mode": mode,
                "network_condition": net_cond,
                "rtt_ms": rtt_ms,
                "chosen_architecture": chosen_arch,
                "pass_rate": entry["pass_rate"],
                "avg_time_s": entry["avg_time_s"],
                "est_energy_j": energy,
                "avg_tokens": entry["avg_tokens"],
                "avg_iterations": entry.get("avg_iterations", 1),
            })

        results[policy_name] = policy_results

    # Oracle: for each eval point, pick the best architecture
    oracle_results = []
    for model, mode, net_cond in eval_points:
        best_entry = None
        best_score = (-1.0, -float("inf"))
        for arch in ("local", "cloud", "hybrid"):
            key = (model, mode, arch, net_cond)
            entry = lookup.get(key)
            if entry is None:
                continue
            # Oracle maximizes pass_rate, breaks ties by minimizing time
            score = (entry["pass_rate"], -entry["avg_time_s"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry:
            energy = estimate_energy(best_entry, cost_params, radio_data)
            sample = lookup.get((model, mode, "cloud", net_cond))
            rtt_ms = sample["rtt_ms"] if sample else 0.0
            oracle_results.append({
                "model": model,
                "mode": mode,
                "network_condition": net_cond,
                "rtt_ms": rtt_ms,
                "chosen_architecture": best_entry["architecture"],
                "pass_rate": best_entry["pass_rate"],
                "avg_time_s": best_entry["avg_time_s"],
                "est_energy_j": energy,
                "avg_tokens": best_entry["avg_tokens"],
                "avg_iterations": best_entry.get("avg_iterations", 1),
            })

    results["ORACLE"] = oracle_results

    return results


def compute_summary(results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Compute aggregate metrics per policy."""
    summary = {}
    for policy_name, entries in results.items():
        if not entries:
            continue
        n = len(entries)
        avg_pass = sum(e["pass_rate"] for e in entries) / n
        avg_time = sum(e["avg_time_s"] for e in entries) / n
        avg_energy = sum(e["est_energy_j"] for e in entries) / n

        # Architecture distribution
        arch_counts = {}
        for e in entries:
            arch = e["chosen_architecture"]
            arch_counts[arch] = arch_counts.get(arch, 0) + 1

        summary[policy_name] = {
            "avg_pass_rate": round(avg_pass, 4),
            "avg_time_s": round(avg_time, 2),
            "avg_energy_j": round(avg_energy, 2),
            "n_evaluations": n,
            "architecture_distribution": {
                k: round(v / n * 100, 1)
                for k, v in sorted(arch_counts.items())
            },
        }

    # Compute vs_oracle for each policy
    oracle = summary.get("ORACLE", {})
    oracle_pass = oracle.get("avg_pass_rate", 1.0)
    oracle_energy = oracle.get("avg_energy_j", 1.0)

    for policy_name, s in summary.items():
        if policy_name == "ORACLE":
            s["vs_oracle_pass_pct"] = 100.0
            s["vs_oracle_energy_pct"] = 100.0
        else:
            # Pass rate ratio (higher is better)
            s["vs_oracle_pass_pct"] = round(
                s["avg_pass_rate"] / oracle_pass * 100, 1
            ) if oracle_pass > 0 else 0.0
            # Energy ratio (lower is better; express as % of oracle energy)
            s["vs_oracle_energy_pct"] = round(
                oracle_energy / s["avg_energy_j"] * 100, 1
            ) if s["avg_energy_j"] > 0 else 0.0

    return summary


def build_decision_matrix(results: Dict[str, List[Dict]]) -> Dict:
    """Build matrix showing which architecture each policy picks at each
    (model, mode, network_condition) combination."""
    matrix = {}
    for policy_name, entries in results.items():
        for e in entries:
            key = f"{e['model']}|{e['mode']}|{e['network_condition']}"
            if key not in matrix:
                matrix[key] = {}
            matrix[key][policy_name] = {
                "architecture": e["chosen_architecture"],
                "pass_rate": e["pass_rate"],
                "est_energy_j": e["est_energy_j"],
            }
    return matrix


def build_figure_data(results: Dict[str, List[Dict]],
                      summary: Dict[str, Dict]) -> Dict:
    """Build structured data for paper figures."""
    figures = {}

    # Figure 1: Policy comparison bar chart
    figures["policy_comparison"] = {
        "description": "Bar chart comparing all policies on pass_rate, time, energy",
        "policies": [],
    }
    for policy_name in ["ALWAYS_LOCAL", "ALWAYS_CLOUD", "ALWAYS_HYBRID",
                        "NETWORK_AWARE", "COST_AWARE", "ORACLE"]:
        s = summary.get(policy_name, {})
        figures["policy_comparison"]["policies"].append({
            "policy": policy_name,
            "avg_pass_rate": s.get("avg_pass_rate", 0),
            "avg_time_s": s.get("avg_time_s", 0),
            "avg_energy_j": s.get("avg_energy_j", 0),
            "vs_oracle_pass_pct": s.get("vs_oracle_pass_pct", 0),
        })

    # Figure 2: Energy vs accuracy tradeoff (scatter)
    figures["energy_accuracy_tradeoff"] = {
        "description": "Scatter: x=avg_energy_j, y=avg_pass_rate per policy",
        "points": [],
    }
    for policy_name in ["ALWAYS_LOCAL", "ALWAYS_CLOUD", "ALWAYS_HYBRID",
                        "NETWORK_AWARE", "COST_AWARE", "ORACLE"]:
        s = summary.get(policy_name, {})
        figures["energy_accuracy_tradeoff"]["points"].append({
            "policy": policy_name,
            "avg_energy_j": s.get("avg_energy_j", 0),
            "avg_pass_rate": s.get("avg_pass_rate", 0),
        })

    # Figure 3: Network condition heatmap — which policy wins at each condition
    # For each (network_condition, model), show pass_rate of COST_AWARE vs others
    figures["network_heatmap"] = {
        "description": "Heatmap: rows=network_conditions, cols=models, "
                       "values=COST_AWARE pass_rate vs best static policy",
        "cells": [],
    }

    models = sorted(set(e["model"] for e in results.get("COST_AWARE", [])))
    net_conds = sorted(set(e["network_condition"]
                           for e in results.get("COST_AWARE", [])))

    for net_cond in net_conds:
        for model in models:
            # COST_AWARE result
            ca_entries = [e for e in results["COST_AWARE"]
                          if e["model"] == model
                          and e["network_condition"] == net_cond]
            # Best static result
            best_static_pass = 0
            best_static_policy = ""
            for pname in ["ALWAYS_LOCAL", "ALWAYS_CLOUD", "ALWAYS_HYBRID"]:
                entries = [e for e in results[pname]
                           if e["model"] == model
                           and e["network_condition"] == net_cond]
                if entries:
                    avg_p = sum(e["pass_rate"] for e in entries) / len(entries)
                    if avg_p > best_static_pass:
                        best_static_pass = avg_p
                        best_static_policy = pname

            if ca_entries:
                ca_avg = sum(e["pass_rate"] for e in ca_entries) / len(ca_entries)
                figures["network_heatmap"]["cells"].append({
                    "network_condition": net_cond,
                    "model": model,
                    "cost_aware_pass_rate": round(ca_avg, 4),
                    "cost_aware_arch": ca_entries[0]["chosen_architecture"],
                    "best_static_pass_rate": round(best_static_pass, 4),
                    "best_static_policy": best_static_policy,
                    "improvement_pp": round((ca_avg - best_static_pass) * 100, 1),
                })

    # Figure 4: Per-model policy effectiveness
    figures["per_model_summary"] = {
        "description": "Per-model breakdown of each policy's performance",
        "models": {},
    }
    for model in models:
        model_data = {}
        for policy_name in ["ALWAYS_LOCAL", "ALWAYS_CLOUD", "ALWAYS_HYBRID",
                            "NETWORK_AWARE", "COST_AWARE", "ORACLE"]:
            entries = [e for e in results.get(policy_name, [])
                       if e["model"] == model]
            if entries:
                n = len(entries)
                model_data[policy_name] = {
                    "avg_pass_rate": round(
                        sum(e["pass_rate"] for e in entries) / n, 4
                    ),
                    "avg_time_s": round(
                        sum(e["avg_time_s"] for e in entries) / n, 2
                    ),
                    "avg_energy_j": round(
                        sum(e["est_energy_j"] for e in entries) / n, 2
                    ),
                }
        figures["per_model_summary"]["models"][model] = model_data

    # Figure 5: Mode comparison (base vs full_tool)
    figures["mode_comparison"] = {
        "description": "How each policy performs in base vs full_tool mode",
        "modes": {},
    }
    for mode in ["base", "full_tool"]:
        mode_data = {}
        for policy_name in ["ALWAYS_LOCAL", "ALWAYS_CLOUD", "ALWAYS_HYBRID",
                            "NETWORK_AWARE", "COST_AWARE", "ORACLE"]:
            entries = [e for e in results.get(policy_name, [])
                       if e["mode"] == mode]
            if entries:
                n = len(entries)
                mode_data[policy_name] = {
                    "avg_pass_rate": round(
                        sum(e["pass_rate"] for e in entries) / n, 4
                    ),
                    "avg_energy_j": round(
                        sum(e["est_energy_j"] for e in entries) / n, 2
                    ),
                }
        figures["mode_comparison"]["modes"][mode] = mode_data

    return figures


def sweep_thresholds(
    arch_data: List[Dict],
    cost_params: Dict,
    radio_data: Dict,
    energy_grid=None,
    latency_grid=None,
):
    """Evaluate policy_cost_aware at each (energy_threshold, latency_threshold)
    combination.

    Returns a pandas.DataFrame with columns:
        [energy_threshold, latency_threshold, pass_rate, mean_energy_J,
         oracle_pass_rate, energy_reduction_vs_local, pct_of_oracle_pass].
    """
    import numpy as np
    import pandas as pd

    if energy_grid is None:
        energy_grid = np.linspace(0.5, 5.0, 19).tolist()
    if latency_grid is None:
        latency_grid = np.linspace(0.2, 3.0, 15).tolist()

    lookup = build_lookup(arch_data)

    # Unique evaluation points (model, mode, network_condition)
    eval_points = set()
    for entry in arch_data:
        eval_points.add((entry["model"], entry["mode"],
                         entry["network_condition"]))
    eval_points = sorted(eval_points)

    # Oracle + always-local baselines are threshold-independent — compute once.
    oracle_pass_rates = []
    local_energies = []
    for model, mode, net_cond in eval_points:
        # Oracle
        best_entry = None
        best_score = (-1.0, -float("inf"))
        for arch in ("local", "cloud", "hybrid"):
            e = lookup.get((model, mode, arch, net_cond))
            if e is None:
                continue
            score = (e["pass_rate"], -e["avg_time_s"])
            if score > best_score:
                best_score = score
                best_entry = e
        if best_entry:
            oracle_pass_rates.append(best_entry["pass_rate"])

        # Always-local
        local_entry = lookup.get((model, mode, "local", net_cond))
        if local_entry is not None:
            local_energies.append(
                estimate_energy(local_entry, cost_params, radio_data)
            )

    oracle_pass_rate = (
        sum(oracle_pass_rates) / len(oracle_pass_rates)
        if oracle_pass_rates else 1.0
    )
    mean_energy_local = (
        sum(local_energies) / len(local_energies)
        if local_energies else 1.0
    )

    rows = []
    for e_thr in energy_grid:
        for l_thr in latency_grid:
            pass_rates = []
            energies = []
            for model, mode, net_cond in eval_points:
                sample = lookup.get((model, mode, "cloud", net_cond))
                rtt_ms = sample["rtt_ms"] if sample else 0.0
                chosen = policy_cost_aware(
                    model, mode, net_cond, rtt_ms,
                    cost_params, radio_data,
                    energy_threshold_J=float(e_thr),
                    latency_threshold_s=float(l_thr),
                )
                entry = lookup.get((model, mode, chosen, net_cond))
                if entry is None:
                    continue
                pass_rates.append(entry["pass_rate"])
                energies.append(estimate_energy(entry, cost_params, radio_data))

            if not pass_rates:
                continue
            mean_pass = sum(pass_rates) / len(pass_rates)
            mean_energy = sum(energies) / len(energies)
            rows.append({
                "energy_threshold": float(e_thr),
                "latency_threshold": float(l_thr),
                "pass_rate": mean_pass,
                "mean_energy_J": mean_energy,
                "oracle_pass_rate": oracle_pass_rate,
                "energy_reduction_vs_local": (
                    1.0 - mean_energy / mean_energy_local
                    if mean_energy_local > 0 else 0.0
                ),
                "pct_of_oracle_pass": (
                    mean_pass / oracle_pass_rate
                    if oracle_pass_rate > 0 else 0.0
                ),
            })

    return pd.DataFrame(rows)


def print_summary_table(summary: Dict[str, Dict]) -> None:
    """Print a formatted summary table."""
    header = (f"{'Policy':<18} {'Pass@1':>8} {'Avg Time':>10} "
              f"{'Est Energy':>12} {'vs Oracle':>10}")
    print("\n" + "=" * 64)
    print("  PLACEMENT POLICY EVALUATION RESULTS")
    print("=" * 64)
    print(header)
    print("-" * 64)

    order = ["ALWAYS_LOCAL", "ALWAYS_CLOUD", "ALWAYS_HYBRID",
             "NETWORK_AWARE", "COST_AWARE", "ORACLE"]

    for policy_name in order:
        s = summary.get(policy_name, {})
        pass_rate = s.get("avg_pass_rate", 0)
        avg_time = s.get("avg_time_s", 0)
        avg_energy = s.get("avg_energy_j", 0)
        vs_oracle = s.get("vs_oracle_pass_pct", 0)

        marker = " <--" if policy_name == "COST_AWARE" else ""
        if policy_name == "ORACLE":
            marker = " (upper bound)"

        print(f"{policy_name:<18} {pass_rate:>7.1%} {avg_time:>9.1f}s "
              f"{avg_energy:>11.1f}J {vs_oracle:>9.1f}%{marker}")

    print("-" * 64)

    # Architecture distribution
    print("\nArchitecture Distribution:")
    for policy_name in order:
        s = summary.get(policy_name, {})
        dist = s.get("architecture_distribution", {})
        dist_str = ", ".join(f"{k}: {v}%" for k, v in dist.items())
        print(f"  {policy_name:<18} {dist_str}")

    # Key finding
    print("\n" + "-" * 64)
    ca = summary.get("COST_AWARE", {})
    oracle = summary.get("ORACLE", {})
    if ca and oracle and oracle["avg_pass_rate"] > 0:
        capture_pct = ca["vs_oracle_pass_pct"]
        energy_vs_local = 0
        local_s = summary.get("ALWAYS_LOCAL", {})
        if local_s and local_s["avg_energy_j"] > 0:
            energy_vs_local = round(
                (1 - ca["avg_energy_j"] / local_s["avg_energy_j"]) * 100, 1
            )
        print(f"KEY FINDING: The cost-aware policy captures {capture_pct:.1f}% "
              f"of oracle pass rate")
        print(f"             while using {energy_vs_local}% less energy "
              f"than always-local.")

    # Per-model breakdown
    print("\n" + "=" * 64)
    print("  PER-MODEL BREAKDOWN (COST_AWARE)")
    print("=" * 64)

    # Reconstruct from results (we need the raw results for this)
    # This is printed from the figure data in main()


def print_per_model_table(figures: Dict) -> None:
    """Print per-model breakdown."""
    per_model = figures.get("per_model_summary", {}).get("models", {})
    if not per_model:
        return

    for model, policies in sorted(per_model.items()):
        print(f"\n  {model}:")
        for pname in ["ALWAYS_LOCAL", "ALWAYS_CLOUD", "ALWAYS_HYBRID",
                       "NETWORK_AWARE", "COST_AWARE", "ORACLE"]:
            p = policies.get(pname, {})
            if p:
                print(f"    {pname:<18} pass={p['avg_pass_rate']:.1%}  "
                      f"time={p['avg_time_s']:.1f}s  "
                      f"energy={p['avg_energy_j']:.1f}J")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    arch_data, cost_params, radio_data = load_data()
    print(f"  3-arch entries: {len(arch_data)}")
    print(f"  Models in cost model: {list(cost_params['parameters'].keys())}")

    print("\nEvaluating policies...")
    results = evaluate_policies(arch_data, cost_params, radio_data)

    for policy_name, entries in results.items():
        print(f"  {policy_name}: {len(entries)} evaluations")

    summary = compute_summary(results)
    decision_matrix = build_decision_matrix(results)
    figures = build_figure_data(results, summary)

    # Print results
    print_summary_table(summary)
    print_per_model_table(figures)

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "policy_evaluation.json", "w") as f:
        json.dump({
            "description": "Placement policy evaluation using 3-arch experiment data",
            "summary": summary,
            "per_policy_results": {
                k: v for k, v in results.items()
            },
        }, f, indent=2)

    with open(OUTPUT_DIR / "decision_matrix.json", "w") as f:
        json.dump({
            "description": "Which architecture each policy chooses at each "
                           "(model, mode, network_condition) combination",
            "matrix": decision_matrix,
        }, f, indent=2)

    with open(OUTPUT_DIR / "figures_data.json", "w") as f:
        json.dump(figures, f, indent=2)

    print(f"\nOutput saved to {OUTPUT_DIR}/")
    print(f"  policy_evaluation.json  ({os.path.getsize(OUTPUT_DIR / 'policy_evaluation.json') // 1024} KB)")
    print(f"  decision_matrix.json    ({os.path.getsize(OUTPUT_DIR / 'decision_matrix.json') // 1024} KB)")
    print(f"  figures_data.json       ({os.path.getsize(OUTPUT_DIR / 'figures_data.json') // 1024} KB)")

    # ── Threshold sensitivity sweep ──────────────────────────────────────────
    print("\nRunning threshold sensitivity sweep (19 x 15 = 285 grid cells)...")
    df = sweep_thresholds(arch_data, cost_params, radio_data)
    sweep_out_dir = DATA_DIR / "placement_evaluation"
    sweep_out_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv = sweep_out_dir / "threshold_sensitivity.csv"
    df.to_csv(sweep_csv, index=False)
    print(f"  wrote {sweep_csv} ({len(df)} rows)")
    print(f"  pct_of_oracle_pass: min={df['pct_of_oracle_pass'].min():.3f} "
          f"max={df['pct_of_oracle_pass'].max():.3f} "
          f"mean={df['pct_of_oracle_pass'].mean():.3f}")
    print(f"  energy_reduction_vs_local: min={df['energy_reduction_vs_local'].min():.3f} "
          f"max={df['energy_reduction_vs_local'].max():.3f}")


if __name__ == "__main__":
    main()
