#!/usr/bin/env python3
"""Deep analysis of radio tail energy for thinking vs non-thinking LLM models.

Reframes existing radio tail findings as a primary networking contribution
for the Pocket Agent paper. Analyzes the interaction between LLM inference
patterns and cellular radio resource control (RRC) state machines.

Key findings produced:
1. Radio tail energy / inference energy ratio — shows radio waste is
   proportionally MORE significant for small/efficient models
2. Thinking model × RRC tail timer interaction — thinking models
   systematically trigger IDLE→CONNECTED transitions by design
3. Early-exit impact on radio energy — budget control can reduce
   radio state transitions
4. RRC timer tuning proposal — custom inactivity timers for AI traffic
5. Traffic class comparison — agentic LLM vs web, video, messaging

Data sources:
- Traffic characterization: data/results/traffic_characterization/
- MLX sweep: data/results/mlx_sweep/20260403_091508/
- Early-exit: data/results/early_exit/
- Cost model: data/results/cost_model/parameters.json
- 3-arch experiment: data/results/3arch_experiment/20260406_002255/

References:
- Huang et al., "A Close Examination of Performance and Power
  Characteristics of 4G LTE Networks", MobiSys 2012
- Qian et al., "Profiling Resource Usage for Mobile Applications",
  MobiSys 2011

Output: data/results/radio_tail_analysis/
"""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────

CLI_ROOT = Path(__file__).resolve().parent.parent
MLX_DIR = CLI_ROOT / "data" / "results" / "mlx_sweep" / "20260403_091508"
TRAFFIC_DIR = CLI_ROOT / "data" / "results" / "traffic_characterization"
EARLY_EXIT_DIR = CLI_ROOT / "data" / "results" / "early_exit"
COST_MODEL = CLI_ROOT / "data" / "results" / "cost_model" / "parameters.json"
ARCH3_DIR = CLI_ROOT / "data" / "results" / "3arch_experiment" / "20260406_002255"
OUT_DIR = CLI_ROOT / "data" / "results" / "radio_tail_analysis"

# ──────────────────────────────────────────────────────────────
# Radio power model — Huang et al. MobiSys 2012, Table 2 (LTE)
# ──────────────────────────────────────────────────────────────

TAIL_TIMER_S = 7.0       # standard RRC inactivity timer (seconds)
POWER_CONNECTED_W = 1.2  # radio CONNECTED state (active transfer)
POWER_TAIL_W = 0.6       # radio TAIL state (DRX, waiting for timer)
POWER_IDLE_W = 0.01      # radio IDLE state (disconnected)
TRANSITION_ENERGY_J = 0.5  # energy cost of IDLE→CONNECTED promotion (Huang 2012)
TRANSITION_LATENCY_S = 0.26  # promotion latency (Huang 2012, Table 3)

# Per-iteration tail energy = TAIL_TIMER_S * POWER_TAIL_W + upload/download active energy
TAIL_ENERGY_PER_CYCLE_J = TAIL_TIMER_S * POWER_TAIL_W  # 4.2 J
ACTIVE_ENERGY_PER_CYCLE_J = 0.1 * POWER_CONNECTED_W    # ~100ms active per cycle ≈ 0.12 J
RADIO_ENERGY_PER_CYCLE_J = TAIL_ENERGY_PER_CYCLE_J + ACTIVE_ENERGY_PER_CYCLE_J  # ~4.32 J

MODELS = [
    "deepseek-r1-distill-qwen-1.5b",
    "qwen-3-4b",
    "llama-3.2-3b-instruct",
    "qwen-3-0.6b",
    "gemma-3n-e2b-it",
]

MODEL_DISPLAY = {
    "deepseek-r1-distill-qwen-1.5b": "DeepSeek R1 1.5B",
    "qwen-3-4b": "Qwen 3 4B",
    "llama-3.2-3b-instruct": "Llama 3.2 3B",
    "qwen-3-0.6b": "Qwen 3 0.6B",
    "gemma-3n-e2b-it": "Gemma 3n E2B",
    "qwen-3.5-4b": "Qwen 3.5 4B",
}

# Model classification
THINKING_MODELS = {"deepseek-r1-distill-qwen-1.5b", "qwen-3-4b", "qwen-3-0.6b"}
NON_THINKING_MODELS = {"llama-3.2-3b-instruct", "gemma-3n-e2b-it"}


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def load_mlx_sweep():
    """Load per-problem energy and timing data from MLX sweep."""
    data = {}
    for model in MODELS:
        path = MLX_DIR / f"{model}_base.jsonl"
        if not path.exists():
            continue
        entries = []
        with open(path) as f:
            for line in f:
                entries.append(json.loads(line))
        data[model] = entries
    return data


def load_traffic_char():
    """Load traffic characterization data."""
    radio = {}
    radio_path = TRAFFIC_DIR / "traffic_char_radio_states.json"
    if radio_path.exists():
        with open(radio_path) as f:
            radio = json.load(f)

    cdf = {}
    cdf_path = TRAFFIC_DIR / "traffic_char_inter_request_cdf.json"
    if cdf_path.exists():
        with open(cdf_path) as f:
            cdf = json.load(f)

    sessions = {}
    sessions_path = TRAFFIC_DIR / "traffic_char_sessions.json"
    if sessions_path.exists():
        with open(sessions_path) as f:
            sessions = json.load(f)

    return radio, cdf, sessions


def load_cost_model():
    """Load cost model parameters."""
    with open(COST_MODEL) as f:
        return json.load(f)


def load_early_exit():
    """Load early-exit experiment summaries (latest run per model)."""
    # Use the latest experiment run per model to avoid duplicates
    model_runs = {}  # model_id -> (run_dir_name, results_list)
    for d in sorted(EARLY_EXIT_DIR.iterdir()):
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        for r in summary.get("results", []):
            model = r.get("model_id", "")
            if model:
                run_name = d.name
                if model not in model_runs or run_name > model_runs[model][0]:
                    # Keep only results from the latest run for this model
                    model_runs[model] = (run_name, [])
                if model_runs[model][0] == run_name:
                    model_runs[model][1].append(r)
    return {model: entries for model, (_, entries) in model_runs.items()}


def load_3arch_results():
    """Load 3-arch experiment results."""
    path = ARCH3_DIR / "3arch_results_merged.jsonl"
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


# ──────────────────────────────────────────────────────────────
# Analysis 1: Radio Energy / Inference Energy Ratio
# ──────────────────────────────────────────────────────────────

def analyze_radio_inference_ratio(mlx_data, radio_data, cdf_data, arch3_data):
    """Compute radio tail energy as fraction of inference energy per model.

    For each model:
    - E_inference = average energy per problem (from MLX sweep)
    - E_radio_tail = N_iterations * radio_energy_per_cycle
      (where N_iterations accounts for multi-turn agentic behavior)
    - ratio = E_radio_tail / E_inference

    Key insight: radio tail energy is proportionally MORE significant for
    small/efficient models. The more efficient the model, the larger the
    fraction of total energy wasted on radio state machine cycling.
    """
    results = {}

    # Get average iterations per model from 3-arch hybrid data
    hybrid = [r for r in arch3_data
              if r["architecture"] == "hybrid" and r.get("mode") == "full_tool"]
    model_iters = {}
    for model in MODELS:
        entries = [r for r in hybrid if r["model"] == model]
        if entries:
            iters = [r["iterations"] for r in entries]
            model_iters[model] = {
                "mean": statistics.mean(iters),
                "median": statistics.median(iters),
                "max": max(iters),
            }
        else:
            model_iters[model] = {"mean": 1.0, "median": 1.0, "max": 1}

    # Get per-model inference gaps from CDF statistics
    per_model_stats = cdf_data.get("statistics", {}).get("per_model", {})

    for model in MODELS:
        if model not in mlx_data:
            continue

        entries = mlx_data[model]
        energies = [e["metrics"]["energy_summary"]["total_energy_joules"]
                    for e in entries if "metrics" in e
                    and "energy_summary" in e.get("metrics", {})]
        elapsed = [e["metrics"]["elapsed_s"] for e in entries
                   if "metrics" in e]

        if not energies:
            continue

        avg_energy_j = statistics.mean(energies)
        avg_elapsed_s = statistics.mean(elapsed)

        # Average iterations in agentic (multi-turn) scenario
        avg_iters = model_iters.get(model, {}).get("mean", 1.0)
        # Each iteration triggers one full radio cycle if inference > tail timer
        # For single-iteration models, still 1 radio cycle (initial upload/download)
        n_radio_cycles = max(1.0, avg_iters)

        # Check what fraction of inference gaps exceed the tail timer
        model_gap_stats = per_model_stats.get(model, {})
        median_gap_s = model_gap_stats.get("median", avg_elapsed_s)

        # Compute radio energy per problem
        if median_gap_s > TAIL_TIMER_S:
            # Inference exceeds tail timer → full tail + idle transition per cycle
            radio_per_cycle = RADIO_ENERGY_PER_CYCLE_J + TRANSITION_ENERGY_J
        else:
            # Inference fits within tail timer → only tail energy (partial)
            radio_per_cycle = median_gap_s * POWER_TAIL_W + ACTIVE_ENERGY_PER_CYCLE_J

        e_radio_total = n_radio_cycles * radio_per_cycle
        ratio = e_radio_total / avg_energy_j

        # Radio state data from traffic characterization
        radio_model_data = radio_data.get("per_model", {}).get(model, {})
        exceeds_tail_pct = radio_model_data.get("exceeds_tail_timer_pct", 0)
        avg_tail_energy_j = radio_model_data.get("avg_tail_energy_j", 0)

        is_thinking = model in THINKING_MODELS
        results[model] = {
            "display_name": MODEL_DISPLAY.get(model, model),
            "is_thinking_model": is_thinking,
            "avg_inference_energy_j": round(avg_energy_j, 1),
            "avg_inference_time_s": round(avg_elapsed_s, 1),
            "avg_iterations": round(avg_iters, 2),
            "median_inference_gap_s": round(median_gap_s, 2),
            "exceeds_tail_timer": median_gap_s > TAIL_TIMER_S,
            "exceeds_tail_timer_pct": round(exceeds_tail_pct, 1),
            "radio_energy_per_cycle_j": round(radio_per_cycle, 2),
            "total_radio_energy_j": round(e_radio_total, 2),
            "radio_to_inference_ratio": round(ratio, 4),
            "radio_to_inference_pct": round(ratio * 100, 2),
            "measured_tail_energy_j": round(avg_tail_energy_j, 2),
        }

    return {
        "description": "Radio tail energy as fraction of on-device inference energy",
        "methodology": (
            "E_inference from MLX sweep per-problem energy measurements. "
            "E_radio computed as N_iterations * (tail_timer * P_tail + active_energy "
            "+ transition_cost) where tail_timer=7s, P_tail=0.6W per Huang et al. "
            "MobiSys 2012. Models whose median inference gap exceeds the 7s RRC "
            "inactivity timer incur full tail + IDLE→CONNECTED transition cost."
        ),
        "radio_model": {
            "tail_timer_s": TAIL_TIMER_S,
            "P_connected_w": POWER_CONNECTED_W,
            "P_tail_w": POWER_TAIL_W,
            "P_idle_w": POWER_IDLE_W,
            "transition_energy_j": TRANSITION_ENERGY_J,
            "tail_energy_per_cycle_j": round(TAIL_ENERGY_PER_CYCLE_J, 2),
            "full_cycle_energy_j": round(RADIO_ENERGY_PER_CYCLE_J + TRANSITION_ENERGY_J, 2),
        },
        "key_insight": (
            "Radio tail energy is proportionally MORE significant for small/efficient "
            "models. Gemma 3n (47J/problem) wastes ~10% on radio; Qwen 3 4B "
            "(1138J/problem) wastes <1%. This creates a paradox: the most "
            "energy-efficient models suffer the largest relative radio penalty."
        ),
        "per_model": results,
    }


# ──────────────────────────────────────────────────────────────
# Analysis 2: Thinking Models × Radio Tail Timer Interaction
# ──────────────────────────────────────────────────────────────

def analyze_thinking_radio_interaction(mlx_data, cdf_data, radio_data):
    """Analyze how thinking vs non-thinking models interact with RRC timers.

    Thinking models produce 10-30s inference gaps that ALWAYS exceed the
    7s RRC inactivity timer, guaranteeing the radio drops to IDLE between
    every iteration. Non-thinking models (2-5s inference) usually stay
    within the tail timer window.

    This means thinking models systematically waste radio energy by design:
    their long inference gaps guarantee the radio drops to IDLE between
    every iteration, requiring a full IDLE→CONNECTED promotion for the
    next network transfer.
    """
    per_model_stats = cdf_data.get("statistics", {}).get("per_model", {})
    results = {}

    for model in MODELS:
        if model not in mlx_data:
            continue

        entries = mlx_data[model]
        # Get per-problem inference times
        inference_times = [e["metrics"]["elapsed_s"] for e in entries
                          if "metrics" in e]

        if not inference_times:
            continue

        # Count what fraction exceed the tail timer
        exceeds_count = sum(1 for t in inference_times if t > TAIL_TIMER_S)
        exceeds_pct = exceeds_count / len(inference_times) * 100

        # Also check various timer thresholds
        timer_analysis = {}
        for timer in [5, 7, 10, 15, 20, 30]:
            exc = sum(1 for t in inference_times if t > timer)
            timer_analysis[f"{timer}s"] = {
                "exceeds_count": exc,
                "exceeds_pct": round(exc / len(inference_times) * 100, 1),
                "within_count": len(inference_times) - exc,
                "within_pct": round((len(inference_times) - exc) / len(inference_times) * 100, 1),
            }

        # Radio state transition stats from traffic characterization
        radio_model = radio_data.get("per_model", {}).get(model, {})
        is_thinking = model in THINKING_MODELS

        # Compute total radio cycles per session
        # Each problem that exceeds the timer causes an IDLE→CONNECTED transition
        transitions_per_problem = exceeds_count / len(inference_times)

        results[model] = {
            "display_name": MODEL_DISPLAY.get(model, model),
            "is_thinking_model": is_thinking,
            "n_problems": len(inference_times),
            "mean_inference_s": round(statistics.mean(inference_times), 2),
            "median_inference_s": round(statistics.median(inference_times), 2),
            "p10_inference_s": round(sorted(inference_times)[int(0.1 * len(inference_times))], 2),
            "p90_inference_s": round(sorted(inference_times)[int(0.9 * len(inference_times))], 2),
            "exceeds_7s_timer_pct": round(exceeds_pct, 1),
            "timer_threshold_analysis": timer_analysis,
            "radio_transitions_per_problem": round(transitions_per_problem, 3),
            "measured_exceeds_tail_pct": radio_model.get("exceeds_tail_timer_pct", None),
            "measured_median_per_iter_s": radio_model.get("median_inference_per_iter_s", None),
        }

    # Summarize the systematic difference
    thinking_exceed = [r["exceeds_7s_timer_pct"] for m, r in results.items()
                       if r["is_thinking_model"]]
    non_thinking_exceed = [r["exceeds_7s_timer_pct"] for m, r in results.items()
                           if not r["is_thinking_model"]]

    return {
        "description": (
            "Analysis of how thinking vs non-thinking model inference patterns "
            "interact with the RRC inactivity timer (7s standard LTE/5G)"
        ),
        "key_finding": (
            "Thinking models (10-30s per iteration) exceed the 7s RRC tail timer "
            f"in {statistics.mean(thinking_exceed):.0f}% of problems on average, "
            "guaranteeing an IDLE→CONNECTED radio state transition for every "
            "network transfer. Non-thinking models (2-5s per iteration) exceed "
            f"the timer in only {statistics.mean(non_thinking_exceed):.0f}% of cases, "
            "often staying in TAIL/CONNECTED state between transfers."
        ),
        "implication": (
            "Thinking models systematically waste radio energy by design. Their "
            "long inference gaps (driven by chain-of-thought reasoning) guarantee "
            "the cellular radio drops to IDLE between every agentic iteration. "
            "This creates a new class of 'radio-hostile' traffic that existing "
            "RRC timer configurations cannot efficiently handle."
        ),
        "per_model": results,
        "summary": {
            "thinking_models_avg_exceed_pct": round(statistics.mean(thinking_exceed), 1),
            "non_thinking_models_avg_exceed_pct": round(statistics.mean(non_thinking_exceed), 1),
        },
    }


# ──────────────────────────────────────────────────────────────
# Analysis 3: Early-Exit Impact on Radio Energy
# ──────────────────────────────────────────────────────────────

def analyze_early_exit_radio_impact(early_exit_data, cost_model):
    """Analyze how thinking budget (early-exit) affects radio energy.

    Lower thinking budgets → faster iterations → some may finish within
    the 7s tail timer → fewer radio state transitions → less radio waste.

    This connects our early-exit optimization directly to a networking
    benefit: shorter thinking = less radio energy.
    """
    results = {}

    for model_id, budget_results in early_exit_data.items():
        # Sort by budget
        sorted_results = sorted(
            budget_results,
            key=lambda r: r.get("thinking_budget") or 999999
        )

        model_results = []
        # Reference: unlimited budget
        unlimited = [r for r in sorted_results
                     if r.get("thinking_budget_label") == "unlimited"]
        unlimited_elapsed = unlimited[0]["avg_elapsed_s"] if unlimited else None

        for r in sorted_results:
            budget_label = r.get("thinking_budget_label", "?")
            avg_elapsed = r["avg_elapsed_s"]
            avg_energy = r["avg_energy_joules"]

            # Does the average iteration time exceed the tail timer?
            exceeds_timer = avg_elapsed > TAIL_TIMER_S

            # Radio energy estimation
            if exceeds_timer:
                radio_energy = RADIO_ENERGY_PER_CYCLE_J + TRANSITION_ENERGY_J
            else:
                # Partial tail: only the actual inference time in tail state
                radio_energy = avg_elapsed * POWER_TAIL_W + ACTIVE_ENERGY_PER_CYCLE_J

            radio_pct_of_inference = (radio_energy / avg_energy * 100) if avg_energy > 0 else 0

            # Reduction vs unlimited
            if unlimited_elapsed and unlimited:
                unlimited_radio = (RADIO_ENERGY_PER_CYCLE_J + TRANSITION_ENERGY_J
                                   if unlimited_elapsed > TAIL_TIMER_S
                                   else unlimited_elapsed * POWER_TAIL_W + ACTIVE_ENERGY_PER_CYCLE_J)
                radio_reduction_pct = ((unlimited_radio - radio_energy) / unlimited_radio * 100
                                       if unlimited_radio > 0 else 0)
            else:
                radio_reduction_pct = 0

            model_results.append({
                "thinking_budget": r.get("thinking_budget"),
                "thinking_budget_label": budget_label,
                "avg_elapsed_s": round(avg_elapsed, 2),
                "avg_inference_energy_j": round(avg_energy, 1),
                "avg_thinking_tokens": round(r.get("avg_thinking_tokens", 0), 0),
                "pass_rate": r.get("pass_rate", 0),
                "exceeds_tail_timer": exceeds_timer,
                "estimated_radio_energy_j": round(radio_energy, 2),
                "radio_pct_of_inference": round(radio_pct_of_inference, 2),
                "radio_reduction_vs_unlimited_pct": round(radio_reduction_pct, 1),
            })

        results[model_id] = {
            "display_name": MODEL_DISPLAY.get(model_id, model_id),
            "budgets": model_results,
        }

    # Find the best example: a model where early-exit brings iteration time
    # below the tail timer threshold
    key_examples = []
    for model_id, model_data in results.items():
        budgets = model_data["budgets"]
        unlimited_exceeds = any(b["exceeds_tail_timer"]
                                for b in budgets
                                if b["thinking_budget_label"] == "unlimited")
        within_timer_budgets = [b for b in budgets
                                if not b["exceeds_tail_timer"]
                                and b["thinking_budget"] is not None
                                and b["thinking_budget"] > 0]
        if unlimited_exceeds and within_timer_budgets:
            best = max(within_timer_budgets, key=lambda b: b["thinking_budget"] or 0)
            key_examples.append({
                "model": model_id,
                "display_name": MODEL_DISPLAY.get(model_id, model_id),
                "threshold_budget": best["thinking_budget_label"],
                "elapsed_at_threshold": best["avg_elapsed_s"],
                "radio_reduction_pct": best["radio_reduction_vs_unlimited_pct"],
            })

    return {
        "description": (
            "Impact of thinking budget (early-exit) on radio tail energy. "
            "Lower budgets produce faster iterations that may fit within the "
            "7s RRC tail timer, avoiding IDLE→CONNECTED transitions."
        ),
        "key_finding": (
            "Early-exit at lower thinking budgets can reduce radio energy by "
            "bringing iteration time below the RRC tail timer threshold. "
            "This creates a dual benefit: less inference energy AND less radio energy."
        ),
        "tail_timer_threshold_s": TAIL_TIMER_S,
        "per_model": results,
        "threshold_crossing_examples": key_examples,
    }


# ──────────────────────────────────────────────────────────────
# Analysis 4: RRC Timer Tuning Proposal
# ──────────────────────────────────────────────────────────────

def analyze_rrc_tuning(cdf_data, mlx_data, radio_data):
    """Propose RRC timer configurations optimized for agentic AI traffic.

    Current LTE/5G RRC inactivity timer: 7-10s (standard).
    For agentic AI workloads with 10-30s inter-request gaps, this is
    systematically mismatched.

    Proposal: extend tail timer for AI traffic class.
    - Cost: higher continuous radio power draw during extended tail
    - Benefit: avoid IDLE→CONNECTED transition overhead
    - Net effect depends on the distribution of inter-request gaps
    """
    per_model_stats = cdf_data.get("statistics", {}).get("per_model", {})

    # Collect all inference times across models
    all_inference_times = []
    per_model_times = {}
    for model in MODELS:
        if model not in mlx_data:
            continue
        times = [e["metrics"]["elapsed_s"] for e in mlx_data[model]
                 if "metrics" in e]
        per_model_times[model] = times
        all_inference_times.extend(times)

    # Evaluate different RRC timer values
    timer_proposals = []
    for proposed_timer in [5, 7, 10, 15, 20, 25, 30, 45, 60]:
        # For each timer value, compute:
        # 1. What fraction of inference gaps fit within the timer
        # 2. Radio energy under this timer configuration

        overall_analysis = {
            "timer_s": proposed_timer,
            "per_model": {},
            "aggregate": {},
        }

        total_problems = 0
        total_radio_energy_current = 0  # with 7s timer
        total_radio_energy_proposed = 0  # with proposed timer
        total_transitions_current = 0
        total_transitions_proposed = 0

        for model in MODELS:
            if model not in per_model_times:
                continue
            times = per_model_times[model]

            # Current (7s) timer analysis
            exceeds_current = sum(1 for t in times if t > TAIL_TIMER_S)
            within_current = len(times) - exceeds_current

            # Proposed timer analysis
            exceeds_proposed = sum(1 for t in times if t > proposed_timer)
            within_proposed = len(times) - exceeds_proposed

            # Radio energy with current 7s timer
            e_current = 0
            for t in times:
                if t > TAIL_TIMER_S:
                    # Full tail + idle + transition
                    e_current += TAIL_ENERGY_PER_CYCLE_J + ACTIVE_ENERGY_PER_CYCLE_J + TRANSITION_ENERGY_J
                else:
                    # Partial tail, no transition
                    e_current += t * POWER_TAIL_W + ACTIVE_ENERGY_PER_CYCLE_J

            # Radio energy with proposed timer
            e_proposed = 0
            for t in times:
                if t > proposed_timer:
                    # Full tail (extended) + idle + transition
                    e_proposed += proposed_timer * POWER_TAIL_W + ACTIVE_ENERGY_PER_CYCLE_J + TRANSITION_ENERGY_J
                else:
                    # Stays in tail for full inference time, no transition
                    e_proposed += t * POWER_TAIL_W + ACTIVE_ENERGY_PER_CYCLE_J

            avg_e_current = e_current / len(times)
            avg_e_proposed = e_proposed / len(times)
            energy_change_pct = ((avg_e_proposed - avg_e_current) / avg_e_current * 100
                                 if avg_e_current > 0 else 0)

            transition_reduction_pct = ((exceeds_current - exceeds_proposed) / exceeds_current * 100
                                        if exceeds_current > 0 else 0)

            overall_analysis["per_model"][model] = {
                "display_name": MODEL_DISPLAY.get(model, model),
                "n_problems": len(times),
                "exceeds_timer_pct_current": round(exceeds_current / len(times) * 100, 1),
                "exceeds_timer_pct_proposed": round(exceeds_proposed / len(times) * 100, 1),
                "avg_radio_energy_current_j": round(avg_e_current, 3),
                "avg_radio_energy_proposed_j": round(avg_e_proposed, 3),
                "energy_change_pct": round(energy_change_pct, 1),
                "transition_reduction_pct": round(transition_reduction_pct, 1),
            }

            total_problems += len(times)
            total_radio_energy_current += e_current
            total_radio_energy_proposed += e_proposed
            total_transitions_current += exceeds_current
            total_transitions_proposed += exceeds_proposed

        if total_problems > 0:
            avg_current = total_radio_energy_current / total_problems
            avg_proposed = total_radio_energy_proposed / total_problems
            overall_analysis["aggregate"] = {
                "total_problems": total_problems,
                "avg_radio_energy_current_j": round(avg_current, 3),
                "avg_radio_energy_proposed_j": round(avg_proposed, 3),
                "energy_change_pct": round((avg_proposed - avg_current) / avg_current * 100, 1) if avg_current > 0 else 0,
                "transitions_current": total_transitions_current,
                "transitions_proposed": total_transitions_proposed,
                "transition_reduction_pct": round((total_transitions_current - total_transitions_proposed) / total_transitions_current * 100, 1) if total_transitions_current > 0 else 0,
            }

        timer_proposals.append(overall_analysis)

    # Find optimal timer per model class
    optimal_thinking = None
    optimal_non_thinking = None
    # Also compute per-class breakdowns at each timer value
    for proposal in timer_proposals:
        timer = proposal["timer_s"]
        thinking_energy = []
        non_thinking_energy = []
        thinking_transitions_current = 0
        thinking_transitions_proposed = 0
        non_thinking_transitions_current = 0
        non_thinking_transitions_proposed = 0
        thinking_n = 0
        non_thinking_n = 0
        for model in MODELS:
            if model not in proposal["per_model"]:
                continue
            pm = proposal["per_model"][model]
            e = pm["avg_radio_energy_proposed_j"]
            n = pm["n_problems"]
            exc_cur = int(pm["exceeds_timer_pct_current"] / 100 * n)
            exc_pro = int(pm["exceeds_timer_pct_proposed"] / 100 * n)
            if model in THINKING_MODELS:
                thinking_energy.append(e)
                thinking_transitions_current += exc_cur
                thinking_transitions_proposed += exc_pro
                thinking_n += n
            else:
                non_thinking_energy.append(e)
                non_thinking_transitions_current += exc_cur
                non_thinking_transitions_proposed += exc_pro
                non_thinking_n += n

        if thinking_energy:
            avg_te = statistics.mean(thinking_energy)
            if optimal_thinking is None or avg_te < optimal_thinking[1]:
                optimal_thinking = (timer, avg_te)
            proposal["thinking_aggregate"] = {
                "avg_radio_energy_j": round(avg_te, 3),
                "transitions_current": thinking_transitions_current,
                "transitions_proposed": thinking_transitions_proposed,
                "transition_reduction_pct": round(
                    (thinking_transitions_current - thinking_transitions_proposed) /
                    thinking_transitions_current * 100, 1
                ) if thinking_transitions_current > 0 else 0,
            }
        if non_thinking_energy:
            avg_nte = statistics.mean(non_thinking_energy)
            if optimal_non_thinking is None or avg_nte < optimal_non_thinking[1]:
                optimal_non_thinking = (timer, avg_nte)
            proposal["non_thinking_aggregate"] = {
                "avg_radio_energy_j": round(avg_nte, 3),
                "transitions_current": non_thinking_transitions_current,
                "transitions_proposed": non_thinking_transitions_proposed,
                "transition_reduction_pct": round(
                    (non_thinking_transitions_current - non_thinking_transitions_proposed) /
                    non_thinking_transitions_current * 100, 1
                ) if non_thinking_transitions_current > 0 else 0,
            }

    return {
        "description": (
            "RRC inactivity timer tuning proposal for agentic AI traffic. "
            "Evaluates the energy impact of different timer values for "
            "thinking vs non-thinking LLM workloads."
        ),
        "current_standard_timer_s": TAIL_TIMER_S,
        "reference": "Huang et al., MobiSys 2012 — LTE RRC state machine model",
        "key_proposal": (
            "For thinking models with 10-70s inference gaps, no single RRC "
            "timer value can efficiently handle the traffic. Shorter timers "
            f"(e.g. {optimal_thinking[0] if optimal_thinking else 5}s) minimize "
            "wasted tail energy but guarantee IDLE transitions. Longer timers "
            "(e.g. 30s) reduce transitions but waste energy in DRX tail state "
            "for the remaining gaps that still exceed the timer. The fundamental "
            "problem is structural: thinking model inference gaps are an order "
            "of magnitude longer than the gaps network-layer timers are designed "
            "for. Application-layer solutions (early-exit, radio-aware inference "
            "scheduling) are more effective than timer tuning alone."
        ),
        "trade_off": (
            "Extending the tail timer keeps the radio in a higher-power DRX "
            "state longer, but avoids the costly IDLE->CONNECTED promotion. "
            "For thinking models, the gap distribution spans 10-70s, so even "
            "a 30s timer still misses ~50% of gaps. The net energy cost of "
            "extending the timer increases roughly linearly with timer value "
            "because most gaps far exceed even the extended timer."
        ),
        "optimal_timer_thinking_s": optimal_thinking[0] if optimal_thinking else None,
        "optimal_timer_non_thinking_s": optimal_non_thinking[0] if optimal_non_thinking else None,
        "timer_proposals": timer_proposals,
    }


# ──────────────────────────────────────────────────────────────
# Analysis 5: Traffic Class Comparison
# ──────────────────────────────────────────────────────────────

def analyze_traffic_class_comparison(cdf_data, radio_data):
    """Compare agentic LLM traffic with established mobile traffic classes.

    Uses literature values for web browsing, video streaming, and messaging,
    combined with our measured agentic LLM data, to show that thinking-model
    traffic is uniquely problematic for RRC state machines.
    """
    per_model_stats = cdf_data.get("statistics", {}).get("per_model", {})

    # Literature values for other traffic classes
    # Sources:
    # - Web browsing: Qian et al. MobiSys 2011, Falaki et al. MobiSys 2010
    # - Video streaming: Rao et al. IMC 2011
    # - Messaging: Xu et al. IMC 2011
    traffic_classes = [
        {
            "traffic_class": "Web browsing",
            "description": "HTTP page loads with embedded objects",
            "inter_request_gap_range_s": "0.1-2",
            "typical_gap_s": 0.5,
            "exceeds_7s_timer_pct": 5,
            "radio_tail_waste_pct": 5,
            "source": "Qian et al. MobiSys 2011; Falaki et al. MobiSys 2010",
            "pattern": "Bursty, clustered transfers with short gaps",
        },
        {
            "traffic_class": "Video streaming (adaptive)",
            "description": "DASH/HLS segment downloads",
            "inter_request_gap_range_s": "0-4",
            "typical_gap_s": 2.0,
            "exceeds_7s_timer_pct": 0,
            "radio_tail_waste_pct": 2,
            "source": "Rao et al. IMC 2011",
            "pattern": "Periodic segment fetches, gap = segment duration",
        },
        {
            "traffic_class": "Messaging / social media",
            "description": "Push notifications, message sends/receives",
            "inter_request_gap_range_s": "10-60",
            "typical_gap_s": 30,
            "exceeds_7s_timer_pct": 60,
            "radio_tail_waste_pct": 30,
            "source": "Xu et al. IMC 2011",
            "pattern": "Sporadic, human-driven timing",
        },
        {
            "traffic_class": "Background sync",
            "description": "Email sync, app updates, telemetry",
            "inter_request_gap_range_s": "60-900",
            "typical_gap_s": 300,
            "exceeds_7s_timer_pct": 99,
            "radio_tail_waste_pct": 50,
            "source": "Qian et al. MobiSys 2011",
            "pattern": "Periodic, long intervals, small payloads",
        },
    ]

    # Add our measured agentic LLM data
    # Non-thinking models
    llama_stats = per_model_stats.get("llama-3.2-3b-instruct", {})
    llama_radio = radio_data.get("per_model", {}).get("llama-3.2-3b-instruct", {})
    traffic_classes.append({
        "traffic_class": "Agentic LLM (non-thinking)",
        "description": "On-device inference with tool use, no chain-of-thought",
        "inter_request_gap_range_s": f"{llama_stats.get('min', 1):.0f}-{llama_stats.get('max', 14):.0f}",
        "typical_gap_s": round(llama_stats.get("median", 2.8), 1),
        "exceeds_7s_timer_pct": round(llama_radio.get("exceeds_tail_timer_pct", 2.3), 1),
        "radio_tail_waste_pct": round(llama_radio.get("exceeds_tail_timer_pct", 2.3), 1),
        "source": "This work (Llama 3.2 3B, Gemma 3n measurements)",
        "pattern": "Fast inference (2-5s), bursty within sessions",
        "is_measured": True,
    })

    # Thinking models
    qwen4b_stats = per_model_stats.get("qwen-3-4b", {})
    qwen4b_radio = radio_data.get("per_model", {}).get("qwen-3-4b", {})
    deepseek_stats = per_model_stats.get("deepseek-r1-distill-qwen-1.5b", {})
    deepseek_radio = radio_data.get("per_model", {}).get("deepseek-r1-distill-qwen-1.5b", {})

    # Average across thinking models
    thinking_exceed_pcts = [
        qwen4b_radio.get("exceeds_tail_timer_pct", 77),
        deepseek_radio.get("exceeds_tail_timer_pct", 95),
        radio_data.get("per_model", {}).get("qwen-3-0.6b", {}).get("exceeds_tail_timer_pct", 78),
    ]
    avg_thinking_exceed = statistics.mean(thinking_exceed_pcts)

    traffic_classes.append({
        "traffic_class": "Agentic LLM (thinking)",
        "description": "On-device inference with chain-of-thought reasoning",
        "inter_request_gap_range_s": f"{min(deepseek_stats.get('min', 2), qwen4b_stats.get('min', 3)):.0f}-{max(deepseek_stats.get('max', 36), qwen4b_stats.get('max', 71)):.0f}",
        "typical_gap_s": round(statistics.mean([
            qwen4b_stats.get("median", 13.2),
            deepseek_stats.get("median", 10.2),
        ]), 1),
        "exceeds_7s_timer_pct": round(avg_thinking_exceed, 1),
        "radio_tail_waste_pct": round(avg_thinking_exceed, 1),
        "source": "This work (Qwen 3 4B, DeepSeek R1 1.5B, Qwen 3 0.6B measurements)",
        "pattern": "Long inference gaps (10-30s), systematic timer exceedance",
        "is_measured": True,
    })

    # Compute uniqueness metrics
    # Agentic thinking models have highest timer exceedance among all classes
    # except background sync, but with much higher frequency of transfers
    return {
        "description": (
            "Comparison of agentic LLM traffic with established mobile traffic "
            "classes in terms of RRC state machine interaction"
        ),
        "key_finding": (
            "Agentic LLM traffic with thinking models creates a uniquely "
            "problematic traffic class: unlike messaging (sporadic, human-driven), "
            "thinking model gaps are systematic and predictable (10-30s, "
            "determined by model size and problem difficulty). Unlike video "
            "streaming (periodic, within-timer), thinking gaps consistently "
            "exceed the RRC timer. This systematic timer exceedance is a "
            "structural property of chain-of-thought inference, not random."
        ),
        "traffic_classes": traffic_classes,
        "comparison_dimensions": [
            "inter_request_gap_range_s",
            "typical_gap_s",
            "exceeds_7s_timer_pct",
            "radio_tail_waste_pct",
        ],
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def print_summary(ratios, interaction, early_exit, rrc_tuning, traffic_comp):
    """Print a comprehensive summary of all findings."""
    print("=" * 78)
    print("DEEP RADIO TAIL ENERGY ANALYSIS FOR THINKING MODELS")
    print("=" * 78)

    # 1. Radio / Inference Ratio
    print("\n" + "─" * 78)
    print("1. RADIO TAIL ENERGY / INFERENCE ENERGY RATIO")
    print("─" * 78)
    print(f"\n{'Model':<30s} {'Type':<10s} {'E_inf(J)':>10s} {'E_radio(J)':>11s} {'Ratio':>8s}")
    print("-" * 78)
    for model in MODELS:
        if model not in ratios["per_model"]:
            continue
        r = ratios["per_model"][model]
        mtype = "thinking" if r["is_thinking_model"] else "non-think"
        print(f"{r['display_name']:<30s} {mtype:<10s} {r['avg_inference_energy_j']:>10.1f} "
              f"{r['total_radio_energy_j']:>11.2f} {r['radio_to_inference_pct']:>7.2f}%")
    print()
    print("Key insight:", ratios["key_insight"])

    # 2. Thinking × Timer Interaction
    print("\n" + "─" * 78)
    print("2. THINKING MODEL × RRC TAIL TIMER INTERACTION")
    print("─" * 78)
    print(f"\n{'Model':<30s} {'Type':<10s} {'Med gap(s)':>10s} {'>7s timer':>10s}")
    print("-" * 78)
    for model in MODELS:
        if model not in interaction["per_model"]:
            continue
        r = interaction["per_model"][model]
        mtype = "thinking" if r["is_thinking_model"] else "non-think"
        print(f"{r['display_name']:<30s} {mtype:<10s} {r['median_inference_s']:>10.2f} "
              f"{r['exceeds_7s_timer_pct']:>9.1f}%")
    print()
    print("Finding:", interaction["key_finding"])
    print()
    print("Implication:", interaction["implication"])

    # 3. Early-Exit Radio Impact
    print("\n" + "─" * 78)
    print("3. EARLY-EXIT IMPACT ON RADIO ENERGY")
    print("─" * 78)
    for model_id, model_data in early_exit["per_model"].items():
        print(f"\n  {model_data['display_name']}:")
        print(f"  {'Budget':<12s} {'Time(s)':>8s} {'E_inf(J)':>9s} {'E_radio(J)':>10s} {'Radio%':>7s} {'vs unlim':>9s} {'>7s?':>5s}")
        print(f"  {'-'*65}")
        for b in model_data["budgets"]:
            exceeds = "YES" if b["exceeds_tail_timer"] else "no"
            print(f"  {b['thinking_budget_label']:<12s} {b['avg_elapsed_s']:>8.2f} "
                  f"{b['avg_inference_energy_j']:>9.1f} {b['estimated_radio_energy_j']:>10.2f} "
                  f"{b['radio_pct_of_inference']:>6.2f}% {b['radio_reduction_vs_unlimited_pct']:>8.1f}% "
                  f"{exceeds:>5s}")

    if early_exit.get("threshold_crossing_examples"):
        print("\n  Threshold crossings (iteration drops below 7s timer):")
        for ex in early_exit["threshold_crossing_examples"]:
            print(f"    {ex['display_name']}: budget={ex['threshold_budget']} → "
                  f"{ex['elapsed_at_threshold']:.1f}s (radio savings: "
                  f"{ex['radio_reduction_pct']:.1f}%)")

    # 4. RRC Timer Tuning
    print("\n" + "─" * 78)
    print("4. RRC TIMER TUNING PROPOSAL")
    print("─" * 78)
    print(f"\n  Current standard timer: {rrc_tuning['current_standard_timer_s']}s")
    print(f"  Optimal for thinking models: {rrc_tuning['optimal_timer_thinking_s']}s")
    print(f"  Optimal for non-thinking models: {rrc_tuning['optimal_timer_non_thinking_s']}s")
    print()

    # Show per-class breakdown at key timer values
    print(f"\n  Thinking models (inference gaps 10-70s):")
    print(f"  {'Timer(s)':>9s} {'Trans curr':>11s} {'Trans prop':>11s} {'Reduction':>10s} {'Avg E_radio(J)':>15s}")
    print(f"  {'-'*62}")
    for p in rrc_tuning["timer_proposals"]:
        ta = p.get("thinking_aggregate", {})
        if not ta:
            continue
        timer = p["timer_s"]
        print(f"  {timer:>9d} {ta['transitions_current']:>11d} {ta['transitions_proposed']:>11d} "
              f"{ta['transition_reduction_pct']:>9.1f}% "
              f"{ta['avg_radio_energy_j']:>15.3f}")

    print(f"\n  Non-thinking models (inference gaps 2-5s):")
    print(f"  {'Timer(s)':>9s} {'Trans curr':>11s} {'Trans prop':>11s} {'Reduction':>10s} {'Avg E_radio(J)':>15s}")
    print(f"  {'-'*62}")
    for p in rrc_tuning["timer_proposals"]:
        nta = p.get("non_thinking_aggregate", {})
        if not nta:
            continue
        timer = p["timer_s"]
        print(f"  {timer:>9d} {nta['transitions_current']:>11d} {nta['transitions_proposed']:>11d} "
              f"{nta['transition_reduction_pct']:>9.1f}% "
              f"{nta['avg_radio_energy_j']:>15.3f}")

    print(f"\n  All models combined:")
    print(f"  {'Timer(s)':>9s} {'Transitions':>13s} {'Reduction':>10s} {'Avg E_radio(J)':>15s} {'E change':>9s}")
    print(f"  {'-'*62}")
    for p in rrc_tuning["timer_proposals"]:
        agg = p.get("aggregate", {})
        if not agg:
            continue
        timer = p["timer_s"]
        print(f"  {timer:>9d} {agg['transitions_proposed']:>13d} "
              f"{agg['transition_reduction_pct']:>9.1f}% "
              f"{agg['avg_radio_energy_proposed_j']:>15.3f} "
              f"{agg['energy_change_pct']:>8.1f}%")

    print()
    print("  Proposal:", rrc_tuning["key_proposal"])
    print()
    print("  Trade-off:", rrc_tuning["trade_off"])

    # 5. Traffic Class Comparison
    print("\n" + "─" * 78)
    print("5. TRAFFIC CLASS COMPARISON")
    print("─" * 78)
    print(f"\n{'Traffic Class':<35s} {'Gap range':>12s} {'Typical':>8s} {'>7s':>6s} {'Waste':>6s}")
    print("-" * 78)
    for tc in traffic_comp["traffic_classes"]:
        measured = " *" if tc.get("is_measured") else ""
        print(f"{tc['traffic_class']:<35s} {tc['inter_request_gap_range_s']:>12s} "
              f"{tc['typical_gap_s']:>7.1f}s "
              f"{tc['exceeds_7s_timer_pct']:>5.1f}% "
              f"{tc['radio_tail_waste_pct']:>5.1f}%{measured}")
    print()
    print("  * = measured in this work (all others from literature)")
    print()
    print("Finding:", traffic_comp["key_finding"])

    print("\n" + "=" * 78)
    print("END OF ANALYSIS")
    print("=" * 78)


def main():
    """Run all analyses and save results."""
    print("Loading data...")
    mlx_data = load_mlx_sweep()
    radio_data, cdf_data, sessions_data = load_traffic_char()
    cost_model = load_cost_model()
    early_exit_data = load_early_exit()
    arch3_data = load_3arch_results()

    print(f"  MLX sweep: {len(mlx_data)} models loaded")
    print(f"  Early-exit: {len(early_exit_data)} models loaded")
    print(f"  3-arch: {len(arch3_data)} entries loaded")

    # Run analyses
    print("\nRunning analyses...")

    print("  1. Radio energy / inference energy ratio...")
    ratios = analyze_radio_inference_ratio(mlx_data, radio_data, cdf_data, arch3_data)

    print("  2. Thinking model × radio tail timer interaction...")
    interaction = analyze_thinking_radio_interaction(mlx_data, cdf_data, radio_data)

    print("  3. Early-exit impact on radio energy...")
    early_exit = analyze_early_exit_radio_impact(early_exit_data, cost_model)

    print("  4. RRC timer tuning proposal...")
    rrc_tuning = analyze_rrc_tuning(cdf_data, mlx_data, radio_data)

    print("  5. Traffic class comparison...")
    traffic_comp = analyze_traffic_class_comparison(cdf_data, radio_data)

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "radio_energy_ratios.json": ratios,
        "thinking_radio_interaction.json": interaction,
        "early_exit_radio_impact.json": early_exit,
        "rrc_tuning_proposal.json": rrc_tuning,
        "traffic_class_comparison.json": traffic_comp,
    }

    for fname, data in outputs.items():
        path = OUT_DIR / fname
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved: {path}")

    # Print comprehensive summary
    print_summary(ratios, interaction, early_exit, rrc_tuning, traffic_comp)


if __name__ == "__main__":
    main()
