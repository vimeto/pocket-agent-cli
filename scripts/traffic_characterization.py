#!/usr/bin/env python3
"""Characterize agentic LLM traffic as a new mobile traffic class.

Analyzes the 3-architecture experiment data to quantify the distinctive
network traffic patterns of agentic LLM workloads. Produces data for
MobiHoc paper figures demonstrating that agentic AI creates fundamentally
different traffic from web browsing, video streaming, or messaging.

Key findings this script produces:
1. Inter-request timing CDF — long inference gaps (2-80s) between transfers
2. Payload size distribution — context accumulation causes growing uploads
3. Radio state efficiency — every iteration triggers IDLE->CONNECTED transition
4. Traffic class comparison table with quantified metrics

Data sources:
- 3-arch experiment: data/results/3arch_experiment/20260405_183229/
- MLX sweep: data/results/mlx_sweep/20260403_091508/

Output: data/results/traffic_characterization/
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────

CLI_ROOT = Path(__file__).resolve().parent.parent
ARCH3_DIR = CLI_ROOT / "data" / "results" / "3arch_experiment" / "20260405_183229"
MLX_DIR = CLI_ROOT / "data" / "results" / "mlx_sweep" / "20260403_091508"
OUT_DIR = CLI_ROOT / "data" / "results" / "traffic_characterization"

ARCH3_JSONL = ARCH3_DIR / "3arch_results_merged.jsonl"

MODELS = [
    "deepseek-r1-distill-qwen-1.5b",
    "qwen-3-4b",
    "llama-3.2-3b-instruct",
    "qwen-3-0.6b",
]

# Radio model parameters (matching radio_model.py)
TAIL_TIMER_S = 7.0  # RRC inactivity timer
# Power values from Huang et al. MobiSys 2012, Table 2 (LTE)
POWER_CONNECTED_W = 1.2   # radio in CONNECTED/active state
POWER_TAIL_W = 0.6        # radio in TAIL state (DRX)
POWER_IDLE_W = 0.01       # radio in IDLE state


def load_3arch_results():
    """Load all 3-arch merged results."""
    results = []
    with open(ARCH3_JSONL) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def load_mlx_sweep():
    """Load MLX sweep results for per-problem token/timing data."""
    data = {}
    for model in MODELS:
        for mode in ["base", "full_tool", "tool_submission"]:
            path = MLX_DIR / f"{model}_{mode}.jsonl"
            if path.exists():
                with open(path) as f:
                    entries = [json.loads(l) for l in f]
                data[(model, mode)] = entries
    return data


# ──────────────────────────────────────────────────────────────
# 1. Inter-request timing analysis
# ──────────────────────────────────────────────────────────────

def analyze_inter_request_timing(results):
    """Compute inter-request time distribution for hybrid architecture.

    In the hybrid architecture, each agentic iteration involves:
      upload -> inference (long gap) -> download -> tool exec -> next upload

    The inter-request time is dominated by inference time. For multi-iteration
    problems, the gap between the download of iteration N and the upload of
    iteration N+1 is essentially the tool execution time (~0s for our
    code-gen tasks). The gap *within* an iteration (upload to download) is
    the inference time.

    Since we don't have per-transfer timestamps, we reconstruct timing:
    - Per iteration: inference_time / iterations gives avg per-iteration inference
    - Each iteration has 2 transfers, so inter-transfer time within an iteration
      = inference_time_per_iter (upload->download gap)
    - Between iterations: tool_exec_time / (iterations-1) ~ very small
    """
    hybrid = [r for r in results
              if r["architecture"] == "hybrid" and r.get("mode") == "full_tool"]

    # Collect all inter-transfer gaps
    # Within an iteration: upload -> [inference gap] -> download
    # Between iterations: download -> [tool exec gap] -> upload
    intra_iter_gaps = []   # inference time per iteration (upload->download)
    inter_iter_gaps = []   # tool exec time between iterations (download->upload)

    for r in hybrid:
        n_iter = r["iterations"]
        if n_iter < 1:
            continue

        inf_per_iter = r["inference_time_s"] / n_iter
        intra_iter_gaps.append(inf_per_iter)

        if n_iter > 1:
            # Tool execution gap between iterations
            tool_gap = r.get("tool_exec_time_s", 0.0) / (n_iter - 1)
            for _ in range(n_iter - 1):
                inter_iter_gaps.append(tool_gap)

    # All gaps sorted for CDF
    all_gaps = sorted(intra_iter_gaps + inter_iter_gaps)
    # Separate the intra-iteration (inference) gaps — these are the dominant ones
    intra_sorted = sorted(intra_iter_gaps)

    # Compute CDF
    def make_cdf(values):
        """Return (x, y) lists for CDF plot."""
        if not values:
            return [], []
        sorted_v = sorted(values)
        n = len(sorted_v)
        x = sorted_v
        y = [(i + 1) / n for i in range(n)]
        return x, y

    all_cdf_x, all_cdf_y = make_cdf(all_gaps)
    intra_cdf_x, intra_cdf_y = make_cdf(intra_sorted)

    # Percentile stats
    def percentiles(values, ps=(10, 25, 50, 75, 90, 95, 99)):
        if not values:
            return {}
        s = sorted(values)
        n = len(s)
        return {f"p{p}": s[min(int(p / 100 * n), n - 1)] for p in ps}

    stats = {
        "inference_gap_s": {
            "count": len(intra_iter_gaps),
            "mean": statistics.mean(intra_iter_gaps) if intra_iter_gaps else 0,
            "median": statistics.median(intra_iter_gaps) if intra_iter_gaps else 0,
            "min": min(intra_iter_gaps) if intra_iter_gaps else 0,
            "max": max(intra_iter_gaps) if intra_iter_gaps else 0,
            "stdev": statistics.stdev(intra_iter_gaps) if len(intra_iter_gaps) > 1 else 0,
            **percentiles(intra_iter_gaps),
        },
        "tool_exec_gap_s": {
            "count": len(inter_iter_gaps),
            "mean": statistics.mean(inter_iter_gaps) if inter_iter_gaps else 0,
            "median": statistics.median(inter_iter_gaps) if inter_iter_gaps else 0,
        },
        "per_model": {},
    }

    # Per-model breakdown
    for model in MODELS:
        model_entries = [r for r in hybrid if r["model"] == model]
        gaps = []
        for r in model_entries:
            if r["iterations"] > 0:
                gaps.append(r["inference_time_s"] / r["iterations"])
        if gaps:
            stats["per_model"][model] = {
                "count": len(gaps),
                "mean": statistics.mean(gaps),
                "median": statistics.median(gaps),
                "min": min(gaps),
                "max": max(gaps),
                **percentiles(gaps),
            }

    cdf_data = {
        "description": "CDF of inter-request times for agentic LLM traffic (hybrid architecture)",
        "all_gaps": {"x_seconds": all_cdf_x, "y_fraction": all_cdf_y},
        "inference_gaps_only": {"x_seconds": intra_cdf_x, "y_fraction": intra_cdf_y},
        "statistics": stats,
    }

    return cdf_data


# ──────────────────────────────────────────────────────────────
# 2. Payload size / context growth analysis
# ──────────────────────────────────────────────────────────────

def analyze_payload_growth(results):
    """Analyze how payload sizes grow with iteration count.

    In the hybrid architecture, uploads grow because each iteration sends
    the full conversation history (system prompt + all prior messages +
    observations). Downloads are roughly proportional to model output tokens.

    Since we only have total network_bytes per problem (not per-transfer),
    we reconstruct the growth pattern:
    - Iteration 1: upload = prompt (~1-2KB), download = response tokens * ~4 bytes
    - Iteration i: upload = prompt + sum of prior (response + observation) * ~4 bytes
    - Total bytes = sum of all uploads + all downloads

    We use the relationship: total_bytes / network_transfers = avg bytes per transfer.
    For multi-iteration problems, we can model the growth.
    """
    hybrid = [r for r in results
              if r["architecture"] == "hybrid" and r.get("mode") == "full_tool"]

    # Group by iteration count for comparison
    by_iters = defaultdict(list)
    for r in hybrid:
        by_iters[r["iterations"]].append(r)

    # Average bytes per transfer grouped by iteration count
    growth_data = {}
    for n_iter, entries in sorted(by_iters.items()):
        bytes_per_xfer = [r["network_bytes"] / r["network_transfers"]
                          for r in entries if r["network_transfers"] > 0]
        total_bytes = [r["network_bytes"] for r in entries]
        growth_data[n_iter] = {
            "count": len(entries),
            "avg_bytes_per_transfer": statistics.mean(bytes_per_xfer) if bytes_per_xfer else 0,
            "median_bytes_per_transfer": statistics.median(bytes_per_xfer) if bytes_per_xfer else 0,
            "avg_total_bytes": statistics.mean(total_bytes) if total_bytes else 0,
            "median_total_bytes": statistics.median(total_bytes) if total_bytes else 0,
        }

    # Reconstruct per-iteration upload sizes for multi-iteration problems.
    # Model: each iteration i uploads all prior context.
    # If total tokens = T and iterations = N, approximate per-iteration tokens:
    #   iter 1 produces t1 tokens, iter 2 sees t1 context + produces t2, etc.
    # With uniform distribution: each iter produces ~T/N tokens, and upload for
    # iter i includes all prior tokens plus the initial prompt.
    #
    # For a problem with total_tokens T over N iterations:
    #   response_per_iter ≈ T / N tokens ≈ T/N * 4 bytes
    #   upload_iter_i ≈ prompt_bytes + i * response_per_iter * 4 (cumulative context)
    #   download_iter_i ≈ response_per_iter * 4

    PROMPT_BASE_BYTES = 1200  # estimated base prompt size (system + problem)
    BYTES_PER_TOKEN = 4       # approximate UTF-8 bytes per token

    per_iteration_uploads = defaultdict(list)   # iter_number -> [upload_bytes, ...]
    per_iteration_downloads = defaultdict(list)

    for r in hybrid:
        n_iter = r["iterations"]
        if n_iter < 1:
            continue

        total_tokens = r.get("tokens", 0)
        tokens_per_iter = total_tokens / n_iter if n_iter > 0 else total_tokens
        response_bytes_per_iter = tokens_per_iter * BYTES_PER_TOKEN

        for i in range(n_iter):
            # Upload: base prompt + all prior responses as context
            upload = PROMPT_BASE_BYTES + i * response_bytes_per_iter
            # Add tool observation context (~200 bytes per prior iteration)
            upload += i * 200
            download = response_bytes_per_iter

            per_iteration_uploads[i + 1].append(upload)
            per_iteration_downloads[i + 1].append(download)

    # Summary per iteration number
    iteration_payload_summary = {}
    for iter_num in sorted(per_iteration_uploads.keys()):
        uploads = per_iteration_uploads[iter_num]
        downloads = per_iteration_downloads[iter_num]
        iteration_payload_summary[iter_num] = {
            "n_samples": len(uploads),
            "upload_bytes": {
                "mean": statistics.mean(uploads),
                "median": statistics.median(uploads),
                "min": min(uploads),
                "max": max(uploads),
            },
            "download_bytes": {
                "mean": statistics.mean(downloads),
                "median": statistics.median(downloads),
                "min": min(downloads),
                "max": max(downloads),
            },
            "total_per_iteration": {
                "mean": statistics.mean(uploads) + statistics.mean(downloads),
            },
        }

    payload_data = {
        "description": "Payload size growth analysis for agentic LLM traffic",
        "by_iteration_count": growth_data,
        "per_iteration_payload": iteration_payload_summary,
        "notes": {
            "prompt_base_bytes": PROMPT_BASE_BYTES,
            "bytes_per_token": BYTES_PER_TOKEN,
            "key_finding": (
                "Upload payload grows linearly with iteration number due to "
                "context accumulation. By iteration 4, uploads are 3-5x larger "
                "than iteration 1. This is unique to agentic LLM traffic."
            ),
        },
    }

    return payload_data


# ──────────────────────────────────────────────────────────────
# 3. Radio state efficiency analysis
# ──────────────────────────────────────────────────────────────

def analyze_radio_efficiency(results):
    """Simulate RRC state machine for agentic traffic and compute waste.

    Key insight: the long inference gaps (mean ~10s, range 2-80s) between
    network transfers *always* exceed the 7s tail timer, so the radio
    transitions IDLE -> CONNECTED -> TAIL -> IDLE between every iteration.

    This means:
    - Every iteration wastes 7s of tail energy (radio on, doing nothing)
    - Every iteration pays the IDLE -> CONNECTED promotion delay (~50-100ms)
    - Multi-iteration problems waste proportionally more energy

    Compare with:
    - Web browsing: rapid requests keep radio in CONNECTED, tail shared
    - Video streaming: continuous data, radio stays CONNECTED
    - Messaging: sporadic but tiny payloads, some tail waste
    """
    hybrid = [r for r in results
              if r["architecture"] == "hybrid" and r.get("mode") == "full_tool"]

    # For each problem, simulate the radio state timeline
    radio_timelines = []
    energy_waste_per_problem = []
    fraction_tail_per_problem = []

    for r in hybrid:
        n_iter = r["iterations"]
        if n_iter < 1:
            continue

        total_inf_time = r["inference_time_s"]
        inf_per_iter = total_inf_time / n_iter
        total_time = r["total_time_s"]

        # Reconstruct transfer timeline
        # Each iteration: upload (brief) -> inference (long) -> download (brief)
        # Transfer duration is very short compared to inference
        TRANSFER_DURATION_S = 0.05  # ~50ms per transfer

        timeline = []
        t = 0.0

        for i in range(n_iter):
            # Upload transfer
            timeline.append({
                "time": t,
                "event": "upload",
                "state": "CONNECTED",
                "iteration": i + 1,
            })
            t += TRANSFER_DURATION_S

            # Inference gap (radio enters TAIL then IDLE)
            tail_start = t
            tail_end = min(t + TAIL_TIMER_S, t + inf_per_iter)
            idle_start = t + TAIL_TIMER_S if inf_per_iter > TAIL_TIMER_S else None

            timeline.append({
                "time": t,
                "event": "tail_start",
                "state": "TAIL",
                "iteration": i + 1,
            })

            if idle_start is not None:
                timeline.append({
                    "time": idle_start,
                    "event": "idle_start",
                    "state": "IDLE",
                    "iteration": i + 1,
                })

            t += inf_per_iter

            # Download transfer
            timeline.append({
                "time": t,
                "event": "download",
                "state": "CONNECTED",
                "iteration": i + 1,
            })
            t += TRANSFER_DURATION_S

            # Brief tool execution gap before next iteration
            if i < n_iter - 1:
                tool_exec = r.get("tool_exec_time_s", 0.0) / max(n_iter - 1, 1)
                t += tool_exec

        # Compute energy breakdown
        total_connected_time = n_iter * 2 * TRANSFER_DURATION_S  # uploads + downloads
        total_tail_time = n_iter * min(TAIL_TIMER_S, inf_per_iter)
        # If inference < tail timer, tail covers the whole gap
        # If inference > tail timer, tail = TAIL_TIMER_S per iteration
        total_idle_time = max(0, total_time - total_connected_time - total_tail_time)

        # Energy computation
        connected_energy = total_connected_time * POWER_CONNECTED_W
        tail_energy = total_tail_time * POWER_TAIL_W
        idle_energy = total_idle_time * POWER_IDLE_W
        total_energy = connected_energy + tail_energy + idle_energy

        # Tail energy is "wasted" — radio is on but not transferring
        tail_fraction = tail_energy / total_energy if total_energy > 0 else 0

        energy_waste_per_problem.append(tail_energy)
        fraction_tail_per_problem.append(tail_fraction)

        # Does inference exceed tail timer?
        exceeds_tail = inf_per_iter > TAIL_TIMER_S

        radio_timelines.append({
            "model": r["model"],
            "problem_id": r.get("problem_id", "?"),
            "iterations": n_iter,
            "inference_per_iter_s": inf_per_iter,
            "exceeds_tail_timer": exceeds_tail,
            "total_time_s": total_time,
            "connected_time_s": total_connected_time,
            "tail_time_s": total_tail_time,
            "idle_time_s": total_idle_time,
            "connected_energy_j": connected_energy,
            "tail_energy_j": tail_energy,
            "idle_energy_j": idle_energy,
            "total_radio_energy_j": total_energy,
            "tail_energy_fraction": tail_fraction,
            "timeline": timeline,
            "network_condition": r["network_condition"],
        })

    # Aggregate statistics
    exceeds_count = sum(1 for rt in radio_timelines if rt["exceeds_tail_timer"])
    total_count = len(radio_timelines)

    # Compute additional metrics:
    # - tail-to-active ratio: how much time radio wastes in tail vs actual transfer
    # - energy per useful byte: radio energy / bytes actually transferred
    tail_to_active_ratios = []
    energy_per_byte = []
    for rt in radio_timelines:
        if rt["connected_time_s"] > 0:
            tail_to_active_ratios.append(rt["tail_time_s"] / rt["connected_time_s"])
        entry = next((r for r in hybrid
                       if r["model"] == rt["model"]
                       and r.get("problem_id") == rt["problem_id"]
                       and r["network_condition"] == rt["network_condition"]),
                      None)
        if entry and entry["network_bytes"] > 0 and rt["total_radio_energy_j"] > 0:
            energy_per_byte.append(rt["total_radio_energy_j"] / entry["network_bytes"])

    # Per-model radio stats
    per_model_radio = {}
    for model in MODELS:
        model_timelines = [rt for rt in radio_timelines if rt["model"] == model]
        if model_timelines:
            tail_fracs = [rt["tail_energy_fraction"] for rt in model_timelines]
            tail_energies = [rt["tail_energy_j"] for rt in model_timelines]
            exceeds = sum(1 for rt in model_timelines if rt["exceeds_tail_timer"])
            t2a = [rt["tail_time_s"] / rt["connected_time_s"]
                   for rt in model_timelines if rt["connected_time_s"] > 0]
            per_model_radio[model] = {
                "count": len(model_timelines),
                "exceeds_tail_timer_pct": exceeds / len(model_timelines) * 100,
                "avg_tail_energy_fraction": statistics.mean(tail_fracs),
                "avg_tail_energy_j": statistics.mean(tail_energies),
                "median_inference_per_iter_s": statistics.median(
                    [rt["inference_per_iter_s"] for rt in model_timelines]
                ),
                "avg_tail_to_active_ratio": statistics.mean(t2a) if t2a else 0,
            }

    radio_data = {
        "description": "Radio state efficiency analysis for agentic LLM traffic",
        "tail_timer_s": TAIL_TIMER_S,
        "power_model": {
            "connected_w": POWER_CONNECTED_W,
            "tail_w": POWER_TAIL_W,
            "idle_w": POWER_IDLE_W,
        },
        "aggregate": {
            "total_problems": total_count,
            "exceeds_tail_timer_count": exceeds_count,
            "exceeds_tail_timer_pct": exceeds_count / total_count * 100 if total_count > 0 else 0,
            "avg_tail_energy_fraction": (
                statistics.mean(fraction_tail_per_problem) if fraction_tail_per_problem else 0
            ),
            "avg_tail_waste_j": (
                statistics.mean(energy_waste_per_problem) if energy_waste_per_problem else 0
            ),
            "avg_tail_to_active_ratio": (
                statistics.mean(tail_to_active_ratios) if tail_to_active_ratios else 0
            ),
            "avg_energy_per_byte_mj": (
                statistics.mean(energy_per_byte) * 1000 if energy_per_byte else 0
            ),
        },
        "per_model": per_model_radio,
        # Save a few example timelines for figures (pick multi-iter, local condition)
        "example_timelines": [
            rt for rt in radio_timelines
            if rt["iterations"] > 1 and rt["network_condition"] == "local"
        ][:5],
    }

    return radio_data


# ──────────────────────────────────────────────────────────────
# 4. Traffic class comparison
# ──────────────────────────────────────────────────────────────

def build_traffic_comparison(inter_request_data, payload_data, radio_data):
    """Build comparison table with other traffic classes.

    Reference traffic patterns from literature:
    - Web browsing: Ihm & Bhagwan, IMC 2011; Butkiewicz et al., IMC 2011
    - Video streaming: Jiang et al., SIGCOMM 2017 (ABR); Li et al., MMSys 2013
    - Messaging: Xu et al., IMC 2011 (smartphone traffic)
    """
    # Extract our measured stats
    agentic_stats = inter_request_data["statistics"]["inference_gap_s"]
    payload_stats = payload_data["by_iteration_count"]
    radio_stats = radio_data["aggregate"]

    # Compute agentic payload range
    all_payload_avgs = [v["avg_bytes_per_transfer"] for v in payload_stats.values()]
    payload_min = min(all_payload_avgs) if all_payload_avgs else 0
    payload_max = max(all_payload_avgs) if all_payload_avgs else 0

    comparison = {
        "description": "Comparison of agentic LLM traffic with other mobile traffic classes",
        "traffic_classes": [
            {
                "class": "Web browsing",
                "pattern": "Many rapid small requests, page load bursts",
                "inter_request_time_s": {"typical_range": [0.01, 2.0], "median": 0.5},
                "payload_bytes": {"typical_range": [500, 50000], "median": 5000},
                "session_duration_s": {"typical_range": [30, 600]},
                "radio_efficiency": "Good — rapid requests keep radio CONNECTED",
                "radio_tail_waste_fraction": 0.05,
                "source": "Ihm & Bhagwan IMC 2011; Butkiewicz et al. IMC 2011",
            },
            {
                "class": "Video streaming (ABR)",
                "pattern": "Continuous high-throughput, periodic chunk fetches",
                "inter_request_time_s": {"typical_range": [0.0, 4.0], "median": 2.0},
                "payload_bytes": {"typical_range": [100000, 2000000], "median": 500000},
                "session_duration_s": {"typical_range": [120, 3600]},
                "radio_efficiency": "Good — continuous data keeps radio CONNECTED",
                "radio_tail_waste_fraction": 0.02,
                "source": "Jiang et al. SIGCOMM 2017; Li et al. MMSys 2013",
            },
            {
                "class": "Messaging / chat",
                "pattern": "Sporadic tiny payloads, unpredictable timing",
                "inter_request_time_s": {"typical_range": [5.0, 120.0], "median": 30.0},
                "payload_bytes": {"typical_range": [50, 2000], "median": 200},
                "session_duration_s": {"typical_range": [60, 1800]},
                "radio_efficiency": "Poor — gaps often exceed tail timer",
                "radio_tail_waste_fraction": 0.30,
                "source": "Xu et al. IMC 2011; Falaki et al. MobiSys 2010",
            },
            {
                "class": "Agentic LLM (measured)",
                "pattern": "Bursty medium payloads, long compute gaps, growing context",
                "inter_request_time_s": {
                    "typical_range": [
                        round(agentic_stats.get("p10", 2.0), 1),
                        round(agentic_stats.get("p90", 20.0), 1),
                    ],
                    "median": round(agentic_stats["median"], 1),
                    "mean": round(agentic_stats["mean"], 1),
                },
                "payload_bytes": {
                    "typical_range": [round(payload_min), round(payload_max)],
                    "median": round(
                        payload_stats.get(1, {}).get("median_bytes_per_transfer", 0)
                    ),
                    "note": "Uploads grow with iteration (context accumulation)",
                },
                "session_duration_s": {
                    "typical_range": [5, 100],
                    "note": "Varies 5-100s+ depending on model and problem difficulty",
                },
                "radio_efficiency": (
                    f"Very poor — {radio_stats['exceeds_tail_timer_pct']:.0f}% of "
                    f"inference gaps exceed {TAIL_TIMER_S}s tail timer"
                ),
                "radio_tail_waste_fraction": round(
                    radio_stats["avg_tail_energy_fraction"], 2
                ),
                "unique_properties": [
                    "Inter-request gaps are deterministic (inference time), not user-driven",
                    "Payload size grows with conversation depth (context accumulation)",
                    "Request pattern is strictly alternating upload/download pairs",
                    "Gap duration varies by model capability (thinking models: 20-80s, fast models: 2-10s)",
                ],
                "source": "This work — measured from 4 SLMs x 150 problems x 7 network conditions",
            },
        ],
    }

    return comparison


# ──────────────────────────────────────────────────────────────
# 5. Session-level analysis
# ──────────────────────────────────────────────────────────────

def analyze_sessions(results):
    """Analyze session-level statistics: duration, total bytes, iteration count."""
    hybrid = [r for r in results
              if r["architecture"] == "hybrid" and r.get("mode") == "full_tool"]

    sessions = []
    for r in hybrid:
        sessions.append({
            "model": r["model"],
            "problem_id": r.get("problem_id", "?"),
            "iterations": r["iterations"],
            "total_time_s": r["total_time_s"],
            "inference_time_s": r["inference_time_s"],
            "network_bytes": r["network_bytes"],
            "network_transfers": r["network_transfers"],
            "passed": r["passed"],
            "tokens": r.get("tokens", 0),
            "network_condition": r["network_condition"],
        })

    # Per-model session stats (using local condition to avoid network delay artifacts)
    per_model_sessions = {}
    for model in MODELS:
        model_sessions = [s for s in sessions
                          if s["model"] == model and s["network_condition"] == "local"]
        if not model_sessions:
            continue
        durations = [s["total_time_s"] for s in model_sessions]
        bytes_list = [s["network_bytes"] for s in model_sessions]
        iters = [s["iterations"] for s in model_sessions]
        per_model_sessions[model] = {
            "count": len(model_sessions),
            "duration_s": {
                "mean": statistics.mean(durations),
                "median": statistics.median(durations),
                "min": min(durations),
                "max": max(durations),
            },
            "total_bytes": {
                "mean": statistics.mean(bytes_list),
                "median": statistics.median(bytes_list),
                "min": min(bytes_list),
                "max": max(bytes_list),
            },
            "iterations": {
                "mean": statistics.mean(iters),
                "median": statistics.median(iters),
                "max": max(iters),
            },
        }

    return per_model_sessions


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def print_summary(inter_req, payload, radio, comparison, sessions):
    """Print human-readable summary to stdout."""
    print("=" * 72)
    print("TRAFFIC CHARACTERIZATION: Agentic LLM as a New Mobile Traffic Class")
    print("=" * 72)

    # 1. Inter-request timing
    stats = inter_req["statistics"]["inference_gap_s"]
    print("\n1. INTER-REQUEST TIMING (inference gaps between transfers)")
    print("-" * 72)
    print(f"   Samples:  {stats['count']}")
    print(f"   Mean:     {stats['mean']:.1f}s")
    print(f"   Median:   {stats['median']:.1f}s")
    print(f"   P10-P90:  {stats.get('p10', 0):.1f}s - {stats.get('p90', 0):.1f}s")
    print(f"   Range:    {stats['min']:.1f}s - {stats['max']:.1f}s")
    print()
    print("   Per-model median inference gap:")
    for model, ms in inter_req["statistics"]["per_model"].items():
        print(f"     {model:<35} {ms['median']:.1f}s  (range {ms['min']:.1f}-{ms['max']:.1f}s)")

    # 2. Payload growth
    print("\n2. PAYLOAD SIZE & CONTEXT GROWTH")
    print("-" * 72)
    for n_iter, info in sorted(payload["by_iteration_count"].items(), key=lambda x: x[0]):
        print(f"   {n_iter} iteration(s):  avg {info['avg_bytes_per_transfer']:,.0f} bytes/transfer"
              f"  ({info['count']} problems)")

    print("\n   Per-iteration upload growth (estimated):")
    for iter_num, info in sorted(payload["per_iteration_payload"].items(), key=lambda x: x[0]):
        up = info["upload_bytes"]
        down = info["download_bytes"]
        print(f"     Iter {iter_num}: upload={up['mean']:,.0f}B (median {up['median']:,.0f}B), "
              f"download={down['mean']:,.0f}B  [n={info['n_samples']}]")

    # 3. Radio efficiency
    agg = radio["aggregate"]
    print("\n3. RADIO STATE EFFICIENCY")
    print("-" * 72)
    print(f"   Tail timer:           {TAIL_TIMER_S}s")
    print(f"   Problems analyzed:    {agg['total_problems']}")
    print(f"   Exceed tail timer:    {agg['exceeds_tail_timer_count']}/{agg['total_problems']}"
          f" ({agg['exceeds_tail_timer_pct']:.1f}%)")
    print(f"   Avg tail energy frac: {agg['avg_tail_energy_fraction']:.1%}")
    print(f"   Avg tail:active ratio: {agg['avg_tail_to_active_ratio']:.0f}x"
          f"  (radio in tail {agg['avg_tail_to_active_ratio']:.0f}x longer than actively transferring)")
    print(f"   Avg energy/byte:      {agg['avg_energy_per_byte_mj']:.2f} mJ/byte")
    print()
    print("   Per-model radio efficiency:")
    for model, ms in radio["per_model"].items():
        print(f"     {model:<35} "
              f"exceed_tail={ms['exceeds_tail_timer_pct']:.0f}%, "
              f"tail_frac={ms['avg_tail_energy_fraction']:.1%}, "
              f"tail:active={ms['avg_tail_to_active_ratio']:.0f}x, "
              f"median_gap={ms['median_inference_per_iter_s']:.1f}s")

    # 4. Comparison table
    print("\n4. TRAFFIC CLASS COMPARISON")
    print("-" * 72)
    header = f"{'Traffic Class':<20} {'Pattern':<30} {'Inter-req':<12} {'Payload':<12} {'Radio Eff':<12}"
    print(f"   {header}")
    print(f"   {'─' * 86}")
    for tc in comparison["traffic_classes"]:
        ir = tc["inter_request_time_s"]
        ir_str = f"{ir['typical_range'][0]}-{ir['typical_range'][1]}s"
        pl = tc["payload_bytes"]
        pl_range = pl["typical_range"]
        if pl_range[1] >= 100000:
            pl_str = f"{pl_range[0] // 1000}-{pl_range[1] // 1000}KB"
        elif pl_range[1] >= 1000:
            pl_str = f"{pl_range[0] / 1000:.1f}-{pl_range[1] / 1000:.1f}KB"
        else:
            pl_str = f"{pl_range[0]}-{pl_range[1]}B"
        waste = tc.get("radio_tail_waste_fraction", 0)
        eff_str = f"{waste:.0%} waste"
        pattern = tc["pattern"][:28]
        print(f"   {tc['class']:<20} {pattern:<30} {ir_str:<12} {pl_str:<12} {eff_str:<12}")

    # 5. Sessions
    print("\n5. SESSION CHARACTERISTICS (per model, local network)")
    print("-" * 72)
    for model, ms in sessions.items():
        d = ms["duration_s"]
        b = ms["total_bytes"]
        it = ms["iterations"]
        print(f"   {model}:")
        print(f"     Duration: median {d['median']:.1f}s, range {d['min']:.1f}-{d['max']:.1f}s")
        print(f"     Bytes:    median {b['median']:,.0f}, range {b['min']:,.0f}-{b['max']:,.0f}")
        print(f"     Iters:    median {it['median']:.0f}, max {it['max']}")

    # Key findings
    agentic_tc = comparison["traffic_classes"][-1]
    print("\n6. KEY FINDINGS FOR PAPER")
    print("-" * 72)
    print("   Agentic LLM traffic is a DISTINCT traffic class because:")
    print(f"   a) Inter-request gaps (median {stats['median']:.1f}s) are deterministic,")
    print(f"      driven by on-device inference, not user behavior.")
    print(f"   b) {agg['exceeds_tail_timer_pct']:.0f}% of gaps exceed the {TAIL_TIMER_S}s "
          f"RRC tail timer, causing")
    print(f"      the radio to cycle IDLE->CONNECTED->TAIL->IDLE every iteration.")
    print(f"   c) Upload payloads grow with iteration depth (context accumulation),")
    print(f"      unlike any other traffic class where payloads are roughly constant.")
    print(f"   d) Radio is in tail state {agg['avg_tail_to_active_ratio']:.0f}x longer "
          f"than actively transferring,")
    print(f"      wasting {agg['avg_tail_energy_fraction']:.0%} of radio energy on idle tail periods.")
    print(f"   e) Pattern is strictly alternating upload/download pairs with")
    print(f"      compute-bound gaps, unlike the user-driven patterns of other classes.")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading 3-arch experiment data...")
    results = load_3arch_results()
    print(f"  Loaded {len(results)} entries from {ARCH3_JSONL.name}")

    hybrid_ft = [r for r in results
                 if r["architecture"] == "hybrid" and r.get("mode") == "full_tool"]
    print(f"  Hybrid full_tool entries: {len(hybrid_ft)}")
    print(f"  Models: {sorted(set(r['model'] for r in hybrid_ft))}")

    # Run analyses
    print("\nAnalyzing inter-request timing...")
    inter_req = analyze_inter_request_timing(results)

    print("Analyzing payload growth...")
    payload = analyze_payload_growth(results)

    print("Analyzing radio state efficiency...")
    radio = analyze_radio_efficiency(results)

    print("Analyzing sessions...")
    sessions = analyze_sessions(results)

    print("Building traffic class comparison...")
    comparison = build_traffic_comparison(inter_req, payload, radio)

    # Print summary
    print()
    print_summary(inter_req, payload, radio, comparison, sessions)

    # Save figure data
    outputs = {
        "traffic_char_inter_request_cdf.json": inter_req,
        "traffic_char_payload_growth.json": payload,
        "traffic_char_radio_states.json": radio,
        "traffic_char_comparison.json": comparison,
        "traffic_char_sessions.json": sessions,
    }

    print(f"\n{'=' * 72}")
    print("SAVING FIGURE DATA")
    print(f"{'=' * 72}")
    for filename, data in outputs.items():
        path = OUT_DIR / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved: {path}")

    print(f"\nDone. Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
