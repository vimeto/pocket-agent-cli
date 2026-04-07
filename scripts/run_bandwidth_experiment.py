#!/usr/bin/env python3
"""Bandwidth × RTT Sweep Experiment (COL-213).

Extends the 3-architecture framework to measure the impact of bandwidth
constraints on the Local vs Hybrid vs Cloud tradeoff. The key question:
at what bandwidth does on-device inference (MLX) become faster than
hybrid (cloud inference + network transfer)?

Sweep: 5 bandwidth levels × 3 RTT levels = 15 network conditions
Models: Qwen 3 4B (thinking, large context) and Llama 3.2 3B (non-thinking)
Mode: full_tool only (most iterations = most context growth = most network)

Per-iteration logging:
  upload_bytes, download_bytes, bandwidth_delay_ms, rtt_delay_ms, total_network_ms

Usage:
    python scripts/run_bandwidth_experiment.py --problems 50 --node g1301
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from pocket_agent_cli.network.network_simulator import (
    NetworkSimulator,
    NetworkConfig,
    NETWORK_PRESETS,
)
from pocket_agent_cli.network.radio_model import RadioStateModel
from pocket_agent_cli.utils.tool_extractor import ToolExtractor
from pocket_agent_cli.utils.optimized_prompts import get_optimized_prompt
from pocket_agent_cli.datasets.registry import DatasetRegistry

import httpx


# ── Experiment Design ────────────────────────────────────────────────────

# 2 representative models
BANDWIDTH_MODELS = [
    {
        "id": "qwen-3-4b",
        "name": "Qwen 3 4B",
        "arch": "qwen",
        "hf_id": "Qwen/Qwen3-4B",
        "local_port": 30001,
        "is_thinking": True,
    },
    {
        "id": "llama-3.2-3b-instruct",
        "name": "Llama 3.2 3B",
        "arch": "llama",
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "local_port": 30003,
        "is_thinking": False,
    },
]

# 5 bandwidth levels (Mbps)
BANDWIDTH_LEVELS = [0.5, 1.0, 5.0, 20.0, 100.0]

# 3 RTT levels (ms) with corresponding jitter and loss
RTT_CONFIGS = [
    {"rtt_ms": 20, "jitter_ms": 5, "packet_loss_rate": 0.001, "radio_tail_energy_j": 0.0},   # WiFi
    {"rtt_ms": 80, "jitter_ms": 30, "packet_loss_rate": 0.005, "radio_tail_energy_j": 0.5},   # 4G
    {"rtt_ms": 200, "jitter_ms": 100, "packet_loss_rate": 0.02, "radio_tail_energy_j": 0.8},  # Poor cellular
]

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "Execute Python code",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_python_solution",
            "description": "Submit final solution",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
]


# ── Helpers (same as run_3arch_experiment.py) ────────────────────────────


def strip_thinking(text):
    c = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return re.sub(r"<think>.*", "", c, flags=re.DOTALL).strip()


def extract_code(response, tool_calls=None):
    if tool_calls:
        for tc in tool_calls if isinstance(tool_calls, list) else []:
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    continue
            code = args.get("code", "")
            if code and len(code) > 10:
                return code

    text = strip_thinking(response) or response
    te = ToolExtractor()
    tcs, _ = te.extract_tools(text)
    if tcs:
        for tc in tcs:
            params = tc.get("parameters", tc.get("arguments", {}))
            code = params.get("code", "")
            if code and len(code) > 10:
                return code

    matches = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    match = re.search(
        r"(def \w+\([^)]*\):.*?)(?=\n(?:def |\n\n[A-Z]|\Z))", text, re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return None


def evaluate_code(code, test_cases):
    if not code:
        return {"passed": False, "error": "No code extracted"}
    test_code = code + "\n\n" + "\n".join(test_cases)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            result = subprocess.run(
                [sys.executable, f.name], capture_output=True, text=True, timeout=10
            )
            os.unlink(f.name)
        if result.returncode == 0:
            return {"passed": True}
        return {"passed": False, "error": result.stderr[:300]}
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "Timeout"}
    except Exception as e:
        return {"passed": False, "error": str(e)[:200]}


def build_messages(problem, prompt_config):
    messages = []
    if prompt_config.get("system"):
        messages.append({"role": "system", "content": prompt_config["system"]})
    text = problem.prompt
    if hasattr(problem, "test_cases") and problem.test_cases:
        code_tests = [t for t in problem.test_cases if not t.startswith("EXPECTED_ANSWER")]
        if code_tests:
            text += "\n\nTest cases:\n" + "\n".join(code_tests[:3])
    suffix = prompt_config.get("user_suffix", "")
    messages.append({"role": "user", "content": text + suffix})
    return messages


def sglang_chat(base_url, model_hf_id, messages, tools=None, max_tokens=2048,
                temperature=0.7, retries=3):
    payload = {
        "model": model_hf_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
    for attempt in range(retries):
        try:
            resp = httpx.post(
                f"{base_url}/v1/chat/completions", json=payload, timeout=300
            )
            resp.raise_for_status()
            d = resp.json()
            tokens = d.get("usage", {}).get("completion_tokens", 0)
            if tokens > 0:
                return d
            if attempt < retries - 1:
                time.sleep(2)
                continue
            return d
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
                continue
            return {"error": str(e)[:200]}


def execute_tool_locally(code):
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            r = subprocess.run(
                [sys.executable, f.name], capture_output=True, text=True, timeout=10
            )
            os.unlink(f.name)
        return (
            (r.stdout[:300] or "(no output)")
            if r.returncode == 0
            else f"Error:\n{r.stderr[:300]}"
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout"
    except Exception as e:
        return f"Error: {e}"


# ── Hybrid Runner with Per-Iteration Network Logging ────────────────────


def run_hybrid_bandwidth(problem, model_def, network_config, base_url):
    """Architecture B: Cloud inference + local tools + network sim.

    Logs per-iteration network details for bandwidth analysis.
    Returns result dict plus a list of per-iteration records.
    """
    prompt_config = get_optimized_prompt(model_def["id"], "full_tool")
    messages = build_messages(problem, prompt_config)
    no_api_tools = prompt_config.get("no_api_tools", False)

    max_tokens = 8192 if model_def["is_thinking"] else 2048

    net_sim = NetworkSimulator(network_config, seed=42)
    t0 = time.time()
    total_inference_time = 0
    total_tokens = 0
    iterations = 0
    tool_calls_count = 0
    submitted_code = None

    max_iterations = 5
    iteration_logs = []

    for iteration in range(max_iterations):
        iterations += 1
        iter_log = {
            "iteration": iteration,
            "upload_bytes": 0,
            "download_bytes": 0,
            "upload_bandwidth_delay_ms": 0.0,
            "upload_rtt_delay_ms": 0.0,
            "download_bandwidth_delay_ms": 0.0,
            "download_rtt_delay_ms": 0.0,
            "total_network_ms": 0.0,
            "inference_time_s": 0.0,
            "tokens": 0,
        }

        # 1. Simulate upload (prompt -> cloud)
        upload_bytes = len(json.dumps(messages).encode("utf-8"))
        upload_event = net_sim.simulate_transfer_sync(upload_bytes, "upload")

        iter_log["upload_bytes"] = upload_bytes
        iter_log["upload_bandwidth_delay_ms"] = round(upload_event.bandwidth_delay_ms, 3)
        iter_log["upload_rtt_delay_ms"] = round(upload_event.simulated_rtt_ms, 3)

        # 2. Cloud inference (real call)
        api_tools = None if no_api_tools else TOOL_DEFS
        t_inf = time.time()
        resp = sglang_chat(
            base_url, model_def["hf_id"], messages, tools=api_tools,
            max_tokens=max_tokens,
        )
        inference_time = time.time() - t_inf
        total_inference_time += inference_time
        iter_log["inference_time_s"] = round(inference_time, 3)

        if "error" in resp:
            iteration_logs.append(iter_log)
            break

        choice = resp["choices"][0]
        msg = choice["message"]
        content = msg.get("content", "") or ""
        api_tool_calls = msg.get("tool_calls")
        usage = resp.get("usage", {})
        iter_tokens = usage.get("completion_tokens", 0)
        total_tokens += iter_tokens
        iter_log["tokens"] = iter_tokens

        # 3. Simulate download (response -> device)
        download_bytes = len(content.encode("utf-8"))
        if api_tool_calls:
            download_bytes += len(json.dumps(api_tool_calls).encode("utf-8"))
        download_event = net_sim.simulate_transfer_sync(download_bytes, "download")

        iter_log["download_bytes"] = download_bytes
        iter_log["download_bandwidth_delay_ms"] = round(download_event.bandwidth_delay_ms, 3)
        iter_log["download_rtt_delay_ms"] = round(download_event.simulated_rtt_ms, 3)
        iter_log["total_network_ms"] = round(
            upload_event.total_delay_ms + download_event.total_delay_ms, 3
        )

        iteration_logs.append(iter_log)

        # Parse tool calls
        tool_calls = api_tool_calls
        if not tool_calls and content:
            te = ToolExtractor()
            cleaned = strip_thinking(content) or content
            parsed, _ = te.extract_tools(cleaned)
            if parsed:
                tool_calls = [
                    {
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": json.dumps(
                                tc.get("parameters", tc.get("arguments", {}))
                            ),
                        },
                        "id": f"parsed_{i}",
                    }
                    for i, tc in enumerate(parsed)
                ]

        messages.append({"role": "assistant", "content": content})

        if not tool_calls:
            code = extract_code(content, api_tool_calls)
            if code:
                submitted_code = code
            break

        # 4. Execute tools locally
        for tc in tool_calls:
            fn = tc.get("function", {})
            tc_name = fn.get("name", "")
            tool_calls_count += 1

            if tc_name == "submit_python_solution":
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                submitted_code = args.get("code", "")
                break

            if tc_name == "run_python_code":
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                code = args.get("code", "")
                obs = execute_tool_locally(code)
                if no_api_tools:
                    messages.append(
                        {"role": "user", "content": f"Tool result ({tc_name}):\n{obs}"}
                    )
                else:
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.get("id", ""), "content": obs}
                    )

        if submitted_code:
            break

    elapsed = time.time() - t0

    if not submitted_code:
        for msg_item in reversed(messages):
            c = msg_item.get("content", "")
            submitted_code = extract_code(c)
            if submitted_code:
                break

    evaluation = evaluate_code(submitted_code, problem.test_cases)
    net_summary = net_sim.get_summary()

    # Aggregate per-iteration network stats
    total_upload_bytes = sum(il["upload_bytes"] for il in iteration_logs)
    total_download_bytes = sum(il["download_bytes"] for il in iteration_logs)
    total_bandwidth_delay_ms = sum(
        il["upload_bandwidth_delay_ms"] + il["download_bandwidth_delay_ms"]
        for il in iteration_logs
    )
    total_rtt_delay_ms = sum(
        il["upload_rtt_delay_ms"] + il["download_rtt_delay_ms"]
        for il in iteration_logs
    )

    result = {
        "architecture": "hybrid",
        "passed": evaluation["passed"],
        "total_time_s": round(elapsed, 3),
        "inference_time_s": round(total_inference_time, 3),
        "network_time_s": round(net_summary["total_delay_ms"] / 1000, 3),
        "tool_exec_time_s": round(
            elapsed - total_inference_time - net_summary["total_delay_ms"] / 1000, 3
        ),
        "tokens": total_tokens,
        "iterations": iterations,
        "tool_calls": tool_calls_count,
        "network_transfers": net_summary["total_transfers"],
        "total_upload_bytes": total_upload_bytes,
        "total_download_bytes": total_download_bytes,
        "total_network_bytes": total_upload_bytes + total_download_bytes,
        "total_bandwidth_delay_ms": round(total_bandwidth_delay_ms, 3),
        "total_rtt_delay_ms": round(total_rtt_delay_ms, 3),
        "radio_transitions": net_summary.get("radio_transitions", 0),
        "radio_tail_energy_j": net_summary.get("radio_energy", {}).get(
            "total_tail_energy_j", 0
        ),
        "iteration_logs": iteration_logs,
    }

    return result


def run_fully_local(problem, model_def):
    """Architecture A: Read from pre-computed MLX results."""
    mlx_dir = Path("data/results/mlx_sweep/20260403_091508")
    jsonl = mlx_dir / f"{model_def['id']}_full_tool.jsonl"
    if not jsonl.exists():
        return None

    with open(jsonl) as f:
        for line in f:
            r = json.loads(line)
            if str(r["problem_id"]) == str(problem.task_id):
                return {
                    "architecture": "local",
                    "passed": r["passed"],
                    "total_time_s": r["metrics"]["elapsed_s"],
                    "inference_time_s": r["metrics"]["elapsed_s"],
                    "network_time_s": 0,
                    "tool_exec_time_s": 0,
                    "tokens": r["metrics"]["total_tokens"],
                    "thinking_tokens": r["metrics"].get("thinking_tokens", 0),
                    "ttft_ms": r["metrics"].get("ttft_ms"),
                    "tps": r["metrics"].get("tps"),
                    "iterations": r.get("iterations", 1),
                    "tool_calls": r.get("tool_call_count", 0),
                    "total_upload_bytes": 0,
                    "total_download_bytes": 0,
                    "total_network_bytes": 0,
                    "total_bandwidth_delay_ms": 0,
                    "total_rtt_delay_ms": 0,
                }
    return None


# ── Main Experiment ──────────────────────────────────────────────────────


def build_network_configs(bw_levels=None, rtt_levels=None):
    """Build NetworkConfig objects for bandwidth x RTT sweep.

    Args:
        bw_levels: List of bandwidth levels in Mbps. Defaults to BANDWIDTH_LEVELS.
        rtt_levels: List of RTT levels in ms. Defaults to all RTT_CONFIGS.
    """
    bw_list = bw_levels or BANDWIDTH_LEVELS
    rtt_list = RTT_CONFIGS
    if rtt_levels:
        rtt_list = [r for r in RTT_CONFIGS if r["rtt_ms"] in rtt_levels]
    configs = []
    for bw in bw_list:
        for rtt_cfg in rtt_list:
            name = f"bw{bw}mbps_rtt{int(rtt_cfg['rtt_ms'])}ms"
            cfg = NetworkConfig(
                name=name,
                rtt_ms=rtt_cfg["rtt_ms"],
                jitter_ms=rtt_cfg["jitter_ms"],
                packet_loss_rate=rtt_cfg["packet_loss_rate"],
                bandwidth_mbps=bw,
                radio_tail_energy_j=rtt_cfg["radio_tail_energy_j"],
            )
            configs.append(cfg)
    return configs


def run_experiment(args):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load MBPP problems
    ds_cls = DatasetRegistry.get("mbpp")
    from pocket_agent_cli.config import DATA_DIR
    ds = ds_cls(DATA_DIR)
    if not ds.is_downloaded():
        ds.download()
    problems = ds.load(split="test", limit=args.problems)

    # Filter models if specified
    models_to_run = BANDWIDTH_MODELS
    if args.models:
        models_to_run = [m for m in BANDWIDTH_MODELS if m["id"] in args.models]
        if not models_to_run:
            print(f"ERROR: No matching models for {args.models}")
            print(f"Available: {[m['id'] for m in BANDWIDTH_MODELS]}")
            sys.exit(1)

    bw_levels = [float(b) for b in args.bandwidths] if args.bandwidths else None
    rtt_levels = [float(r) for r in args.rtts] if args.rtts else None
    network_configs = build_network_configs(bw_levels=bw_levels, rtt_levels=rtt_levels)

    actual_bws = sorted(set(c.bandwidth_mbps for c in network_configs))
    actual_rtts = sorted(set(c.rtt_ms for c in network_configs))

    print(f"Bandwidth x RTT Sweep Experiment")
    print(f"Problems: {len(problems)}")
    print(f"Models: {[m['id'] for m in models_to_run]}")
    print(f"Network conditions: {len(network_configs)} ({len(actual_bws)} BW x {len(actual_rtts)} RTT)")
    print(f"  Bandwidths: {actual_bws} Mbps")
    print(f"  RTTs: {actual_rtts} ms")
    print(f"Output: {out_dir}\n")

    all_results = []
    per_problem_records = []

    for model_def in models_to_run:
        base_url = f"http://localhost:{model_def['local_port']}"

        # Verify server
        try:
            r = httpx.get(f"{base_url}/v1/models", timeout=5)
            print(f"Model: {model_def['name']} -- server OK")
        except Exception:
            print(f"Model: {model_def['name']} -- server NOT available, skipping")
            continue

        # Pre-load local results for this model
        local_results_by_pid = {}
        mlx_dir = Path("data/results/mlx_sweep/20260403_091508")
        jsonl_path = mlx_dir / f"{model_def['id']}_full_tool.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path) as f:
                for line in f:
                    r = json.loads(line)
                    local_results_by_pid[str(r["problem_id"])] = r

        for nc_idx, net_config in enumerate(network_configs):
            print(f"\n  [{nc_idx+1}/{len(network_configs)}] "
                  f"BW={net_config.bandwidth_mbps}Mbps RTT={net_config.rtt_ms}ms "
                  f"({net_config.name})")

            condition_results = []

            for i, problem in enumerate(problems):
                # Architecture A: Local (from MLX data)
                local_result = run_fully_local(problem, model_def)

                # Architecture B: Hybrid with bandwidth simulation
                hybrid_result = run_hybrid_bandwidth(
                    problem, model_def, net_config, base_url
                )

                # Store iteration logs separately for per-problem JSONL
                iteration_logs = hybrid_result.pop("iteration_logs", [])

                # Annotate results
                for result in [local_result, hybrid_result]:
                    if result is None:
                        continue
                    result["model"] = model_def["id"]
                    result["mode"] = "full_tool"
                    result["bandwidth_mbps"] = net_config.bandwidth_mbps
                    result["rtt_ms"] = net_config.rtt_ms
                    result["network_condition"] = net_config.name
                    result["problem_id"] = problem.task_id

                if local_result:
                    all_results.append(local_result)
                all_results.append(hybrid_result)

                # Per-problem record with iteration details
                per_problem_record = {
                    "model": model_def["id"],
                    "problem_id": problem.task_id,
                    "bandwidth_mbps": net_config.bandwidth_mbps,
                    "rtt_ms": net_config.rtt_ms,
                    "network_condition": net_config.name,
                    "local_time_s": local_result["total_time_s"] if local_result else None,
                    "local_passed": local_result["passed"] if local_result else None,
                    "hybrid_time_s": hybrid_result["total_time_s"],
                    "hybrid_passed": hybrid_result["passed"],
                    "hybrid_inference_time_s": hybrid_result["inference_time_s"],
                    "hybrid_network_time_s": hybrid_result["network_time_s"],
                    "hybrid_total_upload_bytes": hybrid_result["total_upload_bytes"],
                    "hybrid_total_download_bytes": hybrid_result["total_download_bytes"],
                    "hybrid_total_bandwidth_delay_ms": hybrid_result["total_bandwidth_delay_ms"],
                    "hybrid_total_rtt_delay_ms": hybrid_result["total_rtt_delay_ms"],
                    "hybrid_iterations": hybrid_result["iterations"],
                    "hybrid_tokens": hybrid_result["tokens"],
                    "iteration_logs": iteration_logs,
                }
                per_problem_records.append(per_problem_record)
                condition_results.append(per_problem_record)

                if (i + 1) % 10 == 0 or (i + 1) == len(problems):
                    h_pass = sum(1 for r in condition_results if r["hybrid_passed"])
                    l_pass = sum(
                        1 for r in condition_results
                        if r["local_passed"] is not None and r["local_passed"]
                    )
                    print(
                        f"    [{i+1}/{len(problems)}] "
                        f"hybrid={h_pass}/{len(condition_results)} "
                        f"local={l_pass}/{len(condition_results)}"
                    )

    # ── Save results ─────────────────────────────────────────────────────

    # 1. Summary results (all_results, one line per architecture x condition x problem)
    results_file = out_dir / "bandwidth_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results to {results_file}")

    # 2. Per-problem JSONL with iteration logs
    per_problem_file = out_dir / "bandwidth_per_problem.jsonl"
    with open(per_problem_file, "w") as f:
        for rec in per_problem_records:
            f.write(json.dumps(rec, default=str) + "\n")
    print(f"Saved {len(per_problem_records)} per-problem records to {per_problem_file}")

    # 3. Generate figure data and analysis
    generate_figure_data(per_problem_records, all_results, out_dir)

    return out_dir


def generate_figure_data(per_problem_records, all_results, out_dir):
    """Generate heatmap data and crossover analysis."""
    import statistics
    from collections import defaultdict

    print(f"\n{'='*70}")
    print("BANDWIDTH x RTT SWEEP ANALYSIS")
    print(f"{'='*70}\n")

    # Group per-problem records by model x bandwidth x rtt
    groups = defaultdict(list)
    for rec in per_problem_records:
        key = (rec["model"], rec["bandwidth_mbps"], rec["rtt_ms"])
        groups[key].append(rec)

    # Build heatmap data: rows=bandwidth, cols=RTT, cells=time ratio
    heatmap_entries = []

    print(f"{'Model':<25} {'BW(Mbps)':>8} {'RTT(ms)':>7} {'HybTime':>8} {'LocTime':>8} "
          f"{'Ratio':>6} {'HybNet':>8} {'BWDly':>8} {'RTTDly':>8} {'HPass%':>7} {'LPass%':>7}")
    print("-" * 120)

    for (model, bw, rtt), recs in sorted(groups.items()):
        n = len(recs)

        # Hybrid stats
        h_times = [r["hybrid_time_s"] for r in recs]
        h_net_times = [r["hybrid_network_time_s"] for r in recs]
        h_bw_delays = [r["hybrid_total_bandwidth_delay_ms"] for r in recs]
        h_rtt_delays = [r["hybrid_total_rtt_delay_ms"] for r in recs]
        h_passed = sum(1 for r in recs if r["hybrid_passed"])

        # Local stats (may not be available for all problems)
        l_times = [r["local_time_s"] for r in recs if r["local_time_s"] is not None]
        l_passed = sum(1 for r in recs if r["local_passed"] is not None and r["local_passed"])

        avg_h_time = statistics.mean(h_times) if h_times else 0
        avg_l_time = statistics.mean(l_times) if l_times else 0
        avg_h_net = statistics.mean(h_net_times) if h_net_times else 0
        avg_h_bw_delay = statistics.mean(h_bw_delays) if h_bw_delays else 0
        avg_h_rtt_delay = statistics.mean(h_rtt_delays) if h_rtt_delays else 0

        # Time ratio: hybrid / local (>1 means local is faster)
        time_ratio = avg_h_time / avg_l_time if avg_l_time > 0 else None

        h_pass_rate = h_passed / n if n > 0 else 0
        l_pass_rate = l_passed / len(l_times) if l_times else 0

        ratio_str = f"{time_ratio:.2f}" if time_ratio else "N/A"
        print(f"{model:<25} {bw:>8.1f} {rtt:>7.0f} {avg_h_time:>7.1f}s {avg_l_time:>7.1f}s "
              f"{ratio_str:>6} {avg_h_net:>7.3f}s "
              f"{avg_h_bw_delay:>7.1f}ms {avg_h_rtt_delay:>7.1f}ms "
              f"{h_pass_rate:>6.0%} {l_pass_rate:>6.0%}")

        heatmap_entry = {
            "model": model,
            "bandwidth_mbps": bw,
            "rtt_ms": rtt,
            "n_problems": n,
            "avg_hybrid_time_s": round(avg_h_time, 3),
            "avg_local_time_s": round(avg_l_time, 3),
            "avg_hybrid_network_time_s": round(avg_h_net, 3),
            "avg_hybrid_inference_time_s": round(
                statistics.mean(r["hybrid_inference_time_s"] for r in recs), 3
            ),
            "avg_bandwidth_delay_ms": round(avg_h_bw_delay, 3),
            "avg_rtt_delay_ms": round(avg_h_rtt_delay, 3),
            "time_ratio_hybrid_over_local": round(time_ratio, 4) if time_ratio else None,
            "hybrid_pass_rate": round(h_pass_rate, 4),
            "local_pass_rate": round(l_pass_rate, 4),
            "avg_upload_bytes": round(
                statistics.mean(r["hybrid_total_upload_bytes"] for r in recs), 1
            ),
            "avg_download_bytes": round(
                statistics.mean(r["hybrid_total_download_bytes"] for r in recs), 1
            ),
            "avg_iterations": round(
                statistics.mean(r["hybrid_iterations"] for r in recs), 2
            ),
            "avg_tokens": round(
                statistics.mean(r["hybrid_tokens"] for r in recs), 1
            ),
        }
        heatmap_entries.append(heatmap_entry)

    # Save figure data
    figure_data = {
        "experiment": "bandwidth_rtt_sweep",
        "bandwidth_levels": BANDWIDTH_LEVELS,
        "rtt_levels": [r["rtt_ms"] for r in RTT_CONFIGS],
        "models": [m["id"] for m in BANDWIDTH_MODELS],
        "heatmap": heatmap_entries,
    }

    fig_file = out_dir / "bandwidth_figure_data.json"
    with open(fig_file, "w") as f:
        json.dump(figure_data, f, indent=2)
    print(f"\nFigure data saved to {fig_file}")

    # ── Crossover analysis ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CROSSOVER ANALYSIS: Where does Local beat Hybrid?")
    print(f"{'='*70}\n")

    for model in set(e["model"] for e in heatmap_entries):
        print(f"\n  Model: {model}")
        model_entries = [e for e in heatmap_entries if e["model"] == model]

        if not model_entries or model_entries[0]["avg_local_time_s"] == 0:
            print("    (no local data available)")
            continue

        local_time = model_entries[0]["avg_local_time_s"]
        print(f"    Local (MLX) avg time: {local_time:.1f}s\n")

        print(f"    {'BW(Mbps)':>8} {'RTT(ms)':>7} {'HybTime':>8} {'NetOverhead':>11} "
              f"{'NetPct':>7} {'Winner':>8}")
        print(f"    {'-'*60}")

        for entry in sorted(model_entries, key=lambda e: (e["bandwidth_mbps"], e["rtt_ms"])):
            h_time = entry["avg_hybrid_time_s"]
            net_time = entry["avg_hybrid_network_time_s"]
            inf_time = entry["avg_hybrid_inference_time_s"]
            net_pct = (net_time / h_time * 100) if h_time > 0 else 0
            winner = "LOCAL" if local_time < h_time else "HYBRID"

            print(f"    {entry['bandwidth_mbps']:>8.1f} {entry['rtt_ms']:>7.0f} "
                  f"{h_time:>7.1f}s {net_time:>10.3f}s {net_pct:>6.1f}% {winner:>8}")

        # Estimate crossover bandwidth
        print(f"\n    Crossover estimation:")
        print(f"    Local time = {local_time:.1f}s")

        for rtt_cfg in RTT_CONFIGS:
            rtt = rtt_cfg["rtt_ms"]
            rtt_entries = [e for e in model_entries if e["rtt_ms"] == rtt]
            if not rtt_entries:
                continue

            # Find bandwidth where hybrid crosses local
            crossover_found = False
            for i in range(len(rtt_entries) - 1):
                e1 = rtt_entries[i]
                e2 = rtt_entries[i + 1]
                if (e1["avg_hybrid_time_s"] > local_time and
                        e2["avg_hybrid_time_s"] <= local_time):
                    print(f"    RTT={rtt}ms: crossover between "
                          f"BW={e1['bandwidth_mbps']}-{e2['bandwidth_mbps']} Mbps")
                    crossover_found = True
                    break

            if not crossover_found:
                # Is hybrid always faster or always slower?
                if all(e["avg_hybrid_time_s"] < local_time for e in rtt_entries):
                    print(f"    RTT={rtt}ms: hybrid always faster (even at {rtt_entries[0]['bandwidth_mbps']} Mbps)")
                elif all(e["avg_hybrid_time_s"] > local_time for e in rtt_entries):
                    min_bw_entry = min(rtt_entries, key=lambda e: e["bandwidth_mbps"])
                    print(f"    RTT={rtt}ms: local always faster (hybrid slower even at "
                          f"{max(e['bandwidth_mbps'] for e in rtt_entries)} Mbps, "
                          f"net overhead at {min_bw_entry['bandwidth_mbps']} Mbps = "
                          f"{min_bw_entry['avg_hybrid_network_time_s']:.3f}s)")
                else:
                    # Mixed results
                    print(f"    RTT={rtt}ms: mixed results (no clean crossover)")


def main():
    parser = argparse.ArgumentParser(
        description="Bandwidth x RTT Sweep Experiment (COL-213)"
    )
    parser.add_argument("--problems", type=int, default=50)
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Filter to specific model IDs",
    )
    parser.add_argument(
        "--output-dir", default="data/results/bandwidth_experiment",
    )
    parser.add_argument(
        "--bandwidths", nargs="*", default=None,
        help="Bandwidth levels in Mbps (default: 0.5 1.0 5.0 20.0 100.0)",
    )
    parser.add_argument(
        "--rtts", nargs="*", default=None,
        help="RTT levels in ms (default: 20 80 200)",
    )
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
