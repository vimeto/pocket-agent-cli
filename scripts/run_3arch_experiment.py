#!/usr/bin/env python3
"""3-Architecture Deployment Experiment (Section 6 of the paper).

Compares Local vs Hybrid vs Cloud deployment under varying network conditions.
Produces the energy-vs-RTT crossover plots that are the core MobiHoc contribution.

Architecture A: FULLY LOCAL — MLX on-device inference + local tool execution
Architecture B: HYBRID — Cloud inference (SGLang) + local tool execution + network simulation
Architecture C: FULLY CLOUD — Cloud inference + cloud tool execution (no network cost per iteration)

Usage:
    python scripts/run_3arch_experiment.py --problems 50 --node g1301
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
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from pocket_agent_cli.network.network_simulator import NetworkSimulator, NetworkConfig, NETWORK_PRESETS
from pocket_agent_cli.network.radio_model import RadioStateModel
from pocket_agent_cli.utils.tool_extractor import ToolExtractor
from pocket_agent_cli.utils.optimized_prompts import get_optimized_prompt
from pocket_agent_cli.datasets.registry import DatasetRegistry
from pocket_agent_cli.datasets import mbpp

import httpx

# ── Models for the experiment ─────────────────────────────────────────────
# Use 2 representative models: one that benefits from tools, one that doesn't

EXPERIMENT_MODELS = [
    {"id": "qwen-3-4b", "name": "Qwen 3 4B", "arch": "qwen",
     "hf_id": "Qwen/Qwen3-4B", "local_port": 30001,
     "tool_benefit": True},  # +36pp with tools in paper
    {"id": "llama-3.2-3b-instruct", "name": "Llama 3.2 3B", "arch": "llama",
     "hf_id": "meta-llama/Llama-3.2-3B-Instruct", "local_port": 30003,
     "tool_benefit": False},  # -8pp with tools in paper
]

NETWORK_CONDITIONS = list(NETWORK_PRESETS.keys())  # 7 conditions

TOOL_DEFS = [
    {"type": "function", "function": {
        "name": "run_python_code", "description": "Execute Python code",
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string"}},
                       "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "submit_python_solution", "description": "Submit final solution",
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string"}},
                       "required": ["code"]}}},
]


# ── Helpers ───────────────────────────────────────────────────────────────

def strip_thinking(text):
    c = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return re.sub(r'<think>.*', '', c, flags=re.DOTALL).strip()


def extract_code(response, tool_calls=None):
    if tool_calls:
        for tc in (tool_calls if isinstance(tool_calls, list) else []):
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
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

    matches = re.findall(r'```python\s*(.*?)```', text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    match = re.search(r'(def \w+\([^)]*\):.*?)(?=\n(?:def |\n\n[A-Z]|\Z))', text, re.DOTALL)
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
            result = subprocess.run([sys.executable, f.name],
                                    capture_output=True, text=True, timeout=10)
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


def sglang_chat(base_url, model_hf_id, messages, tools=None,
                max_tokens=2048, temperature=0.7):
    payload = {"model": model_hf_id, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    if tools:
        payload["tools"] = tools
    try:
        resp = httpx.post(f"{base_url}/v1/chat/completions",
                          json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)[:200]}


def execute_tool_locally(code):
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            r = subprocess.run([sys.executable, f.name],
                               capture_output=True, text=True, timeout=10)
            os.unlink(f.name)
        return (r.stdout[:300] or "(no output)") if r.returncode == 0 else f"Error:\n{r.stderr[:300]}"
    except subprocess.TimeoutExpired:
        return "Error: Timeout"
    except Exception as e:
        return f"Error: {e}"


# ── Architecture Runners ──────────────────────────────────────────────────

def run_fully_local(problem, model_def, mode):
    """Architecture A: Read from pre-computed MLX results."""
    # We already have MLX results — load from JSONL
    mlx_dir = Path("data/results/mlx_sweep/20260403_091508")
    jsonl = mlx_dir / f"{model_def['id']}_{mode}.jsonl"
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
                    "tool_exec_time_s": 0,  # included in elapsed
                    "tokens": r["metrics"]["total_tokens"],
                    "thinking_tokens": r["metrics"].get("thinking_tokens", 0),
                    "ttft_ms": r["metrics"].get("ttft_ms"),
                    "tps": r["metrics"].get("tps"),
                    "energy_j": r["metrics"].get("energy_summary", {}).get("total_energy_joules", 0),
                    "power_w": r["metrics"].get("energy_summary", {}).get("avg_power_watts", 0),
                    "iterations": r.get("iterations", 1),
                    "tool_calls": r.get("tool_call_count", 0),
                    "network_transfers": 0,
                    "network_bytes": 0,
                    "radio_transitions": 0,
                    "radio_tail_energy_j": 0,
                }
    return None


def run_hybrid(problem, model_def, mode, network_config, base_url):
    """Architecture B: Cloud inference + local tool execution + network sim.

    Each agentic iteration:
    1. Upload prompt to cloud (simulated network delay)
    2. Cloud runs inference (real SGLang call)
    3. Download response (simulated network delay)
    4. Execute tools locally (real subprocess)
    5. Repeat
    """
    prompt_config = get_optimized_prompt(model_def["id"], mode)
    messages = build_messages(problem, prompt_config)
    no_api_tools = prompt_config.get("no_api_tools", False)

    is_thinking = model_def["arch"] == "qwen"
    max_tokens = 8192 if is_thinking else 2048

    net_sim = NetworkSimulator(network_config, seed=42)
    t0 = time.time()
    total_inference_time = 0
    total_tokens = 0
    iterations = 0
    tool_calls_count = 0
    submitted_code = None

    max_iterations = 5 if mode == "full_tool" else 1

    for iteration in range(max_iterations):
        iterations += 1

        # 1. Simulate upload (prompt → cloud)
        upload_bytes = len(json.dumps(messages).encode("utf-8"))
        net_sim.simulate_transfer_sync(upload_bytes, "upload")

        # 2. Cloud inference (real call)
        api_tools = None if no_api_tools else (TOOL_DEFS if mode != "base" else None)
        t_inf = time.time()
        resp = sglang_chat(base_url, model_def["hf_id"], messages,
                           tools=api_tools, max_tokens=max_tokens)
        inference_time = time.time() - t_inf
        total_inference_time += inference_time

        if "error" in resp:
            break

        choice = resp["choices"][0]
        msg = choice["message"]
        content = msg.get("content", "") or ""
        api_tool_calls = msg.get("tool_calls")
        usage = resp.get("usage", {})
        total_tokens += usage.get("completion_tokens", 0)

        # 3. Simulate download (response → device)
        download_bytes = len(content.encode("utf-8"))
        if api_tool_calls:
            download_bytes += len(json.dumps(api_tool_calls).encode("utf-8"))
        net_sim.simulate_transfer_sync(download_bytes, "download")

        # Parse tool calls
        tool_calls = api_tool_calls
        if not tool_calls and content:
            te = ToolExtractor()
            cleaned = strip_thinking(content) or content
            parsed, _ = te.extract_tools(cleaned)
            if parsed:
                tool_calls = [{"function": {"name": tc.get("name", ""),
                               "arguments": json.dumps(tc.get("parameters", tc.get("arguments", {})))},
                               "id": f"parsed_{i}"} for i, tc in enumerate(parsed)]

        messages.append({"role": "assistant", "content": content})

        if not tool_calls or mode == "base":
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
                    except:
                        args = {}
                submitted_code = args.get("code", "")
                break

            if tc_name == "run_python_code":
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}
                code = args.get("code", "")
                obs = execute_tool_locally(code)
                if no_api_tools:
                    messages.append({"role": "user",
                                     "content": f"Tool result ({tc_name}):\n{obs}"})
                else:
                    messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                                     "content": obs})

        if submitted_code:
            break

    elapsed = time.time() - t0

    if not submitted_code:
        for msg in reversed(messages):
            c = msg.get("content", "")
            submitted_code = extract_code(c)
            if submitted_code:
                break

    evaluation = evaluate_code(submitted_code, problem.test_cases)
    net_summary = net_sim.get_summary()

    return {
        "architecture": "hybrid",
        "network_condition": network_config.name,
        "passed": evaluation["passed"],
        "total_time_s": round(elapsed, 2),
        "inference_time_s": round(total_inference_time, 2),
        "network_time_s": round(net_summary["total_delay_ms"] / 1000, 3),
        "tool_exec_time_s": round(elapsed - total_inference_time - net_summary["total_delay_ms"] / 1000, 2),
        "tokens": total_tokens,
        "thinking_tokens": 0,  # can't measure from API
        "ttft_ms": None,  # not available in non-streaming
        "tps": round(total_tokens / total_inference_time, 1) if total_inference_time > 0 else 0,
        "energy_j": None,  # would need device-side measurement
        "power_w": None,
        "iterations": iterations,
        "tool_calls": tool_calls_count,
        "network_transfers": net_summary["total_transfers"],
        "network_bytes": net_summary["total_bytes"],
        "radio_transitions": net_summary.get("radio_transitions", 0),
        "radio_tail_energy_j": net_summary.get("radio_energy", {}).get("total_tail_energy_j", 0),
    }


def run_fully_cloud(problem, model_def, mode, base_url):
    """Architecture C: Everything on cloud, single upload/download.

    One upload of the task, all iterations server-side, one download of result.
    Network cost = 1 RTT (upload task) + 1 RTT (download result).
    """
    prompt_config = get_optimized_prompt(model_def["id"], mode)
    messages = build_messages(problem, prompt_config)
    no_api_tools = prompt_config.get("no_api_tools", False)

    is_thinking = model_def["arch"] == "qwen"
    max_tokens = 8192 if is_thinking else 2048
    api_tools = None if no_api_tools else (TOOL_DEFS if mode != "base" else None)

    t0 = time.time()
    resp = sglang_chat(base_url, model_def["hf_id"], messages,
                       tools=api_tools, max_tokens=max_tokens)
    inference_time = time.time() - t0

    if "error" in resp:
        return {"architecture": "cloud", "passed": False, "error": resp["error"],
                "total_time_s": inference_time}

    choice = resp["choices"][0]
    msg = choice["message"]
    content = msg.get("content", "") or ""
    tool_calls = msg.get("tool_calls")
    usage = resp.get("usage", {})
    tokens = usage.get("completion_tokens", 0)

    code = extract_code(content, tool_calls)

    # For full_tool on cloud, we'd need server-side tool execution
    # For now, extract from single response (no iteration)
    if not code:
        te = ToolExtractor()
        tcs, _ = te.extract_tools(strip_thinking(content) or content)
        if tcs:
            for tc in tcs:
                params = tc.get("parameters", tc.get("arguments", {}))
                c = params.get("code", "")
                if c and len(c) > 10:
                    code = c
                    break

    evaluation = evaluate_code(code, problem.test_cases)

    # Network cost: upload + download (minimal, just 2 transfers)
    upload_bytes = len(json.dumps(messages).encode("utf-8"))
    download_bytes = len(content.encode("utf-8"))

    return {
        "architecture": "cloud",
        "passed": evaluation["passed"],
        "total_time_s": round(inference_time, 2),
        "inference_time_s": round(inference_time, 2),
        "network_time_s": 0,  # will be added per network condition
        "tool_exec_time_s": 0,
        "tokens": tokens,
        "thinking_tokens": 0,
        "ttft_ms": None,
        "tps": round(tokens / inference_time, 1) if inference_time > 0 else 0,
        "energy_j": None,
        "power_w": None,
        "iterations": 1,
        "tool_calls": 0,
        "network_transfers": 2,  # 1 upload + 1 download
        "network_bytes": upload_bytes + download_bytes,
        "radio_transitions": 1,
        "radio_tail_energy_j": 0,
    }


# ── Main Experiment ───────────────────────────────────────────────────────

def run_experiment(args):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_cls = DatasetRegistry.get("mbpp")
    from pocket_agent_cli.config import DATA_DIR
    ds = ds_cls(DATA_DIR)
    if not ds.is_downloaded():
        ds.download()
    problems = ds.load(split="test", limit=args.problems)
    print(f"3-Architecture Experiment")
    print(f"Problems: {len(problems)}, Models: {len(EXPERIMENT_MODELS)}")
    print(f"Network conditions: {NETWORK_CONDITIONS}")
    print(f"Output: {out_dir}\n")

    selected_modes = args.modes or ["base", "full_tool"]
    all_results = []

    for model_def in EXPERIMENT_MODELS:
        base_url = f"http://localhost:{model_def['local_port']}"

        # Verify server is running
        try:
            r = httpx.get(f"{base_url}/v1/models", timeout=5)
            print(f"Model: {model_def['name']} — server OK")
        except Exception:
            print(f"Model: {model_def['name']} — server NOT available, skipping")
            continue

        for mode in selected_modes:
            print(f"\n  Mode: {mode}")
            print(f"  {'='*50}")

            for net_name in NETWORK_CONDITIONS:
                net_config = NETWORK_PRESETS[net_name]

                print(f"\n    Network: {net_name} (RTT={net_config.rtt_ms}ms)")

                for i, problem in enumerate(problems):
                    # Architecture A: Local (from pre-computed MLX data)
                    local_result = run_fully_local(problem, model_def, mode)
                    if local_result:
                        local_result["model"] = model_def["id"]
                        local_result["mode"] = mode
                        local_result["network_condition"] = net_name
                        local_result["problem_id"] = problem.task_id
                        all_results.append(local_result)

                    # Architecture B: Hybrid (cloud inference + network sim + local tools)
                    hybrid_result = run_hybrid(problem, model_def, mode, net_config, base_url)
                    hybrid_result["model"] = model_def["id"]
                    hybrid_result["mode"] = mode
                    hybrid_result["problem_id"] = problem.task_id
                    all_results.append(hybrid_result)

                    # Architecture C: Cloud (single request, add simulated network cost)
                    if net_name == NETWORK_CONDITIONS[0]:  # Only run cloud once (no per-iteration network)
                        cloud_result = run_fully_cloud(problem, model_def, mode, base_url)
                        cloud_result["model"] = model_def["id"]
                        cloud_result["mode"] = mode
                        cloud_result["problem_id"] = problem.task_id
                        # Add simulated network cost for each condition
                        for nc_name in NETWORK_CONDITIONS:
                            nc = NETWORK_PRESETS[nc_name]
                            cloud_copy = dict(cloud_result)
                            cloud_copy["network_condition"] = nc_name
                            # Cloud: just 1 upload + 1 download
                            sim = NetworkSimulator(nc, seed=42)
                            up = sim.simulate_transfer_sync(cloud_copy["network_bytes"] // 2, "upload")
                            down = sim.simulate_transfer_sync(cloud_copy["network_bytes"] // 2, "download")
                            cloud_copy["network_time_s"] = round((up.total_delay_ms + down.total_delay_ms) / 1000, 3)
                            cloud_copy["total_time_s"] = round(cloud_copy["inference_time_s"] + cloud_copy["network_time_s"], 2)
                            cloud_copy["radio_tail_energy_j"] = nc.radio_tail_energy_j
                            all_results.append(cloud_copy)

                    if (i + 1) % 10 == 0 or (i + 1) == len(problems):
                        # Count results for this condition
                        local_pass = sum(1 for r in all_results
                                         if r.get("architecture") == "local"
                                         and r.get("network_condition") == net_name
                                         and r.get("model") == model_def["id"]
                                         and r.get("mode") == mode and r.get("passed"))
                        hybrid_pass = sum(1 for r in all_results
                                          if r.get("architecture") == "hybrid"
                                          and r.get("network_condition") == net_name
                                          and r.get("model") == model_def["id"]
                                          and r.get("mode") == mode and r.get("passed"))
                        print(f"      [{i+1}/{len(problems)}] local={local_pass} hybrid={hybrid_pass}")

    # Save all results
    out_file = out_dir / "3arch_results.jsonl"
    with open(out_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nSaved {len(all_results)} results to {out_file}")

    # Generate summary and figures
    generate_summary(all_results, out_dir)


def generate_summary(results, out_dir):
    """Generate summary tables and figure data."""
    import statistics

    print(f"\n{'='*70}")
    print("3-ARCHITECTURE EXPERIMENT SUMMARY")
    print(f"{'='*70}\n")

    # Group by model × mode × architecture × network
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        key = (r.get("model", "?"), r.get("mode", "?"),
               r.get("architecture", "?"), r.get("network_condition", "?"))
        groups[key].append(r)

    # Summary table
    print(f"{'Model':<20} {'Mode':<10} {'Arch':<8} {'Network':<15} {'Pass%':>6} {'Time':>7} {'NetTime':>8} {'Tokens':>7}")
    print("-" * 85)

    figure_data = []

    for (model, mode, arch, net), group in sorted(groups.items()):
        n = len(group)
        passed = sum(1 for r in group if r.get("passed"))
        pass_rate = passed / n if n > 0 else 0
        avg_time = statistics.mean(r.get("total_time_s", 0) for r in group)
        avg_net = statistics.mean(r.get("network_time_s", 0) for r in group)
        avg_tokens = statistics.mean(r.get("tokens", 0) for r in group)

        print(f"{model:<20} {mode:<10} {arch:<8} {net:<15} {pass_rate:>5.0%} {avg_time:>6.1f}s {avg_net:>7.3f}s {avg_tokens:>6.0f}")

        figure_data.append({
            "model": model, "mode": mode, "architecture": arch,
            "network_condition": net,
            "rtt_ms": NETWORK_PRESETS.get(net, NetworkConfig(name="?", rtt_ms=0)).rtt_ms,
            "pass_rate": pass_rate, "avg_time_s": avg_time,
            "avg_network_time_s": avg_net, "avg_tokens": avg_tokens,
            "n_problems": n,
            "avg_radio_tail_energy_j": statistics.mean(r.get("radio_tail_energy_j", 0) for r in group),
            "avg_iterations": statistics.mean(r.get("iterations", 1) for r in group),
            "avg_tool_calls": statistics.mean(r.get("tool_calls", 0) for r in group),
            "avg_network_transfers": statistics.mean(r.get("network_transfers", 0) for r in group),
        })

    # Save figure data
    with open(out_dir / "figure_data.json", "w") as f:
        json.dump(figure_data, f, indent=2)
    print(f"\nFigure data saved to {out_dir / 'figure_data.json'}")

    # Generate crossover analysis
    print(f"\n{'='*70}")
    print("CROSSOVER ANALYSIS: Where does Local beat Hybrid?")
    print(f"{'='*70}\n")

    for model in set(r["model"] for r in results):
        for mode in set(r["mode"] for r in results):
            local_times = {}
            hybrid_times = {}
            cloud_times = {}

            for fd in figure_data:
                if fd["model"] == model and fd["mode"] == mode:
                    if fd["architecture"] == "local":
                        local_times[fd["network_condition"]] = fd["avg_time_s"]
                    elif fd["architecture"] == "hybrid":
                        hybrid_times[fd["network_condition"]] = fd["avg_time_s"]
                    elif fd["architecture"] == "cloud":
                        cloud_times[fd["network_condition"]] = fd["avg_time_s"]

            if local_times and hybrid_times:
                print(f"{model} / {mode}:")
                local_t = list(local_times.values())[0] if local_times else 0
                for net in NETWORK_CONDITIONS:
                    h = hybrid_times.get(net, 0)
                    c = cloud_times.get(net, 0)
                    winner = "LOCAL" if local_t < h else "HYBRID"
                    rtt = NETWORK_PRESETS[net].rtt_ms
                    print(f"  {net:<15} (RTT={rtt:>3}ms): local={local_t:.1f}s hybrid={h:.1f}s cloud={c:.1f}s → {winner}")
                print()


def main():
    parser = argparse.ArgumentParser(description="3-Architecture Deployment Experiment")
    parser.add_argument("--problems", type=int, default=50)
    parser.add_argument("--modes", nargs="*", default=["base", "full_tool"])
    parser.add_argument("--output-dir", default="data/results/3arch_experiment")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
