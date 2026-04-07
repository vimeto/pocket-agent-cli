#!/usr/bin/env python3
"""Context Compression Experiment for Hybrid Agentic Deployment.

Measures the bandwidth savings vs accuracy tradeoff of compressing context
before uploading to a cloud inference server in hybrid mode.

Compression Levels:
    0 (none):   Full context uploaded — baseline.
    1 (light):  Strip <think>...</think> from all previous assistant turns.
    2 (medium): Strip thinking + keep only last 2 assistant/tool turns.
    3 (heavy):  Strip thinking + keep only last 1 turn + truncate tool to 200 chars.

Bandwidth Conditions:
    50 Mbps (good WiFi), 5 Mbps (poor 4G), 1 Mbps (edge case)
    RTT fixed at 80ms (4G).

Total: 4 levels x 3 bandwidths x 50 problems = 600 agentic runs.

Usage:
    # Qwen 4B on SGLang (Mahti):
    python scripts/run_compression_experiment.py --node g1301 --problems 50

    # Quick smoke test:
    python scripts/run_compression_experiment.py --node g1301 --problems 3 --bandwidths 5
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from pocket_agent_cli.network.network_simulator import NetworkSimulator, NetworkConfig
from pocket_agent_cli.utils.tool_extractor import ToolExtractor
from pocket_agent_cli.utils.optimized_prompts import get_optimized_prompt
from pocket_agent_cli.datasets.registry import DatasetRegistry
from pocket_agent_cli.datasets import mbpp

import httpx

# ── Model config ─────────────────────────────────────────────────────────

MODEL_DEF = {
    "id": "qwen-3.5-4b",
    "name": "Qwen 3.5 4B",
    "arch": "qwen",
    "hf_id": "Qwen/Qwen3.5-4B",
    "local_port": 30006,
}

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

# ── Bandwidth conditions ─────────────────────────────────────────────────
# All use 4G RTT (80ms) — bandwidth is the variable that makes compression matter.

BANDWIDTH_CONFIGS = {
    "50mbps": NetworkConfig(
        name="50mbps", rtt_ms=80, jitter_ms=30,
        packet_loss_rate=0.005, bandwidth_mbps=50.0, radio_tail_energy_j=0.5,
    ),
    "5mbps": NetworkConfig(
        name="5mbps", rtt_ms=80, jitter_ms=30,
        packet_loss_rate=0.005, bandwidth_mbps=5.0, radio_tail_energy_j=0.5,
    ),
    "1mbps": NetworkConfig(
        name="1mbps", rtt_ms=80, jitter_ms=30,
        packet_loss_rate=0.005, bandwidth_mbps=1.0, radio_tail_energy_j=0.5,
    ),
}


# ── Context compression ─────────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    c = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return re.sub(r'<think>.*', '', c, flags=re.DOTALL).strip()


def compress_context(messages: List[Dict], level: int) -> List[Dict]:
    """Compress message context at the given level.

    Level 0: No compression (baseline).
    Level 1: Strip <think>...</think> from all previous assistant turns.
    Level 2: Strip thinking + keep only last 2 assistant/tool turns,
             summarize older turns.
    Level 3: Strip thinking + keep only last 1 turn + truncate tool
             results to 200 chars.

    Args:
        messages: Full message history.
        level: Compression level (0-3).

    Returns:
        Compressed copy of messages.
    """
    if level == 0:
        return [dict(m) for m in messages]

    result = []

    if level == 1:
        # Strip thinking from all previous assistant turns (not the last one
        # since we haven't added it yet; in practice, the last message in the
        # history being sent is always a user/tool message).
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant" and i < len(messages) - 1:
                content = re.sub(
                    r'<think>.*?</think>', '', msg["content"], flags=re.DOTALL
                )
                # Also strip unclosed thinking blocks
                content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
                result.append({**msg, "content": content.strip()})
            else:
                result.append(dict(msg))
        return result

    if level == 2:
        # Strip thinking + keep only last 2 assistant/tool turns.
        # Find system + first user message, then summarize old turns,
        # keep last 2 assistant+tool pairs.
        system_and_first_user = []
        conversation_turns = []

        for msg in messages:
            if msg["role"] in ("system",) and not conversation_turns:
                system_and_first_user.append(dict(msg))
            elif msg["role"] == "user" and not conversation_turns:
                system_and_first_user.append(dict(msg))
            else:
                conversation_turns.append(dict(msg))

        # Strip thinking from all conversation turns
        stripped_turns = []
        for msg in conversation_turns:
            if msg["role"] == "assistant":
                content = re.sub(
                    r'<think>.*?</think>', '', msg["content"], flags=re.DOTALL
                )
                content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
                stripped_turns.append({**msg, "content": content.strip()})
            else:
                stripped_turns.append(msg)

        # Count assistant turns to determine what to keep
        assistant_indices = [
            i for i, m in enumerate(stripped_turns) if m["role"] == "assistant"
        ]

        if len(assistant_indices) <= 2:
            # Few enough turns — keep all
            return system_and_first_user + stripped_turns

        # Keep last 2 assistant turns and everything after the cut point
        cut_idx = assistant_indices[-2]
        old_turns = stripped_turns[:cut_idx]
        kept_turns = stripped_turns[cut_idx:]

        # Summarize old turns
        n_attempts = sum(1 for m in old_turns if m["role"] == "assistant")
        last_error = ""
        for m in reversed(old_turns):
            if m["role"] in ("tool", "user") and "Error" in m.get("content", ""):
                last_error = m["content"][:100]
                break

        summary = f"[Previous: {n_attempts} attempts"
        if last_error:
            summary += f", last error: {last_error}"
        summary += "]"

        result = system_and_first_user
        result.append({"role": "user", "content": summary})
        result.extend(kept_turns)
        return result

    if level == 3:
        # Strip thinking + keep only last 1 turn + truncate tool results to 200 chars.
        system_and_first_user = []
        conversation_turns = []

        for msg in messages:
            if msg["role"] in ("system",) and not conversation_turns:
                system_and_first_user.append(dict(msg))
            elif msg["role"] == "user" and not conversation_turns:
                system_and_first_user.append(dict(msg))
            else:
                conversation_turns.append(dict(msg))

        # Strip thinking
        stripped_turns = []
        for msg in conversation_turns:
            if msg["role"] == "assistant":
                content = re.sub(
                    r'<think>.*?</think>', '', msg["content"], flags=re.DOTALL
                )
                content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
                stripped_turns.append({**msg, "content": content.strip()})
            elif msg["role"] in ("tool", "user") and msg.get("content"):
                # Truncate tool/user messages with tool results
                content = msg["content"]
                if len(content) > 200:
                    content = content[:200] + "...[truncated]"
                stripped_turns.append({**msg, "content": content})
            else:
                stripped_turns.append(msg)

        # Keep only last 1 assistant turn and everything after
        assistant_indices = [
            i for i, m in enumerate(stripped_turns) if m["role"] == "assistant"
        ]

        if len(assistant_indices) <= 1:
            return system_and_first_user + stripped_turns

        cut_idx = assistant_indices[-1]
        old_turns = stripped_turns[:cut_idx]
        kept_turns = stripped_turns[cut_idx:]

        n_attempts = sum(1 for m in old_turns if m["role"] == "assistant")
        last_error = ""
        for m in reversed(old_turns):
            if m["role"] in ("tool", "user") and "Error" in m.get("content", ""):
                last_error = m["content"][:80]
                break

        summary = f"[Previous: {n_attempts} attempts"
        if last_error:
            summary += f", last error: {last_error}"
        summary += "]"

        result = system_and_first_user
        result.append({"role": "user", "content": summary})
        result.extend(kept_turns)
        return result

    raise ValueError(f"Unknown compression level: {level}")


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for code/English mix."""
    return max(1, len(text) // 4)


# ── Helpers (reused from run_3arch_experiment.py) ────────────────────────

def extract_code(response, tool_calls=None):
    if tool_calls:
        for tc in (tool_calls if isinstance(tool_calls, list) else []):
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

    matches = re.findall(r'```python\s*(.*?)```', text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    match = re.search(
        r'(def \w+\([^)]*\):.*?)(?=\n(?:def |\n\n[A-Z]|\Z))', text, re.DOTALL
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
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=10,
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


def sglang_chat(base_url, model_hf_id, messages, tools=None,
                max_tokens=8192, temperature=0.7, retries=3):
    payload = {
        "model": model_hf_id, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
    for attempt in range(retries):
        try:
            resp = httpx.post(
                f"{base_url}/v1/chat/completions",
                json=payload, timeout=300,
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
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=10,
            )
            os.unlink(f.name)
        return (r.stdout[:300] or "(no output)") if r.returncode == 0 else f"Error:\n{r.stderr[:300]}"
    except subprocess.TimeoutExpired:
        return "Error: Timeout"
    except Exception as e:
        return f"Error: {e}"


def compute_network_delay_ms(payload_bytes: int, config: NetworkConfig,
                              seed: int = 42) -> float:
    """Compute simulated network delay WITHOUT sleeping.

    Uses the same math as NetworkSimulator._compute_transfer but returns
    immediately. This lets us run 600 experiment loops without spending
    hours on sleep().
    """
    import random
    rng = random.Random(seed)
    jitter = rng.gauss(0, config.jitter_ms) if config.jitter_ms > 0 else 0.0
    simulated_rtt_ms = max(0.0, config.rtt_ms + jitter)

    if config.bandwidth_mbps > 0:
        bytes_per_ms = config.bandwidth_mbps * 125.0
        bandwidth_delay_ms = payload_bytes / bytes_per_ms
    else:
        bandwidth_delay_ms = 0.0

    retransmit = rng.random() < config.packet_loss_rate
    retransmit_delay_ms = simulated_rtt_ms if retransmit else 0.0

    return simulated_rtt_ms + bandwidth_delay_ms + retransmit_delay_ms


# ── Core: run one problem with a specific compression level ──────────────

def run_hybrid_compressed(
    problem, compression_level: int, base_url: str,
    bandwidth_configs: Dict[str, NetworkConfig],
) -> Dict[str, Any]:
    """Run hybrid agentic loop with context compression.

    Performs the real inference (one set of API calls per problem/compression
    level), then computes simulated network times for each bandwidth condition.

    Returns per-problem result with per-iteration breakdown.
    """
    model_def = MODEL_DEF
    mode = "full_tool"
    prompt_config = get_optimized_prompt(model_def["id"], mode)
    messages = build_messages(problem, prompt_config)

    max_tokens = 8192  # Qwen thinking model
    max_iterations = 5

    t0 = time.time()
    total_inference_time = 0.0
    total_tokens = 0
    iterations = 0
    tool_calls_count = 0
    submitted_code = None

    iteration_data = []

    for iteration in range(max_iterations):
        iterations += 1

        # ── Compute upload sizes BEFORE and AFTER compression ──
        messages_original_json = json.dumps(messages)
        upload_bytes_uncompressed = len(messages_original_json.encode("utf-8"))

        compressed_messages = compress_context(messages, compression_level)
        compressed_json = json.dumps(compressed_messages)
        upload_bytes_compressed = len(compressed_json.encode("utf-8"))

        compression_ratio = (
            upload_bytes_compressed / upload_bytes_uncompressed
            if upload_bytes_uncompressed > 0 else 1.0
        )

        total_context_tokens_estimate = estimate_tokens(compressed_json)

        # ── Cloud inference (send COMPRESSED context) ──
        t_inf = time.time()
        resp = sglang_chat(
            base_url, model_def["hf_id"], compressed_messages,
            tools=TOOL_DEFS, max_tokens=max_tokens,
        )
        inference_time = time.time() - t_inf
        total_inference_time += inference_time

        if "error" in resp:
            # Record the failed iteration
            iter_record = {
                "iteration_number": iteration + 1,
                "upload_bytes_uncompressed": upload_bytes_uncompressed,
                "upload_bytes_compressed": upload_bytes_compressed,
                "compression_ratio": round(compression_ratio, 4),
                "download_bytes": 0,
                "inference_time_s": round(inference_time, 3),
                "total_context_tokens_estimate": total_context_tokens_estimate,
                "error": resp.get("error", "unknown"),
                "network_time_by_bandwidth": {},
            }
            # Compute network times for each bandwidth
            for bw_name, bw_config in bandwidth_configs.items():
                up_delay = compute_network_delay_ms(
                    upload_bytes_compressed, bw_config, seed=42 + iteration
                )
                iter_record["network_time_by_bandwidth"][bw_name] = {
                    "upload_ms": round(up_delay, 2),
                    "download_ms": 0,
                    "total_ms": round(up_delay, 2),
                }
            iteration_data.append(iter_record)
            break

        choice = resp["choices"][0]
        msg = choice["message"]
        content = msg.get("content", "") or ""
        api_tool_calls = msg.get("tool_calls")
        usage = resp.get("usage", {})
        total_tokens += usage.get("completion_tokens", 0)

        # ── Compute download size ──
        download_bytes = len(content.encode("utf-8"))
        if api_tool_calls:
            download_bytes += len(json.dumps(api_tool_calls).encode("utf-8"))

        # ── Compute simulated network times for each bandwidth ──
        network_times = {}
        for bw_name, bw_config in bandwidth_configs.items():
            up_delay = compute_network_delay_ms(
                upload_bytes_compressed, bw_config, seed=42 + iteration
            )
            down_delay = compute_network_delay_ms(
                download_bytes, bw_config, seed=43 + iteration
            )
            network_times[bw_name] = {
                "upload_ms": round(up_delay, 2),
                "download_ms": round(down_delay, 2),
                "total_ms": round(up_delay + down_delay, 2),
            }

        iter_record = {
            "iteration_number": iteration + 1,
            "upload_bytes_uncompressed": upload_bytes_uncompressed,
            "upload_bytes_compressed": upload_bytes_compressed,
            "compression_ratio": round(compression_ratio, 4),
            "download_bytes": download_bytes,
            "inference_time_s": round(inference_time, 3),
            "total_context_tokens_estimate": total_context_tokens_estimate,
            "network_time_by_bandwidth": network_times,
        }
        iteration_data.append(iter_record)

        # ── Parse tool calls ──
        tool_calls = api_tool_calls
        if not tool_calls and content:
            te = ToolExtractor()
            cleaned = strip_thinking(content) or content
            parsed, _ = te.extract_tools(cleaned)
            if parsed:
                tool_calls = [
                    {"function": {"name": tc.get("name", ""),
                                  "arguments": json.dumps(
                                      tc.get("parameters", tc.get("arguments", {}))
                                  )},
                     "id": f"parsed_{i}"}
                    for i, tc in enumerate(parsed)
                ]

        # Add assistant message to the FULL (uncompressed) history
        messages.append({"role": "assistant", "content": content})

        if not tool_calls:
            code = extract_code(content, api_tool_calls)
            if code:
                submitted_code = code
            break

        # ── Execute tools locally ──
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
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": obs,
                })

        if submitted_code:
            break

    elapsed = time.time() - t0

    # Fallback: try to extract code from last message
    if not submitted_code:
        for msg in reversed(messages):
            c = msg.get("content", "")
            submitted_code = extract_code(c)
            if submitted_code:
                break

    evaluation = evaluate_code(submitted_code, problem.test_cases)

    # Aggregate per-problem totals
    total_upload_uncompressed = sum(
        it["upload_bytes_uncompressed"] for it in iteration_data
    )
    total_upload_compressed = sum(
        it["upload_bytes_compressed"] for it in iteration_data
    )
    total_download = sum(it["download_bytes"] for it in iteration_data)

    # Aggregate network times per bandwidth
    total_network_time_by_bw = {}
    for bw_name in bandwidth_configs:
        total_ms = sum(
            it["network_time_by_bandwidth"].get(bw_name, {}).get("total_ms", 0)
            for it in iteration_data
        )
        total_network_time_by_bw[bw_name] = round(total_ms, 2)

    return {
        "problem_id": problem.task_id,
        "compression_level": compression_level,
        "passed": evaluation["passed"],
        "error": evaluation.get("error"),
        "iterations": iterations,
        "tool_calls": tool_calls_count,
        "elapsed_s": round(elapsed, 2),
        "inference_time_s": round(total_inference_time, 2),
        "total_tokens": total_tokens,
        "total_upload_bytes_uncompressed": total_upload_uncompressed,
        "total_upload_bytes_compressed": total_upload_compressed,
        "total_download_bytes": total_download,
        "total_network_time_by_bandwidth_ms": total_network_time_by_bw,
        "iteration_data": iteration_data,
    }


# ── Main experiment ──────────────────────────────────────────────────────

def run_experiment(args):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds_cls = DatasetRegistry.get("mbpp")
    from pocket_agent_cli.config import DATA_DIR
    ds = ds_cls(DATA_DIR)
    if not ds.is_downloaded():
        ds.download()
    problems = ds.load(split="test", limit=args.problems)

    base_url = f"http://localhost:{args.port}"

    # Parse bandwidth conditions
    bw_names = [f"{b}mbps" for b in args.bandwidths]
    selected_bw = {k: v for k, v in BANDWIDTH_CONFIGS.items() if k in bw_names}
    if not selected_bw:
        print(f"ERROR: No matching bandwidth configs for {args.bandwidths}")
        print(f"Available: {list(BANDWIDTH_CONFIGS.keys())}")
        sys.exit(1)

    compression_levels = args.levels

    # Verify server
    try:
        r = httpx.get(f"{base_url}/v1/models", timeout=10)
        r.raise_for_status()
        print(f"SGLang server OK at {base_url}")
    except Exception as e:
        print(f"ERROR: SGLang server not reachable at {base_url}: {e}")
        print("Start the server first (see docstring).")
        sys.exit(1)

    total_runs = len(compression_levels) * len(problems)
    print(f"\nCompression Experiment")
    print(f"  Problems:     {len(problems)}")
    print(f"  Levels:       {compression_levels}")
    print(f"  Bandwidths:   {list(selected_bw.keys())}")
    print(f"  Inference:    {total_runs} agentic loops")
    print(f"  Total combos: {total_runs * len(selected_bw)} (levels x bw x problems)")
    print(f"  Concurrency:  {args.concurrency}")
    print(f"  Output:       {out_dir}\n")

    all_results = []
    jsonl_path = out_dir / "compression_per_problem.jsonl"

    for level in compression_levels:
        print(f"\n{'=' * 70}")
        print(f"Compression Level {level}")
        print(f"{'=' * 70}")

        level_results = []

        def run_one(problem):
            return run_hybrid_compressed(
                problem, level, base_url, selected_bw,
            )

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(run_one, p): p for p in problems
            }
            passed = 0
            done_count = 0
            for future in as_completed(futures):
                result = future.result()
                level_results.append(result)
                all_results.append(result)
                done_count += 1
                if result["passed"]:
                    passed += 1

                # Append to JSONL incrementally
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(result, default=str) + "\n")

                if done_count % 10 == 0 or done_count == len(problems):
                    avg_ratio = 0
                    ratios = [
                        it["compression_ratio"]
                        for r in level_results
                        for it in r["iteration_data"]
                    ]
                    if ratios:
                        avg_ratio = sum(ratios) / len(ratios)
                    print(
                        f"  [{done_count}/{len(problems)}] "
                        f"pass={passed}/{done_count} "
                        f"avg_ratio={avg_ratio:.3f}"
                    )

        pass_rate = passed / len(problems) if problems else 0
        print(
            f"\n  Level {level}: {passed}/{len(problems)} = {pass_rate:.0%}"
        )

    # ── Generate summary files ───────────────────────────────────────────

    # 1. compression_results.json — per level x bandwidth
    summary = {}
    for level in compression_levels:
        level_results = [r for r in all_results if r["compression_level"] == level]
        n = len(level_results)
        if n == 0:
            continue
        passed = sum(1 for r in level_results if r["passed"])
        pass_rate = passed / n

        for bw_name in selected_bw:
            key = f"level_{level}_{bw_name}"

            avg_upload_compressed = (
                sum(r["total_upload_bytes_compressed"] for r in level_results) / n
            )
            avg_upload_uncompressed = (
                sum(r["total_upload_bytes_uncompressed"] for r in level_results) / n
            )
            avg_download = sum(r["total_download_bytes"] for r in level_results) / n
            avg_network_ms = (
                sum(
                    r["total_network_time_by_bandwidth_ms"].get(bw_name, 0)
                    for r in level_results
                ) / n
            )
            avg_iterations = sum(r["iterations"] for r in level_results) / n
            avg_tool_calls = sum(r["tool_calls"] for r in level_results) / n
            avg_inference_s = sum(r["inference_time_s"] for r in level_results) / n
            avg_elapsed_s = sum(r["elapsed_s"] for r in level_results) / n

            # Per-iteration average compression ratio
            all_ratios = [
                it["compression_ratio"]
                for r in level_results
                for it in r["iteration_data"]
            ]
            avg_compression_ratio = (
                sum(all_ratios) / len(all_ratios) if all_ratios else 1.0
            )

            summary[key] = {
                "compression_level": level,
                "bandwidth": bw_name,
                "pass_rate": round(pass_rate, 4),
                "passed": passed,
                "total": n,
                "avg_upload_kb_compressed": round(avg_upload_compressed / 1024, 2),
                "avg_upload_kb_uncompressed": round(avg_upload_uncompressed / 1024, 2),
                "avg_download_kb": round(avg_download / 1024, 2),
                "avg_network_ms": round(avg_network_ms, 2),
                "avg_compression_ratio": round(avg_compression_ratio, 4),
                "avg_iterations": round(avg_iterations, 2),
                "avg_tool_calls": round(avg_tool_calls, 2),
                "avg_inference_s": round(avg_inference_s, 2),
                "avg_elapsed_s": round(avg_elapsed_s, 2),
            }

    summary_path = out_dir / "compression_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")

    # 2. compression_figure_data.json — for paper figures
    figure_data = {
        "pareto_data": [],
        "context_growth_curves": {},
        "bandwidth_heatmap": [],
    }

    # Pareto: compression ratio vs accuracy at each bandwidth
    for level in compression_levels:
        level_results = [r for r in all_results if r["compression_level"] == level]
        n = len(level_results)
        if n == 0:
            continue
        passed = sum(1 for r in level_results if r["passed"])
        pass_rate = passed / n

        all_ratios = [
            it["compression_ratio"]
            for r in level_results
            for it in r["iteration_data"]
        ]
        avg_ratio = sum(all_ratios) / len(all_ratios) if all_ratios else 1.0

        for bw_name in selected_bw:
            avg_net_ms = (
                sum(
                    r["total_network_time_by_bandwidth_ms"].get(bw_name, 0)
                    for r in level_results
                ) / n
            )
            figure_data["pareto_data"].append({
                "compression_level": level,
                "bandwidth": bw_name,
                "pass_rate": round(pass_rate, 4),
                "avg_compression_ratio": round(avg_ratio, 4),
                "avg_network_ms": round(avg_net_ms, 2),
                "avg_upload_kb": round(
                    sum(r["total_upload_bytes_compressed"] for r in level_results)
                    / n / 1024, 2
                ),
            })

    # Context growth curves: per-iteration context size for each level
    for level in compression_levels:
        level_results = [r for r in all_results if r["compression_level"] == level]
        if not level_results:
            continue

        # Group iteration data by iteration number
        by_iter = defaultdict(list)
        for r in level_results:
            for it in r["iteration_data"]:
                by_iter[it["iteration_number"]].append(it)

        curve = []
        for iter_num in sorted(by_iter.keys()):
            items = by_iter[iter_num]
            avg_uncompressed = (
                sum(it["upload_bytes_uncompressed"] for it in items) / len(items)
            )
            avg_compressed = (
                sum(it["upload_bytes_compressed"] for it in items) / len(items)
            )
            avg_ratio = (
                sum(it["compression_ratio"] for it in items) / len(items)
            )
            curve.append({
                "iteration": iter_num,
                "avg_uncompressed_bytes": round(avg_uncompressed),
                "avg_compressed_bytes": round(avg_compressed),
                "avg_compression_ratio": round(avg_ratio, 4),
                "n_samples": len(items),
            })

        figure_data["context_growth_curves"][f"level_{level}"] = curve

    # Bandwidth x compression heatmap
    for level in compression_levels:
        for bw_name in selected_bw:
            key = f"level_{level}_{bw_name}"
            if key in summary:
                figure_data["bandwidth_heatmap"].append({
                    "compression_level": level,
                    "bandwidth": bw_name,
                    "pass_rate": summary[key]["pass_rate"],
                    "avg_network_ms": summary[key]["avg_network_ms"],
                    "avg_upload_kb": summary[key]["avg_upload_kb_compressed"],
                })

    figure_path = out_dir / "compression_figure_data.json"
    with open(figure_path, "w") as f:
        json.dump(figure_data, f, indent=2)
    print(f"Figure data: {figure_path}")

    # ── Print summary table ──────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"COMPRESSION EXPERIMENT SUMMARY")
    print(f"{'=' * 100}")

    header = (
        f"{'Level':>5} {'BW':>8} {'Pass%':>7} {'UploadKB':>10} "
        f"{'Ratio':>7} {'NetMS':>8} {'Iters':>6} {'InfS':>6}"
    )
    print(header)
    print("-" * 100)

    for level in compression_levels:
        for bw_name in selected_bw:
            key = f"level_{level}_{bw_name}"
            if key not in summary:
                continue
            s = summary[key]
            print(
                f"{level:>5} {bw_name:>8} {s['pass_rate']:>6.0%} "
                f"{s['avg_upload_kb_compressed']:>9.1f} "
                f"{s['avg_compression_ratio']:>6.2f} "
                f"{s['avg_network_ms']:>7.1f} "
                f"{s['avg_iterations']:>5.1f} "
                f"{s['avg_inference_s']:>5.1f}"
            )
        if level < max(compression_levels):
            print()

    # ── Key findings ─────────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("KEY FINDINGS")
    print(f"{'=' * 100}")

    # Compare level 0 vs level 1 at each bandwidth
    for bw_name in selected_bw:
        k0 = f"level_0_{bw_name}"
        k1 = f"level_1_{bw_name}"
        if k0 in summary and k1 in summary:
            s0, s1 = summary[k0], summary[k1]
            bw_savings = (1 - s1["avg_compression_ratio"]) * 100
            acc_diff = (s1["pass_rate"] - s0["pass_rate"]) * 100
            net_reduction = (
                (s0["avg_network_ms"] - s1["avg_network_ms"])
                / s0["avg_network_ms"] * 100
                if s0["avg_network_ms"] > 0 else 0
            )
            print(
                f"\n  [{bw_name}] Level 1 (strip thinking) vs Level 0 (baseline):"
            )
            print(
                f"    Upload reduction: {bw_savings:.0f}%  "
                f"({s0['avg_upload_kb_compressed']:.1f}KB -> "
                f"{s1['avg_upload_kb_compressed']:.1f}KB)"
            )
            print(f"    Accuracy change:  {acc_diff:+.1f}pp")
            print(
                f"    Network time:     {s0['avg_network_ms']:.0f}ms -> "
                f"{s1['avg_network_ms']:.0f}ms ({net_reduction:.0f}% reduction)"
            )

    # Config used
    config_path = out_dir / "experiment_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_DEF,
            "problems": args.problems,
            "compression_levels": compression_levels,
            "bandwidths": list(selected_bw.keys()),
            "concurrency": args.concurrency,
            "port": args.port,
        }, f, indent=2)

    print(f"\nAll results: {out_dir}")
    print(f"  compression_results.json       — summary per level x bandwidth")
    print(f"  compression_per_problem.jsonl   — per-problem with iterations")
    print(f"  compression_figure_data.json    — for paper figures")
    print(f"  experiment_config.json          — experiment parameters")


def main():
    parser = argparse.ArgumentParser(
        description="Context Compression Experiment for Hybrid Deployment"
    )
    parser.add_argument("--problems", type=int, default=50)
    parser.add_argument("--port", type=int, default=30001,
                        help="SGLang server port (default: 30001)")
    parser.add_argument("--node", type=str, default=None,
                        help="Mahti node name (for reference only)")
    parser.add_argument("--levels", type=int, nargs="*", default=[0, 1, 2, 3],
                        help="Compression levels to test (default: 0 1 2 3)")
    parser.add_argument("--bandwidths", type=int, nargs="*", default=[50, 5, 1],
                        help="Bandwidth conditions in Mbps (default: 50 5 1)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Parallel agentic loops per compression level")
    parser.add_argument("--output-dir", default="data/results/compression_experiment",
                        help="Output directory")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
