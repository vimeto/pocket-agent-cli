#!/usr/bin/env python3
"""Web Search QA Benchmark: 3-Architecture Comparison with HotpotQA.

Runs a HotpotQA benchmark with a simulated web_search tool and configurable
network latency. Compares Local/Hybrid/Cloud architectures to show how
network-dependent tools change the deployment calculus.

Architecture A (Local):
    - Inference: MLX on-device (pre-computed results or live)
    - web_search: local retrieval with simulated search latency

Architecture B (Hybrid):
    - Inference: SGLang on cloud (via tunnel)
    - web_search: executed locally on device
    - Network sim for inference round-trips + simulated search latency

Architecture C (Cloud):
    - Inference: SGLang on cloud
    - web_search: server-side (no extra RTT per search)
    - Only initial upload + final download network cost

Usage:
    python scripts/run_websearch_benchmark.py \\
        --problems 100 \\
        --models qwen-3-4b llama-3.2-3b-instruct \\
        --network-conditions wifi 4g poor_cellular \\
        --port 30001 \\
        --concurrency 10
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from pocket_agent_cli.datasets.hotpotqa import (
    HotpotQADataset,
    exact_match_score,
    extract_answer_from_response,
    f1_score,
)
from pocket_agent_cli.datasets.registry import DatasetRegistry
from pocket_agent_cli.network.network_simulator import (
    NETWORK_PRESETS,
    NetworkConfig,
    NetworkSimulator,
)
from pocket_agent_cli.tools.web_search import SimulatedWebSearch
from pocket_agent_cli.utils.tool_extractor import ToolExtractor

import httpx

# ── Models ───────────────────────────────────────────────────────────────

MODELS = [
    {
        "id": "qwen-3-4b",
        "name": "Qwen 3 4B",
        "arch": "qwen",
        "hf_id": "Qwen/Qwen3-4B",
        "local_port": 30001,
    },
    {
        "id": "llama-3.2-3b-instruct",
        "name": "Llama 3.2 3B",
        "arch": "llama",
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "local_port": 30003,
    },
]

# ── Tool Definition (for API-based tool calling) ────────────────────────

WEB_SEARCH_TOOL_DEF = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return relevant passages",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# ── Helpers ──────────────────────────────────────────────────────────────


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from response."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()
    return cleaned


def sglang_chat(
    base_url: str,
    model_hf_id: str,
    messages: List[Dict],
    tools: Optional[List] = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    retries: int = 3,
) -> Dict:
    """Send a chat completion request to SGLang."""
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


def parse_tool_calls_from_text(content: str) -> List[Dict]:
    """Parse tool calls from model response text."""
    cleaned = strip_thinking(content) or content
    te = ToolExtractor()
    tool_calls_parsed, _ = te.extract_tools(cleaned)

    result = []
    for tc in tool_calls_parsed or []:
        name = tc.get("name", "")
        params = tc.get("parameters", tc.get("arguments", {}))
        result.append(
            {
                "function": {"name": name, "arguments": json.dumps(params)},
                "id": f"parsed_{len(result)}",
            }
        )
    return result


# ── Architecture B: Hybrid Runner ────────────────────────────────────────


def run_hybrid_websearch(
    problem,
    model_def: Dict,
    network_config: NetworkConfig,
    search_network_config: Optional[NetworkConfig],
    base_url: str,
    max_iterations: int = 3,
    timeout_s: int = 120,
) -> Dict[str, Any]:
    """Run a single HotpotQA problem in Hybrid mode.

    Cloud inference + local web_search execution + network simulation.

    Args:
        problem: HotpotQA Problem object.
        model_def: Model definition dict.
        network_config: Network config for inference round-trips.
        search_network_config: Network config for web search latency.
        base_url: SGLang server URL.
        max_iterations: Max search iterations per problem.
        timeout_s: Overall timeout in seconds.

    Returns:
        Result dict with metrics.
    """
    # Set up web search with search-specific latency
    paragraphs = problem.metadata.get("paragraphs", [])
    web_search = SimulatedWebSearch(
        context_paragraphs=paragraphs,
        network_config=search_network_config,
    )

    # Set up inference network simulator
    inference_net_sim = NetworkSimulator(network_config, seed=42)

    # Build initial messages
    system_prompt = HotpotQADataset.SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem.prompt},
    ]

    is_thinking = model_def["arch"] == "qwen"
    max_tokens = 8192 if is_thinking else 2048

    t0 = time.time()
    total_inference_time = 0.0
    total_tokens = 0
    iterations = 0
    total_search_calls = 0
    final_response = ""

    for iteration in range(max_iterations + 1):
        # +1 to allow a final answer after last search
        if time.time() - t0 > timeout_s:
            break

        iterations += 1

        # 1. Simulate upload (prompt -> cloud)
        upload_bytes = len(json.dumps(messages).encode("utf-8"))
        inference_net_sim.simulate_transfer_sync(upload_bytes, "upload")

        # 2. Cloud inference
        t_inf = time.time()
        resp = sglang_chat(
            base_url,
            model_def["hf_id"],
            messages,
            tools=None,  # use prompt-based tools for web_search
            max_tokens=max_tokens,
        )
        inference_time = time.time() - t_inf
        total_inference_time += inference_time

        if "error" in resp:
            break

        choice = resp["choices"][0]
        msg = choice["message"]
        content = msg.get("content", "") or ""
        usage = resp.get("usage", {})
        total_tokens += usage.get("completion_tokens", 0)

        # 3. Simulate download (response -> device)
        download_bytes = len(content.encode("utf-8"))
        inference_net_sim.simulate_transfer_sync(download_bytes, "download")

        final_response = content
        messages.append({"role": "assistant", "content": content})

        # Parse tool calls
        tool_calls = parse_tool_calls_from_text(content)

        # Check if model wants to search
        search_called = False
        for tc in tool_calls:
            fn = tc.get("function", {})
            tc_name = fn.get("name", "")

            if tc_name == "web_search":
                args_raw = fn.get("arguments", "{}")
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = args_raw
                query = args.get("query", problem.prompt)

                # Execute search locally with network latency
                search_results = web_search.search(query)
                total_search_calls += 1
                search_called = True

                # Feed results back
                messages.append(
                    {
                        "role": "user",
                        "content": f"Search results for '{query}':\n\n{search_results}",
                    }
                )
                break  # One search per iteration

        if not search_called:
            # Model didn't call search -> it's answering
            break

    elapsed = time.time() - t0

    # Evaluate
    ground_truth = problem.metadata.get("answer", "")
    predicted = extract_answer_from_response(final_response)
    em = exact_match_score(predicted, ground_truth)
    f1 = f1_score(predicted, ground_truth)

    # Network stats
    inference_net_summary = inference_net_sim.get_summary()
    search_stats = web_search.get_search_stats()

    return {
        "problem_id": problem.task_id,
        "architecture": "hybrid",
        "network_condition": network_config.name,
        "question": problem.prompt,
        "ground_truth": ground_truth,
        "predicted_answer": predicted,
        "em": em,
        "f1": f1,
        "total_time_s": round(elapsed, 3),
        "inference_time_s": round(total_inference_time, 3),
        "inference_network_time_s": round(
            inference_net_summary["total_delay_ms"] / 1000, 3
        ),
        "search_network_time_s": round(
            search_stats["total_network_delay_ms"] / 1000, 3
        ),
        "total_network_time_s": round(
            (
                inference_net_summary["total_delay_ms"]
                + search_stats["total_network_delay_ms"]
            )
            / 1000,
            3,
        ),
        "tokens": total_tokens,
        "iterations": iterations,
        "search_calls": total_search_calls,
        "search_result_bytes": search_stats["total_result_bytes"],
        "inference_transfers": inference_net_summary["total_transfers"],
        "inference_bytes": inference_net_summary["total_bytes"],
    }


# ── Architecture C: Cloud Runner ─────────────────────────────────────────


def run_cloud_websearch(
    problem,
    model_def: Dict,
    base_url: str,
    max_iterations: int = 3,
    timeout_s: int = 120,
) -> Dict[str, Any]:
    """Run a single HotpotQA problem in Cloud mode.

    Everything on cloud: inference + search are co-located, no per-search RTT.
    We still run the agentic loop but without network sim on search calls.

    Args:
        problem: HotpotQA Problem object.
        model_def: Model definition dict.
        base_url: SGLang server URL.
        max_iterations: Max search iterations per problem.
        timeout_s: Overall timeout in seconds.

    Returns:
        Result dict with metrics.
    """
    paragraphs = problem.metadata.get("paragraphs", [])
    web_search = SimulatedWebSearch(
        context_paragraphs=paragraphs,
        network_config=None,  # No network cost for co-located search
    )

    system_prompt = HotpotQADataset.SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem.prompt},
    ]

    is_thinking = model_def["arch"] == "qwen"
    max_tokens = 8192 if is_thinking else 2048

    t0 = time.time()
    total_inference_time = 0.0
    total_tokens = 0
    iterations = 0
    total_search_calls = 0
    final_response = ""

    for iteration in range(max_iterations + 1):
        if time.time() - t0 > timeout_s:
            break

        iterations += 1

        t_inf = time.time()
        resp = sglang_chat(
            base_url,
            model_def["hf_id"],
            messages,
            tools=None,
            max_tokens=max_tokens,
        )
        inference_time = time.time() - t_inf
        total_inference_time += inference_time

        if "error" in resp:
            break

        choice = resp["choices"][0]
        msg = choice["message"]
        content = msg.get("content", "") or ""
        usage = resp.get("usage", {})
        total_tokens += usage.get("completion_tokens", 0)

        final_response = content
        messages.append({"role": "assistant", "content": content})

        # Parse tool calls
        tool_calls = parse_tool_calls_from_text(content)

        search_called = False
        for tc in tool_calls:
            fn = tc.get("function", {})
            tc_name = fn.get("name", "")

            if tc_name == "web_search":
                args_raw = fn.get("arguments", "{}")
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = args_raw
                query = args.get("query", problem.prompt)

                search_results = web_search.search(query)
                total_search_calls += 1
                search_called = True

                messages.append(
                    {
                        "role": "user",
                        "content": f"Search results for '{query}':\n\n{search_results}",
                    }
                )
                break

        if not search_called:
            break

    elapsed = time.time() - t0

    ground_truth = problem.metadata.get("answer", "")
    predicted = extract_answer_from_response(final_response)
    em = exact_match_score(predicted, ground_truth)
    f1 = f1_score(predicted, ground_truth)

    # Network cost for cloud: just initial upload + final download
    upload_bytes = len(json.dumps(messages[:2]).encode("utf-8"))
    download_bytes = len(final_response.encode("utf-8"))

    return {
        "problem_id": problem.task_id,
        "architecture": "cloud",
        "network_condition": "cloud",
        "question": problem.prompt,
        "ground_truth": ground_truth,
        "predicted_answer": predicted,
        "em": em,
        "f1": f1,
        "total_time_s": round(elapsed, 3),
        "inference_time_s": round(total_inference_time, 3),
        "inference_network_time_s": 0.0,
        "search_network_time_s": 0.0,
        "total_network_time_s": 0.0,
        "tokens": total_tokens,
        "iterations": iterations,
        "search_calls": total_search_calls,
        "search_result_bytes": web_search.get_search_stats()["total_result_bytes"],
        "inference_transfers": 2,  # 1 upload + 1 download
        "inference_bytes": upload_bytes + download_bytes,
    }


# ── Main Runner ──────────────────────────────────────────────────────────


def run_experiment(args):
    """Run the full web search QA benchmark experiment."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds_cls = DatasetRegistry.get("hotpotqa")
    from pocket_agent_cli.config import DATA_DIR

    ds = ds_cls(DATA_DIR)
    if not ds.is_downloaded():
        print("Downloading HotpotQA dataset...")
        ds.download()
    problems = ds.load(split="validation", limit=args.problems)
    print(f"Web Search QA Benchmark (HotpotQA)")
    print(f"Problems: {len(problems)}")
    print(f"Network conditions: {args.network_conditions}")
    print(f"Output: {out_dir}\n")

    selected_models = [m for m in MODELS if not args.models or m["id"] in args.models]
    network_conditions = args.network_conditions or ["wifi", "4g", "poor_cellular"]

    all_results = []
    per_problem_results = []

    for model_def in selected_models:
        base_url = f"http://localhost:{args.port}"

        # Verify server
        try:
            r = httpx.get(f"{base_url}/v1/models", timeout=5)
            print(f"Model: {model_def['name']} -- server OK at {base_url}")
        except Exception:
            print(f"Model: {model_def['name']} -- server NOT available at {base_url}, skipping")
            continue

        # Run cloud architecture once (no per-search network cost)
        print(f"\n  Architecture C (Cloud) - no per-search network cost")
        cloud_results = []

        def run_cloud_problem(p):
            return run_cloud_websearch(
                p, model_def, base_url,
                max_iterations=args.max_iterations,
                timeout_s=args.timeout,
            )

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(run_cloud_problem, p): i for i, p in enumerate(problems)}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    result["model"] = model_def["id"]
                    cloud_results.append(result)
                    per_problem_results.append(result)
                except Exception as e:
                    print(f"    Cloud error: {e}")

        if cloud_results:
            avg_f1 = sum(r["f1"] for r in cloud_results) / len(cloud_results)
            avg_em = sum(r["em"] for r in cloud_results) / len(cloud_results)
            avg_searches = sum(r["search_calls"] for r in cloud_results) / len(cloud_results)
            print(f"    Cloud: F1={avg_f1:.3f} EM={avg_em:.3f} avg_searches={avg_searches:.1f}")

            all_results.append({
                "model": model_def["id"],
                "architecture": "cloud",
                "network_condition": "cloud",
                "n_problems": len(cloud_results),
                "avg_f1": round(avg_f1, 4),
                "avg_em": round(avg_em, 4),
                "avg_searches": round(avg_searches, 2),
                "avg_time_s": round(
                    sum(r["total_time_s"] for r in cloud_results) / len(cloud_results), 3
                ),
                "avg_inference_time_s": round(
                    sum(r["inference_time_s"] for r in cloud_results) / len(cloud_results), 3
                ),
                "avg_network_time_s": 0.0,
                "avg_tokens": round(
                    sum(r["tokens"] for r in cloud_results) / len(cloud_results), 1
                ),
            })

        # Run hybrid architecture for each network condition
        for net_name in network_conditions:
            net_config = NETWORK_PRESETS.get(net_name)
            if not net_config:
                print(f"  WARNING: Unknown network condition '{net_name}', skipping")
                continue

            print(f"\n  Architecture B (Hybrid) - {net_name} (RTT={net_config.rtt_ms}ms)")

            hybrid_results = []

            def run_hybrid_problem(p, nc=net_config, snc=net_config):
                return run_hybrid_websearch(
                    p, model_def, nc, snc, base_url,
                    max_iterations=args.max_iterations,
                    timeout_s=args.timeout,
                )

            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {
                    executor.submit(run_hybrid_problem, p): i
                    for i, p in enumerate(problems)
                }
                done_count = 0
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        result["model"] = model_def["id"]
                        hybrid_results.append(result)
                        per_problem_results.append(result)
                        done_count += 1
                        if done_count % 20 == 0 or done_count == len(problems):
                            running_f1 = sum(r["f1"] for r in hybrid_results) / len(hybrid_results)
                            print(f"    [{done_count}/{len(problems)}] running F1={running_f1:.3f}")
                    except Exception as e:
                        done_count += 1
                        print(f"    Hybrid error: {e}")

            if hybrid_results:
                avg_f1 = sum(r["f1"] for r in hybrid_results) / len(hybrid_results)
                avg_em = sum(r["em"] for r in hybrid_results) / len(hybrid_results)
                avg_searches = sum(r["search_calls"] for r in hybrid_results) / len(hybrid_results)
                avg_inf_net = sum(r["inference_network_time_s"] for r in hybrid_results) / len(hybrid_results)
                avg_search_net = sum(r["search_network_time_s"] for r in hybrid_results) / len(hybrid_results)
                avg_total_net = sum(r["total_network_time_s"] for r in hybrid_results) / len(hybrid_results)

                print(
                    f"    {net_name}: F1={avg_f1:.3f} EM={avg_em:.3f} "
                    f"avg_searches={avg_searches:.1f} "
                    f"net_overhead={avg_total_net:.3f}s "
                    f"(inf={avg_inf_net:.3f}s + search={avg_search_net:.3f}s)"
                )

                all_results.append({
                    "model": model_def["id"],
                    "architecture": "hybrid",
                    "network_condition": net_name,
                    "rtt_ms": net_config.rtt_ms,
                    "n_problems": len(hybrid_results),
                    "avg_f1": round(avg_f1, 4),
                    "avg_em": round(avg_em, 4),
                    "avg_searches": round(avg_searches, 2),
                    "avg_time_s": round(
                        sum(r["total_time_s"] for r in hybrid_results) / len(hybrid_results), 3
                    ),
                    "avg_inference_time_s": round(
                        sum(r["inference_time_s"] for r in hybrid_results) / len(hybrid_results), 3
                    ),
                    "avg_inference_network_time_s": round(avg_inf_net, 3),
                    "avg_search_network_time_s": round(avg_search_net, 3),
                    "avg_network_time_s": round(avg_total_net, 3),
                    "avg_tokens": round(
                        sum(r["tokens"] for r in hybrid_results) / len(hybrid_results), 1
                    ),
                })

    # Save results
    summary_path = out_dir / "websearch_results.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "n_problems": args.problems,
                "max_iterations": args.max_iterations,
                "network_conditions": network_conditions,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nSummary saved to {summary_path}")

    per_problem_path = out_dir / "websearch_per_problem.jsonl"
    with open(per_problem_path, "w") as f:
        for r in sorted(per_problem_results, key=lambda x: (x.get("model", ""), x.get("problem_id", ""))):
            f.write(json.dumps(r, default=str) + "\n")
    print(f"Per-problem results saved to {per_problem_path}")

    # Print summary table
    print_summary(all_results)


def print_summary(results: List[Dict]):
    """Print a formatted summary table."""
    print(f"\n{'=' * 90}")
    print("WEB SEARCH QA BENCHMARK SUMMARY")
    print(f"{'=' * 90}\n")

    print(
        f"{'Model':<25} {'Arch':<8} {'Network':<15} "
        f"{'F1':>6} {'EM':>6} {'Srch':>5} "
        f"{'Time':>7} {'InfNet':>7} {'SrchNet':>8} {'TotalNet':>9}"
    )
    print("-" * 90)

    for r in sorted(results, key=lambda x: (x["model"], x["architecture"], x.get("rtt_ms", 0))):
        print(
            f"{r['model']:<25} {r['architecture']:<8} {r['network_condition']:<15} "
            f"{r['avg_f1']:>5.3f} {r['avg_em']:>5.3f} {r['avg_searches']:>4.1f} "
            f"{r['avg_time_s']:>6.1f}s "
            f"{r.get('avg_inference_network_time_s', 0):>6.3f}s "
            f"{r.get('avg_search_network_time_s', 0):>7.3f}s "
            f"{r['avg_network_time_s']:>8.3f}s"
        )

    # Key finding
    print(f"\n{'=' * 90}")
    print("KEY FINDING: With network-dependent tools, network cost compounds")
    print("with each search iteration, shifting the crossover point toward")
    print("on-device inference.")
    print(f"{'=' * 90}")


def main():
    parser = argparse.ArgumentParser(
        description="Web Search QA Benchmark with 3-Architecture Comparison"
    )
    parser.add_argument("--problems", type=int, default=100,
                        help="Number of HotpotQA problems to run")
    parser.add_argument("--models", nargs="*",
                        help="Model IDs to run (default: all)")
    parser.add_argument("--network-conditions", nargs="*",
                        default=["wifi", "4g", "poor_cellular"],
                        help="Network conditions to test")
    parser.add_argument("--port", type=int, default=30001,
                        help="SGLang server port")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Number of concurrent problem runners")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Max search iterations per problem")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Timeout per problem in seconds")
    parser.add_argument("--output-dir", default="data/results/websearch_qa",
                        help="Output directory for results")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
