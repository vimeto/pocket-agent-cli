#!/usr/bin/env python3
"""BFCL (Berkeley Function Calling Leaderboard) benchmark runner.

Evaluates tool-calling proficiency of on-device models via SGLang servers.

Usage:
    # Single model (server must be running):
    python scripts/run_bfcl_benchmark.py \\
        --models qwen-3-4b \\
        --port 30001 \\
        --concurrency 30

    # All models:
    python scripts/run_bfcl_benchmark.py \\
        --port 30001 \\
        --concurrency 30

    # Specific categories:
    python scripts/run_bfcl_benchmark.py \\
        --categories simple multiple \\
        --limit 50
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

from pocket_agent_cli.datasets.bfcl import BFCLDataset
from pocket_agent_cli.evaluation.bfcl_eval import (
    aggregate_results,
    evaluate_example,
)
from pocket_agent_cli.utils.tool_extractor import ToolExtractor

# ---------------------------------------------------------------------------
# Model definitions (mirrors run_benchmarks_sglang.py)
# ---------------------------------------------------------------------------

MODELS = [
    {"id": "qwen-3-4b", "name": "Qwen 3 4B", "arch": "qwen",
     "hf_id": "Qwen/Qwen3-4B", "local_port": 30001},
    {"id": "qwen-3-0.6b", "name": "Qwen 3 0.6B", "arch": "qwen",
     "hf_id": "Qwen/Qwen3-0.6B", "local_port": 30002},
    {"id": "llama-3.2-3b-instruct", "name": "Llama 3.2 3B", "arch": "llama",
     "hf_id": "meta-llama/Llama-3.2-3B-Instruct", "local_port": 30003,
     "no_api_tools": True},
    {"id": "deepseek-r1-distill-qwen-1.5b", "name": "DeepSeek R1 1.5B",
     "arch": "qwen", "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
     "local_port": 30004, "no_api_tools": True, "max_tokens": 2048},
    {"id": "gemma-3n-e2b-it", "name": "Gemma 3n E2B", "arch": "gemma",
     "hf_id": "google/gemma-3n-E2B-it", "local_port": 30005,
     "no_api_tools": True},
    {"id": "qwen-3.5-4b", "name": "Qwen 3.5 4B", "arch": "qwen",
     "hf_id": "Qwen/Qwen3.5-4B", "local_port": 30006},
    {"id": "gemma-4-e2b-it", "name": "Gemma 4 E2B", "arch": "gemma",
     "hf_id": "google/gemma-4-E2B-it", "local_port": 30007,
     "max_tokens": 3072},
]


# ---------------------------------------------------------------------------
# Chat helper
# ---------------------------------------------------------------------------

def sglang_chat(
    base_url: str,
    model_hf_id: str,
    messages: List[Dict],
    tools: Optional[List] = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> Dict:
    """Send a chat completion request to an SGLang or OpenAI-compatible server."""
    import httpx

    payload: Dict[str, Any] = {
        "model": model_hf_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools

    try:
        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)[:300]}


# ---------------------------------------------------------------------------
# Build messages for an example
# ---------------------------------------------------------------------------

def _functions_to_prompt_text(functions: List[Dict]) -> str:
    """Convert OpenAI tool definitions into a textual description for
    models that cannot use the API tools parameter."""
    parts = ["You have access to the following functions:\n"]
    for tool_def in functions:
        fn = tool_def.get("function", tool_def)
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        parts.append(f"- {name}: {desc}")
        props = params.get("properties", {})
        req = set(params.get("required", []))
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "any")
            pdesc = pinfo.get("description", "")
            opt = "" if pname in req else " (optional)"
            parts.append(f"    {pname} ({ptype}{opt}): {pdesc}")
    parts.append(
        "\nTo call a function, respond with a JSON object: "
        '{"name": "<function>", "arguments": {<args>}}'
    )
    parts.append(
        "If no function is relevant to the request, respond normally "
        "WITHOUT calling any function."
    )
    return "\n".join(parts)


def build_messages(
    example: Dict[str, Any],
    model_def: Dict,
) -> tuple:
    """Build chat messages and optional tools parameter.

    Returns (messages, tools_or_none).
    """
    prompt = example["prompt"]
    functions = example["functions"]
    no_api = model_def.get("no_api_tools", False)

    if no_api:
        # Embed function definitions in system prompt
        system = _functions_to_prompt_text(functions)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return messages, None
    else:
        system = (
            "You are a helpful assistant. When the user's request matches "
            "an available function, call it with the correct arguments. "
            "If no function is relevant, respond normally without calling any."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return messages, functions


# ---------------------------------------------------------------------------
# Parse tool calls from response
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks."""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL).strip()
    return cleaned


def extract_tool_calls(
    response_json: Dict,
    model_arch: str = "qwen",
) -> List[Dict[str, Any]]:
    """Extract tool calls from a chat completion response.

    Checks both API-level tool_calls and text-based extraction.
    Returns list of {"name": ..., "arguments": {...}}.
    """
    calls = []

    choice = response_json.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content", "") or ""
    api_tool_calls = msg.get("tool_calls")

    # 1. API-level tool calls
    if api_tool_calls:
        for tc in api_tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args_raw = fn.get("arguments", "{}")
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = args_raw
            calls.append({"name": name, "arguments": args})

    # 2. Text-based extraction (always try — models may embed calls in text)
    if content and not calls:
        cleaned = _strip_thinking(content) or content
        te = ToolExtractor()
        parsed, _ = te.extract_tools(cleaned, model_architecture=model_arch)
        for tc in parsed:
            name = tc.get("name", "")
            args = tc.get("parameters", tc.get("arguments", {}))
            # Skip Python code submission tools — not relevant for BFCL
            if name in ("run_python_code", "submit_python_solution"):
                continue
            calls.append({"name": name, "arguments": args})

    return calls


# ---------------------------------------------------------------------------
# Run one example
# ---------------------------------------------------------------------------

def run_single_example(
    example: Dict[str, Any],
    model_def: Dict,
    base_url: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Send one BFCL example to the model and evaluate the response."""
    messages, tools = build_messages(example, model_def)

    t0 = time.time()
    resp = sglang_chat(
        base_url,
        model_def["hf_id"],
        messages,
        tools=tools,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    elapsed = time.time() - t0

    if "error" in resp:
        return {
            "id": example["id"],
            "category": example["category"],
            "score": "no_match",
            "full_match": False,
            "partial_match": False,
            "no_match": True,
            "error": resp["error"],
            "elapsed_s": round(elapsed, 2),
        }

    # Parse tool calls from response
    actual_calls = extract_tool_calls(resp, model_arch=model_def.get("arch", "qwen"))

    # Evaluate
    eval_result = evaluate_example(
        actual_calls,
        example["expected"],
        category=example["category"],
    )

    usage = resp.get("usage", {})
    choice = resp.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content", "") or ""
    api_tool_calls = msg.get("tool_calls")

    # Debug: log raw response details for empty content
    raw_debug = ""
    if not content and not actual_calls:
        raw_debug = json.dumps({
            "content_type": str(type(msg.get("content"))),
            "content_raw": str(msg.get("content"))[:200],
            "tool_calls_raw": str(api_tool_calls)[:200] if api_tool_calls else None,
            "msg_keys": list(msg.keys()),
            "finish_reason": choice.get("finish_reason"),
        })

    return {
        "id": example["id"],
        "category": example["category"],
        "score": eval_result["score"],
        "full_match": eval_result["full_match"],
        "partial_match": eval_result["partial_match"],
        "no_match": eval_result["no_match"],
        "expected_count": eval_result["expected_count"],
        "actual_count": eval_result["actual_count"],
        "actual_calls": actual_calls,
        "expected_calls": example["expected"],
        "call_results": eval_result.get("call_results", []),
        "response_preview": content[:300],
        "raw_debug": raw_debug,
        "tokens": usage.get("completion_tokens", 0),
        "elapsed_s": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Run all examples for one model
# ---------------------------------------------------------------------------

def run_model_bfcl(
    model_def: Dict,
    base_url: str,
    examples: List[Dict[str, Any]],
    concurrency: int = 30,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    """Run BFCL examples for a single model with concurrent requests."""
    results = []
    full = 0

    def _run(ex):
        return run_single_example(ex, model_def, base_url, max_tokens, temperature)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_run, ex): ex["id"] for ex in examples}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                eid = futures[future]
                result = {
                    "id": eid,
                    "category": "unknown",
                    "score": "no_match",
                    "full_match": False,
                    "partial_match": False,
                    "no_match": True,
                    "error": str(e)[:200],
                }
            results.append(result)
            if result.get("full_match"):
                full += 1
            done = len(results)
            if done % 10 == 0 or done == len(examples):
                print(f"    [{done}/{len(examples)}] {full} full_match so far")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BFCL benchmark runner")
    parser.add_argument("--models", nargs="*", help="Model IDs to evaluate")
    parser.add_argument("--port", type=int, default=30001,
                        help="Server port (used when running one model)")
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--categories", nargs="*",
                        default=["simple", "multiple", "parallel", "relevance"],
                        help="BFCL categories to evaluate")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max examples per category (None = all)")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", default="data/results/bfcl")
    args = parser.parse_args()

    # Load dataset
    data_dir = Path("data")
    dataset = BFCLDataset(data_dir)
    if not dataset.is_downloaded():
        dataset.download()

    examples = dataset.load_raw(
        split="test",
        categories=args.categories,
        limit=args.limit,
    )
    print(f"Loaded {len(examples)} BFCL examples")
    for cat in args.categories:
        n = sum(1 for e in examples if e["category"] == cat)
        print(f"  {cat}: {n}")

    # Select models
    selected = [m for m in MODELS if not args.models or m["id"] in args.models]
    if not selected:
        print(f"No models matched: {args.models}")
        print(f"Available: {[m['id'] for m in MODELS]}")
        return

    # Output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}

    for model_def in selected:
        mid = model_def["id"]
        port = args.port if args.port != 30001 else model_def.get("local_port", args.port)
        base_url = f"http://localhost:{port}"
        max_tok = model_def.get("max_tokens", args.max_tokens)

        print(f"\n{'='*60}")
        print(f"Model: {model_def['name']} ({mid})")
        print(f"  URL: {base_url}")
        print(f"  API tools: {'No' if model_def.get('no_api_tools') else 'Yes'}")
        print(f"{'='*60}")

        results = run_model_bfcl(
            model_def,
            base_url,
            examples,
            concurrency=args.concurrency,
            max_tokens=max_tok,
            temperature=args.temperature,
        )

        # Save per-example results
        per_example_path = out_dir / f"{mid}_per_example.jsonl"
        with open(per_example_path, "w") as f:
            for r in sorted(results, key=lambda x: x["id"]):
                # Convert call_results for JSON serialization
                r_copy = dict(r)
                f.write(json.dumps(r_copy, default=str) + "\n")

        # Aggregate
        summary = aggregate_results(results)
        summary["model_id"] = mid
        summary["model_name"] = model_def["name"]
        all_summaries[mid] = summary

        print(f"\n  Overall: {summary['full_match_pct']}% full match "
              f"({summary['full_match']}/{summary['total']})")
        for cat, cat_info in summary.get("per_category", {}).items():
            print(f"    {cat}: {cat_info['full_match_pct']}% "
                  f"({cat_info['full_match']}/{cat_info['total']})")

    # Save combined summary
    summary_path = out_dir / "bfcl_results.json"
    with open(summary_path, "w") as f:
        json.dump({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "categories": args.categories,
            "n_examples": len(examples),
            "models": all_summaries,
        }, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print("BFCL BENCHMARK SUMMARY")
    print(f"{'='*70}")

    # Build header from categories
    cats = args.categories
    header = f"{'Model':<30}"
    for c in cats:
        header += f" {c.capitalize():>10}%"
    header += f" {'Overall':>10}%"
    print(header)
    print("-" * 70)

    for mid, s in all_summaries.items():
        row = f"{s['model_name']:<30}"
        for c in cats:
            pct = s.get("per_category", {}).get(c, {}).get("full_match_pct", 0)
            row += f" {pct:>10.1f}"
        row += f" {s['full_match_pct']:>10.1f}"
        print(row)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
