#!/usr/bin/env python3
"""Validate web search tool calling for each model.

Runs 5 HotpotQA problems per model and checks:
1. Does the model output web_search calls?
2. Does ToolExtractor parse them?
3. Does the model use search results in its answer?

Usage:
    python scripts/validate_websearch_models.py --models qwen-3-4b llama-3.2-3b-instruct
    python scripts/validate_websearch_models.py  # all models with available servers
"""

import argparse
import json
import re
import sys
import time
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
from pocket_agent_cli.tools.web_search import SimulatedWebSearch
from pocket_agent_cli.utils.tool_extractor import ToolExtractor

import httpx

# Import model definitions and prompts from the benchmark script
from run_websearch_benchmark import MODELS, get_system_prompt, strip_thinking, parse_tool_calls_from_text


def validate_model(
    model_def: Dict,
    problems: List,
    base_url: str,
    max_iterations: int = 3,
) -> Dict[str, Any]:
    """Run validation for a single model.

    Returns detailed results for each problem including raw responses
    and tool call parsing diagnostics.
    """
    results = []
    is_thinking = model_def["arch"] == "qwen" or "deepseek" in model_def["id"]
    max_tokens = 8192 if is_thinking else 2048
    system_prompt = get_system_prompt(model_def)

    print(f"\n{'='*70}")
    print(f"VALIDATING: {model_def['name']} ({model_def['id']})")
    print(f"  Server: {base_url}")
    print(f"  Arch: {model_def['arch']}")
    print(f"  Thinking model: {is_thinking}")
    print(f"  System prompt preview: {system_prompt[:120]}...")
    print(f"{'='*70}")

    for i, problem in enumerate(problems):
        print(f"\n  Problem {i+1}/5: {problem.prompt[:80]}...")
        paragraphs = problem.metadata.get("paragraphs", [])
        web_search = SimulatedWebSearch(context_paragraphs=paragraphs, network_config=None)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem.prompt},
        ]

        total_search_calls = 0
        all_responses = []
        tool_calls_found = []
        final_response = ""

        for iteration in range(max_iterations + 1):
            payload = {
                "model": model_def["hf_id"],
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }

            try:
                resp = httpx.post(
                    f"{base_url}/v1/chat/completions",
                    json=payload,
                    timeout=300,
                )
                resp.raise_for_status()
                d = resp.json()
            except Exception as e:
                print(f"    ERROR: {e}")
                break

            choice = d["choices"][0]
            content = choice["message"].get("content", "") or ""
            usage = d.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)

            # Store raw response (trimmed for display)
            cleaned = strip_thinking(content) or content
            all_responses.append({
                "iteration": iteration,
                "raw_length": len(content),
                "cleaned_length": len(cleaned),
                "tokens": completion_tokens,
                "content_preview": cleaned[:300],
            })

            final_response = content
            messages.append({"role": "assistant", "content": content})

            # Parse tool calls
            tool_calls = parse_tool_calls_from_text(content)

            search_called = False
            for tc in tool_calls:
                fn = tc.get("function", {})
                tc_name = fn.get("name", "")
                tc_args = fn.get("arguments", "{}")

                tool_calls_found.append({
                    "iteration": iteration,
                    "name": tc_name,
                    "arguments": tc_args,
                })

                if tc_name == "web_search":
                    if isinstance(tc_args, str):
                        try:
                            args = json.loads(tc_args)
                        except json.JSONDecodeError:
                            args = {}
                    else:
                        args = tc_args
                    query = args.get("query", problem.prompt)

                    search_results = web_search.search(query)
                    total_search_calls += 1
                    search_called = True

                    messages.append({
                        "role": "user",
                        "content": f"Search results for '{query}':\n\n{search_results}",
                    })
                    print(f"    Iteration {iteration}: web_search('{query[:60]}...')")
                    break

            if not search_called:
                if tool_calls:
                    print(f"    Iteration {iteration}: Tool call found but NOT web_search: {[tc.get('function', {}).get('name') for tc in tool_calls]}")
                else:
                    print(f"    Iteration {iteration}: No tool call found (answering directly)")
                break

        # Evaluate answer
        ground_truth = problem.metadata.get("answer", "")
        predicted = extract_answer_from_response(final_response)
        em = exact_match_score(predicted, ground_truth)
        f1 = f1_score(predicted, ground_truth)

        result = {
            "problem_id": problem.task_id,
            "question": problem.prompt,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "em": em,
            "f1": f1,
            "search_calls": total_search_calls,
            "tool_calls_found": tool_calls_found,
            "responses": all_responses,
        }
        results.append(result)

        print(f"    Answer: '{predicted[:80]}' (GT: '{ground_truth}')")
        print(f"    F1={f1:.3f} EM={em:.0f} searches={total_search_calls}")

    # Summary
    n_searched = sum(1 for r in results if r["search_calls"] > 0)
    avg_f1 = sum(r["f1"] for r in results) / len(results) if results else 0
    avg_searches = sum(r["search_calls"] for r in results) / len(results) if results else 0

    summary = {
        "model_id": model_def["id"],
        "model_name": model_def["name"],
        "arch": model_def["arch"],
        "n_problems": len(results),
        "n_with_search": n_searched,
        "search_rate": n_searched / len(results) if results else 0,
        "avg_searches": avg_searches,
        "avg_f1": avg_f1,
        "validation_passed": n_searched >= 2,  # At least 2/5 problems should search
        "details": results,
    }

    print(f"\n  SUMMARY for {model_def['name']}:")
    print(f"    Search rate: {n_searched}/{len(results)} ({summary['search_rate']:.0%})")
    print(f"    Avg searches: {avg_searches:.1f}")
    print(f"    Avg F1: {avg_f1:.3f}")
    print(f"    VALIDATION: {'PASSED' if summary['validation_passed'] else 'FAILED'}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Validate web search models")
    parser.add_argument("--models", nargs="*", help="Model IDs to validate")
    parser.add_argument("--problems", type=int, default=5, help="Number of problems")
    args = parser.parse_args()

    # Load dataset
    ds_cls = DatasetRegistry.get("hotpotqa")
    from pocket_agent_cli.config import DATA_DIR
    ds = ds_cls(DATA_DIR)
    if not ds.is_downloaded():
        print("Downloading HotpotQA dataset...")
        ds.download()
    problems = ds.load(split="validation", limit=args.problems)
    print(f"Loaded {len(problems)} HotpotQA problems for validation")

    selected_models = [m for m in MODELS if not args.models or m["id"] in args.models]

    all_summaries = []
    for model_def in selected_models:
        port = model_def["local_port"]
        base_url = f"http://localhost:{port}"

        # Check server availability
        try:
            r = httpx.get(f"{base_url}/v1/models", timeout=5)
            r.raise_for_status()
        except Exception:
            print(f"\nSKIPPING {model_def['name']}: server not available at {base_url}")
            continue

        summary = validate_model(model_def, problems, base_url)
        all_summaries.append(summary)

    # Final report
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'Search Rate':>12} {'Avg F1':>8} {'Status':>10}")
    print("-" * 65)
    for s in all_summaries:
        status = "PASSED" if s["validation_passed"] else "FAILED"
        print(f"{s['model_name']:<30} {s['n_with_search']}/{s['n_problems']:>8} {s['avg_f1']:>8.3f} {status:>10}")

    # Save detailed results
    out_path = Path("data/results/websearch_qa/validation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove non-serializable details for the summary
    save_summaries = []
    for s in all_summaries:
        save_summaries.append(s)
    with open(out_path, "w") as f:
        json.dump(save_summaries, f, indent=2, default=str)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
