#!/usr/bin/env python3
"""Benchmark runner using remote SGLang servers on Mahti.

Launches one SGLang server per model on separate GPUs, sets up SSH tunnels,
runs all benchmarks concurrently. Supports all 3 modes including full_tool
(multi-turn agentic loop with local tool execution).

Usage:
    # All models in parallel (submits 4 SLURM jobs):
    python scripts/run_benchmarks_sglang.py --dataset mbpp --problems 50 --parallel

    # Single model (must already be running):
    python scripts/run_benchmarks_sglang.py --dataset mbpp --problems 50 \
        --models qwen-3-4b --node g2301
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from pocket_agent_cli.utils.tool_extractor import ToolExtractor
from pocket_agent_cli.utils.optimized_prompts import get_optimized_prompt
from pocket_agent_cli.datasets.registry import DatasetRegistry
from pocket_agent_cli.datasets import mbpp, humaneval, gsm8k

MODELS = [
    {"id": "qwen-3-4b", "name": "Qwen 3 4B", "arch": "qwen",
     "hf_id": "Qwen/Qwen3-4B", "parser": "qwen", "local_port": 30001},
    {"id": "qwen-3-0.6b", "name": "Qwen 3 0.6B", "arch": "qwen",
     "hf_id": "Qwen/Qwen3-0.6B", "parser": "qwen", "local_port": 30002},
    {"id": "llama-3.2-3b-instruct", "name": "Llama 3.2 3B", "arch": "llama",
     "hf_id": "meta-llama/Llama-3.2-3B-Instruct", "parser": "llama3", "local_port": 30003},
    {"id": "deepseek-r1-distill-qwen-1.5b", "name": "DeepSeek R1 1.5B", "arch": "qwen",
     "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "parser": "qwen", "local_port": 30004},
]

TOOL_DEFS = [
    {"type": "function", "function": {
        "name": "run_python_code", "description": "Execute Python code and return output",
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string", "description": "Python code"}},
                       "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "submit_python_solution", "description": "Submit your final Python solution",
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string", "description": "Complete function code"}},
                       "required": ["code"]}}},
]

# ── Server Management ─────────────────────────────────────────────────────


def submit_sglang_jobs(models: List[Dict], time_limit: str = "02:00:00") -> Dict[str, str]:
    """Submit SGLang SLURM jobs for each model. Returns {model_id: job_id}."""
    jobs = {}
    for model in models:
        port = model["local_port"]  # use as remote port too
        cmd = (f"sbatch --partition=gpusmall --time={time_limit} "
               f"~/sglang-server.sh {model['hf_id']} {model['parser']} {port}")
        result = subprocess.run(
            ["ssh", "mahti", cmd], capture_output=True, text=True, timeout=15
        )
        if "Submitted batch job" in result.stdout:
            job_id = result.stdout.strip().split()[-1]
            jobs[model["id"]] = job_id
            print(f"  Submitted {model['name']} -> job {job_id} (port {port})")
        else:
            print(f"  FAILED to submit {model['name']}: {result.stderr[:100]}")
    return jobs


def wait_for_servers(models: List[Dict], jobs: Dict[str, str],
                     timeout: int = 300) -> Dict[str, str]:
    """Wait for all SGLang servers to start. Returns {model_id: node_name}."""
    import httpx

    print(f"\nWaiting for {len(jobs)} servers to start (timeout {timeout}s)...")
    nodes = {}
    start = time.time()

    while time.time() - start < timeout and len(nodes) < len(jobs):
        # Check SLURM for running jobs and their nodes
        result = subprocess.run(
            ["ssh", "mahti", "squeue -u vtoivone -o '%i %N %T' --noheader"],
            capture_output=True, text=True, timeout=15
        )
        for line in result.stdout.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 3 and parts[2] == "RUNNING":
                job_id, node = parts[0], parts[1]
                for mid, jid in jobs.items():
                    if jid == job_id and mid not in nodes:
                        nodes[mid] = node
                        print(f"  {mid} running on {node}")

        if len(nodes) < len(jobs):
            time.sleep(10)

    # Set up tunnels and verify servers respond
    base_urls = {}
    for model in models:
        mid = model["id"]
        if mid not in nodes:
            print(f"  {mid}: NOT RUNNING")
            continue

        node = nodes[mid]
        local_port = model["local_port"]
        remote_port = model["local_port"]

        # Kill existing tunnel on this port
        subprocess.run(["lsof", "-ti", f":{local_port}"], capture_output=True)
        result = subprocess.run(["lsof", "-ti", f":{local_port}"],
                                capture_output=True, text=True)
        for pid in result.stdout.strip().split("\n"):
            if pid:
                subprocess.run(["kill", pid], capture_output=True)

        # Create tunnel
        subprocess.Popen(
            ["ssh", "-f", "-N", "-L",
             f"{local_port}:{node}.mahti.csc.fi:{remote_port}", "mahti"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(1)

        # Wait for server to respond
        url = f"http://localhost:{local_port}"
        for attempt in range(30):
            try:
                resp = httpx.get(f"{url}/v1/models", timeout=5)
                if resp.status_code == 200:
                    base_urls[mid] = url
                    print(f"  {mid}: ready at {url}")
                    break
            except Exception:
                pass
            time.sleep(5)
        else:
            print(f"  {mid}: server not responding after tunnel setup")

    return base_urls


def cancel_all_jobs():
    """Cancel all SGLang jobs on Mahti."""
    subprocess.run(["ssh", "mahti", "scancel -u vtoivone"],
                   capture_output=True, timeout=15)


# ── Chat & Tool Execution ────────────────────────────────────────────────


def sglang_chat(base_url: str, model_hf_id: str,
                messages: List[Dict], tools: Optional[List] = None,
                max_tokens: int = 2048, temperature: float = 0.7) -> Dict:
    """Send a chat completion request to SGLang via tunnel."""
    import httpx

    payload = {
        "model": model_hf_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools

    try:
        resp = httpx.post(f"{base_url}/v1/chat/completions",
                          json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except httpx.TimeoutException:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)[:200]}


def format_problem(problem, prompt_config: Dict) -> List[Dict]:
    """Build chat messages for a problem."""
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


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks."""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL).strip()
    return cleaned


def extract_code(response: str, tool_calls: Optional[List] = None) -> Optional[str]:
    """Extract Python code from response or tool calls."""
    if tool_calls:
        # First pass: look for submit_python_solution / run_python_code
        # Second pass: any tool call with a "code" parameter
        for priority_names in [("submit_python_solution", "run_python_code"), None]:
            for tc in tool_calls:
                fn = tc.get("function", {})
                if priority_names and fn.get("name") not in priority_names:
                    continue
                args = fn.get("arguments", "")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
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


def evaluate_code(code: str, test_cases: List[str]) -> Dict:
    """Run code against test cases in subprocess."""
    if not code:
        return {"passed": False, "error": "No code extracted"}

    test_code = code + "\n\n" + "\n".join(test_cases)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=10
            )
            os.unlink(f.name)
        if result.returncode == 0:
            return {"passed": True}
        return {"passed": False, "error": result.stderr[:300]}
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "Timeout"}
    except Exception as e:
        return {"passed": False, "error": str(e)[:200]}


def evaluate_gsm8k(response: str, ground_truth: float) -> Dict:
    """Evaluate GSM8K numeric answer."""
    from pocket_agent_cli.datasets.gsm8k import extract_numeric_answer, numeric_answers_match
    extracted = extract_numeric_answer(response)
    if extracted is None:
        return {"passed": False, "error": "No number found"}
    return {"passed": numeric_answers_match(extracted, ground_truth),
            "extracted": extracted, "expected": ground_truth}


def execute_tool_locally(tool_call: Dict) -> str:
    """Execute a tool call locally and return the observation string."""
    fn = tool_call.get("function", {})
    name = fn.get("name", "")
    args_raw = fn.get("arguments", "{}")
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            return f"Error: Could not parse arguments: {args_raw[:100]}"
    else:
        args = args_raw

    if name == "run_python_code":
        code = args.get("code", "")
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                f.flush()
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True, text=True, timeout=10
                )
                os.unlink(f.name)
            if result.returncode == 0:
                return result.stdout[:500] or "(no output)"
            return f"Error:\n{result.stderr[:500]}"
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out (10s)"
        except Exception as e:
            return f"Error: {e}"

    elif name == "submit_python_solution":
        return "SUBMITTED"

    elif name in ("upsert_file", "read_file", "run_submission_tests"):
        return f"Tool '{name}' not available in this evaluation mode."

    return f"Unknown tool: {name}"


# ── Problem Runners ───────────────────────────────────────────────────────


def run_problem_single_turn(problem, model_def: Dict, mode: str,
                            dataset_name: str, base_url: str) -> Dict:
    """Run a problem with single-turn generation (base / tool_submission)."""
    prompt_config = get_optimized_prompt(model_def["id"], mode)
    messages = format_problem(problem, prompt_config)

    is_thinking = model_def["arch"] == "qwen"
    max_tokens = 8192 if is_thinking else 2048
    # Some models use prompt-based tool calling (no API tools)
    use_api_tools = mode == "tool_submission" and not prompt_config.get("no_api_tools")
    tools = TOOL_DEFS if use_api_tools else None

    t0 = time.time()
    resp = sglang_chat(base_url, model_def["hf_id"], messages,
                       tools=tools, max_tokens=max_tokens)
    elapsed = time.time() - t0

    if "error" in resp:
        return {"problem_id": problem.task_id, "passed": False,
                "error": resp["error"], "elapsed_s": elapsed}

    choice = resp["choices"][0]
    msg = choice["message"]
    content = msg.get("content", "") or ""
    tool_calls = msg.get("tool_calls")
    usage = resp.get("usage", {})

    code = extract_code(content, tool_calls)

    if dataset_name == "gsm8k":
        gt = problem.metadata.get("ground_truth_answer", 0)
        evaluation = evaluate_gsm8k(content, gt)
    else:
        evaluation = evaluate_code(code, problem.test_cases)

    return {
        "problem_id": problem.task_id,
        "passed": evaluation["passed"],
        "error": evaluation.get("error"),
        "response_preview": content[:200],
        "tool_calls_found": tool_calls is not None,
        "code_extracted": code is not None,
        "tokens": usage.get("completion_tokens", 0),
        "elapsed_s": round(elapsed, 1),
        "iterations": 1,
    }


def _parse_tool_calls_from_text(content: str) -> List[Dict]:
    """Parse tool calls from response text (for prompt-based tool calling).

    Handles:
    - <tool_call>{"name":...,"arguments":{...}}</tool_call>  (Qwen)
    - ```tool_call\n{"name":...,"parameters":{...}}\n```     (Llama/paper format)
    - Raw JSON {"name":...,"arguments":{...}}
    """
    cleaned = strip_thinking(content) or content
    te = ToolExtractor()
    tool_calls_parsed, _ = te.extract_tools(cleaned)

    # Convert to API-like format
    result = []
    for tc in (tool_calls_parsed or []):
        name = tc.get("name", "")
        params = tc.get("parameters", tc.get("arguments", {}))
        result.append({
            "function": {"name": name, "arguments": json.dumps(params)},
            "id": f"parsed_{len(result)}",
        })
    return result


def run_problem_agentic(problem, model_def: Dict, dataset_name: str,
                        base_url: str, max_iterations: int = 5,
                        timeout_s: int = 300) -> Dict:
    """Run a problem with multi-turn agentic loop (full_tool mode).

    Loop: generate → parse tool call → execute locally → feed observation → repeat.
    Stops when: submit_python_solution is called, max iterations, or timeout.
    """
    prompt_config = get_optimized_prompt(model_def["id"], "full_tool")
    messages = format_problem(problem, prompt_config)
    use_prompt_tools = prompt_config.get("no_api_tools", False)

    is_thinking = model_def["arch"] == "qwen"
    max_tokens = 8192 if is_thinking else 2048

    t0 = time.time()
    submitted_code = None
    iterations = 0
    total_tokens = 0
    total_tool_calls = 0

    for iteration in range(max_iterations):
        if time.time() - t0 > timeout_s:
            break

        iterations += 1
        api_tools = None if use_prompt_tools else TOOL_DEFS
        resp = sglang_chat(base_url, model_def["hf_id"], messages,
                           tools=api_tools, max_tokens=max_tokens)

        if "error" in resp:
            break

        choice = resp["choices"][0]
        msg = choice["message"]
        content = msg.get("content", "") or ""
        usage = resp.get("usage", {})
        total_tokens += usage.get("completion_tokens", 0)

        # Get tool calls — from API response or by parsing text
        tool_calls = msg.get("tool_calls")
        if not tool_calls and content:
            tool_calls = _parse_tool_calls_from_text(content)

        # Add assistant message to conversation
        messages.append({"role": "assistant", "content": content})

        if not tool_calls:
            # No tool calls found at all — model is done (or stuck)
            break

        # Process each tool call
        any_submitted = False
        for tc in tool_calls:
            fn = tc.get("function", {})
            tc_name = fn.get("name", "")
            tc_id = tc.get("id", f"call_{iteration}")
            total_tool_calls += 1

            # Execute tool locally
            observation = execute_tool_locally(tc)

            # Check for submission
            if tc_name == "submit_python_solution":
                args_raw = fn.get("arguments", "{}")
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = args_raw
                submitted_code = args.get("code", "")
                any_submitted = True

            # Feed observation back as user message (for prompt-based tools)
            # or as tool message (for API tools)
            if use_prompt_tools:
                messages.append({
                    "role": "user",
                    "content": f"Tool result ({tc_name}):\n{observation}",
                })
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": observation,
                })

        if any_submitted:
            break

    elapsed = time.time() - t0

    # If no explicit submission, try to extract from conversation
    if not submitted_code:
        for msg in reversed(messages):
            c = msg.get("content", "")
            if c:
                submitted_code = extract_code(c)
                if submitted_code:
                    break

    # Evaluate
    if dataset_name == "gsm8k":
        gt = problem.metadata.get("ground_truth_answer", 0)
        evaluation = evaluate_gsm8k(last_content, gt)
    else:
        evaluation = evaluate_code(submitted_code, problem.test_cases)

    return {
        "problem_id": problem.task_id,
        "passed": evaluation["passed"],
        "error": evaluation.get("error"),
        "code_extracted": submitted_code is not None,
        "tokens": total_tokens,
        "elapsed_s": round(elapsed, 1),
        "iterations": iterations,
        "tool_calls": total_tool_calls,
    }


# ── Main Runner ───────────────────────────────────────────────────────────


def run_model_benchmark(model_def: Dict, base_url: str, problems: List,
                        modes: List[str], dataset_name: str,
                        concurrency: int, out_dir: Path) -> Dict:
    """Run all modes for a single model."""
    model_results = {}

    for mode in modes:
        print(f"\n  Mode: {mode} (concurrency={concurrency})")

        if mode == "full_tool":
            runner = lambda p: run_problem_agentic(p, model_def, dataset_name, base_url)
        else:
            runner = lambda p: run_problem_single_turn(p, model_def, mode, dataset_name, base_url)

        results = []
        passed = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(runner, p): i for i, p in enumerate(problems)}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    result = {"problem_id": "?", "passed": False,
                              "error": str(e)[:200], "elapsed_s": 0}
                results.append(result)
                if result["passed"]:
                    passed += 1
                done = len(results)
                if done % 10 == 0 or done == len(problems):
                    print(f"    [{done}/{len(problems)}] {passed}/{done} passed")

        elapsed = time.time() - t0
        pass_rate = passed / len(problems) if problems else 0
        print(f"  {mode}: {passed}/{len(problems)} = {pass_rate:.0%} ({elapsed:.1f}s)")

        model_results[mode] = {
            "pass_rate": pass_rate, "passed": passed,
            "total": len(problems), "elapsed_s": round(elapsed, 1),
        }

        out_file = out_dir / f"{model_def['id']}_{mode}.jsonl"
        with open(out_file, "w") as f:
            for r in sorted(results, key=lambda x: str(x["problem_id"])):
                f.write(json.dumps(r, default=str) + "\n")

    return model_results


def run_sweep(args):
    """Run benchmark sweep — parallel or single model."""
    ds_cls = DatasetRegistry.get(args.dataset)
    from pocket_agent_cli.config import DATA_DIR
    ds = ds_cls(DATA_DIR)
    if not ds.is_downloaded():
        print(f"Downloading {args.dataset}...")
        ds.download()
    problems = ds.load(split="test", limit=args.problems)
    print(f"Loaded {len(problems)} {args.dataset} problems")

    selected_models = [m for m in MODELS if not args.models or m["id"] in args.models]
    selected_modes = args.modes or ["base", "tool_submission", "full_tool"]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"sglang_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.parallel:
        # Submit jobs for all models, wait, then run in parallel
        print(f"\nSubmitting {len(selected_models)} SGLang jobs...")
        jobs = submit_sglang_jobs(selected_models, args.time_limit)
        base_urls = wait_for_servers(selected_models, jobs, timeout=300)

        if not base_urls:
            print("ERROR: No servers started")
            return

        all_results = {}
        # Run all models concurrently (each model on its own thread)
        with ThreadPoolExecutor(max_workers=len(base_urls)) as executor:
            futures = {}
            for model_def in selected_models:
                mid = model_def["id"]
                if mid in base_urls:
                    print(f"\n{'='*60}")
                    print(f"Model: {model_def['name']}")
                    futures[executor.submit(
                        run_model_benchmark, model_def, base_urls[mid],
                        problems, selected_modes, args.dataset,
                        args.concurrency, out_dir
                    )] = mid

            for future in as_completed(futures):
                mid = futures[future]
                try:
                    all_results[mid] = future.result()
                except Exception as e:
                    print(f"  {mid} FAILED: {e}")

    else:
        # Single model mode with --node
        import httpx
        base_url = f"http://localhost:{args.port}"

        # Check if tunnel exists, set up if not
        try:
            httpx.get(f"{base_url}/v1/models", timeout=3)
        except Exception:
            if not args.node:
                print("ERROR: No --node specified and no tunnel active")
                print("Use --parallel to auto-submit, or specify --node")
                return
            print(f"Setting up tunnel to {args.node}:{args.port}...")
            subprocess.Popen(
                ["ssh", "-f", "-N", "-L",
                 f"{args.port}:{args.node}.mahti.csc.fi:{args.port}", "mahti"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            time.sleep(3)

        all_results = {}
        for model_def in selected_models:
            print(f"\n{'='*60}")
            print(f"Model: {model_def['name']}")
            all_results[model_def["id"]] = run_model_benchmark(
                model_def, base_url, problems, selected_modes,
                args.dataset, args.concurrency, out_dir
            )

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ({args.dataset}, {args.problems} problems)")
    print(f"{'='*60}")
    print(f"\n{'Model':<35} {'Base':>6} {'ToolSub':>8} {'FullTool':>9}")
    print("-" * 60)
    for mid, mr in all_results.items():
        b = mr.get("base", {}).get("pass_rate", 0)
        t = mr.get("tool_submission", {}).get("pass_rate", 0)
        f = mr.get("full_tool", {}).get("pass_rate", 0)
        print(f"{mid:<35} {b:>5.0%} {t:>7.0%} {f:>8.0%}")

    summary = {
        "run_id": run_id, "dataset": args.dataset,
        "n_problems": args.problems, "parallel": args.parallel,
        "timestamp": datetime.now().isoformat(), "results": all_results,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="SGLang benchmark runner")
    parser.add_argument("--dataset", default="mbpp")
    parser.add_argument("--problems", type=int, default=50)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--modes", nargs="*", default=["base", "tool_submission", "full_tool"])
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--parallel", action="store_true",
                        help="Submit separate SLURM jobs for each model")
    parser.add_argument("--node", help="Mahti node (for single-model mode)")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--time-limit", default="02:00:00")
    parser.add_argument("--output-dir", default="data/results")
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
