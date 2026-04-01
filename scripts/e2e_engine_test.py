#!/usr/bin/env python3
"""End-to-end engine throughput test for MLX and SGLang.

Tests both inference engines with actual model loading, generation,
tool calling, and batch inference. Measures throughput for experiment
time estimation.

Usage:
    cd /Users/vilhelmtoivonen/code/phd/pocket-agent/cli
    .venv/bin/python scripts/e2e_engine_test.py
"""

import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Test problems ──────────────────────────────────────────────────────────

MATH_PROBLEMS = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
    "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. If she gives her chickens 20 cups of feed in the final meal of the day, how many chickens does Wendi have?",
    "Kylar went to the store to get his 2 dozens of eggs but realized that each egg costs $0.50 more than last month. If a dozen eggs used to cost $2, how much does a dozen eggs cost now?",
]

MATH_ANSWERS = [18.0, 3.0, 70000.0, 20.0, 8.0]

CODE_PROBLEMS = [
    "Write a function that returns the sum of all even numbers from 1 to n.",
    "Write a function that checks if a string is a palindrome.",
    "Write a function that returns the factorial of a number.",
]

TOOL_CALL_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a math problem solver. You have access to a Python code "
            "execution tool. Use it to compute the answer, then state your final "
            "answer clearly as 'The answer is X'."
        ),
    },
    {
        "role": "user",
        "content": MATH_PROBLEMS[0],  # Janet's ducks
    },
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "Execute Python code and return the output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
                "required": ["code"],
            },
        },
    }
]

BATCH_MESSAGES = [
    [
        {"role": "system", "content": "You are a math solver. Solve this and state 'The answer is X'."},
        {"role": "user", "content": p},
    ]
    for p in MATH_PROBLEMS
]


def test_mlx_engine():
    """Test MLX inference engine with streaming, tool calling, and batching."""
    print("\n" + "=" * 70)
    print("MLX ENGINE TEST (Apple Silicon)")
    print("=" * 70)

    from pocket_agent_cli.services.mlx_inference_service import MLXInferenceService
    from pocket_agent_cli.config import InferenceConfig, Model

    service = MLXInferenceService()

    model = Model(
        id="qwen-3-4b",
        name="Qwen 3 4B",
        architecture="qwen",
        downloaded=True,
        default_version="Q4_K_M",
    )
    config = InferenceConfig(
        temperature=0.7,
        max_tokens=512,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        context_length=4096,
    )

    # ── Load model ──
    print("\n[1/4] Loading Qwen3-4B-4bit via MLX...")
    t0 = time.time()
    service.load_model(model, config)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    info = service.get_model_info()
    print(f"  Model: {info['hf_model_id']}, peak memory: {info['peak_memory_gb']:.2f} GB")

    # ── Streaming generation ──
    print("\n[2/4] Streaming generation (single prompt)...")
    messages = [
        {"role": "user", "content": "What is 15 * 37? Just give the number."}
    ]
    t0 = time.time()
    full_text = ""
    metrics = {}
    for chunk in service.generate(messages, stream=True):
        full_text += chunk["token"]
        metrics = chunk["metrics"]
    elapsed = time.time() - t0
    print(f"  Response: {full_text.strip()[:100]}")
    gen_tps = metrics.get('generation_tps', metrics.get('tps', 0))
    ttft = metrics.get('ttft', 0)
    print(f"  Tokens: {metrics.get('tokens', '?')}, TPS: {gen_tps:.1f}")
    print(f"  TTFT: {ttft:.0f}ms, Total: {elapsed:.2f}s")

    # ── Tool calling ──
    print("\n[3/4] Tool calling test (Janet's ducks problem)...")
    t0 = time.time()
    response_text, tool_calls, tc_metrics = service.generate_with_tools(
        TOOL_CALL_MESSAGES, TOOLS
    )
    tc_elapsed = time.time() - t0
    print(f"  Response (first 200 chars): {response_text.strip()[:200]}")
    print(f"  Tool calls found: {tool_calls is not None}")
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", tc)
            print(f"    -> {fn.get('name', '?')}({str(fn.get('arguments', ''))[:80]})")
    print(f"  Time: {tc_elapsed:.2f}s")

    # ── Batch generation ──
    print("\n[4/4] Batch generation (5 math problems)...")
    batch_sizes = [1, 5]
    for bs in batch_sizes:
        prompts = BATCH_MESSAGES[:bs]
        t0 = time.time()
        results = service.batch_generate(prompts, max_tokens=256)
        batch_elapsed = time.time() - t0

        total_gen_tokens = sum(r["metrics"].get("tokens", 0) for r in results)
        stats = results[0]["metrics"]

        print(f"\n  Batch size {bs}:")
        print(f"    Wall time: {batch_elapsed:.2f}s")
        print(f"    Prefill TPS: {stats.get('prompt_tps', '?'):.1f}")
        print(f"    Generation TPS: {stats.get('generation_tps', '?'):.1f}")
        print(f"    Peak memory: {stats.get('peak_memory_gb', '?'):.2f} GB")

        for i, r in enumerate(results):
            from pocket_agent_cli.datasets.gsm8k import extract_numeric_answer
            extracted = extract_numeric_answer(r["text"])
            correct = "?" if i >= len(MATH_ANSWERS) else ("OK" if extracted == MATH_ANSWERS[i] else f"WRONG (got {extracted}, expected {MATH_ANSWERS[i]})")
            print(f"    Problem {i}: answer={extracted}, {correct}")

    # ── Max batch test ──
    print("\n  Max batch test (5 problems x 3 = 15 prompts)...")
    big_batch = BATCH_MESSAGES * 3  # 15 prompts
    t0 = time.time()
    results = service.batch_generate(big_batch, max_tokens=256)
    big_batch_elapsed = time.time() - t0
    stats = results[0]["metrics"]
    print(f"    15 prompts in {big_batch_elapsed:.2f}s")
    print(f"    Prefill TPS: {stats.get('prompt_tps', '?'):.1f}")
    print(f"    Generation TPS: {stats.get('generation_tps', '?'):.1f}")
    print(f"    Peak memory: {stats.get('peak_memory_gb', '?'):.2f} GB")
    print(f"    Effective throughput: {15 / big_batch_elapsed:.1f} problems/s")

    service.unload_model()

    return {
        "engine": "mlx",
        "model": "Qwen3-4B-4bit",
        "load_time_s": load_time,
        "single_tps": metrics.get("generation_tps", metrics.get("tps", 0)),
        "batch5_time_s": batch_elapsed if bs == 5 else None,
        "batch15_time_s": big_batch_elapsed,
        "batch15_gen_tps": stats.get("generation_tps", 0),
        "batch15_prefill_tps": stats.get("prompt_tps", 0),
        "tool_calling_works": tool_calls is not None,
        "problems_per_sec_batch15": 15 / big_batch_elapsed,
    }


def test_sglang_engine():
    """Test SGLang inference engine on Mahti via HTTP."""
    print("\n" + "=" * 70)
    print("SGLANG ENGINE TEST (Mahti A100)")
    print("=" * 70)

    import subprocess
    import httpx

    # Check if there's a running SGLang server on Mahti
    print("\n[1/4] Checking for running SGLang server on Mahti...")
    try:
        result = subprocess.run(
            ["ssh", "mahti", "squeue -u vtoivone -o '%j %N %T' --noheader"],
            capture_output=True, text=True, timeout=15
        )
        jobs = result.stdout.strip()
        print(f"  Running jobs: {jobs if jobs else 'none'}")
    except Exception as e:
        print(f"  Could not check jobs: {e}")
        jobs = ""

    if not jobs or "sglang" not in jobs.lower():
        print("\n  No SGLang server running. Submitting job...")
        try:
            result = subprocess.run(
                ["ssh", "mahti", "sbatch --partition=gputest --time=00:15:00 ~/sglang-server.sh"],
                capture_output=True, text=True, timeout=15
            )
            print(f"  {result.stdout.strip()}")
            print("  Waiting 120s for server startup (flashinfer JIT compilation)...")
            time.sleep(120)
            # Get the node
            result = subprocess.run(
                ["ssh", "mahti", "squeue -u vtoivone -o '%N' --noheader"],
                capture_output=True, text=True, timeout=15
            )
            node = result.stdout.strip().split()[0] if result.stdout.strip() else None
        except Exception as e:
            print(f"  Failed to submit: {e}")
            return {"engine": "sglang", "error": str(e)}
    else:
        # Extract node from running job
        parts = jobs.strip().split()
        node = parts[1] if len(parts) > 1 else None

    if not node:
        print("  Could not determine node. Skipping SGLang test.")
        return {"engine": "sglang", "error": "no node found"}

    base_url = f"http://{node}.mahti.csc.fi:30000"
    print(f"  Server URL: {base_url}")

    # Test connectivity via SSH tunnel curl
    def sglang_request(endpoint, payload, timeout=120):
        """Send request to SGLang via SSH."""
        cmd = f"curl -s -m {timeout} {base_url}{endpoint} -H 'Content-Type: application/json' -d '{json.dumps(payload)}'"
        result = subprocess.run(
            ["ssh", "mahti", cmd],
            capture_output=True, text=True, timeout=timeout + 30
        )
        if result.returncode != 0:
            raise RuntimeError(f"curl failed: {result.stderr}")
        return json.loads(result.stdout)

    # ── Check models ──
    print("\n[2/4] Checking model availability...")
    try:
        result = subprocess.run(
            ["ssh", "mahti", f"curl -s {base_url}/v1/models"],
            capture_output=True, text=True, timeout=20
        )
        models_resp = json.loads(result.stdout)
        model_id = models_resp["data"][0]["id"]
        print(f"  Model: {model_id}")
    except Exception as e:
        print(f"  Failed to get models: {e}")
        return {"engine": "sglang", "error": str(e)}

    # ── Single completion ──
    print("\n[3/4] Single completion test...")
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "What is 15 * 37? Just give the number."}],
        "max_tokens": 128,
        "temperature": 0.7,
    }
    t0 = time.time()
    resp = sglang_request("/v1/chat/completions", payload)
    single_time = time.time() - t0
    content = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    tps = completion_tokens / single_time if single_time > 0 else 0
    print(f"  Response: {content.strip()[:100]}")
    print(f"  Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
    print(f"  Time: {single_time:.2f}s, TPS: {tps:.1f}")

    # ── Tool calling ──
    print("\n[4/4] Tool calling test...")
    tool_payload = {
        "model": model_id,
        "messages": TOOL_CALL_MESSAGES,
        "tools": TOOLS,
        "max_tokens": 512,
        "temperature": 0.7,
    }
    t0 = time.time()
    resp = sglang_request("/v1/chat/completions", tool_payload)
    tc_time = time.time() - t0
    msg = resp["choices"][0]["message"]
    content = msg.get("content", "")
    tool_calls = msg.get("tool_calls", None)
    usage = resp.get("usage", {})
    print(f"  Content: {(content or '').strip()[:200]}")
    print(f"  Tool calls: {tool_calls is not None}")
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            print(f"    -> {fn.get('name', '?')}({str(fn.get('arguments', ''))[:80]})")
    print(f"  Time: {tc_time:.2f}s")
    print(f"  Tokens: {usage.get('completion_tokens', '?')}")

    # ── Concurrent requests (simulated batch) ──
    print("\n  Concurrent request test (5 problems)...")
    import concurrent.futures

    def run_single_problem(idx):
        p = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "Solve this math problem. State 'The answer is X'."},
                {"role": "user", "content": MATH_PROBLEMS[idx]},
            ],
            "max_tokens": 256,
            "temperature": 0.7,
        }
        t = time.time()
        r = sglang_request("/v1/chat/completions", p)
        return time.time() - t, r

    # Sequential baseline
    t0 = time.time()
    for i in range(5):
        run_single_problem(i)
    seq_time = time.time() - t0
    print(f"  Sequential (5 problems): {seq_time:.2f}s")

    # Concurrent
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_single_problem, i) for i in range(5)]
        results = [f.result() for f in futures]
    conc_time = time.time() - t0
    print(f"  Concurrent (5 problems): {conc_time:.2f}s")
    print(f"  Speedup: {seq_time / conc_time:.1f}x")

    return {
        "engine": "sglang",
        "model": model_id,
        "single_time_s": single_time,
        "single_tps": tps,
        "tool_calling_works": tool_calls is not None,
        "sequential_5_s": seq_time,
        "concurrent_5_s": conc_time,
        "concurrency_speedup": seq_time / conc_time,
        "problems_per_sec_concurrent": 5 / conc_time,
    }


def estimate_experiment_time(mlx_results, sglang_results):
    """Estimate total experiment time based on measured throughput."""
    print("\n" + "=" * 70)
    print("EXPERIMENT TIME ESTIMATION")
    print("=" * 70)

    # Experiment matrix from the revision plan:
    # 3 architectures x 7 network conditions x 2 models x 2 tasks = 84 configs
    # Each: 150 problems x 3 attempts = 450 runs
    # Plus: cross-platform baseline (5 models x 4 tasks x 150 problems x 5 attempts)

    print("\n── Deployment Architecture Experiment (Section 6) ──")
    print("  3 arch x 7 RTT x 2 models x 2 tasks = 84 configs")
    print("  150 problems x 3 attempts per config = 450 runs/config")
    print("  Total runs: 84 x 450 = 37,800")

    if mlx_results and "problems_per_sec_batch15" in mlx_results:
        mlx_rate = mlx_results["problems_per_sec_batch15"]
        mlx_total_s = 37800 / mlx_rate
        mlx_total_h = mlx_total_s / 3600
        print(f"\n  MLX (M2 Max, batch=15): {mlx_rate:.1f} problems/s")
        print(f"    -> {mlx_total_h:.1f} hours for full matrix")
        print(f"    -> With 20-batch: ~{mlx_total_h * 0.75:.1f} hours (est.)")

    if sglang_results and "problems_per_sec_concurrent" in sglang_results:
        sg_rate = sglang_results["problems_per_sec_concurrent"]
        sg_total_s = 37800 / sg_rate
        sg_total_h = sg_total_s / 3600
        print(f"\n  SGLang (A100, concurrent=5): {sg_rate:.1f} problems/s")
        print(f"    -> {sg_total_h:.1f} hours for full matrix")
        print(f"    -> With concurrent=20: ~{sg_total_h * 0.25:.1f} hours (est.)")

    print("\n── On-Device Characterization (Section 5) ──")
    print("  5 models x 4 tasks x 150 problems x 5 attempts = 15,000 runs")
    print("  (MacBook only — mobile is separate, slower)")

    if mlx_results and "problems_per_sec_batch15" in mlx_results:
        char_total_h = 15000 / mlx_rate / 3600
        print(f"\n  MLX (M2 Max, batch=15): {char_total_h:.1f} hours")

    print("\n── Mobile Evaluation (separate) ──")
    print("  5 models x 2 tasks x 150 problems x 5 attempts = 7,500 runs")
    print("  iPhone 16 Pro: ~30s/run -> ~62.5 hours (must run in background)")
    print("  iPhone 15: ~45s/run -> ~93.8 hours")

    print("\n── TOTAL ESTIMATED COMPUTE ──")
    total_h = 0
    if mlx_results and "problems_per_sec_batch15" in mlx_results:
        mlx_h = (37800 + 15000) / mlx_rate / 3600
        total_h += mlx_h
        print(f"  MacBook M2 Max (MLX):   {mlx_h:.1f}h")
    if sglang_results and "problems_per_sec_concurrent" in sglang_results:
        sg_h = 37800 / sg_rate / 3600
        total_h += sg_h
        print(f"  Mahti A100 (SGLang):    {sg_h:.1f}h")
    print(f"  Mobile (background):    ~160h (multiple devices in parallel)")
    print(f"\n  MacBook + A100 total:   {total_h:.1f}h wall-clock (parallelizable)")


if __name__ == "__main__":
    print("Pocket Agent E2E Engine Test")
    print("Testing MLX and SGLang with tool calling and batching\n")

    mlx_results = None
    sglang_results = None

    # MLX test (local)
    try:
        mlx_results = test_mlx_engine()
    except Exception as e:
        print(f"\nMLX TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

    # SGLang test (Mahti)
    try:
        sglang_results = test_sglang_engine()
    except Exception as e:
        print(f"\nSGLANG TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Estimation
    estimate_experiment_time(mlx_results, sglang_results)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    if mlx_results:
        print(f"\nMLX ({mlx_results.get('model', '?')}):")
        for k, v in mlx_results.items():
            if k != "engine":
                print(f"  {k}: {v}")
    if sglang_results:
        print(f"\nSGLang ({sglang_results.get('model', '?')}):")
        for k, v in sglang_results.items():
            if k != "engine":
                print(f"  {k}: {v}")
