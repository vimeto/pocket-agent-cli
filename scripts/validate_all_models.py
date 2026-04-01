#!/usr/bin/env python3
"""Validate all model × quantization variants produce valid tool calls via MLX.

Tests each of the 5 models × 2 quantizations (Q4_K_M, F16) = 10 variants.
For each: load model, run a tool-calling prompt, verify tool call is parsed.

Usage:
    cd /Users/vilhelmtoivonen/code/phd/pocket-agent/cli
    .venv/bin/python scripts/validate_all_models.py
"""

import sys
import os
import time
import json
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pocket_agent_cli.services.mlx_inference_service import MLXInferenceService, MLX_MODEL_MAP
from pocket_agent_cli.config import InferenceConfig, Model

MODELS = [
    {"id": "qwen-3-4b", "name": "Qwen 3 4B", "arch": "qwen"},
    {"id": "qwen-3-0.6b", "name": "Qwen 3 0.6B", "arch": "qwen"},
    {"id": "llama-3.2-3b-instruct", "name": "Llama 3.2 3B Instruct", "arch": "llama"},
    {"id": "deepseek-r1-distill-qwen-1.5b", "name": "DeepSeek R1 Distill Qwen 1.5B", "arch": "qwen"},
    {"id": "gemma-3n-e2b-it", "name": "Gemma 3n E2B IT", "arch": "gemma"},
]

VERSIONS = ["Q4_K_M", "F16"]

TOOL_PROMPT = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. You MUST use the run_python_code tool to answer. "
            "Do not answer directly — always call the tool first."
        ),
    },
    {
        "role": "user",
        "content": "Use the run_python_code tool to calculate 16 - 3 - 4 and then multiply by 2.",
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


def validate_variant(model_def, version):
    """Test one model+version variant. Returns a result dict."""
    model_id = model_def["id"]
    hf_id = MLX_MODEL_MAP.get(model_id, {}).get(version, "???")

    result = {
        "model": model_id,
        "version": version,
        "hf_id": hf_id,
        "load_ok": False,
        "generate_ok": False,
        "tool_call_found": False,
        "tool_call_details": None,
        "response_preview": "",
        "tokens": 0,
        "tps": 0,
        "elapsed_s": 0,
        "error": None,
    }

    service = MLXInferenceService()
    model = Model(
        id=model_id,
        name=model_def["name"],
        architecture=model_def["arch"],
        downloaded=True,
        default_version=version,
        current_version=version,
    )
    config = InferenceConfig(
        temperature=0.7,
        max_tokens=1024,  # High enough for thinking models
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        context_length=4096,
        jinja=True,
    )

    try:
        # Load
        t0 = time.time()
        service.load_model(model, config)
        load_time = time.time() - t0
        result["load_ok"] = True
        print(f"    Loaded in {load_time:.1f}s")

        # Generate with tools
        t0 = time.time()
        response_text, tool_calls, metrics = service.generate_with_tools(
            TOOL_PROMPT, TOOLS
        )
        elapsed = time.time() - t0
        result["generate_ok"] = True
        result["elapsed_s"] = round(elapsed, 1)
        result["tokens"] = metrics.get("tokens", metrics.get("thinking_stats", {}).get("total_tokens", 0))
        result["tps"] = round(metrics.get("tps", 0), 1)
        result["response_preview"] = (response_text or "")[:200]

        # Check tool calls
        if tool_calls:
            result["tool_call_found"] = True
            result["tool_call_details"] = []
            for tc in tool_calls:
                fn = tc.get("function", tc)
                result["tool_call_details"].append({
                    "name": fn.get("name", "?"),
                    "args_preview": str(fn.get("arguments", ""))[:100],
                })

    except Exception as e:
        result["error"] = str(e)

    finally:
        service.unload_model()
        gc.collect()

    return result


def main():
    print("=" * 70)
    print("POCKET AGENT: Full Model Validation (MLX)")
    print(f"Models: {len(MODELS)}, Versions: {len(VERSIONS)}, Total: {len(MODELS) * len(VERSIONS)}")
    print("=" * 70)

    results = []

    for model_def in MODELS:
        for version in VERSIONS:
            hf_id = MLX_MODEL_MAP.get(model_def["id"], {}).get(version, "???")
            print(f"\n[{len(results)+1}/{len(MODELS)*len(VERSIONS)}] {model_def['name']} ({version})")
            print(f"    HF: {hf_id}")

            r = validate_variant(model_def, version)
            results.append(r)

            # Print result
            status = ""
            if r["error"]:
                status = f"ERROR: {r['error'][:80]}"
            elif r["tool_call_found"]:
                tc = r["tool_call_details"][0]
                status = f"TOOL CALL OK: {tc['name']}({tc['args_preview'][:50]})"
            elif r["generate_ok"]:
                status = f"NO TOOL CALL (response: {r['response_preview'][:80]})"
            else:
                status = "GENERATION FAILED"

            emoji = "OK" if r["tool_call_found"] else ("WARN" if r["generate_ok"] else "FAIL")
            print(f"    [{emoji}] {status}")
            if r["generate_ok"]:
                print(f"    Tokens: {r['tokens']}, TPS: {r['tps']}, Time: {r['elapsed_s']}s")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<35} {'Version':<8} {'Load':>5} {'Gen':>5} {'Tool':>5} {'TPS':>6} {'Time':>6}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['model']:<35} {r['version']:<8} "
            f"{'OK' if r['load_ok'] else 'FAIL':>5} "
            f"{'OK' if r['generate_ok'] else 'FAIL':>5} "
            f"{'OK' if r['tool_call_found'] else 'MISS':>5} "
            f"{r['tps']:>6.1f} "
            f"{r['elapsed_s']:>5.1f}s"
        )

    ok = sum(1 for r in results if r["tool_call_found"])
    gen_ok = sum(1 for r in results if r["generate_ok"])
    total = len(results)
    print(f"\nTool calls: {ok}/{total}  |  Generation: {gen_ok}/{total}  |  Errors: {total - gen_ok}/{total}")

    if ok < total:
        print("\nVariants without tool calls:")
        for r in results:
            if not r["tool_call_found"] and r["generate_ok"]:
                print(f"  {r['model']} ({r['version']}): {r['response_preview'][:120]}")

    # Save results
    out_path = "scripts/model_validation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
