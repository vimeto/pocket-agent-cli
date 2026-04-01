"""Tests for the MLX inference service.

Run with: cd /Users/vilhelmtoivonen/code/phd/pocket-agent/cli && .venv/bin/python -m pytest tests/test_mlx_inference.py -v

These tests require Apple Silicon and a working mlx-lm installation.
The first run will download the Qwen3-0.6B-4bit model (~0.5GB).
"""

import os
import sys
import time
import pytest

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pocket_agent_cli.services.mlx_inference_service import (
    MLXInferenceService,
    MLX_MODEL_MAP,
    get_mlx_model_id,
)
from pocket_agent_cli.config import InferenceConfig, Model


# ---- Fixtures ----


@pytest.fixture(scope="module")
def model_config():
    """Return a Model + InferenceConfig for the smallest test model."""
    model = Model(
        id="qwen-3-0.6b",
        name="Qwen 3 0.6B",
        architecture="qwen",
        downloaded=True,
        default_version="Q4_K_M",
    )
    config = InferenceConfig(
        temperature=0.7,
        max_tokens=256,
        top_p=0.8,
        top_k=40,
        repeat_penalty=1.1,
        context_length=4096,
    )
    return model, config


@pytest.fixture(scope="module")
def loaded_service(model_config):
    """Load the MLX service once for all tests in this module."""
    model, config = model_config
    service = MLXInferenceService()
    service.load_model(model, config)
    yield service
    service.unload_model()


# ---- Unit tests (no model needed) ----


class TestModelMapping:
    def test_all_models_have_q4_and_f16(self):
        for model_id, versions in MLX_MODEL_MAP.items():
            assert "Q4_K_M" in versions, f"{model_id} missing Q4_K_M"
            assert "F16" in versions, f"{model_id} missing F16"

    def test_get_mlx_model_id_valid(self):
        hf_id = get_mlx_model_id("qwen-3-0.6b", "Q4_K_M")
        assert hf_id == "mlx-community/Qwen3-0.6B-4bit"

    def test_get_mlx_model_id_default_version(self):
        hf_id = get_mlx_model_id("qwen-3-4b")
        assert hf_id == "mlx-community/Qwen3-4B-4bit"

    def test_get_mlx_model_id_invalid_model(self):
        with pytest.raises(ValueError, match="not found"):
            get_mlx_model_id("nonexistent-model")

    def test_get_mlx_model_id_invalid_version(self):
        with pytest.raises(ValueError, match="not available"):
            get_mlx_model_id("qwen-3-0.6b", "Q8_0")


class TestServiceInit:
    def test_init_state(self):
        service = MLXInferenceService()
        assert service.model is None
        assert service.tokenizer is None
        assert service.current_model is None
        assert service.config is None

    def test_generate_without_model_raises(self):
        service = MLXInferenceService()
        with pytest.raises(RuntimeError, match="No model loaded"):
            list(service.generate([{"role": "user", "content": "hi"}]))

    def test_get_model_info_without_model(self):
        service = MLXInferenceService()
        assert service.get_model_info() is None


# ---- Integration tests (require model download + Apple Silicon) ----


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="MLX tests require macOS with Apple Silicon",
)
class TestMLXInference:
    def test_load_model(self, loaded_service):
        """Test that model loads and reports info."""
        info = loaded_service.get_model_info()
        assert info is not None
        assert info["model_id"] == "qwen-3-0.6b"
        assert info["backend"] == "mlx"
        assert info["loaded"] is True
        assert "mlx-community" in info["hf_model_id"]

    def test_streaming_generation(self, loaded_service):
        """Test streaming generation produces tokens with metrics."""
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 2+2? Reply with just the number."},
        ]

        chunks = list(loaded_service.generate(messages, stream=True, max_tokens=256))
        assert len(chunks) > 0

        # Check that we got an energy_summary at the end
        energy_chunks = [c for c in chunks if c.get("finish_reason") == "energy_summary"]
        assert len(energy_chunks) == 1

        # Check metrics structure
        last_normal = [c for c in chunks if c.get("finish_reason") != "energy_summary"][-1]
        m = last_normal["metrics"]
        assert "ttft" in m
        assert "tps" in m
        assert "tokens" in m
        assert m["tokens"] > 0
        assert m["tps"] > 0

    def test_thinking_filter(self, loaded_service):
        """Test that thinking tokens are filtered from output."""
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        raw_text = ""
        filtered_text = ""
        has_thinking = False

        for chunk in loaded_service.generate(messages, stream=True, max_tokens=256):
            if chunk.get("finish_reason") == "energy_summary":
                break
            raw_text += chunk.get("raw_token", "")
            filtered_text += chunk.get("token", "")
            if chunk.get("is_thinking"):
                has_thinking = True

        # Qwen3 uses <think> blocks, so thinking should be detected
        assert has_thinking, "Expected thinking tokens from Qwen3"
        # Raw text should contain <think> tags
        assert "<think>" in raw_text.lower()
        # Filtered text should not contain <think> tags
        assert "<think>" not in filtered_text.lower()

    def test_tool_calling(self, loaded_service):
        """Test tool call parsing from generation output."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Python expert. Submit solutions using the tool.\n"
                    "Respond with:\n"
                    "```tool_call\n"
                    '{"name": "submit_python_solution", "parameters": {"solution": "code"}}\n'
                    "```"
                ),
            },
            {
                "role": "user",
                "content": "Write a function add(a, b) that returns a + b.",
            },
        ]

        response_text, tool_calls, metrics = loaded_service.generate_with_tools(
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "submit_python_solution",
                        "description": "Submit a Python solution",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "solution": {
                                    "type": "string",
                                    "description": "Python code",
                                }
                            },
                            "required": ["solution"],
                        },
                    },
                }
            ],
            tool_choice="auto",
            max_tokens=512,
        )

        # The model should produce a tool call
        assert tool_calls is not None, f"Expected tool calls but got None. Response: {response_text[:200]}"
        assert len(tool_calls) > 0
        assert tool_calls[0]["name"] == "submit_python_solution"
        assert "solution" in tool_calls[0].get("parameters", {})

    def test_batch_generation(self, loaded_service):
        """Test batch generation produces correct results."""
        batch_prompts = [
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "What is 2+2? Just the number."},
            ],
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "What is 3*7? Just the number."},
            ],
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Capital of France? One word."},
            ],
        ]

        results = loaded_service.batch_generate(batch_prompts, max_tokens=256)

        assert len(results) == 3
        for r in results:
            assert "text" in r
            assert "metrics" in r
            assert r["metrics"]["batch_size"] == 3

        # Check that answers are reasonable
        texts = [r["text"].strip().lower() for r in results]
        assert "4" in texts[0]
        assert "21" in texts[1]
        assert "paris" in texts[2]

    def test_batch_faster_than_sequential(self, loaded_service):
        """Test that batch generation is faster than sequential."""
        prompts = [
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": f"What is {i}+{i}? Just the number."},
            ]
            for i in range(5)
        ]

        # Batch
        t0 = time.time()
        loaded_service.batch_generate(prompts, max_tokens=128)
        batch_time = time.time() - t0

        # Sequential
        t0 = time.time()
        for p in prompts:
            for chunk in loaded_service.generate(p, stream=True, max_tokens=128):
                if chunk.get("finish_reason") == "energy_summary":
                    break
        seq_time = time.time() - t0

        speedup = seq_time / batch_time
        print(f"\nBatch: {batch_time:.2f}s, Sequential: {seq_time:.2f}s, Speedup: {speedup:.2f}x")
        # Batch should be at least 1.5x faster for 5 prompts
        assert speedup > 1.5, f"Expected batch speedup > 1.5x, got {speedup:.2f}x"

    def test_unload_model(self, model_config):
        """Test that unloading properly cleans up."""
        model, config = model_config
        service = MLXInferenceService()
        service.load_model(model, config)
        assert service.get_model_info() is not None

        service.unload_model()
        assert service.model is None
        assert service.tokenizer is None
        assert service.get_model_info() is None
