"""Inference service using MLX (mlx-lm) for Apple Silicon."""

import time
import os
from typing import Dict, Any, List, Optional, Generator, Tuple
from ..config import InferenceConfig, Model
from ..utils.tool_extractor import ToolExtractor
from ..utils.thinking_filter import ThinkingFilter
from ..monitoring.unified_monitor import UnifiedMonitor

DEBUG_INFERENCE = os.environ.get("DEBUG_INFERENCE", "").lower() == "true"

# Import profile decorator
try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func


# Mapping from pocket-agent model IDs to HuggingFace MLX model IDs
# Uses mlx-community INT4 quantized models where available (analogous to Q4_K_M GGUF)
MLX_MODEL_MAP = {
    "llama-3.2-3b-instruct": {
        "Q4_K_M": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "F16": "meta-llama/Llama-3.2-3B-Instruct",
    },
    "gemma-3n-e2b-it": {
        "Q4_K_M": "google/gemma-3n-E2B-it",  # May need manual conversion
        "F16": "google/gemma-3n-E2B-it",
    },
    "deepseek-r1-distill-qwen-1.5b": {
        "Q4_K_M": "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
        "F16": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    },
    "qwen-3-4b": {
        "Q4_K_M": "mlx-community/Qwen3-4B-4bit",
        "F16": "Qwen/Qwen3-4B",
    },
    "qwen-3-0.6b": {
        "Q4_K_M": "mlx-community/Qwen3-0.6B-4bit",
        "F16": "Qwen/Qwen3-0.6B",
    },
}


def get_mlx_model_id(model_id: str, version: Optional[str] = None) -> str:
    """Resolve a pocket-agent model ID to an MLX HuggingFace model ID.

    Args:
        model_id: The pocket-agent model ID (e.g., 'qwen-3-4b')
        version: The quantization version (e.g., 'Q4_K_M', 'F16'). Defaults to Q4_K_M.

    Returns:
        HuggingFace model ID suitable for mlx_lm.load()
    """
    version = version or "Q4_K_M"
    if model_id not in MLX_MODEL_MAP:
        raise ValueError(
            f"Model '{model_id}' not found in MLX model map. "
            f"Available: {list(MLX_MODEL_MAP.keys())}"
        )
    versions = MLX_MODEL_MAP[model_id]
    if version not in versions:
        raise ValueError(
            f"Version '{version}' not available for model '{model_id}'. "
            f"Available: {list(versions.keys())}"
        )
    return versions[version]


class MLXInferenceService:
    """Service for LLM inference using MLX on Apple Silicon.

    This service mirrors the InferenceService interface but uses mlx-lm
    instead of llama-cpp-python. MLX leverages Apple's unified memory
    architecture for efficient GPU-accelerated inference on M-series chips.

    Key differences from llama.cpp backend:
    - Uses HuggingFace safetensors models (not GGUF)
    - Supports native batch inference via mlx_lm.batch_generate
    - Uses tokenizer's built-in chat template (apply_chat_template)
    - Sampling configured via make_sampler / make_logits_processors
    """

    def __init__(self):
        self.model = None  # mlx nn.Module
        self.tokenizer = None  # TokenizerWrapper
        self.current_model: Optional[Model] = None
        self.config: Optional[InferenceConfig] = None
        self.tool_extractor = ToolExtractor()
        self.thinking_filter = ThinkingFilter()
        self.unified_monitor = UnifiedMonitor(sample_interval=2.0)
        self._hf_model_id: Optional[str] = None

    def load_model(self, model: Model, config: InferenceConfig) -> None:
        """Load a model into memory using MLX.

        Args:
            model: Model configuration object
            config: Inference configuration
        """
        from mlx_lm import load

        # Unload current model if any
        if self.model is not None:
            self.unload_model()

        # Resolve HuggingFace model ID
        version = model.current_version or model.default_version
        hf_model_id = get_mlx_model_id(model.id, version)
        self._hf_model_id = hf_model_id

        print(f"[INFO] Loading MLX model: {hf_model_id} (from {model.id}, version={version})")

        load_start = time.time()
        self.model, self.tokenizer = load(hf_model_id)
        load_elapsed = time.time() - load_start

        print(f"[INFO] MLX model loaded in {load_elapsed:.1f}s")

        self.current_model = model
        self.config = config

    def unload_model(self) -> None:
        """Unload the current model from memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
        self.current_model = None
        self.config = None
        self._hf_model_id = None

    @profile
    def generate(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        **kwargs,
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate a response from the model.

        Args:
            messages: List of chat messages
            stream: Whether to stream tokens (always streams internally for metrics)
            **kwargs: Override inference config parameters

        Yields:
            Token chunks with metadata matching InferenceService format
        """
        if self.model is None or self.config is None:
            raise RuntimeError("No model loaded")

        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors

        # Merge kwargs with config
        config = self.config.model_copy()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Format prompt using tokenizer's chat template
        prompt = self._format_prompt(messages, config)

        if DEBUG_INFERENCE:
            print(f"[DEBUG MLXInferenceService] Prompt length: {len(prompt)} characters")

        # Build sampler with temperature, top_p, top_k
        sampler = make_sampler(
            temp=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else 0,
        )

        # Build logits processors for repetition penalty
        logits_processors = None
        if config.repeat_penalty and config.repeat_penalty != 1.0:
            logits_processors = make_logits_processors(
                repetition_penalty=config.repeat_penalty,
            )

        # Track metrics
        start_time = time.time()
        first_token_time = None
        token_count = 0

        # Reset thinking filter for new generation
        self.thinking_filter.reset()

        # Start unified monitoring
        if DEBUG_INFERENCE:
            print(f"[DEBUG MLXInferenceService] Starting unified monitoring...")
        self.unified_monitor.start_monitoring()

        if DEBUG_INFERENCE:
            print(f"[DEBUG MLXInferenceService] Starting MLX generation with config:")
            print(f"  - max_tokens: {config.max_tokens}")
            print(f"  - temperature: {config.temperature}")
            print(f"  - stream: {stream}")

        # Generate with MLX stream_generate
        gen_kwargs = {
            "max_tokens": config.max_tokens,
            "sampler": sampler,
        }
        if logits_processors:
            gen_kwargs["logits_processors"] = logits_processors

        last_metrics_check = start_time
        metrics_check_interval = 1.0
        current_power = None

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            **gen_kwargs,
        ):
            if first_token_time is None:
                first_token_time = time.time()
                if DEBUG_INFERENCE:
                    print(f"[DEBUG MLXInferenceService] First token received! TTFT: {(first_token_time - start_time) * 1000:.1f}ms")

            raw_token = response.text
            filtered_token, is_thinking = self.thinking_filter.filter_token(raw_token)

            token_count += 1

            # Calculate metrics
            current_time = time.time()
            ttft = (first_token_time - start_time) * 1000 if first_token_time else None
            elapsed = current_time - start_time
            tps = token_count / elapsed if elapsed > 0 else 0

            # Only check power metrics periodically
            if current_time - last_metrics_check >= metrics_check_interval:
                current_metrics = self.unified_monitor.get_current_metrics()
                current_power = current_metrics["power_watts"] if current_metrics else None
                last_metrics_check = current_time

            thinking_stats = self.thinking_filter.get_stats()

            yield {
                "token": filtered_token,
                "raw_token": raw_token,
                "is_thinking": is_thinking,
                "finish_reason": response.finish_reason,
                "metrics": {
                    "ttft": ttft,
                    "tps": tps,
                    "tokens": token_count,
                    "elapsed": elapsed,
                    "current_power_watts": current_power,
                    "thinking_tokens": thinking_stats["thinking_tokens"],
                    "regular_tokens": thinking_stats["regular_tokens"],
                    # MLX-specific metrics
                    "prompt_tokens": response.prompt_tokens,
                    "prompt_tps": response.prompt_tps,
                    "generation_tps": response.generation_tps,
                    "peak_memory_gb": response.peak_memory,
                },
            }

            # Stop monitoring after completion
            if response.finish_reason is not None:
                # Flush any remaining buffered content
                remaining, was_thinking = self.thinking_filter.flush()
                if remaining:
                    yield {
                        "token": remaining,
                        "raw_token": remaining,
                        "is_thinking": was_thinking,
                        "finish_reason": None,
                        "metrics": {
                            "ttft": ttft,
                            "tps": tps,
                            "tokens": token_count,
                            "elapsed": elapsed,
                            "current_power_watts": current_power,
                        },
                    }

                # Get final thinking stats
                final_thinking_stats = self.thinking_filter.get_stats()
                thinking_content = self.thinking_filter.get_thinking_content()

                energy_summary = self.unified_monitor.stop_monitoring()
                yield {
                    "token": "",
                    "finish_reason": "energy_summary",
                    "metrics": {
                        "energy_summary": energy_summary,
                        "energy_per_token_joules": (
                            energy_summary["total_energy_joules"] / token_count
                            if token_count > 0
                            else 0
                        ),
                        "thinking_stats": final_thinking_stats,
                        "thinking_content": thinking_content if DEBUG_INFERENCE else None,
                        # MLX-specific final metrics
                        "prompt_tokens": response.prompt_tokens,
                        "prompt_tps": response.prompt_tps,
                        "generation_tps": response.generation_tps,
                        "peak_memory_gb": response.peak_memory,
                    },
                }

    def generate_with_thinking_budget(
        self,
        messages: List[Dict[str, str]],
        thinking_budget: Optional[int] = None,
        **kwargs,
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate with a cap on thinking tokens (two-phase approach).

        When ``thinking_budget`` is set, generation proceeds normally until
        the model has produced that many thinking tokens.  At that point
        generation is stopped, ``</think>\\n`` is appended to the accumulated
        output, and a *second* generation pass is started so the model sees
        its thinking as complete and produces the answer directly.

        Special case: ``thinking_budget=0`` pre-fills the prompt with
        ``<think>\\n</think>\\n`` so the model skips thinking entirely.

        When ``thinking_budget is None`` the call is equivalent to the normal
        ``generate()`` (no budget enforced).

        Yields the same dict format as ``generate()`` with extra metadata:
            - ``thinking_budget``: the budget that was set
            - ``thinking_was_truncated``: whether the budget was hit
        """
        if self.model is None or self.config is None:
            raise RuntimeError("No model loaded")

        # Unlimited budget -> delegate to normal generate
        if thinking_budget is None:
            for chunk in self.generate(messages, stream=True, **kwargs):
                chunk["thinking_budget"] = None
                chunk["thinking_was_truncated"] = False
                yield chunk
            return

        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors

        # Merge kwargs with config
        config = self.config.model_copy()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Format prompt using tokenizer's chat template
        prompt = self._format_prompt(messages, config)

        # Build sampler / logits processors (reused for both phases)
        sampler = make_sampler(
            temp=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else 0,
        )
        logits_processors = None
        if config.repeat_penalty and config.repeat_penalty != 1.0:
            logits_processors = make_logits_processors(
                repetition_penalty=config.repeat_penalty,
            )

        gen_kwargs = {"sampler": sampler}
        if logits_processors:
            gen_kwargs["logits_processors"] = logits_processors

        # ── budget == 0: disable thinking via empty think block ──────────
        if thinking_budget == 0:
            # Append empty thinking block so model skips straight to answer
            prompt = prompt + "<think>\n</think>\n"
            remaining_tokens = config.max_tokens

            start_time = time.time()
            first_token_time = None
            token_count = 0
            self.thinking_filter.reset()
            self.unified_monitor.start_monitoring()

            last_metrics_check = start_time
            metrics_check_interval = 1.0
            current_power = None

            for response in stream_generate(
                self.model, self.tokenizer, prompt=prompt,
                max_tokens=remaining_tokens, **gen_kwargs,
            ):
                if first_token_time is None:
                    first_token_time = time.time()

                raw_token = response.text
                filtered_token, is_thinking = self.thinking_filter.filter_token(raw_token)
                token_count += 1

                current_time = time.time()
                ttft = (first_token_time - start_time) * 1000 if first_token_time else None
                elapsed = current_time - start_time
                tps = token_count / elapsed if elapsed > 0 else 0

                if current_time - last_metrics_check >= metrics_check_interval:
                    current_metrics = self.unified_monitor.get_current_metrics()
                    current_power = current_metrics["power_watts"] if current_metrics else None
                    last_metrics_check = current_time

                thinking_stats = self.thinking_filter.get_stats()

                yield {
                    "token": filtered_token,
                    "raw_token": raw_token,
                    "is_thinking": is_thinking,
                    "finish_reason": response.finish_reason,
                    "thinking_budget": 0,
                    "thinking_was_truncated": True,
                    "metrics": {
                        "ttft": ttft, "tps": tps, "tokens": token_count,
                        "elapsed": elapsed, "current_power_watts": current_power,
                        "thinking_tokens": thinking_stats["thinking_tokens"],
                        "regular_tokens": thinking_stats["regular_tokens"],
                        "prompt_tokens": response.prompt_tokens,
                        "prompt_tps": response.prompt_tps,
                        "generation_tps": response.generation_tps,
                        "peak_memory_gb": response.peak_memory,
                    },
                }

                if response.finish_reason is not None:
                    remaining, was_thinking = self.thinking_filter.flush()
                    if remaining:
                        yield {
                            "token": remaining, "raw_token": remaining,
                            "is_thinking": was_thinking, "finish_reason": None,
                            "thinking_budget": 0, "thinking_was_truncated": True,
                            "metrics": {"ttft": ttft, "tps": tps, "tokens": token_count,
                                        "elapsed": elapsed, "current_power_watts": current_power},
                        }
                    final_thinking_stats = self.thinking_filter.get_stats()
                    energy_summary = self.unified_monitor.stop_monitoring()
                    yield {
                        "token": "", "finish_reason": "energy_summary",
                        "thinking_budget": 0, "thinking_was_truncated": True,
                        "metrics": {
                            "energy_summary": energy_summary,
                            "energy_per_token_joules": (
                                energy_summary["total_energy_joules"] / token_count
                                if token_count > 0 else 0
                            ),
                            "thinking_stats": final_thinking_stats,
                            "prompt_tokens": response.prompt_tokens,
                            "prompt_tps": response.prompt_tps,
                            "generation_tps": response.generation_tps,
                            "peak_memory_gb": response.peak_memory,
                        },
                    }
            return

        # ── budget > 0: two-phase generation ─────────────────────────────
        start_time = time.time()
        first_token_time = None
        token_count = 0
        thinking_token_count = 0
        was_truncated = False
        raw_accumulated = ""

        self.thinking_filter.reset()
        self.unified_monitor.start_monitoring()

        last_metrics_check = start_time
        metrics_check_interval = 1.0
        current_power = None

        # Phase 1: generate up to thinking_budget thinking tokens
        phase1_max = thinking_budget + 512  # allow some non-thinking overhead
        for response in stream_generate(
            self.model, self.tokenizer, prompt=prompt,
            max_tokens=phase1_max, **gen_kwargs,
        ):
            if first_token_time is None:
                first_token_time = time.time()

            raw_token = response.text
            raw_accumulated += raw_token
            filtered_token, is_thinking = self.thinking_filter.filter_token(raw_token)
            token_count += 1

            if is_thinking:
                thinking_token_count += 1

            current_time = time.time()
            ttft = (first_token_time - start_time) * 1000 if first_token_time else None
            elapsed = current_time - start_time
            tps = token_count / elapsed if elapsed > 0 else 0

            if current_time - last_metrics_check >= metrics_check_interval:
                current_metrics = self.unified_monitor.get_current_metrics()
                current_power = current_metrics["power_watts"] if current_metrics else None
                last_metrics_check = current_time

            thinking_stats = self.thinking_filter.get_stats()

            yield {
                "token": filtered_token,
                "raw_token": raw_token,
                "is_thinking": is_thinking,
                "finish_reason": None,
                "thinking_budget": thinking_budget,
                "thinking_was_truncated": False,
                "metrics": {
                    "ttft": ttft, "tps": tps, "tokens": token_count,
                    "elapsed": elapsed, "current_power_watts": current_power,
                    "thinking_tokens": thinking_stats["thinking_tokens"],
                    "regular_tokens": thinking_stats["regular_tokens"],
                    "prompt_tokens": response.prompt_tokens,
                    "prompt_tps": response.prompt_tps,
                    "generation_tps": response.generation_tps,
                    "peak_memory_gb": response.peak_memory,
                },
            }

            # Check if model finished on its own
            if response.finish_reason is not None:
                # Model finished within budget -- wrap up normally
                remaining, was_thinking_rem = self.thinking_filter.flush()
                if remaining:
                    yield {
                        "token": remaining, "raw_token": remaining,
                        "is_thinking": was_thinking_rem, "finish_reason": None,
                        "thinking_budget": thinking_budget,
                        "thinking_was_truncated": False,
                        "metrics": {"ttft": ttft, "tps": tps, "tokens": token_count,
                                    "elapsed": elapsed, "current_power_watts": current_power},
                    }
                final_thinking_stats = self.thinking_filter.get_stats()
                energy_summary = self.unified_monitor.stop_monitoring()
                yield {
                    "token": "", "finish_reason": "energy_summary",
                    "thinking_budget": thinking_budget,
                    "thinking_was_truncated": False,
                    "metrics": {
                        "energy_summary": energy_summary,
                        "energy_per_token_joules": (
                            energy_summary["total_energy_joules"] / token_count
                            if token_count > 0 else 0
                        ),
                        "thinking_stats": final_thinking_stats,
                        "prompt_tokens": response.prompt_tokens,
                        "prompt_tps": response.prompt_tps,
                        "generation_tps": response.generation_tps,
                        "peak_memory_gb": response.peak_memory,
                    },
                }
                return

            # Budget hit -- break out of phase 1
            if thinking_token_count >= thinking_budget:
                was_truncated = True
                break

        if not was_truncated:
            # Phase 1 ran out of max_tokens without hitting budget
            # and model didn't finish -- still wrap up
            remaining, was_thinking_rem = self.thinking_filter.flush()
            if remaining:
                yield {
                    "token": remaining, "raw_token": remaining,
                    "is_thinking": was_thinking_rem, "finish_reason": None,
                    "thinking_budget": thinking_budget,
                    "thinking_was_truncated": False,
                    "metrics": {"ttft": ttft, "tps": tps, "tokens": token_count,
                                "elapsed": elapsed, "current_power_watts": current_power},
                }
            final_thinking_stats = self.thinking_filter.get_stats()
            energy_summary = self.unified_monitor.stop_monitoring()
            yield {
                "token": "", "finish_reason": "energy_summary",
                "thinking_budget": thinking_budget,
                "thinking_was_truncated": False,
                "metrics": {
                    "energy_summary": energy_summary,
                    "energy_per_token_joules": (
                        energy_summary["total_energy_joules"] / token_count
                        if token_count > 0 else 0
                    ),
                    "thinking_stats": final_thinking_stats,
                },
            }
            return

        # ── Phase 2: inject </think>\n and continue generation ───────────
        # Build new prompt = original prompt + raw phase-1 output + </think>\n
        phase2_prompt = prompt + raw_accumulated + "</think>\n"
        remaining_tokens = max(config.max_tokens - token_count, 256)

        if DEBUG_INFERENCE:
            print(f"[DEBUG] Thinking budget hit ({thinking_token_count}/{thinking_budget}). "
                  f"Starting phase 2 with {remaining_tokens} remaining tokens.")

        # Reset thinking filter for phase 2 (model should no longer think)
        self.thinking_filter.reset()

        for response in stream_generate(
            self.model, self.tokenizer, prompt=phase2_prompt,
            max_tokens=remaining_tokens, **gen_kwargs,
        ):
            raw_token = response.text
            filtered_token, is_thinking = self.thinking_filter.filter_token(raw_token)
            token_count += 1

            current_time = time.time()
            elapsed = current_time - start_time
            tps = token_count / elapsed if elapsed > 0 else 0

            if current_time - last_metrics_check >= metrics_check_interval:
                current_metrics = self.unified_monitor.get_current_metrics()
                current_power = current_metrics["power_watts"] if current_metrics else None
                last_metrics_check = current_time

            thinking_stats_phase2 = self.thinking_filter.get_stats()

            yield {
                "token": filtered_token,
                "raw_token": raw_token,
                "is_thinking": is_thinking,
                "finish_reason": response.finish_reason,
                "thinking_budget": thinking_budget,
                "thinking_was_truncated": True,
                "metrics": {
                    "ttft": ttft, "tps": tps, "tokens": token_count,
                    "elapsed": elapsed, "current_power_watts": current_power,
                    "thinking_tokens": thinking_token_count + thinking_stats_phase2["thinking_tokens"],
                    "regular_tokens": thinking_stats_phase2["regular_tokens"],
                    "prompt_tokens": response.prompt_tokens,
                    "prompt_tps": response.prompt_tps,
                    "generation_tps": response.generation_tps,
                    "peak_memory_gb": response.peak_memory,
                },
            }

            if response.finish_reason is not None:
                remaining, was_thinking_rem = self.thinking_filter.flush()
                if remaining:
                    yield {
                        "token": remaining, "raw_token": remaining,
                        "is_thinking": was_thinking_rem, "finish_reason": None,
                        "thinking_budget": thinking_budget,
                        "thinking_was_truncated": True,
                        "metrics": {"ttft": ttft, "tps": tps, "tokens": token_count,
                                    "elapsed": elapsed, "current_power_watts": current_power},
                    }

                final_phase2_stats = self.thinking_filter.get_stats()
                energy_summary = self.unified_monitor.stop_monitoring()
                yield {
                    "token": "", "finish_reason": "energy_summary",
                    "thinking_budget": thinking_budget,
                    "thinking_was_truncated": True,
                    "metrics": {
                        "energy_summary": energy_summary,
                        "energy_per_token_joules": (
                            energy_summary["total_energy_joules"] / token_count
                            if token_count > 0 else 0
                        ),
                        "thinking_stats": {
                            "thinking_tokens": thinking_token_count + final_phase2_stats["thinking_tokens"],
                            "regular_tokens": final_phase2_stats["regular_tokens"],
                            "total_tokens": token_count,
                            "thinking_ratio": (
                                (thinking_token_count + final_phase2_stats["thinking_tokens"])
                                / token_count if token_count > 0 else 0
                            ),
                            "thinking_budget": thinking_budget,
                            "thinking_was_truncated": True,
                        },
                        "prompt_tokens": response.prompt_tokens,
                        "prompt_tps": response.prompt_tps,
                        "generation_tps": response.generation_tps,
                        "peak_memory_gb": response.peak_memory,
                    },
                }

    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        **kwargs,
    ) -> Tuple[str, Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        """Generate a response with tool calling support.

        Args:
            messages: List of chat messages
            tools: List of tool definitions
            tool_choice: Tool selection mode ("auto", "required", "none")
            **kwargs: Override inference config parameters

        Returns:
            Tuple of (response_text, tool_calls, metrics)
        """
        if self.model is None or self.config is None:
            raise RuntimeError("No model loaded")

        # Add tools to config
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

        # Collect full response and track inter-token latencies
        response_text = ""
        metrics = {}
        inter_token_latencies = []
        last_token_time = None
        thinking_tokens = 0
        regular_tokens = 0

        for chunk in self.generate(messages, stream=True, **kwargs):
            current_time = time.time()
            if last_token_time is not None:
                inter_token_latencies.append((current_time - last_token_time) * 1000)
            last_token_time = current_time

            # Only add non-thinking tokens to response
            response_text += chunk["token"]  # Already filtered

            # Track thinking tokens
            if chunk.get("is_thinking", False):
                thinking_tokens += 1
            else:
                regular_tokens += 1

            metrics = chunk["metrics"]

        # Parse tool calls from response
        tool_calls = self._parse_tool_calls(response_text)

        # Add inter-token latencies to metrics
        if inter_token_latencies:
            metrics["inter_token_latencies"] = inter_token_latencies

        # Ensure thinking stats are in metrics
        if "thinking_stats" not in metrics:
            metrics["thinking_stats"] = {
                "thinking_tokens": thinking_tokens,
                "regular_tokens": regular_tokens,
                "total_tokens": thinking_tokens + regular_tokens,
                "thinking_ratio": (
                    thinking_tokens / (thinking_tokens + regular_tokens)
                    if (thinking_tokens + regular_tokens) > 0
                    else 0
                ),
            }

        return response_text, tool_calls, metrics

    def batch_generate(
        self,
        prompts: List[List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts in a single batch.

        This uses MLX's native batch_generate which processes multiple prompts
        efficiently on Apple Silicon's unified memory. The GPU can process
        prefill for multiple sequences and interleave decoding.

        Args:
            prompts: List of message lists, each being a conversation
            max_tokens: Maximum tokens per response (overrides config)
            **kwargs: Additional parameters

        Returns:
            List of result dicts, each containing:
                - text: Generated text
                - metrics: Generation metrics
                - tool_calls: Parsed tool calls (if any)

        Notes on batching performance:
            - MLX batch_generate handles prefill and decoding efficiently
            - On M2 Max (64GB), can batch 20+ prompts for a 4B model
            - Prefill is batched (configurable via prefill_batch_size)
            - Decoding interleaves across sequences
            - Memory scales linearly with batch_size * context_length
        """
        if self.model is None or self.config is None or self.tokenizer is None:
            raise RuntimeError("No model loaded")

        from mlx_lm import batch_generate as mlx_batch_generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors

        config = self.config.model_copy()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        effective_max_tokens = max_tokens or config.max_tokens

        # Build sampler
        sampler = make_sampler(
            temp=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else 0,
        )

        # Build logits processors
        logits_processors = None
        if config.repeat_penalty and config.repeat_penalty != 1.0:
            logits_processors = make_logits_processors(
                repetition_penalty=config.repeat_penalty,
            )

        # Format all prompts and tokenize
        formatted_prompts = []
        for messages in prompts:
            prompt_str = self._format_prompt(messages, config)
            # Tokenize: mlx_lm batch_generate expects List[List[int]]
            add_special_tokens = (
                self.tokenizer.bos_token is None
                or not prompt_str.startswith(self.tokenizer.bos_token)
            )
            tokens = self.tokenizer.encode(prompt_str, add_special_tokens=add_special_tokens)
            formatted_prompts.append(tokens)

        if DEBUG_INFERENCE:
            print(f"[DEBUG MLXInferenceService] Batch generating {len(prompts)} prompts")
            print(f"  - max_tokens: {effective_max_tokens}")
            print(f"  - temperature: {config.temperature}")
            prompt_lengths = [len(p) for p in formatted_prompts]
            print(f"  - prompt token lengths: min={min(prompt_lengths)}, max={max(prompt_lengths)}, avg={sum(prompt_lengths)/len(prompt_lengths):.0f}")

        # Start monitoring
        self.unified_monitor.start_monitoring()
        start_time = time.time()

        # Run batch generation
        batch_kwargs = {
            "max_tokens": effective_max_tokens,
            "sampler": sampler,
        }
        if logits_processors:
            batch_kwargs["logits_processors"] = logits_processors

        batch_response = mlx_batch_generate(
            self.model,
            self.tokenizer,
            prompts=formatted_prompts,
            **batch_kwargs,
        )

        elapsed = time.time() - start_time
        energy_summary = self.unified_monitor.stop_monitoring()

        # Process results
        results = []
        stats = batch_response.stats

        for i, text in enumerate(batch_response.texts):
            # Filter thinking tokens from completed text
            self.thinking_filter.reset()
            filtered_text = text
            thinking_stats_data = {"thinking_tokens": 0, "regular_tokens": 0}

            # Use post-processing filter for batch (no streaming)
            from ..utils.thinking_filter import remove_thinking_blocks
            filtered_text, block_stats = remove_thinking_blocks(text)

            # Parse tool calls
            model_arch = self.current_model.architecture if self.current_model else None
            tool_calls, _ = self.tool_extractor.extract_tools(filtered_text, model_arch)

            result = {
                "text": filtered_text,
                "raw_text": text,
                "tool_calls": tool_calls if tool_calls else None,
                "metrics": {
                    "elapsed": elapsed,
                    "tps": stats.generation_tps,
                    "tokens": stats.generation_tokens,
                    "prompt_tokens": stats.prompt_tokens,
                    "prompt_tps": stats.prompt_tps,
                    "generation_tps": stats.generation_tps,
                    "peak_memory_gb": stats.peak_memory,
                    "prompt_time": stats.prompt_time,
                    "generation_time": stats.generation_time,
                    "energy_summary": energy_summary,
                    "thinking_stats": block_stats,
                    "batch_size": len(prompts),
                    "batch_index": i,
                },
            }
            results.append(result)

        if DEBUG_INFERENCE:
            print(f"[DEBUG MLXInferenceService] Batch complete:")
            print(f"  - Total time: {elapsed:.2f}s")
            print(f"  - Prompt tokens: {stats.prompt_tokens}, prompt TPS: {stats.prompt_tps:.1f}")
            print(f"  - Generation tokens: {stats.generation_tokens}, gen TPS: {stats.generation_tps:.1f}")
            print(f"  - Peak memory: {stats.peak_memory:.2f} GB")

        return results

    @profile
    def _format_prompt(
        self,
        messages: List[Dict[str, str]],
        config: InferenceConfig,
    ) -> str:
        """Format messages into a prompt using the tokenizer's chat template.

        MLX models come with their own tokenizer that includes the correct
        chat template, so we prefer that over our manual Jinja templates.
        Falls back to the pocket-agent templates if apply_chat_template fails.

        Args:
            messages: List of chat messages
            config: Inference configuration

        Returns:
            Formatted prompt string
        """
        if DEBUG_INFERENCE:
            print(f"[DEBUG _format_prompt] Starting MLX prompt formatting...")

        # Try using the tokenizer's built-in chat template first
        try:
            # Build kwargs for apply_chat_template
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }

            # Some tokenizers support tools in apply_chat_template
            if config.tools:
                template_kwargs["tools"] = config.tools

            prompt = self.tokenizer.apply_chat_template(
                messages, **template_kwargs
            )
            if DEBUG_INFERENCE:
                print(f"[DEBUG _format_prompt] Used tokenizer's built-in chat template")
            return prompt
        except Exception as e:
            if DEBUG_INFERENCE:
                print(f"[DEBUG _format_prompt] Tokenizer chat template failed: {e}, falling back")

        # Fallback to pocket-agent templates
        if not self.current_model:
            raise RuntimeError("No model loaded")

        from ..utils.chat_templates import get_chat_template
        from jinja2 import Template

        tools_enabled = bool(config.tools)
        template_str = get_chat_template(
            self.current_model.architecture, tools_enabled=tools_enabled
        )

        if config.jinja:
            template = Template(template_str)
            context = {
                "messages": messages,
                "bos_token": "",
                "eos_token": "",
            }
            if config.tools:
                context["tools"] = config.tools
                context["tool_choice"] = config.tool_choice
            prompt = template.render(**context)
        else:
            prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            if messages[-1]["role"] != "assistant":
                prompt += "Assistant: "

        return prompt

    def _parse_tool_calls(self, response: str) -> Optional[List[Dict[str, Any]]]:
        """Parse tool calls from model response.

        Args:
            response: Model response text

        Returns:
            List of tool calls or None
        """
        model_arch = self.current_model.architecture if self.current_model else None
        tool_calls, error = self.tool_extractor.extract_tools(response, model_arch)

        if not tool_calls and error:
            if DEBUG_INFERENCE:
                print(f"[DEBUG] Tool extraction error: {error}")

        return tool_calls if tool_calls else None

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the loaded model.

        Returns:
            Model information or None if no model loaded
        """
        if self.model is None or self.current_model is None:
            return None

        import mlx.core as mx

        return {
            "model_id": self.current_model.id,
            "model_name": self.current_model.name,
            "architecture": self.current_model.architecture,
            "quantization": self.current_model.quantization,
            "context_length": self.config.context_length if self.config else None,
            "loaded": True,
            "backend": "mlx",
            "hf_model_id": self._hf_model_id,
            "peak_memory_gb": mx.get_peak_memory() / 1e9,
        }
