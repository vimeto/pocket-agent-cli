"""Inference service using llama-cpp-python."""

import time
import json
from typing import Dict, Any, List, Optional, Generator, Tuple
from pathlib import Path
from llama_cpp import Llama, LlamaGrammar
from jinja2 import Template
from ..config import InferenceConfig, Model
from ..utils.chat_templates import get_chat_template
from ..utils.tool_extractor import ToolExtractor
from ..monitoring.unified_monitor import UnifiedMonitor
import os
DEBUG_INFERENCE = os.environ.get("DEBUG_INFERENCE", "").lower() == "true"

# Import profile decorator
try:
    from line_profiler import profile
except ImportError:
    # Define a no-op decorator if line_profiler is not available
    def profile(func):
        return func


class InferenceService:
    """Service for LLM inference using llama.cpp."""

    def __init__(self):
        self.llama: Optional[Llama] = None
        self.current_model: Optional[Model] = None
        self.config: Optional[InferenceConfig] = None
        self.tool_extractor = ToolExtractor()
        self.unified_monitor = UnifiedMonitor(sample_interval=2.0)  # Optimized unified monitor

    def load_model(self, model: Model, config: InferenceConfig) -> None:
        """Load a model into memory.

        Args:
            model: Model to load
            config: Inference configuration
        """
        if not model.downloaded or not model.path or not model.path.exists():
            raise ValueError(f"Model {model.id} is not downloaded")

        # Unload current model if any
        if self.llama:
            del self.llama
            self.llama = None

        # Initialize llama.cpp with GPU acceleration
        import os
        import platform

        # Optimize thread count based on system
        n_threads = config.n_threads
        if n_threads == -1:
            if platform.system() == "Darwin":
                # M3 Max optimal configuration: use performance cores only
                n_threads = 8  # M3 Max has 8 performance cores
            else:
                # Linux/HPC: use available CPU cores
                n_threads = os.cpu_count() // 2  # Use half of available cores

        # Check for CUDA availability
        cuda_available = os.environ.get("CUDA_VISIBLE_DEVICES") is not None
        
        kwargs = {
            "model_path": str(model.path),
            "use_mmap": True,
            "verbose": False,
            "n_ctx": 4096,  # Increased from 2048 to handle full_tool mode
            "n_gpu_layers": -1 if cuda_available or platform.system() == "Darwin" else 0,
            "flash_attn": True,
            "n_threads": n_threads,
            "n_batch": 512,  # Optimize batch size for GPU
        }
        
        # Add CUDA-specific options if available
        if cuda_available:
            kwargs["cuda_device"] = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
            kwargs["tensor_split"] = None  # Let CUDA handle tensor splitting
            print(f"[INFO] Loading model with CUDA support on device {kwargs['cuda_device']}")
        elif platform.system() == "Darwin":
            print("[INFO] Loading model with Metal acceleration on macOS")
        else:
            print("[INFO] Loading model with CPU only")

        self.llama = Llama(**kwargs)

        self.current_model = model
        self.config = config

    def unload_model(self) -> None:
        """Unload the current model from memory."""
        if self.llama:
            del self.llama
            self.llama = None
        self.current_model = None
        self.config = None

    @profile
    def generate(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate a response from the model.

        Args:
            messages: List of chat messages
            stream: Whether to stream tokens
            **kwargs: Override inference config parameters

        Yields:
            Token chunks with metadata
        """
        if not self.llama or not self.config:
            raise RuntimeError("No model loaded")

        # Merge kwargs with config
        config = self.config.model_copy()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Format prompt
        prompt = self._format_prompt(messages, config)
        if DEBUG_INFERENCE:
            print(f"[DEBUG InferenceService] Prompt length: {len(prompt)} characters")

        # Track metrics
        start_time = time.time()
        first_token_time = None
        token_count = 0

        # Start unified monitoring
        if DEBUG_INFERENCE:
            print(f"[DEBUG InferenceService] Starting unified monitoring...")
        self.unified_monitor.start_monitoring()

        # Generate with optimized settings
        if DEBUG_INFERENCE:
            print(f"[DEBUG InferenceService] Starting LLM generation with config:")
            print(f"  - max_tokens: {config.max_tokens}")
            print(f"  - temperature: {config.temperature}")
            print(f"  - stream: {stream}")

        response_iter = self.llama(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=config.stop_tokens,
            stream=stream,
        )

        if DEBUG_INFERENCE:
            print(f"[DEBUG InferenceService] LLM call initiated, waiting for tokens...")

        if stream:
            last_metrics_check = start_time
            metrics_check_interval = 1.0  # Check metrics every 1 second
            current_power = None

            for chunk in response_iter:
                if first_token_time is None:
                    first_token_time = time.time()
                    if DEBUG_INFERENCE:
                        print(f"[DEBUG InferenceService] First token received! TTFT: {(first_token_time - start_time) * 1000:.1f}ms")

                token_count += 1

                # Calculate metrics
                current_time = time.time()
                ttft = (first_token_time - start_time) * 1000 if first_token_time else None
                elapsed = current_time - start_time
                tps = token_count / elapsed if elapsed > 0 else 0

                # Only check power metrics periodically to avoid overhead
                if current_time - last_metrics_check >= metrics_check_interval:
                    current_metrics = self.unified_monitor.get_current_metrics()
                    current_power = current_metrics["power_watts"] if current_metrics else None
                    last_metrics_check = current_time

                yield {
                    "token": chunk["choices"][0]["text"],
                    "finish_reason": chunk["choices"][0].get("finish_reason"),
                    "metrics": {
                        "ttft": ttft,
                        "tps": tps,
                        "tokens": token_count,
                        "elapsed": elapsed,
                        "current_power_watts": current_power,
                    }
                }

                # Stop monitoring after completion
                if chunk["choices"][0].get("finish_reason"):
                    energy_summary = self.unified_monitor.stop_monitoring()
                    yield {
                        "token": "",
                        "finish_reason": "energy_summary",
                        "metrics": {
                            "energy_summary": energy_summary,
                            "energy_per_token_joules": energy_summary["total_energy_joules"] / token_count if token_count > 0 else 0,
                        }
                    }
        else:
            # Non-streaming response
            result = response_iter
            end_time = time.time()
            elapsed = end_time - start_time

            text = result["choices"][0]["text"]
            token_count = result["usage"]["completion_tokens"]

            # Stop unified monitoring
            energy_summary = self.unified_monitor.stop_monitoring()

            yield {
                "token": text,
                "finish_reason": result["choices"][0].get("finish_reason"),
                "metrics": {
                    "ttft": None,  # Not applicable for non-streaming
                    "tps": token_count / elapsed if elapsed > 0 else 0,
                    "tokens": token_count,
                    "elapsed": elapsed,
                    "energy_summary": energy_summary,
                    "energy_per_token_joules": energy_summary["total_energy_joules"] / token_count if token_count > 0 else 0,
                }
            }

    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        **kwargs
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
        if not self.llama or not self.config:
            raise RuntimeError("No model loaded")

        # # Add tools to config
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

        # Collect full response and track inter-token latencies
        response_text = ""
        metrics = {}
        inter_token_latencies = []
        last_token_time = None

        for chunk in self.generate(messages, stream=True, **kwargs):
            current_time = time.time()
            if last_token_time is not None:
                inter_token_latencies.append((current_time - last_token_time) * 1000)
            last_token_time = current_time
            response_text += chunk["token"]
            metrics = chunk["metrics"]

        # Parse tool calls from response
        tool_calls = self._parse_tool_calls(response_text)
        
        # Add inter-token latencies to metrics
        if inter_token_latencies:
            metrics['inter_token_latencies'] = inter_token_latencies

        return response_text, tool_calls, metrics

    @profile
    def _format_prompt(
        self,
        messages: List[Dict[str, str]],
        config: InferenceConfig
    ) -> str:
        """Format messages into a prompt using the appropriate template.

        Args:
            messages: List of chat messages
            config: Inference configuration

        Returns:
            Formatted prompt string
        """
        if DEBUG_INFERENCE:
            print(f"[DEBUG _format_prompt] Starting prompt formatting...")
        if not self.current_model:
            raise RuntimeError("No model loaded")

        # Get chat template for the model architecture
        tools_enabled = bool(config.tools)
        template_str = get_chat_template(self.current_model.architecture, tools_enabled=tools_enabled)

        if config.jinja:
            # Use Jinja2 template
            template = Template(template_str)

            # Add tools if present
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
            # Simple string formatting
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
        # Use the robust tool extractor with model architecture
        model_arch = self.current_model.architecture if self.current_model else None
        tool_calls, error = self.tool_extractor.extract_tools(response, model_arch)

        # Log if no tools found for debugging
        if not tool_calls and error:
            print(f"[DEBUG] Tool extraction error: {error}")

        return tool_calls if tool_calls else None

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the loaded model.

        Returns:
            Model information or None if no model loaded
        """
        if not self.llama or not self.current_model:
            return None

        return {
            "model_id": self.current_model.id,
            "model_name": self.current_model.name,
            "architecture": self.current_model.architecture,
            "quantization": self.current_model.quantization,
            "context_length": self.config.context_length if self.config else None,
            "loaded": True,
        }
