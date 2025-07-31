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


class InferenceService:
    """Service for LLM inference using llama.cpp."""
    
    def __init__(self):
        self.llama: Optional[Llama] = None
        self.current_model: Optional[Model] = None
        self.config: Optional[InferenceConfig] = None
        self.tool_extractor = ToolExtractor()
        
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
        
        # Initialize llama.cpp with Metal acceleration on macOS
        import os
        
        # Auto-detect optimal thread count
        n_threads = config.n_threads
        if n_threads == -1:
            n_threads = os.cpu_count() or 4
            # Leave 1-2 cores for system on high core count machines
            if n_threads > 8:
                n_threads = n_threads - 2
        
        kwargs = {
            "model_path": str(model.path),
            "n_ctx": config.context_length,
            "n_batch": config.n_batch,
            "n_threads": n_threads,
            "use_mlock": config.use_mlock,
            "use_mmap": config.use_mmap,
            "verbose": False,
            "n_threads_batch": n_threads,  # Use same threads for batch processing
            "rope_scaling_type": 1,  # Linear RoPE scaling for better performance
        }
        
        # Enable Metal acceleration on macOS
        import platform
        if platform.system() == "Darwin":
            kwargs["n_gpu_layers"] = -1  # Offload all layers to Metal
            kwargs["offload_kqv"] = True  # Offload KV cache to Metal
            kwargs["f16_kv"] = True  # Use half precision for KV cache
            kwargs["flash_attn"] = True  # Enable Flash Attention if available
            kwargs["mul_mat_q"] = True  # Use quantized matrix multiplication
        
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
        
        
        # Track metrics
        start_time = time.time()
        first_token_time = None
        token_count = 0
        
        # Generate with optimized settings
        response_iter = self.llama(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=config.stop_tokens,
            stream=stream,
            cache_prompt=True,  # Cache prompt for faster subsequent calls
            penalize_nl=False,  # Don't penalize newlines unnecessarily
        )
        
        if stream:
            for chunk in response_iter:
                if first_token_time is None:
                    first_token_time = time.time()
                
                token_count += 1
                
                # Calculate metrics
                current_time = time.time()
                ttft = (first_token_time - start_time) * 1000 if first_token_time else None
                elapsed = current_time - start_time
                tps = token_count / elapsed if elapsed > 0 else 0
                
                yield {
                    "token": chunk["choices"][0]["text"],
                    "finish_reason": chunk["choices"][0].get("finish_reason"),
                    "metrics": {
                        "ttft": ttft,
                        "tps": tps,
                        "tokens": token_count,
                        "elapsed": elapsed,
                    }
                }
        else:
            # Non-streaming response
            result = response_iter
            end_time = time.time()
            elapsed = end_time - start_time
            
            text = result["choices"][0]["text"]
            token_count = result["usage"]["completion_tokens"]
            
            yield {
                "token": text,
                "finish_reason": result["choices"][0].get("finish_reason"),
                "metrics": {
                    "ttft": None,  # Not applicable for non-streaming
                    "tps": token_count / elapsed if elapsed > 0 else 0,
                    "tokens": token_count,
                    "elapsed": elapsed,
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
        
        # Add tools to config
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
        
        # Collect full response
        response_text = ""
        metrics = {}
        
        for chunk in self.generate(messages, stream=True, **kwargs):
            response_text += chunk["token"]
            metrics = chunk["metrics"]
        
        # Parse tool calls from response
        tool_calls = self._parse_tool_calls(response_text)
        
        return response_text, tool_calls, metrics
    
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