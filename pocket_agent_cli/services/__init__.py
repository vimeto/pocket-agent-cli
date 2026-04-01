"""Service modules."""

from .inference_service import InferenceService

# MLX backend is optional (requires macOS Apple Silicon + mlx-lm)
try:
    from .mlx_inference_service import MLXInferenceService
except ImportError:
    MLXInferenceService = None

__all__ = ["InferenceService", "MLXInferenceService"]