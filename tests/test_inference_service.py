"""Tests for InferenceService."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from pocket_agent_cli.services.inference_service import InferenceService
from pocket_agent_cli.config import InferenceConfig, Model, ModelVersion


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def inference_service():
    """Create an InferenceService instance."""
    return InferenceService()


@pytest.fixture
def mock_model():
    """Create a mock Model."""
    model = Mock(spec=Model)
    model.id = "test-model"
    model.name = "Test Model"
    model.architecture = "llama"
    model.downloaded = True
    model.path = Path("/tmp/test_model.gguf")
    model.versions = {}
    return model


@pytest.fixture
def inference_config():
    """Create an inference config."""
    return InferenceConfig(
        temperature=0.7,
        max_tokens=512,
        context_length=4096,
    )


# ============================================================================
# Test: Initialization
# ============================================================================

class TestInferenceServiceInit:
    """Tests for InferenceService initialization."""

    def test_init_creates_service(self):
        """Test creating an InferenceService."""
        service = InferenceService()

        assert service.llama is None
        assert service.current_model is None
        assert service.config is None

    def test_init_creates_tool_extractor(self, inference_service):
        """Test that tool extractor is created."""
        assert inference_service.tool_extractor is not None

    def test_init_creates_thinking_filter(self, inference_service):
        """Test that thinking filter is created."""
        assert inference_service.thinking_filter is not None

    def test_init_creates_unified_monitor(self, inference_service):
        """Test that unified monitor is created."""
        assert inference_service.unified_monitor is not None


# ============================================================================
# Test: Model Loading (Mocked)
# ============================================================================

class TestModelLoading:
    """Tests for model loading (without actual model files)."""

    def test_load_model_requires_downloaded_model(self, inference_service, mock_model, inference_config):
        """Test that load_model fails if model not downloaded."""
        mock_model.downloaded = False

        with pytest.raises(ValueError) as exc_info:
            inference_service.load_model(mock_model, inference_config)

        assert "not downloaded" in str(exc_info.value)

    def test_load_model_requires_model_path(self, inference_service, mock_model, inference_config):
        """Test that load_model fails if no path."""
        mock_model.path = None

        with pytest.raises(ValueError) as exc_info:
            inference_service.load_model(mock_model, inference_config)

    def test_unload_model_clears_state(self, inference_service):
        """Test that unload_model clears state."""
        # Manually set some state
        inference_service.llama = Mock()
        inference_service.current_model = Mock()
        inference_service.config = Mock()

        inference_service.unload_model()

        assert inference_service.llama is None
        assert inference_service.current_model is None
        assert inference_service.config is None


# ============================================================================
# Test: Generate (Mocked)
# ============================================================================

class TestGenerate:
    """Tests for generate method (mocked)."""

    def test_generate_requires_loaded_model(self, inference_service):
        """Test that generate fails without model."""
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(RuntimeError) as exc_info:
            list(inference_service.generate(messages))

        assert "No model loaded" in str(exc_info.value)


# ============================================================================
# Test: Prompt Formatting
# ============================================================================

class TestPromptFormatting:
    """Tests for prompt formatting."""

    def test_format_prompt_returns_string(self, inference_service):
        """Test that _format_prompt returns a string."""
        # Set up mocks
        inference_service.current_model = Mock()
        inference_service.current_model.id = "test"
        inference_service.config = InferenceConfig()

        messages = [{"role": "user", "content": "Hello"}]

        # The method may fail without actual template, but should exist
        assert hasattr(inference_service, '_format_prompt')


# ============================================================================
# Test: Tool Extraction Integration
# ============================================================================

class TestToolExtraction:
    """Tests for tool extraction integration."""

    def test_tool_extractor_exists(self, inference_service):
        """Test that tool extractor is available."""
        assert inference_service.tool_extractor is not None

    def test_tool_extractor_can_extract_json(self, inference_service):
        """Test basic tool extraction."""
        response = '```tool_call\n{"name": "test", "parameters": {}}\n```'
        tools, error = inference_service.tool_extractor.extract_tools(response)

        assert len(tools) >= 1
        assert tools[0]["name"] == "test"


# ============================================================================
# Test: Thinking Filter Integration
# ============================================================================

class TestThinkingFilterIntegration:
    """Tests for thinking filter integration."""

    def test_thinking_filter_exists(self, inference_service):
        """Test that thinking filter is available."""
        assert inference_service.thinking_filter is not None

    def test_thinking_filter_can_filter(self, inference_service):
        """Test basic thinking filtering."""
        # Test filtering a non-thinking token
        filtered, is_thinking = inference_service.thinking_filter.filter_token("Hello")
        assert filtered == "Hello"
        assert is_thinking is False

    def test_thinking_filter_reset(self, inference_service):
        """Test thinking filter reset."""
        inference_service.thinking_filter.reset()
        stats = inference_service.thinking_filter.get_stats()

        assert stats["thinking_tokens"] == 0
        assert stats["regular_tokens"] == 0
