"""Tests for the config module."""

import pytest
from pathlib import Path
from pydantic import ValidationError


class TestInferenceConfig:
    """Tests for InferenceConfig."""

    def test_inference_config_defaults(self):
        """Test default values are set correctly."""
        from pocket_agent_cli.config import InferenceConfig

        config = InferenceConfig()

        assert config.temperature == 0.1
        assert config.max_tokens == 100
        assert config.top_p == 0.8
        assert config.top_k == 40
        assert config.repeat_penalty == 1.1
        assert config.context_length == 4096
        assert config.jinja is True
        assert config.tool_choice == "auto"
        assert config.n_threads == 12
        assert config.n_batch == 512

    def test_inference_config_custom_values(self):
        """Test custom values are accepted."""
        from pocket_agent_cli.config import InferenceConfig

        config = InferenceConfig(
            temperature=0.9,
            max_tokens=4096,
            top_p=0.95,
            context_length=8192,
        )

        assert config.temperature == 0.9
        assert config.max_tokens == 4096
        assert config.top_p == 0.95
        assert config.context_length == 8192

    def test_inference_config_temperature_validation(self):
        """Test temperature bounds validation."""
        from pocket_agent_cli.config import InferenceConfig

        # Valid temperature
        config = InferenceConfig(temperature=1.5)
        assert config.temperature == 1.5

        # Invalid temperature (too high)
        with pytest.raises(ValidationError):
            InferenceConfig(temperature=2.5)

        # Invalid temperature (negative)
        with pytest.raises(ValidationError):
            InferenceConfig(temperature=-0.1)

    def test_inference_config_max_tokens_validation(self):
        """Test max_tokens bounds validation."""
        from pocket_agent_cli.config import InferenceConfig

        # Valid max_tokens
        config = InferenceConfig(max_tokens=8192)
        assert config.max_tokens == 8192

        # Invalid max_tokens (zero)
        with pytest.raises(ValidationError):
            InferenceConfig(max_tokens=0)

    def test_inference_config_stop_tokens(self):
        """Test stop tokens list."""
        from pocket_agent_cli.config import InferenceConfig

        config = InferenceConfig()
        assert isinstance(config.stop_tokens, list)
        assert "<|im_end|>" in config.stop_tokens
        assert "<|eot_id|>" in config.stop_tokens

    def test_inference_config_tool_choice_validation(self):
        """Test tool_choice pattern validation."""
        from pocket_agent_cli.config import InferenceConfig

        # Valid choices
        for choice in ["auto", "required", "none"]:
            config = InferenceConfig(tool_choice=choice)
            assert config.tool_choice == choice

        # Invalid choice
        with pytest.raises(ValidationError):
            InferenceConfig(tool_choice="invalid")


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_benchmark_config_defaults(self, temp_results_dir):
        """Test default values are set correctly."""
        from pocket_agent_cli.config import BenchmarkConfig

        config = BenchmarkConfig(model_name="test-model")

        assert config.model_name == "test-model"
        assert config.mode == "base"
        assert config.num_samples == 10
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.context_length == 4096
        assert config.enable_tools is True
        assert config.system_monitoring is True
        assert config.save_individual_runs is True
        assert config.compute_pass_at_k == [1, 3, 5, 10]

    def test_benchmark_config_custom_values(self, temp_results_dir):
        """Test custom values are accepted."""
        from pocket_agent_cli.config import BenchmarkConfig

        config = BenchmarkConfig(
            model_name="custom-model",
            mode="full_tool",
            num_samples=5,
            temperature=0.5,
            problems_limit=10,
            output_dir=temp_results_dir,
        )

        assert config.model_name == "custom-model"
        assert config.mode == "full_tool"
        assert config.num_samples == 5
        assert config.temperature == 0.5
        assert config.problems_limit == 10
        assert config.output_dir == temp_results_dir

    def test_benchmark_config_problem_ids(self):
        """Test problem_ids list."""
        from pocket_agent_cli.config import BenchmarkConfig

        config = BenchmarkConfig(
            model_name="test-model",
            problem_ids=[1, 2, 3, 5, 8],
        )

        assert config.problem_ids == [1, 2, 3, 5, 8]

    def test_benchmark_config_model_version(self):
        """Test model_version field."""
        from pocket_agent_cli.config import BenchmarkConfig

        config = BenchmarkConfig(
            model_name="test-model",
            model_version="F16",
        )

        assert config.model_version == "F16"


class TestBenchmarkMode:
    """Tests for BenchmarkMode."""

    def test_benchmark_mode_structure(self):
        """Test BenchmarkMode dataclass structure."""
        from pocket_agent_cli.config import BenchmarkMode

        mode = BenchmarkMode(
            name="test_mode",
            description="Test mode description",
            system_prompt="You are a test assistant.",
            user_prompt_template="{problem}",
            requires_tools=True,
            max_iterations=3,
        )

        assert mode.name == "test_mode"
        assert mode.description == "Test mode description"
        assert mode.requires_tools is True
        assert mode.max_iterations == 3


class TestBenchmarkModes:
    """Tests for predefined BENCHMARK_MODES."""

    def test_benchmark_modes_defined(self):
        """Test that all benchmark modes are defined."""
        from pocket_agent_cli.config import BENCHMARK_MODES

        assert "base" in BENCHMARK_MODES
        assert "tool_submission" in BENCHMARK_MODES
        assert "full_tool" in BENCHMARK_MODES

    def test_base_mode(self):
        """Test base mode configuration."""
        from pocket_agent_cli.config import BENCHMARK_MODES

        base = BENCHMARK_MODES["base"]
        assert base.name == "base"
        assert base.requires_tools is False
        assert base.max_iterations == 1
        assert "{problem}" in base.user_prompt_template

    def test_tool_submission_mode(self):
        """Test tool_submission mode configuration."""
        from pocket_agent_cli.config import BENCHMARK_MODES

        mode = BENCHMARK_MODES["tool_submission"]
        assert mode.name == "tool_submission"
        assert mode.requires_tools is True
        assert mode.max_iterations == 1

    def test_full_tool_mode(self):
        """Test full_tool mode configuration."""
        from pocket_agent_cli.config import BENCHMARK_MODES

        mode = BENCHMARK_MODES["full_tool"]
        assert mode.name == "full_tool"
        assert mode.requires_tools is True
        assert mode.max_iterations == 5


class TestAvailableTools:
    """Tests for tool definitions."""

    def test_available_tools_structure(self):
        """Test AVAILABLE_TOOLS has correct structure."""
        from pocket_agent_cli.config import AVAILABLE_TOOLS

        assert isinstance(AVAILABLE_TOOLS, list)
        assert len(AVAILABLE_TOOLS) >= 4  # At least 4 tools

        for tool in AVAILABLE_TOOLS:
            assert "type" in tool
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_submit_tool_defined(self):
        """Test SUBMIT_TOOL is properly defined."""
        from pocket_agent_cli.config import SUBMIT_TOOL

        assert SUBMIT_TOOL["type"] == "function"
        assert SUBMIT_TOOL["function"]["name"] == "submit_python_solution"
        assert "solution" in SUBMIT_TOOL["function"]["parameters"]["properties"]

    def test_tool_names(self):
        """Test expected tool names are present."""
        from pocket_agent_cli.config import AVAILABLE_TOOLS

        tool_names = [t["function"]["name"] for t in AVAILABLE_TOOLS]

        assert "run_python_code" in tool_names
        assert "upsert_file" in tool_names
        assert "read_file" in tool_names
        assert "submit_python_solution" in tool_names


class TestModelVersion:
    """Tests for ModelVersion."""

    def test_model_version_structure(self):
        """Test ModelVersion dataclass structure."""
        from pocket_agent_cli.config import ModelVersion

        version = ModelVersion(
            url="https://example.com/model.gguf",
            size=1000000000,
            quantization="Q4_K_M",
        )

        assert version.url == "https://example.com/model.gguf"
        assert version.size == 1000000000
        assert version.quantization == "Q4_K_M"
        assert version.downloaded is False
        assert version.path is None


class TestModel:
    """Tests for Model."""

    def test_model_structure(self):
        """Test Model dataclass structure."""
        from pocket_agent_cli.config import Model, ModelVersion

        model = Model(
            id="test-model",
            name="Test Model",
            architecture="llama",
            versions={
                "Q4_K_M": ModelVersion(
                    url="https://example.com/model-q4.gguf",
                    size=2000000000,
                    quantization="Q4_K_M",
                )
            },
            default_version="Q4_K_M",
        )

        assert model.id == "test-model"
        assert model.name == "Test Model"
        assert model.architecture == "llama"
        assert "Q4_K_M" in model.versions

    def test_model_get_version(self):
        """Test Model.get_version method."""
        from pocket_agent_cli.config import Model, ModelVersion

        model = Model(
            id="test-model",
            name="Test Model",
            architecture="llama",
            versions={
                "Q4_K_M": ModelVersion(
                    url="https://example.com/model-q4.gguf",
                    size=2000000000,
                    quantization="Q4_K_M",
                ),
                "F16": ModelVersion(
                    url="https://example.com/model-f16.gguf",
                    size=8000000000,
                    quantization="F16",
                ),
            },
            default_version="Q4_K_M",
        )

        # Get default version
        default = model.get_version()
        assert default.quantization == "Q4_K_M"

        # Get specific version
        f16 = model.get_version("F16")
        assert f16.quantization == "F16"

        # Invalid version
        with pytest.raises(ValueError):
            model.get_version("INVALID")

    def test_model_is_downloaded(self):
        """Test Model.is_downloaded method."""
        from pocket_agent_cli.config import Model, ModelVersion

        model = Model(
            id="test-model",
            name="Test Model",
            architecture="llama",
            versions={
                "Q4_K_M": ModelVersion(
                    url="https://example.com/model.gguf",
                    size=2000000000,
                    quantization="Q4_K_M",
                    downloaded=True,
                ),
                "F16": ModelVersion(
                    url="https://example.com/model-f16.gguf",
                    size=8000000000,
                    quantization="F16",
                    downloaded=False,
                ),
            },
            default_version="Q4_K_M",
        )

        # Check specific version
        assert model.is_downloaded("Q4_K_M") is True
        assert model.is_downloaded("F16") is False

        # Check any version
        assert model.is_downloaded() is True


class TestDefaultModels:
    """Tests for DEFAULT_MODELS configuration."""

    def test_default_models_defined(self):
        """Test DEFAULT_MODELS list is populated."""
        from pocket_agent_cli.config import DEFAULT_MODELS

        assert isinstance(DEFAULT_MODELS, list)
        assert len(DEFAULT_MODELS) >= 3  # At least some default models

    def test_default_models_structure(self):
        """Test DEFAULT_MODELS have correct structure."""
        from pocket_agent_cli.config import DEFAULT_MODELS

        for model_config in DEFAULT_MODELS:
            assert "id" in model_config
            assert "name" in model_config
            assert "architecture" in model_config
            assert "versions" in model_config
            assert "default_version" in model_config

            # Check versions structure
            versions = model_config["versions"]
            for version_name, version_data in versions.items():
                assert "url" in version_data
                assert "size" in version_data
                assert "quantization" in version_data

    def test_default_models_architectures(self):
        """Test DEFAULT_MODELS have valid architectures."""
        from pocket_agent_cli.config import DEFAULT_MODELS

        valid_architectures = {"llama", "gemma", "qwen", "deepseek"}

        for model_config in DEFAULT_MODELS:
            assert model_config["architecture"] in valid_architectures


class TestDirectories:
    """Tests for directory configuration."""

    def test_directories_exist(self):
        """Test that directory constants are defined."""
        from pocket_agent_cli.config import APP_DIR, MODELS_DIR, DATA_DIR, SANDBOX_DIR, RESULTS_DIR

        assert APP_DIR is not None
        assert MODELS_DIR is not None
        assert DATA_DIR is not None
        assert SANDBOX_DIR is not None
        assert RESULTS_DIR is not None

    def test_directories_are_paths(self):
        """Test that directories are Path objects."""
        from pocket_agent_cli.config import APP_DIR, MODELS_DIR, DATA_DIR, SANDBOX_DIR, RESULTS_DIR

        assert isinstance(APP_DIR, Path)
        assert isinstance(MODELS_DIR, Path)
        assert isinstance(DATA_DIR, Path)
        assert isinstance(SANDBOX_DIR, Path)
        assert isinstance(RESULTS_DIR, Path)
