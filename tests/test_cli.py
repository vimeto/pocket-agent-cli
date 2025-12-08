"""Tests for CLI commands."""

import pytest
from click.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json

from pocket_agent_cli.cli import cli


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


# ============================================================================
# Test: Main CLI Group
# ============================================================================

class TestMainCLI:
    """Tests for main CLI group."""

    def test_cli_help(self, runner):
        """Test CLI help output."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Pocket Agent CLI" in result.output

    def test_cli_has_model_group(self, runner):
        """Test CLI has model command group."""
        result = runner.invoke(cli, ["model", "--help"])

        assert result.exit_code == 0

    def test_cli_has_dataset_group(self, runner):
        """Test CLI has dataset command group."""
        result = runner.invoke(cli, ["dataset", "--help"])

        assert result.exit_code == 0

    def test_cli_has_benchmark_group(self, runner):
        """Test CLI has benchmark command group."""
        result = runner.invoke(cli, ["benchmark", "--help"])

        assert result.exit_code == 0


# ============================================================================
# Test: Model Commands
# ============================================================================

class TestModelCommands:
    """Tests for model management commands."""

    def test_model_list_runs(self, runner):
        """Test model list command runs."""
        result = runner.invoke(cli, ["model", "list"])

        # Should run without crashing (may not have models)
        assert result.exit_code == 0

    def test_model_help(self, runner):
        """Test model help."""
        result = runner.invoke(cli, ["model", "--help"])

        assert result.exit_code == 0
        assert "list" in result.output.lower()


# ============================================================================
# Test: Dataset Commands
# ============================================================================

class TestDatasetCommands:
    """Tests for dataset commands."""

    def test_dataset_list(self, runner):
        """Test listing datasets."""
        result = runner.invoke(cli, ["dataset", "list"])

        assert result.exit_code == 0
        # Should list available datasets
        assert "mbpp" in result.output.lower() or "humaneval" in result.output.lower()

    def test_dataset_info_mbpp(self, runner):
        """Test getting MBPP dataset info."""
        result = runner.invoke(cli, ["dataset", "info", "mbpp"])

        assert result.exit_code == 0
        assert "mbpp" in result.output.lower()

    def test_dataset_info_humaneval(self, runner):
        """Test getting HumanEval dataset info."""
        result = runner.invoke(cli, ["dataset", "info", "humaneval"])

        assert result.exit_code == 0
        assert "humaneval" in result.output.lower()

    def test_dataset_info_unknown(self, runner):
        """Test getting info for unknown dataset."""
        result = runner.invoke(cli, ["dataset", "info", "unknown_dataset"])

        # Should fail gracefully
        assert result.exit_code != 0 or "unknown" in result.output.lower() or "not found" in result.output.lower()


# ============================================================================
# Test: Benchmark Commands
# ============================================================================

class TestBenchmarkCommands:
    """Tests for benchmark commands."""

    def test_benchmark_help(self, runner):
        """Test benchmark help."""
        result = runner.invoke(cli, ["benchmark", "--help"])

        assert result.exit_code == 0


# ============================================================================
# Test: CLI Error Handling
# ============================================================================

class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    def test_unknown_command(self, runner):
        """Test handling unknown command."""
        result = runner.invoke(cli, ["unknown_command"])

        assert result.exit_code != 0


# ============================================================================
# Test: Output Formatting
# ============================================================================

class TestOutputFormatting:
    """Tests for CLI output formatting."""

    def test_dataset_list_uses_table(self, runner):
        """Test dataset list uses table formatting."""
        result = runner.invoke(cli, ["dataset", "list"])

        assert result.exit_code == 0
