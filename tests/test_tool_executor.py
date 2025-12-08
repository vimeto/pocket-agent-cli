"""Tests for ToolExecutor."""

import pytest
import asyncio
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from pocket_agent_cli.tools.tool_executor import ToolExecutor


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tool_executor():
    """Create a ToolExecutor without Docker."""
    # Ensure Docker is disabled for tests
    with patch.dict(os.environ, {"DISABLE_DOCKER": "true"}):
        executor = ToolExecutor(use_docker=False)
        yield executor
        # Cleanup
        if executor.sandbox_dir and Path(executor.sandbox_dir).exists():
            executor._cleanup_sandbox()


@pytest.fixture
def sample_tool_calls():
    """Sample tool calls for testing."""
    return [
        {"name": "run_python_code", "parameters": {"code": "print('hello')"}},
        {"name": "submit_python_solution", "parameters": {"code": "def add(a, b): return a + b"}},
    ]


# ============================================================================
# Test: Initialization
# ============================================================================

class TestToolExecutorInit:
    """Tests for ToolExecutor initialization."""

    def test_init_without_docker(self):
        """Test creating executor without Docker."""
        with patch.dict(os.environ, {"DISABLE_DOCKER": "true"}):
            executor = ToolExecutor(use_docker=False)

        assert executor.use_docker is False
        assert executor.sandbox_dir is None

    def test_init_stores_test_cases(self):
        """Test that test cases are stored."""
        test_cases = ["assert add(1, 2) == 3"]

        with patch.dict(os.environ, {"DISABLE_DOCKER": "true"}):
            executor = ToolExecutor(use_docker=False, test_cases=test_cases)

        assert executor.test_cases == test_cases

    def test_init_default_test_cases_empty(self):
        """Test default test cases are empty list."""
        with patch.dict(os.environ, {"DISABLE_DOCKER": "true"}):
            executor = ToolExecutor(use_docker=False)

        assert executor.test_cases == []


# ============================================================================
# Test: Sandbox Management
# ============================================================================

class TestSandboxManagement:
    """Tests for sandbox management."""

    def test_create_sandbox(self, tool_executor):
        """Test creating sandbox directory."""
        tool_executor._create_sandbox()

        assert tool_executor.sandbox_dir is not None
        assert Path(tool_executor.sandbox_dir).exists()

    def test_cleanup_removes_sandbox(self, tool_executor):
        """Test cleanup removes sandbox."""
        tool_executor._create_sandbox()
        sandbox_path = Path(tool_executor.sandbox_dir)

        assert sandbox_path.exists()

        tool_executor._cleanup_sandbox()

        assert not sandbox_path.exists()


# ============================================================================
# Test: Tool Execution
# ============================================================================

class TestToolExecution:
    """Tests for tool execution."""

    @pytest.mark.asyncio
    async def test_execute_run_python_code(self, tool_executor):
        """Test executing Python code."""
        tool_executor._create_sandbox()

        result = await tool_executor._run_python_code("print('hello')")

        assert "hello" in result

    @pytest.mark.asyncio
    async def test_execute_run_python_code_with_error(self, tool_executor):
        """Test executing Python code with error."""
        tool_executor._create_sandbox()

        result = await tool_executor._run_python_code("raise ValueError('test error')")

        assert "ValueError" in result or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_single_tool_run_code(self, tool_executor):
        """Test executing a single tool call."""
        tool_executor._create_sandbox()

        call = {"name": "run_python_code", "parameters": {"code": "x = 1 + 1\nprint(x)"}}
        result = await tool_executor._execute_single_tool(call)

        assert "2" in result.get("output", "")

    @pytest.mark.asyncio
    async def test_execute_tools_sequence(self, tool_executor):
        """Test executing multiple tools in sequence."""
        tool_executor._create_sandbox()

        calls = [
            {"name": "run_python_code", "parameters": {"code": "print('first')"}},
            {"name": "run_python_code", "parameters": {"code": "print('second')"}},
        ]

        results = await tool_executor.execute_tools(calls)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_execute_tools_max_iterations(self, tool_executor):
        """Test max iterations limit."""
        tool_executor._create_sandbox()

        calls = [{"name": "run_python_code", "parameters": {"code": f"print({i})"}} for i in range(20)]

        results = await tool_executor.execute_tools(calls, max_iterations=5)

        # Should stop at max_iterations + 1 (for the error message)
        assert len(results) <= 6


# ============================================================================
# Test: File Operations
# ============================================================================

class TestFileOperations:
    """Tests for file operations."""

    @pytest.mark.asyncio
    async def test_upsert_file(self, tool_executor):
        """Test creating/updating a file."""
        tool_executor._create_sandbox()

        call = {
            "name": "upsert_file",
            "parameters": {"filename": "test.py", "content": "print('test')"}
        }
        result = await tool_executor._execute_single_tool(call)

        # Check file was created
        file_path = Path(tool_executor.sandbox_dir) / "test.py"
        assert file_path.exists()
        assert file_path.read_text() == "print('test')"

    @pytest.mark.asyncio
    async def test_read_file(self, tool_executor):
        """Test reading a file."""
        tool_executor._create_sandbox()

        # First create a file
        file_path = Path(tool_executor.sandbox_dir) / "test.py"
        file_path.write_text("test content")

        call = {"name": "read_file", "parameters": {"filename": "test.py"}}
        result = await tool_executor._execute_single_tool(call)

        assert "test content" in result.get("output", "")

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tool_executor):
        """Test reading a non-existent file."""
        tool_executor._create_sandbox()

        call = {"name": "read_file", "parameters": {"filename": "nonexistent.py"}}
        result = await tool_executor._execute_single_tool(call)

        assert "error" in result.get("output", "").lower() or "not found" in str(result).lower()


# ============================================================================
# Test: Submission Tools
# ============================================================================

class TestSubmissionTools:
    """Tests for submission tools."""

    @pytest.mark.asyncio
    async def test_submit_python_solution(self, tool_executor):
        """Test submitting a Python solution."""
        tool_executor._create_sandbox()
        tool_executor.test_cases = ["assert add(1, 2) == 3"]

        call = {
            "name": "submit_python_solution",
            "parameters": {"code": "def add(a, b):\n    return a + b"}
        }
        result = await tool_executor._execute_single_tool(call)

        # Should return submission acknowledgment
        assert "submitted" in str(result).lower() or "code" in str(result).lower()

    @pytest.mark.asyncio
    async def test_run_submission_tests(self, tool_executor):
        """Test running submission tests."""
        tool_executor._create_sandbox()
        tool_executor.test_cases = ["assert add(1, 2) == 3"]

        # First create a solution file
        file_path = Path(tool_executor.sandbox_dir) / "solution.py"
        file_path.write_text("def add(a, b):\n    return a + b")

        call = {"name": "run_submission_tests", "parameters": {"filename": "solution.py"}}
        result = await tool_executor._execute_single_tool(call)

        # Should pass tests
        assert "pass" in str(result).lower() or "success" in str(result).lower() or result.get("output", "") == ""


# ============================================================================
# Test: Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, tool_executor):
        """Test that unknown tool returns error."""
        tool_executor._create_sandbox()

        call = {"name": "unknown_tool", "parameters": {}}
        result = await tool_executor._execute_single_tool(call)

        assert "error" in str(result).lower() or "unknown" in str(result).lower()

    @pytest.mark.asyncio
    async def test_syntax_error_in_code(self, tool_executor):
        """Test handling syntax error in code."""
        tool_executor._create_sandbox()

        result = await tool_executor._run_python_code("def broken(")

        assert "error" in result.lower() or "syntax" in result.lower()

    @pytest.mark.asyncio
    async def test_runtime_error_in_code(self, tool_executor):
        """Test handling runtime error in code."""
        tool_executor._create_sandbox()

        result = await tool_executor._run_python_code("x = 1 / 0")

        assert "error" in result.lower() or "division" in result.lower()
