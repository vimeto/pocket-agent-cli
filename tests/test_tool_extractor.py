"""Tests for the tool extractor module."""

import pytest
from pocket_agent_cli.utils.tool_extractor import ToolExtractor


class TestToolExtractor:
    """Test cases for ToolExtractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = ToolExtractor()
    
    def test_extract_tool_call_blocks(self):
        """Test extraction from ```tool_call blocks."""
        response = '''
        Let me help you with that.
        
        ```tool_call
        {"name": "run_python_code", "parameters": {"code": "print('Hello')"}}
        ```
        
        The output is shown above.
        '''
        
        tools, error = self.extractor.extract_tools(response)
        assert len(tools) == 1
        assert tools[0]["name"] == "run_python_code"
        assert tools[0]["parameters"]["code"] == "print('Hello')"
        assert error is None
    
    def test_extract_tool_code_blocks(self):
        """Test extraction from ```tool_code blocks."""
        response = '''
        ```tool_code
        {"name": "upsert_file", "parameters": {"filename": "test.py", "content": "# test"}}
        ```
        '''
        
        tools, error = self.extractor.extract_tools(response)
        assert len(tools) == 1
        assert tools[0]["name"] == "upsert_file"
        assert tools[0]["parameters"]["filename"] == "test.py"
    
    def test_extract_json_blocks(self):
        """Test extraction from ```json blocks."""
        response = '''
        ```json
        {"name": "read_file", "parameters": {"filename": "data.txt"}}
        ```
        '''
        
        tools, error = self.extractor.extract_tools(response)
        assert len(tools) == 1
        assert tools[0]["name"] == "read_file"
    
    def test_extract_json_array(self):
        """Test extraction from JSON array."""
        response = '''
        ```json
        [
            {"name": "run_python_code", "parameters": {"code": "x = 1"}},
            {"name": "run_python_code", "parameters": {"code": "print(x)"}}
        ]
        ```
        '''
        
        tools, error = self.extractor.extract_tools(response)
        assert len(tools) == 2
        assert tools[0]["name"] == "run_python_code"
        assert tools[1]["name"] == "run_python_code"
    
    def test_extract_direct_json(self):
        """Test extraction from direct JSON."""
        response = '''
        I'll use this tool:
        {"name": "submit_python_solution", "parameters": {"code": "def solve(): pass"}}
        '''
        
        tools, error = self.extractor.extract_tools(response)
        assert len(tools) == 1
        assert tools[0]["name"] == "submit_python_solution"
    
    def test_extract_python_submission(self):
        """Test extraction of Python code as submission."""
        response = '''
        Here's my solution:
        
        ```python
        def min_cost_path(cost, m, n):
            return cost[m][n]
        ```
        '''
        
        tools, error = self.extractor.extract_tools(response)
        assert len(tools) == 1
        assert tools[0]["name"] == "submit_python_solution"
        assert "def min_cost_path" in tools[0]["parameters"]["code"]
    
    def test_python_without_function_not_extracted(self):
        """Test that Python code without function is not extracted."""
        response = '''
        ```python
        print("Hello World")
        ```
        '''
        
        tools, error = self.extractor.extract_tools(response)
        assert len(tools) == 0
        assert error is not None
    
    def test_multiple_formats(self):
        """Test extraction from multiple formats in one response."""
        response = '''
        ```tool_call
        {"name": "run_python_code", "parameters": {"code": "x = 1"}}
        ```
        
        ```json
        {"name": "upsert_file", "parameters": {"filename": "test.py", "content": "x = 2"}}
        ```
        '''
        
        tools, error = self.extractor.extract_tools(response)
        assert len(tools) == 2
        assert tools[0]["name"] == "run_python_code"
        assert tools[1]["name"] == "upsert_file"
    
    def test_deduplicate_tools(self):
        """Test deduplication of identical tool calls."""
        response = '''
        ```tool_call
        {"name": "run_python_code", "parameters": {"code": "print(1)"}}
        ```
        
        ```json
        {"name": "run_python_code", "parameters": {"code": "print(1)"}}
        ```
        '''
        
        tools, error = self.extractor.extract_tools(response)
        assert len(tools) == 1  # Duplicates removed
    
    def test_malformed_json_ignored(self):
        """Test that malformed JSON is ignored."""
        response = '''
        ```tool_call
        {"name": "run_python_code", "parameters": {"code": "valid"}}
        ```
        
        ```tool_call
        {"name": "broken json
        ```
        '''
        
        tools, error = self.extractor.extract_tools(response)
        assert len(tools) == 1  # Only valid one extracted
        assert tools[0]["parameters"]["code"] == "valid"
    
    def test_empty_parameters_added(self):
        """Test that empty parameters are added if missing."""
        response = '''
        ```tool_call
        {"name": "list_files"}
        ```
        '''
        
        tools, error = self.extractor.extract_tools(response)
        assert len(tools) == 1
        assert tools[0]["parameters"] == {}
    
    def test_no_tools_returns_error(self):
        """Test that no tools returns error message."""
        response = "Just some text without any tool calls."
        
        tools, error = self.extractor.extract_tools(response)
        assert len(tools) == 0
        assert error == "No tool calls parsed. Return tool calls in ```tool_call\n{...}``` blocks."
    
    def test_complex_nested_json(self):
        """Test extraction with complex nested parameters."""
        response = '''
        ```tool_call
        {
            "name": "complex_tool",
            "parameters": {
                "nested": {
                    "key": "value",
                    "list": [1, 2, 3]
                },
                "code": "print('test')"
            }
        }
        ```
        '''
        
        tools, error = self.extractor.extract_tools(response)
        assert len(tools) == 1
        assert tools[0]["parameters"]["nested"]["key"] == "value"
        assert tools[0]["parameters"]["nested"]["list"] == [1, 2, 3]
    
    def test_gemma_python_style_single(self):
        """Test Gemma's Python-style function call format."""
        response = '''
        I'll submit the solution now:
        
        [submit_python_solution(code="def min_cost(cost, m, n):\\n    return cost[m][n]")]
        '''
        
        tools, error = self.extractor.extract_tools(response, "gemma")
        assert len(tools) == 1
        assert tools[0]["name"] == "submit_python_solution"
        assert "def min_cost" in tools[0]["parameters"]["code"]
    
    def test_gemma_python_style_multiple_params(self):
        """Test Gemma Python-style with multiple parameters."""
        response = '''
        Let me create a file:
        [upsert_file(filename="test.py", content="print('hello')")]
        '''
        
        tools, error = self.extractor.extract_tools(response, "gemma")
        assert len(tools) == 1
        assert tools[0]["name"] == "upsert_file"
        assert tools[0]["parameters"]["filename"] == "test.py"
        assert tools[0]["parameters"]["content"] == "print('hello')"
    
    def test_gemma_python_style_multiple_calls(self):
        """Test multiple Gemma Python-style calls."""
        response = '''
        [read_file(filename="input.txt")]
        [run_python_code(code="x = 42")]
        [submit_python_solution(code="def solve(): return x")]
        '''
        
        tools, error = self.extractor.extract_tools(response, "gemma")
        assert len(tools) == 3
        assert tools[0]["name"] == "read_file"
        assert tools[1]["name"] == "run_python_code"
        assert tools[2]["name"] == "submit_python_solution"
    
    def test_gemma_json_style(self):
        """Test Gemma's JSON-style function call."""
        response = '''
        I'll read the file:
        {"name": "read_file", "parameters": {"filename": "data.txt"}}
        '''
        
        tools, error = self.extractor.extract_tools(response, "gemma")
        assert len(tools) == 1
        assert tools[0]["name"] == "read_file"
        assert tools[0]["parameters"]["filename"] == "data.txt"
    
    def test_gemma_mixed_styles(self):
        """Test mixing Gemma styles."""
        response = '''
        First, let me read:
        [read_file(filename="test.py")]
        
        Then submit:
        {"name": "submit_python_solution", "parameters": {"code": "def test(): pass"}}
        '''
        
        tools, error = self.extractor.extract_tools(response, "gemma")
        assert len(tools) == 2
        assert tools[0]["name"] == "read_file"
        assert tools[1]["name"] == "submit_python_solution"
    
    def test_gemma_error_message(self):
        """Test Gemma-specific error message."""
        response = "Just some text without function calls"
        
        tools, error = self.extractor.extract_tools(response, "gemma")
        assert len(tools) == 0
        assert "Use format: [function_name(param=value)]" in error