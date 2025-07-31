"""Robust tool extraction module for parsing LLM responses."""

import json
import re
from typing import List, Dict, Any, Optional, Tuple


class ToolExtractor:
    """Extract tool calls from various LLM response formats."""
    
    def extract_tools(self, response: str, model_architecture: str = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Extract tool calls from response.
        
        Args:
            response: The LLM response text
            model_architecture: The model architecture (gemma, llama, etc.)
            allow_python_blocks: Whether to extract from ```python blocks
            
        Returns:
            Tuple of (tool_calls, error_message)
            - tool_calls: List of extracted tool calls
            - error_message: Error message if no tools found, None otherwise
        """
        tool_calls = []
        
        # Try multiple extraction strategies in order of preference
        strategies = [
            self._extract_gemma_python_style,  # Gemma's official format
            self._extract_gemma_json_style,    # Gemma's JSON format
            self._extract_tool_call_blocks,
            self._extract_tool_code_blocks,
            self._extract_json_blocks,
            self._extract_direct_json,
            self._extract_python_submission,
        ]
        
        for strategy in strategies:
            extracted = strategy(response)
            if extracted:
                tool_calls.extend(extracted)
        
        # Deduplicate tool calls
        tool_calls = self._deduplicate_tools(tool_calls)
        
        if not tool_calls:
            if model_architecture == "gemma":
                return [], "No function calls found. Use format: [function_name(param=value)] or {\"name\": \"function\", \"parameters\": {...}}"
            return [], "No tool calls parsed. Return tool calls in ```tool_call\n{...}``` blocks."
        
        return tool_calls, None
    
    def _extract_gemma_python_style(self, response: str) -> List[Dict[str, Any]]:
        """Extract Gemma's Python-style function calls: [func_name(param=value)]"""
        # We need to handle nested brackets and parentheses
        tools = []
        
        # Find all potential function calls
        pattern = r'\[([a-zA-Z_]\w*)\('
        
        for match in re.finditer(pattern, response):
            func_name = match.group(1)
            start = match.end()  # Start after the opening (
            
            # Count parentheses to find the matching closing )
            paren_count = 1
            i = start
            while i < len(response) and paren_count > 0:
                if response[i] == '(' and (i == 0 or response[i-1] != '\\'):
                    paren_count += 1
                elif response[i] == ')' and (i == 0 or response[i-1] != '\\'):
                    paren_count -= 1
                i += 1
            
            if paren_count == 0:
                # Look for the closing bracket, which might have whitespace
                j = i
                while j < len(response) and response[j] in ' \t\n':
                    j += 1
                if j < len(response) and response[j] == ']':
                    # Found a complete function call
                    params_str = response[start:i-1]
                    tools.append((func_name, params_str))
            elif paren_count == 1 and i >= len(response):
                # Handle malformed case where closing ) is missing but we hit end of string
                # Look back for a "] pattern
                if response.rstrip().endswith('"]'):
                    # Extract everything up to the last quote
                    params_str = response[start:].rstrip()
                    if params_str.endswith('"]'):
                        params_str = params_str[:-2]  # Remove the "]
                    tools.append((func_name, params_str))
        
        parsed_tools = []
        for func_name, params_str in tools:
            # Parse parameters - use a simple approach for key=value pairs
            params = {}
            if params_str.strip():
                # For simple cases, try direct parsing
                try:
                    # Create a safe evaluation context
                    # Don't replace escape sequences yet - we need to parse first
                    safe_str = params_str
                    
                    # Use regex to extract key=value pairs more robustly
                    # This handles quoted strings with commas, equals signs, etc.
                    param_parts = []
                    current_param = ""
                    in_quotes = False
                    quote_char = None
                    
                    escape_count = 0
                    for i, char in enumerate(safe_str):
                        if char == '\\':
                            escape_count += 1
                            current_param += char
                            continue
                            
                        if char in ['"', "'"] and (not in_quotes or char == quote_char):
                            # Check if this quote is escaped (odd number of backslashes)
                            if escape_count % 2 == 1:
                                current_param += char
                                escape_count = 0
                                continue
                            
                            in_quotes = not in_quotes
                            if in_quotes:
                                quote_char = char
                            else:
                                quote_char = None
                            current_param += char  # Keep quotes for now
                        elif char == ',' and not in_quotes:
                            param_parts.append(current_param.strip())
                            current_param = ""
                        else:
                            current_param += char
                            
                        # Reset escape count for non-backslash chars
                        if char != '\\':
                            escape_count = 0
                    
                    if current_param.strip():
                        param_parts.append(current_param.strip())
                    
                    
                    # Parse each parameter
                    for part in param_parts:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            # Remove quotes and handle escapes
                            if (value.startswith('"') and value.endswith('"')) or \
                               (value.startswith("'") and value.endswith("'")):
                                value = value[1:-1]
                            elif value.startswith('"') and not value.endswith('"'):
                                # Handle case where closing quote is missing (malformed)
                                value = value[1:]
                            elif not value.startswith('"') and value.endswith('"'):
                                # Handle case where opening quote is missing (malformed)
                                value = value[:-1]
                            
                            # Unescape common escape sequences
                            value = value.replace('\\n', '\n')
                            value = value.replace('\\t', '\t')
                            value = value.replace('\\"', '"')
                            value = value.replace("\\'", "'")
                            value = value.replace('\\\\', '\\')
                            
                            params[key] = value
                
                except Exception:
                    # Fallback to simple parsing
                    pass
            
            parsed_tools.append({
                "name": func_name,
                "parameters": params
            })
        
        return parsed_tools
    
    def _extract_gemma_json_style(self, response: str) -> List[Dict[str, Any]]:
        """Extract Gemma's JSON-style function calls."""
        tools = []
        
        # Look for standalone JSON objects with "name" and "parameters"
        # This is more specific than _extract_direct_json to avoid false positives
        json_pattern = r'(?<!\w)(\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{[^{}]*\}[^{}]*\})(?!\w)'
        
        for match in re.finditer(json_pattern, response):
            try:
                tool = json.loads(match.group(1))
                if self._validate_tool(tool):
                    tools.append(tool)
            except json.JSONDecodeError:
                continue
        
        return tools
    
    def _extract_tool_call_blocks(self, response: str) -> List[Dict[str, Any]]:
        """Extract from ```tool_call blocks."""
        pattern = r'```tool_call\s*(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        tools = []
        for match in matches:
            try:
                # Clean up the JSON string
                cleaned = match.strip()
                # Remove any trailing commas before closing braces/brackets
                cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
                # Try to fix newlines in strings by escaping them
                cleaned = self._fix_json_newlines(cleaned)
                
                tool = json.loads(cleaned)
                if self._validate_tool(tool):
                    tools.append(tool)
            except json.JSONDecodeError:
                # Try alternative parsing for malformed JSON
                alt_tool = self._parse_malformed_json(match.strip())
                if alt_tool and self._validate_tool(alt_tool):
                    tools.append(alt_tool)
        
        return tools
    
    def _extract_tool_code_blocks(self, response: str) -> List[Dict[str, Any]]:
        """Extract from ```tool_code blocks."""
        pattern = r'```tool_code\s*(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        tools = []
        for match in matches:
            try:
                tool = json.loads(match.strip())
                if self._validate_tool(tool):
                    tools.append(tool)
            except json.JSONDecodeError:
                continue
        
        return tools
    
    def _extract_json_blocks(self, response: str) -> List[Dict[str, Any]]:
        """Extract from ```json blocks."""
        pattern = r'```json\s*(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        tools = []
        for match in matches:
            try:
                data = json.loads(match.strip())
                
                # Handle single tool
                if isinstance(data, dict) and self._validate_tool(data):
                    tools.append(data)
                
                # Handle array of tools
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and self._validate_tool(item):
                            tools.append(item)
            except json.JSONDecodeError:
                continue
        
        return tools
    
    def _extract_direct_json(self, response: str) -> List[Dict[str, Any]]:
        """Extract direct JSON objects."""
        # Look for JSON objects that start with { and contain "name"
        pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{[^{}]*\}[^{}]*\}'
        
        tools = []
        for match in re.finditer(pattern, response):
            try:
                tool = json.loads(match.group())
                if self._validate_tool(tool):
                    tools.append(tool)
            except json.JSONDecodeError:
                continue
        
        return tools
    
    def _extract_python_submission(self, response: str) -> List[Dict[str, Any]]:
        """Extract Python code blocks as submit_python_solution calls."""
        # Pattern for ```python blocks
        pattern = r'```python\s*(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        tools = []
        for match in matches:
            # Only treat as submission if it contains function definition
            if 'def ' in match:
                tools.append({
                    "name": "submit_python_solution",
                    "parameters": {
                        "code": match.strip()
                    }
                })
        
        return tools
    
    def _validate_tool(self, tool: Dict[str, Any]) -> bool:
        """Validate tool call structure."""
        if not isinstance(tool, dict):
            return False
        
        # Must have name field
        if "name" not in tool:
            return False
        
        # If parameters exist, must be dict
        if "parameters" in tool and not isinstance(tool["parameters"], dict):
            return False
        
        # Add empty parameters if missing
        if "parameters" not in tool:
            tool["parameters"] = {}
        
        return True
    
    def _deduplicate_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate tool calls."""
        seen = set()
        unique_tools = []
        
        for tool in tools:
            # Create a hashable representation
            tool_str = json.dumps(tool, sort_keys=True)
            if tool_str not in seen:
                seen.add(tool_str)
                unique_tools.append(tool)
        
        return unique_tools
    
    def _fix_json_newlines(self, json_str: str) -> str:
        """Fix newlines in JSON strings."""
        # This is a simple approach - find strings and escape newlines
        result = []
        in_string = False
        escape_next = False
        
        for i, char in enumerate(json_str):
            if escape_next:
                result.append(char)
                escape_next = False
                continue
                
            if char == '\\' and in_string:
                escape_next = True
                result.append(char)
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                result.append(char)
                continue
                
            if char == '\n' and in_string:
                result.append('\\n')
            else:
                result.append(char)
        
        return ''.join(result)
    
    def _parse_malformed_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to parse malformed JSON by extracting key information."""
        # Look for name and parameters
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
        if not name_match:
            return None
            
        tool = {"name": name_match.group(1), "parameters": {}}
        
        # Look for parameters object
        params_match = re.search(r'"parameters"\s*:\s*\{([^}]*)\}', text, re.DOTALL)
        if params_match:
            params_content = params_match.group(1)
            # Extract key-value pairs
            kv_pattern = r'"([^"]+)"\s*:\s*"([^"]*?)"'
            for key, value in re.findall(kv_pattern, params_content):
                # Unescape newlines
                value = value.replace('\\n', '\n')
                tool["parameters"][key] = value
        
        return tool