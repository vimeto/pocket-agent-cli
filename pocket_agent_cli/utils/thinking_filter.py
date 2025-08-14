"""Utility for detecting and filtering thinking tokens from model outputs."""

import re
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ThinkingState:
    """Track the state of thinking token detection."""
    
    in_thinking_block: bool = False
    thinking_depth: int = 0  # For nested thinking blocks
    buffer: str = ""  # Buffer for incomplete tags
    thinking_content: str = ""  # Accumulated thinking content
    thinking_token_count: int = 0
    regular_token_count: int = 0
    
    # Pattern tracking
    open_tag_partial: str = ""  # Track partial <think> or <thinking>
    close_tag_partial: str = ""  # Track partial </think> or </thinking>


class ThinkingFilter:
    """Filter thinking tokens from model outputs."""
    
    # Thinking block patterns (case-insensitive)
    THINKING_PATTERNS = [
        (r'<think>', r'</think>'),
        (r'<thinking>', r'</thinking>'),
        (r'<thought>', r'</thought>'),
        (r'<reflection>', r'</reflection>'),
        (r'<THINK>', r'</THINK>'),  # Some models use uppercase
    ]
    
    def __init__(self):
        self.state = ThinkingState()
    
    def reset(self):
        """Reset the filter state for a new generation."""
        self.state = ThinkingState()
    
    def filter_token(self, token: str) -> Tuple[str, bool]:
        """Filter a single token, removing thinking content.
        
        Args:
            token: The token to filter
            
        Returns:
            Tuple of (filtered_token, is_thinking_token)
        """
        # Add token to buffer for pattern matching
        self.state.buffer += token
        
        # Check if we're currently in a thinking block
        if self.state.in_thinking_block:
            # Look for closing tag
            for open_pattern, close_pattern in self.THINKING_PATTERNS:
                if close_pattern.lower() in self.state.buffer.lower():
                    # Found closing tag
                    idx = self.state.buffer.lower().rfind(close_pattern.lower())
                    thinking_part = self.state.buffer[:idx + len(close_pattern)]
                    remaining = self.state.buffer[idx + len(close_pattern):]
                    
                    # Add to thinking content
                    self.state.thinking_content += thinking_part
                    self.state.thinking_token_count += 1
                    
                    # Exit thinking block
                    self.state.in_thinking_block = False
                    self.state.thinking_depth = max(0, self.state.thinking_depth - 1)
                    self.state.buffer = remaining
                    
                    # Process remaining content (might contain regular tokens)
                    if remaining:
                        filtered, is_thinking = self.filter_token("")
                        return (remaining + filtered, False)
                    return ("", True)
            
            # Still in thinking block
            self.state.thinking_content += token
            self.state.thinking_token_count += 1
            self.state.buffer = ""  # Clear buffer as we consumed it
            return ("", True)
        
        else:
            # Look for opening tag
            for open_pattern, close_pattern in self.THINKING_PATTERNS:
                if open_pattern.lower() in self.state.buffer.lower():
                    # Found opening tag
                    idx = self.state.buffer.lower().find(open_pattern.lower())
                    before = self.state.buffer[:idx]
                    after = self.state.buffer[idx:]
                    
                    # Output content before thinking block
                    if before:
                        self.state.regular_token_count += 1
                        self.state.buffer = after
                        return (before, False)
                    
                    # Enter thinking block
                    self.state.in_thinking_block = True
                    self.state.thinking_depth += 1
                    self.state.thinking_content += after
                    self.state.thinking_token_count += 1
                    self.state.buffer = ""
                    return ("", True)
            
            # Check if buffer might be building up to a thinking tag
            # Keep last few characters in buffer in case a tag spans tokens
            if len(self.state.buffer) > 20:  # Reasonable max tag length
                # Output most of buffer, keep potential partial tag
                output = self.state.buffer[:-10]
                self.state.buffer = self.state.buffer[-10:]
                self.state.regular_token_count += 1
                return (output, False)
            
            # If buffer is small, might be accumulating a tag
            # Check if it looks like a partial tag
            if self._might_be_partial_tag(self.state.buffer):
                return ("", False)  # Wait for more tokens
            
            # Not a tag, output the buffer
            output = self.state.buffer
            self.state.buffer = ""
            if output:
                self.state.regular_token_count += 1
            return (output, False)
    
    def _might_be_partial_tag(self, text: str) -> bool:
        """Check if text might be a partial thinking tag."""
        if not text:
            return False
            
        # Check for partial opening tags
        partial_opens = ['<', '<t', '<th', '<thi', '<thin', '<think', 
                        '<thinking', '<thought', '<reflection',
                        '<T', '<TH', '<THI', '<THIN', '<THINK']
        
        # Check for partial closing tags  
        partial_closes = ['</', '</t', '</th', '</thi', '</thin', '</think',
                         '</thinking', '</thought', '</reflection',
                         '</T', '</TH', '</THI', '</THIN', '</THINK']
        
        text_lower = text.lower()
        for partial in partial_opens + partial_closes:
            if text_lower == partial.lower():
                return True
        
        return False
    
    def flush(self) -> Tuple[str, bool]:
        """Flush any remaining buffer content.
        
        Returns:
            Tuple of (remaining_content, was_thinking)
        """
        if self.state.buffer:
            output = self.state.buffer
            was_thinking = self.state.in_thinking_block
            
            if was_thinking:
                self.state.thinking_content += output
                self.state.thinking_token_count += 1
            else:
                self.state.regular_token_count += 1
            
            self.state.buffer = ""
            return (output if not was_thinking else "", was_thinking)
        
        return ("", False)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about thinking tokens.
        
        Returns:
            Dictionary with thinking token statistics
        """
        total_tokens = self.state.thinking_token_count + self.state.regular_token_count
        
        return {
            "thinking_tokens": self.state.thinking_token_count,
            "regular_tokens": self.state.regular_token_count,
            "total_tokens": total_tokens,
            "thinking_ratio": self.state.thinking_token_count / total_tokens if total_tokens > 0 else 0,
            "thinking_content_length": len(self.state.thinking_content),
            "has_thinking": self.state.thinking_token_count > 0,
        }
    
    def get_thinking_content(self) -> str:
        """Get the accumulated thinking content.
        
        Returns:
            The thinking content that was filtered out
        """
        return self.state.thinking_content


def remove_thinking_blocks(text: str) -> Tuple[str, Dict[str, Any]]:
    """Remove thinking blocks from completed text.
    
    This is a simpler function for post-processing complete responses.
    
    Args:
        text: The complete text to filter
        
    Returns:
        Tuple of (filtered_text, stats)
    """
    original_length = len(text)
    filtered_text = text
    thinking_content = ""
    
    # Remove all thinking patterns
    patterns = [
        (r'<think>.*?</think>', re.DOTALL | re.IGNORECASE),
        (r'<thinking>.*?</thinking>', re.DOTALL | re.IGNORECASE),
        (r'<thought>.*?</thought>', re.DOTALL | re.IGNORECASE),
        (r'<reflection>.*?</reflection>', re.DOTALL | re.IGNORECASE),
    ]
    
    for pattern, flags in patterns:
        matches = re.findall(pattern, filtered_text, flags)
        thinking_content += ' '.join(matches)
        filtered_text = re.sub(pattern, '', filtered_text, flags=flags)
    
    # Clean up extra whitespace
    filtered_text = re.sub(r'\n\n+', '\n\n', filtered_text).strip()
    
    stats = {
        "original_length": original_length,
        "filtered_length": len(filtered_text),
        "thinking_content_length": len(thinking_content),
        "removed_chars": original_length - len(filtered_text),
        "has_thinking": len(thinking_content) > 0,
    }
    
    return filtered_text, stats