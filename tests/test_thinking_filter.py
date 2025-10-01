"""Tests for the thinking filter module."""

import pytest
from pocket_agent_cli.utils.thinking_filter import (
    ThinkingFilter,
    ThinkingState,
    remove_thinking_blocks,
)


class TestThinkingState:
    """Tests for ThinkingState dataclass."""

    def test_default_state(self):
        """Test default state values."""
        state = ThinkingState()

        assert state.in_thinking_block is False
        assert state.thinking_depth == 0
        assert state.buffer == ""
        assert state.thinking_content == ""
        assert state.thinking_token_count == 0
        assert state.regular_token_count == 0


class TestThinkingFilter:
    """Tests for ThinkingFilter class."""

    def test_init(self):
        """Test filter initialization."""
        filter = ThinkingFilter()

        assert filter.state is not None
        assert filter.state.in_thinking_block is False

    def test_reset(self):
        """Test filter reset."""
        filter = ThinkingFilter()
        filter.state.in_thinking_block = True
        filter.state.thinking_token_count = 10
        filter.state.buffer = "some content"

        filter.reset()

        assert filter.state.in_thinking_block is False
        assert filter.state.thinking_token_count == 0
        assert filter.state.buffer == ""

    def test_filter_token_no_thinking(self):
        """Test filtering tokens without thinking blocks."""
        filter = ThinkingFilter()

        # Simple text should pass through
        result, is_thinking = filter.filter_token("Hello")
        # May need flush for short tokens
        final, _ = filter.flush()

        combined = result + final
        assert "Hello" in combined
        assert is_thinking is False

    def test_filter_token_with_think_block(self):
        """Test filtering tokens with <think> block."""
        filter = ThinkingFilter()

        # Process text with thinking block
        tokens = ["Before ", "<think>", "secret", " thoughts", "</think>", " After"]
        outputs = []

        for token in tokens:
            result, _ = filter.filter_token(token)
            if result:
                outputs.append(result)

        final, _ = filter.flush()
        if final:
            outputs.append(final)

        combined = "".join(outputs)

        # Should contain Before and After, but not the thinking content
        assert "Before" in combined
        assert "After" in combined
        assert "secret" not in combined
        assert "thoughts" not in combined

    def test_filter_token_with_thinking_block(self):
        """Test filtering tokens with <thinking> block."""
        filter = ThinkingFilter()

        # Process text with thinking block
        tokens = ["Start ", "<thinking>", "reasoning here", "</thinking>", " End"]
        outputs = []

        for token in tokens:
            result, _ = filter.filter_token(token)
            if result:
                outputs.append(result)

        final, _ = filter.flush()
        if final:
            outputs.append(final)

        combined = "".join(outputs)

        assert "Start" in combined
        assert "End" in combined
        assert "reasoning" not in combined

    def test_filter_partial_tags(self):
        """Test filtering with complete tags that arrive in chunks.

        Note: The current implementation handles complete tags within tokens
        but may not perfectly handle tags split mid-word across tokens.
        This test verifies the realistic case of complete tags in separate tokens.
        """
        filter = ThinkingFilter()

        # Complete tags arriving as separate tokens
        tokens = ["Hello ", "<think>", "hidden content", "</think>", " World"]
        outputs = []

        for token in tokens:
            result, _ = filter.filter_token(token)
            if result:
                outputs.append(result)

        final, _ = filter.flush()
        if final:
            outputs.append(final)

        combined = "".join(outputs)

        assert "Hello" in combined
        assert "World" in combined
        # Hidden content should be filtered
        assert "hidden content" not in combined

    def test_get_stats_no_thinking(self):
        """Test stats with no thinking content."""
        filter = ThinkingFilter()

        filter.filter_token("Regular content")
        filter.flush()

        stats = filter.get_stats()

        assert stats["thinking_tokens"] == 0
        assert stats["regular_tokens"] >= 1
        assert stats["thinking_ratio"] == 0
        assert stats["has_thinking"] is False

    def test_get_stats_with_thinking(self):
        """Test stats with thinking content."""
        filter = ThinkingFilter()

        tokens = ["Before", "<think>", "hidden", "</think>", "After"]
        for token in tokens:
            filter.filter_token(token)
        filter.flush()

        stats = filter.get_stats()

        assert stats["thinking_tokens"] > 0
        assert stats["has_thinking"] is True
        assert stats["thinking_ratio"] > 0

    def test_get_thinking_content(self):
        """Test getting accumulated thinking content."""
        filter = ThinkingFilter()

        tokens = ["<think>", "My secret thoughts", "</think>"]
        for token in tokens:
            filter.filter_token(token)
        filter.flush()

        content = filter.get_thinking_content()

        assert "secret thoughts" in content

    def test_flush_empty(self):
        """Test flushing with empty buffer."""
        filter = ThinkingFilter()

        result, was_thinking = filter.flush()

        assert result == ""
        assert was_thinking is False

    def test_flush_with_content(self):
        """Test flushing with remaining content."""
        filter = ThinkingFilter()

        # Add partial content
        filter.state.buffer = "remaining"

        result, was_thinking = filter.flush()

        assert result == "remaining"
        assert was_thinking is False

    def test_thinking_patterns(self):
        """Test all supported thinking patterns."""
        patterns = [
            ("<think>", "</think>"),
            ("<thinking>", "</thinking>"),
            ("<thought>", "</thought>"),
            ("<reflection>", "</reflection>"),
            ("<THINK>", "</THINK>"),  # Uppercase
        ]

        for open_tag, close_tag in patterns:
            filter = ThinkingFilter()

            tokens = [f"Before {open_tag}", "hidden", f"{close_tag} After"]
            outputs = []

            for token in tokens:
                result, _ = filter.filter_token(token)
                if result:
                    outputs.append(result)

            final, _ = filter.flush()
            if final:
                outputs.append(final)

            combined = "".join(outputs)

            assert "Before" in combined, f"Failed for pattern {open_tag}"
            assert "After" in combined, f"Failed for pattern {open_tag}"
            assert "hidden" not in combined, f"Hidden content leaked for {open_tag}"


class TestRemoveThinkingBlocks:
    """Tests for remove_thinking_blocks function."""

    def test_no_thinking_blocks(self):
        """Test text without thinking blocks."""
        text = "This is regular text without any thinking."

        filtered, stats = remove_thinking_blocks(text)

        assert filtered == text.strip()
        assert stats["has_thinking"] is False
        assert stats["removed_chars"] == 0

    def test_single_think_block(self):
        """Test removing a single <think> block."""
        text = "Before <think>hidden content</think> After"

        filtered, stats = remove_thinking_blocks(text)

        assert "Before" in filtered
        assert "After" in filtered
        assert "hidden" not in filtered
        assert stats["has_thinking"] is True

    def test_single_thinking_block(self):
        """Test removing a single <thinking> block."""
        text = "Start <thinking>reasoning process</thinking> End"

        filtered, stats = remove_thinking_blocks(text)

        assert "Start" in filtered
        assert "End" in filtered
        assert "reasoning" not in filtered
        assert stats["has_thinking"] is True

    def test_multiple_blocks(self):
        """Test removing multiple thinking blocks."""
        text = """
        First <think>thought 1</think> middle
        <thinking>thought 2</thinking> last
        """

        filtered, stats = remove_thinking_blocks(text)

        assert "First" in filtered
        assert "middle" in filtered
        assert "last" in filtered
        assert "thought 1" not in filtered
        assert "thought 2" not in filtered
        assert stats["has_thinking"] is True

    def test_multiline_thinking_block(self):
        """Test removing multiline thinking block."""
        text = """Before
<think>
This is a long
multiline
thinking block
</think>
After"""

        filtered, stats = remove_thinking_blocks(text)

        assert "Before" in filtered
        assert "After" in filtered
        assert "multiline" not in filtered
        assert stats["has_thinking"] is True

    def test_case_insensitive(self):
        """Test case-insensitive pattern matching."""
        text = "A <THINK>uppercase</THINK> B <Think>mixed</Think> C"

        filtered, stats = remove_thinking_blocks(text)

        assert "A" in filtered
        assert "B" in filtered
        assert "C" in filtered
        assert "uppercase" not in filtered
        assert "mixed" not in filtered

    def test_thought_block(self):
        """Test removing <thought> blocks."""
        text = "Start <thought>my thoughts</thought> End"

        filtered, stats = remove_thinking_blocks(text)

        assert "Start" in filtered
        assert "End" in filtered
        assert "my thoughts" not in filtered
        assert stats["has_thinking"] is True

    def test_reflection_block(self):
        """Test removing <reflection> blocks."""
        text = "Before <reflection>reflecting on this</reflection> After"

        filtered, stats = remove_thinking_blocks(text)

        assert "Before" in filtered
        assert "After" in filtered
        assert "reflecting" not in filtered
        assert stats["has_thinking"] is True

    def test_stats_accuracy(self):
        """Test that stats are accurate."""
        text = "ABC <think>XYZ</think> DEF"

        filtered, stats = remove_thinking_blocks(text)

        assert stats["original_length"] == len(text)
        assert stats["filtered_length"] == len(filtered)
        assert stats["removed_chars"] == stats["original_length"] - stats["filtered_length"]
        assert stats["thinking_content_length"] > 0

    def test_empty_thinking_block(self):
        """Test empty thinking block."""
        text = "Before <think></think> After"

        filtered, stats = remove_thinking_blocks(text)

        assert "Before" in filtered
        assert "After" in filtered
        assert stats["has_thinking"] is True

    def test_whitespace_cleanup(self):
        """Test that extra whitespace is cleaned up."""
        text = "Before\n\n\n<think>hidden</think>\n\n\nAfter"

        filtered, stats = remove_thinking_blocks(text)

        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in filtered

    def test_preserves_code_blocks(self):
        """Test that code-like content outside thinking is preserved."""
        text = """
def solution():
    <think>figure out logic</think>
    return 42
"""

        filtered, stats = remove_thinking_blocks(text)

        assert "def solution():" in filtered
        assert "return 42" in filtered
        assert "figure out logic" not in filtered

    def test_nested_angle_brackets(self):
        """Test handling of nested angle brackets (non-thinking)."""
        text = "Compare a < b and c > d <think>hidden</think> done"

        filtered, stats = remove_thinking_blocks(text)

        assert "a < b" in filtered
        assert "c > d" in filtered
        assert "done" in filtered
        assert "hidden" not in filtered
