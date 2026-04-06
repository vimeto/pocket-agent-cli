"""Tests for the BFCL (Berkeley Function Calling Leaderboard) benchmark.

Covers:
- Dataset loading and structure
- AST-based evaluation (exact match, partial match, no match)
- Relevance detection (no-call scenarios)
- Parallel call evaluation
- Value matching with type coercion
- Mock model response parsing
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from pocket_agent_cli.datasets.bfcl import BFCLDataset, _build_bfcl_test_data
from pocket_agent_cli.datasets.registry import DatasetRegistry
from pocket_agent_cli.evaluation.bfcl_eval import (
    aggregate_results,
    evaluate_example,
    evaluate_single_call,
    values_match,
)


# ============================================================================
# Dataset Tests
# ============================================================================


class TestBFCLDataset:
    """Tests for BFCLDataset loading and structure."""

    @pytest.fixture
    def dataset(self, temp_dir):
        return BFCLDataset(temp_dir)

    @pytest.fixture
    def dataset_with_data(self, temp_dir):
        ds = BFCLDataset(temp_dir)
        ds.download()
        return ds

    def test_registered(self):
        """BFCL should be registered in the DatasetRegistry."""
        assert DatasetRegistry.is_registered("bfcl")

    def test_create_via_registry(self, temp_dir):
        ds = DatasetRegistry.create("bfcl", temp_dir)
        assert isinstance(ds, BFCLDataset)

    def test_properties(self, dataset):
        assert dataset.name == "bfcl"
        assert dataset.problem_count > 0
        assert "test" in dataset.available_splits
        assert "sample" in dataset.available_splits

    def test_is_downloaded_false(self, dataset):
        assert dataset.is_downloaded() is False

    def test_download_creates_file(self, dataset):
        result = dataset.download()
        assert result is True
        assert dataset.is_downloaded()
        assert (dataset.data_dir / "bfcl_data.json").exists()
        assert (dataset.data_dir / "bfcl_sample.json").exists()

    def test_load_test_split(self, dataset_with_data):
        problems = dataset_with_data.load(split="test")
        assert len(problems) > 0
        p = problems[0]
        assert p.task_id
        assert p.prompt
        assert p.metadata.get("source") == "bfcl"
        assert p.metadata.get("category") in (
            "simple", "multiple", "parallel", "relevance"
        )
        assert isinstance(p.metadata.get("functions"), list)
        assert isinstance(p.metadata.get("expected"), list)

    def test_load_sample_split(self, dataset_with_data):
        problems = dataset_with_data.load(split="sample")
        assert len(problems) == 5

    def test_load_with_limit(self, dataset_with_data):
        problems = dataset_with_data.load(split="test", limit=3)
        assert len(problems) == 3

    def test_load_invalid_split(self, dataset_with_data):
        with pytest.raises(ValueError, match="Invalid split"):
            dataset_with_data.load(split="nonexistent")

    def test_load_raw(self, dataset_with_data):
        """load_raw returns dicts with all BFCL fields."""
        raw = dataset_with_data.load_raw(split="test", limit=5)
        assert len(raw) == 5
        for r in raw:
            assert "id" in r
            assert "category" in r
            assert "prompt" in r
            assert "functions" in r
            assert "expected" in r

    def test_load_raw_category_filter(self, dataset_with_data):
        raw = dataset_with_data.load_raw(categories=["simple"])
        assert all(r["category"] == "simple" for r in raw)

        raw_multi = dataset_with_data.load_raw(categories=["multiple"])
        assert all(r["category"] == "multiple" for r in raw_multi)

    def test_builtin_data_coverage(self):
        """Built-in data has all four categories."""
        data = _build_bfcl_test_data()
        cats = {d["category"] for d in data}
        assert "simple" in cats
        assert "multiple" in cats
        assert "parallel" in cats
        assert "relevance" in cats

    def test_builtin_data_structure(self):
        """Each built-in example has valid structure."""
        data = _build_bfcl_test_data()
        for ex in data:
            assert "id" in ex
            assert "category" in ex
            assert "prompt" in ex and len(ex["prompt"]) > 0
            assert "functions" in ex and isinstance(ex["functions"], list)
            assert "expected" in ex and isinstance(ex["expected"], list)
            # Each function def should have the tool wrapper
            for fn in ex["functions"]:
                assert fn.get("type") == "function"
                assert "function" in fn
                assert "name" in fn["function"]
            # Each expected call should have name and arguments
            for exp in ex["expected"]:
                assert "name" in exp
                assert "arguments" in exp

    def test_builtin_data_count(self):
        """Should have a reasonable number of examples."""
        data = _build_bfcl_test_data()
        assert len(data) >= 40  # at least 40 curated examples


# ============================================================================
# Value Matching Tests
# ============================================================================


class TestValuesMatch:
    """Tests for the values_match comparison function."""

    def test_exact_string(self):
        assert values_match("hello", "hello")

    def test_case_insensitive_string(self):
        assert values_match("San Francisco", "san francisco")

    def test_substring_match(self):
        assert values_match("San Francisco, CA", "San Francisco")
        assert values_match("San Francisco", "San Francisco, CA")

    def test_different_strings(self):
        assert not values_match("New York", "Los Angeles")

    def test_exact_integer(self):
        assert values_match(42, 42)

    def test_float_tolerance(self):
        assert values_match(85.50, 85.5)
        assert values_match(3.14159, 3.14159)

    def test_string_to_number(self):
        """String '100' should match number 100."""
        assert values_match("100", 100)
        assert values_match(85.5, "85.50")

    def test_different_numbers(self):
        assert not values_match(10, 20)

    def test_list_match(self):
        assert values_match([1, 2, 3], [1, 2, 3])

    def test_list_order_insensitive(self):
        assert values_match(["b", "a"], ["a", "b"])

    def test_list_length_mismatch(self):
        assert not values_match([1, 2], [1, 2, 3])

    def test_dict_match(self):
        assert values_match({"a": 1, "b": 2}, {"a": 1, "b": 2})

    def test_dict_extra_keys_ok(self):
        """Actual can have extra keys; expected keys must all match."""
        assert values_match({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2})

    def test_dict_missing_key(self):
        assert not values_match({"a": 1}, {"a": 1, "b": 2})

    def test_none_handling(self):
        assert values_match(None, None)
        assert not values_match(None, "hello")


# ============================================================================
# Single Call Evaluation Tests
# ============================================================================


class TestEvaluateSingleCall:
    """Tests for evaluate_single_call."""

    def test_full_match(self):
        actual = {"name": "get_weather", "arguments": {"location": "Paris"}}
        expected = {"name": "get_weather", "arguments": {"location": "Paris"}}
        result = evaluate_single_call(actual, expected)
        assert result["full_match"] is True
        assert result["name_match"] is True
        assert result["args_match"] is True
        assert result["partial_match"] is False

    def test_name_match_args_wrong(self):
        actual = {"name": "get_weather", "arguments": {"location": "London"}}
        expected = {"name": "get_weather", "arguments": {"location": "Paris"}}
        result = evaluate_single_call(actual, expected)
        assert result["name_match"] is True
        assert result["args_match"] is False
        assert result["full_match"] is False
        assert result["partial_match"] is True

    def test_name_mismatch(self):
        actual = {"name": "get_forecast", "arguments": {"location": "Paris"}}
        expected = {"name": "get_weather", "arguments": {"location": "Paris"}}
        result = evaluate_single_call(actual, expected)
        assert result["name_match"] is False
        assert result["full_match"] is False

    def test_missing_argument(self):
        actual = {"name": "send_email", "arguments": {"to": "alice@example.com"}}
        expected = {"name": "send_email", "arguments": {
            "to": "alice@example.com", "subject": "Hello", "body": "Hi"
        }}
        result = evaluate_single_call(actual, expected)
        assert result["name_match"] is True
        assert result["args_match"] is False
        assert result["arg_names_match"] is False

    def test_empty_expected_args(self):
        """When expected has no args, name match is sufficient."""
        actual = {"name": "list_processes", "arguments": {}}
        expected = {"name": "list_processes", "arguments": {}}
        result = evaluate_single_call(actual, expected)
        assert result["full_match"] is True

    def test_numeric_coercion(self):
        actual = {"name": "calculate_tip",
                  "arguments": {"bill_amount": "85.5", "tip_percentage": "15"}}
        expected = {"name": "calculate_tip",
                    "arguments": {"bill_amount": 85.50, "tip_percentage": 15}}
        result = evaluate_single_call(actual, expected)
        assert result["full_match"] is True

    def test_arguments_as_json_string(self):
        """Arguments might come as a JSON string from API."""
        actual = {"name": "get_weather",
                  "arguments": '{"location": "Tokyo"}'}
        expected = {"name": "get_weather",
                    "arguments": {"location": "Tokyo"}}
        result = evaluate_single_call(actual, expected)
        assert result["full_match"] is True

    def test_parameters_key_alias(self):
        """Both 'arguments' and 'parameters' should work."""
        actual = {"name": "get_weather",
                  "parameters": {"location": "Tokyo"}}
        expected = {"name": "get_weather",
                    "arguments": {"location": "Tokyo"}}
        result = evaluate_single_call(actual, expected)
        assert result["full_match"] is True


# ============================================================================
# Example-Level Evaluation Tests
# ============================================================================


class TestEvaluateExample:
    """Tests for evaluate_example (full example scoring)."""

    def test_simple_full_match(self):
        actual = [{"name": "get_weather", "arguments": {"location": "NYC"}}]
        expected = [{"name": "get_weather", "arguments": {"location": "NYC"}}]
        result = evaluate_example(actual, expected, category="simple")
        assert result["score"] == "full_match"
        assert result["full_match"] is True

    def test_simple_partial_match(self):
        actual = [{"name": "get_weather", "arguments": {"location": "London"}}]
        expected = [{"name": "get_weather", "arguments": {"location": "NYC"}}]
        result = evaluate_example(actual, expected, category="simple")
        assert result["score"] == "partial_match"

    def test_simple_no_match(self):
        actual = [{"name": "send_email", "arguments": {}}]
        expected = [{"name": "get_weather", "arguments": {"location": "NYC"}}]
        result = evaluate_example(actual, expected, category="simple")
        assert result["score"] == "no_match"

    def test_no_actual_calls(self):
        actual = []
        expected = [{"name": "get_weather", "arguments": {"location": "NYC"}}]
        result = evaluate_example(actual, expected, category="simple")
        assert result["score"] == "no_match"
        assert result["actual_count"] == 0

    def test_relevance_no_call_correct(self):
        """Relevance: model correctly does NOT call any function."""
        actual = []
        expected = []
        result = evaluate_example(actual, expected, category="relevance")
        assert result["score"] == "full_match"
        assert result["full_match"] is True

    def test_relevance_spurious_call(self):
        """Relevance: model incorrectly calls a function."""
        actual = [{"name": "get_weather", "arguments": {"location": "NYC"}}]
        expected = []
        result = evaluate_example(actual, expected, category="relevance")
        assert result["score"] == "no_match"
        assert result["no_match"] is True

    def test_parallel_all_match(self):
        actual = [
            {"name": "get_weather", "arguments": {"location": "Tokyo"}},
            {"name": "get_weather", "arguments": {"location": "NYC"}},
        ]
        expected = [
            {"name": "get_weather", "arguments": {"location": "Tokyo"}},
            {"name": "get_weather", "arguments": {"location": "NYC"}},
        ]
        result = evaluate_example(actual, expected, category="parallel")
        assert result["score"] == "full_match"

    def test_parallel_partial(self):
        """One call matches, one doesn't."""
        actual = [
            {"name": "get_weather", "arguments": {"location": "Tokyo"}},
            {"name": "get_weather", "arguments": {"location": "London"}},
        ]
        expected = [
            {"name": "get_weather", "arguments": {"location": "Tokyo"}},
            {"name": "get_weather", "arguments": {"location": "NYC"}},
        ]
        result = evaluate_example(actual, expected, category="parallel")
        # One full match, one partial (name matches, args don't)
        assert result["score"] == "partial_match"

    def test_parallel_missing_call(self):
        actual = [
            {"name": "get_weather", "arguments": {"location": "Tokyo"}},
        ]
        expected = [
            {"name": "get_weather", "arguments": {"location": "Tokyo"}},
            {"name": "get_weather", "arguments": {"location": "NYC"}},
        ]
        result = evaluate_example(actual, expected, category="parallel")
        # One match, but second expected has no match at all
        assert result["score"] != "full_match"

    def test_multiple_correct_choice(self):
        """Model picks the right function from several."""
        actual = [{"name": "add", "arguments": {"a": 25, "b": 17}}]
        expected = [{"name": "add", "arguments": {"a": 25, "b": 17}}]
        result = evaluate_example(actual, expected, category="multiple")
        assert result["score"] == "full_match"


# ============================================================================
# Aggregate Results Tests
# ============================================================================


class TestAggregateResults:
    """Tests for aggregate_results."""

    def test_empty_results(self):
        summary = aggregate_results([])
        assert summary["total"] == 0

    def test_all_full_match(self):
        results = [
            {"score": "full_match", "category": "simple"},
            {"score": "full_match", "category": "simple"},
            {"score": "full_match", "category": "multiple"},
        ]
        summary = aggregate_results(results)
        assert summary["total"] == 3
        assert summary["full_match"] == 3
        assert summary["full_match_pct"] == 100.0
        assert summary["per_category"]["simple"]["full_match"] == 2
        assert summary["per_category"]["multiple"]["full_match"] == 1

    def test_mixed_results(self):
        results = [
            {"score": "full_match", "category": "simple"},
            {"score": "partial_match", "category": "simple"},
            {"score": "no_match", "category": "simple"},
            {"score": "full_match", "category": "multiple"},
        ]
        summary = aggregate_results(results)
        assert summary["total"] == 4
        assert summary["full_match"] == 2
        assert summary["partial_match"] == 1
        assert summary["no_match"] == 1
        assert summary["full_match_pct"] == 50.0


# ============================================================================
# Mock Model Response Tests
# ============================================================================


class TestMockModelResponse:
    """Test parsing tool calls from various model response formats."""

    def test_api_tool_calls(self):
        """Parse from OpenAI-style API tool_calls field."""
        from scripts.run_bfcl_benchmark import extract_tool_calls

        response = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }],
                },
            }],
        }
        calls = extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments"]["location"] == "Paris"

    def test_xml_tool_call_tags(self):
        """Parse Qwen-style <tool_call> tags."""
        from scripts.run_bfcl_benchmark import extract_tool_calls

        response = {
            "choices": [{
                "message": {
                    "content": (
                        '<tool_call>\n'
                        '{"name": "get_weather", "arguments": {"location": "Berlin"}}\n'
                        '</tool_call>'
                    ),
                    "tool_calls": None,
                },
            }],
        }
        calls = extract_tool_calls(response, model_arch="qwen")
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments"]["location"] == "Berlin"

    def test_json_block(self):
        """Parse from ```json blocks."""
        from scripts.run_bfcl_benchmark import extract_tool_calls

        response = {
            "choices": [{
                "message": {
                    "content": (
                        'I will call the weather function.\n'
                        '```json\n'
                        '{"name": "get_weather", "parameters": {"location": "Tokyo"}}\n'
                        '```'
                    ),
                },
            }],
        }
        calls = extract_tool_calls(response, model_arch="llama")
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"

    def test_no_tool_calls(self):
        """Model responds normally without calling any function."""
        from scripts.run_bfcl_benchmark import extract_tool_calls

        response = {
            "choices": [{
                "message": {
                    "content": "The meaning of life is a philosophical question.",
                },
            }],
        }
        calls = extract_tool_calls(response)
        assert len(calls) == 0

    def test_gemma_python_style(self):
        """Parse Gemma's [func(param=value)] format."""
        from scripts.run_bfcl_benchmark import extract_tool_calls

        response = {
            "choices": [{
                "message": {
                    "content": '[get_weather(location="San Francisco")]',
                },
            }],
        }
        calls = extract_tool_calls(response, model_arch="gemma")
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"

    def test_error_response(self):
        """Error responses should yield empty list."""
        from scripts.run_bfcl_benchmark import extract_tool_calls

        response = {"error": "timeout"}
        calls = extract_tool_calls(response)
        assert len(calls) == 0


# ============================================================================
# Integration: end-to-end single example evaluation
# ============================================================================


class TestEndToEnd:
    """Integration test: build message, parse response, evaluate."""

    def test_full_pipeline_full_match(self):
        from scripts.run_bfcl_benchmark import (
            build_messages,
            extract_tool_calls,
        )

        example = {
            "id": "test_0",
            "category": "simple",
            "prompt": "What's the weather in London?",
            "functions": [
                {"type": "function", "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                }},
            ],
            "expected": [{"name": "get_weather",
                          "arguments": {"location": "London"}}],
        }

        model_def = {"id": "test", "name": "Test", "arch": "qwen",
                     "hf_id": "test/model", "local_port": 9999}

        # Build messages
        messages, tools = build_messages(example, model_def)
        assert len(messages) == 2  # system + user
        assert tools is not None  # API tools mode

        # Simulate API response
        mock_resp = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "London"}',
                        },
                    }],
                },
            }],
        }

        calls = extract_tool_calls(mock_resp, model_arch="qwen")
        result = evaluate_example(calls, example["expected"], category="simple")
        assert result["score"] == "full_match"

    def test_full_pipeline_no_api_tools(self):
        """For no_api_tools models, functions go into the system prompt."""
        from scripts.run_bfcl_benchmark import build_messages

        example = {
            "id": "test_1",
            "category": "simple",
            "prompt": "Get stock price for TSLA.",
            "functions": [
                {"type": "function", "function": {
                    "name": "get_stock_price",
                    "description": "Get stock price",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                        },
                        "required": ["symbol"],
                    },
                }},
            ],
            "expected": [{"name": "get_stock_price",
                          "arguments": {"symbol": "TSLA"}}],
        }

        model_def = {"id": "gemma", "name": "Gemma", "arch": "gemma",
                     "hf_id": "google/gemma", "local_port": 9999,
                     "no_api_tools": True}

        messages, tools = build_messages(example, model_def)
        assert tools is None  # No API tools
        assert "get_stock_price" in messages[0]["content"]  # In system prompt
        assert "symbol" in messages[0]["content"]
