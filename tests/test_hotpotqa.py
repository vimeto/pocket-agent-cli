"""Tests for the HotpotQA dataset and web search tool."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from pocket_agent_cli.datasets import Dataset, Problem, DatasetRegistry
from pocket_agent_cli.datasets.hotpotqa import (
    HotpotQADataset,
    exact_match_score,
    extract_answer_from_response,
    f1_score,
    normalize_answer,
)
from pocket_agent_cli.network.network_simulator import NETWORK_PRESETS, NetworkConfig
from pocket_agent_cli.tools.web_search import SimulatedWebSearch


# ============================================================================
# Sample Data
# ============================================================================

SAMPLE_HOTPOTQA_DATA = [
    {
        "id": "5a8b57f25542995d1e6f1371",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "type": "comparison",
        "level": "hard",
        "supporting_facts": {
            "title": ["Scott Derrickson", "Ed Wood"],
            "sent_id": [0, 0],
        },
        "context": {
            "title": [
                "Scott Derrickson",
                "Ed Wood",
                "Adam Collis",
                "Tyler Bates",
            ],
            "sentences": [
                [
                    "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. ",
                    "He lives in Los Angeles, California. ",
                    "He is best known for directing horror films. ",
                ],
                [
                    "Edward Davis Wood Jr. (October 10, 1924 - December 10, 1978) was an American filmmaker. ",
                    "He is often cited as being the worst professional director of all time. ",
                ],
                [
                    "Adam Collis is an American filmmaker. ",
                    "He directed the 2003 film 'Funny Ha Ha'. ",
                ],
                [
                    "Tyler Bates (born June 5, 1965) is an American musician. ",
                    "He has composed film scores for many movies. ",
                ],
            ],
        },
    },
    {
        "id": "5a8c7595554299240d9c2150",
        "question": "What government position was held by the woman who portrayed Nora combative mother on Desperate Housewives?",
        "answer": "Deputy United States Trade Representative",
        "type": "bridge",
        "level": "medium",
        "supporting_facts": {
            "title": ["Desperate Housewives", "Dana Delany"],
            "sent_id": [0, 1],
        },
        "context": {
            "title": [
                "Desperate Housewives",
                "Dana Delany",
                "Some Other Topic",
            ],
            "sentences": [
                [
                    "Desperate Housewives is an American TV series. ",
                    "Nora Huntington's mother was portrayed by Dana Delany. ",
                ],
                [
                    "Dana Delany is an American actress. ",
                    "She served as Deputy United States Trade Representative from 2015. ",
                ],
                [
                    "This is an unrelated paragraph about something else entirely. ",
                ],
            ],
        },
    },
    {
        "id": "5a7a06935542990198eaf050",
        "question": "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas?",
        "answer": "I'm a Jayhawk",
        "type": "bridge",
        "level": "easy",
        "supporting_facts": {
            "title": ["University of Kansas", "Kansas Jayhawks"],
            "sent_id": [0, 2],
        },
        "context": {
            "title": [
                "University of Kansas",
                "Kansas Jayhawks",
                "Distractor Article",
            ],
            "sentences": [
                [
                    "The University of Kansas, often referred to as KU, is a public research university in Lawrence, Kansas. ",
                    "Founded in 1865, it is the largest university in Kansas. ",
                ],
                [
                    "The Kansas Jayhawks are the athletic teams of the University of Kansas. ",
                    "They compete in the Big 12 Conference. ",
                    "The fight song is 'I'm a Jayhawk'. ",
                ],
                [
                    "This is a distractor article with no relevant information. ",
                ],
            ],
        },
    },
]


# ============================================================================
# Answer Normalization Tests
# ============================================================================


class TestNormalizeAnswer:
    """Tests for answer normalization."""

    def test_lowercase(self):
        assert normalize_answer("Hello World") == "hello world"

    def test_remove_articles(self):
        assert normalize_answer("the quick brown fox") == "quick brown fox"
        assert normalize_answer("a cat and an owl") == "cat and owl"

    def test_remove_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"
        assert normalize_answer("it's a test.") == "its test"

    def test_normalize_whitespace(self):
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_combined(self):
        assert normalize_answer("The Answer Is: Yes!") == "answer is yes"

    def test_empty(self):
        assert normalize_answer("") == ""
        assert normalize_answer(None) == ""


# ============================================================================
# Exact Match Tests
# ============================================================================


class TestExactMatchScore:
    """Tests for exact match scoring."""

    def test_exact_match(self):
        assert exact_match_score("yes", "yes") == 1.0

    def test_case_insensitive(self):
        assert exact_match_score("Yes", "yes") == 1.0

    def test_article_normalization(self):
        assert exact_match_score("the answer", "answer") == 1.0

    def test_no_match(self):
        assert exact_match_score("no", "yes") == 0.0

    def test_partial_no_match(self):
        assert exact_match_score("yes definitely", "yes") == 0.0


# ============================================================================
# F1 Score Tests
# ============================================================================


class TestF1Score:
    """Tests for token-level F1 scoring."""

    def test_perfect_match(self):
        assert f1_score("the cat sat", "the cat sat") == 1.0

    def test_no_overlap(self):
        assert f1_score("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        # "cat sat" vs "cat sat on mat" -> precision=2/2, recall=2/4
        # But after normalization: "cat sat" vs "cat sat on mat"
        f1 = f1_score("the cat sat", "the cat sat on the mat")
        assert 0.0 < f1 < 1.0
        # Tokens after normalization: ["cat", "sat"] and ["cat", "sat", "on", "mat"]
        # common = 2, precision = 2/2 = 1.0, recall = 2/4 = 0.5
        # F1 = 2 * 1.0 * 0.5 / (1.0 + 0.5) = 2/3
        assert abs(f1 - 2 / 3) < 0.01

    def test_both_empty(self):
        assert f1_score("", "") == 1.0

    def test_one_empty(self):
        assert f1_score("hello", "") == 0.0
        assert f1_score("", "hello") == 0.0


# ============================================================================
# Answer Extraction Tests
# ============================================================================


class TestExtractAnswerFromResponse:
    """Tests for extracting answers from model responses."""

    def test_the_answer_is(self):
        resp = "Based on my research, the answer is: yes"
        assert extract_answer_from_response(resp).lower() == "yes"

    def test_answer_colon(self):
        resp = "After searching, I found:\nAnswer: Deputy United States Trade Representative"
        answer = extract_answer_from_response(resp)
        assert "Deputy United States Trade Representative" in answer

    def test_final_answer_is(self):
        resp = "Based on the evidence, the final answer is: I'm a Jayhawk"
        answer = extract_answer_from_response(resp)
        assert "Jayhawk" in answer

    def test_thinking_blocks_removed(self):
        resp = "<think>Let me reason about this...</think>The answer is: yes"
        assert extract_answer_from_response(resp).lower() == "yes"

    def test_fallback_last_line(self):
        resp = "Some reasoning\nSome more reasoning\nYes, they are both American"
        answer = extract_answer_from_response(resp)
        assert "American" in answer

    def test_empty_response(self):
        assert extract_answer_from_response("") == ""


# ============================================================================
# HotpotQA Dataset Tests
# ============================================================================


class TestHotpotQADataset:
    """Tests for HotpotQADataset."""

    @pytest.fixture
    def hotpotqa_dataset(self, temp_dir):
        """Create HotpotQA dataset instance."""
        return HotpotQADataset(temp_dir)

    @pytest.fixture
    def hotpotqa_with_data(self, hotpotqa_dataset):
        """HotpotQA dataset with sample data files created."""
        val_path = hotpotqa_dataset.data_dir / "hotpotqa_validation.json"
        with open(val_path, "w") as f:
            json.dump(SAMPLE_HOTPOTQA_DATA, f)

        sample_path = hotpotqa_dataset.data_dir / "hotpotqa_sample.json"
        with open(sample_path, "w") as f:
            json.dump(SAMPLE_HOTPOTQA_DATA[:2], f)

        return hotpotqa_dataset

    # ---- Properties ----

    def test_properties(self, hotpotqa_dataset):
        assert hotpotqa_dataset.name == "hotpotqa"
        assert hotpotqa_dataset.problem_count == 200
        assert "validation" in hotpotqa_dataset.available_splits
        assert "sample" in hotpotqa_dataset.available_splits

    def test_description(self, hotpotqa_dataset):
        assert "hotpotqa" in hotpotqa_dataset.description.lower()

    # ---- Download detection ----

    def test_is_downloaded_false(self, hotpotqa_dataset):
        assert hotpotqa_dataset.is_downloaded() is False

    def test_is_downloaded_true(self, hotpotqa_with_data):
        assert hotpotqa_with_data.is_downloaded() is True

    # ---- Loading ----

    def test_load_validation_split(self, hotpotqa_with_data):
        problems = hotpotqa_with_data.load(split="validation")
        assert len(problems) == 3
        assert all(isinstance(p, Problem) for p in problems)

    def test_load_sample_split(self, hotpotqa_with_data):
        problems = hotpotqa_with_data.load(split="sample")
        assert len(problems) == 2

    def test_load_with_limit(self, hotpotqa_with_data):
        problems = hotpotqa_with_data.load(split="validation", limit=1)
        assert len(problems) == 1

    def test_load_converts_to_problem(self, hotpotqa_with_data):
        problems = hotpotqa_with_data.load(split="validation", limit=1)
        problem = problems[0]

        assert problem.task_id == "HotpotQA/0"
        assert "Scott Derrickson" in problem.prompt
        assert problem.entry_point == "answer"
        assert problem.metadata["source"] == "hotpotqa"
        assert problem.metadata["answer"] == "yes"
        assert problem.metadata["type"] == "comparison"
        assert len(problem.metadata["paragraphs"]) == 4

    def test_load_invalid_split(self, hotpotqa_with_data):
        with pytest.raises(ValueError) as exc_info:
            hotpotqa_with_data.load(split="invalid")
        assert "Invalid split" in str(exc_info.value)

    def test_load_not_downloaded(self, hotpotqa_dataset):
        with pytest.raises(FileNotFoundError):
            hotpotqa_dataset.load(split="validation")

    # ---- Paragraph extraction ----

    def test_get_paragraphs(self, hotpotqa_with_data):
        problems = hotpotqa_with_data.load(split="validation", limit=1)
        paragraphs = hotpotqa_with_data.get_paragraphs(problems[0])

        assert len(paragraphs) == 4
        assert paragraphs[0]["title"] == "Scott Derrickson"
        assert "American director" in paragraphs[0]["text"]

    def test_paragraph_structure(self, hotpotqa_with_data):
        problems = hotpotqa_with_data.load(split="validation", limit=1)
        paragraphs = problems[0].metadata["paragraphs"]

        for para in paragraphs:
            assert "title" in para
            assert "text" in para
            assert "sentences" in para
            assert isinstance(para["sentences"], list)

    # ---- Evaluation ----

    def test_evaluate_response_exact(self, hotpotqa_with_data):
        problems = hotpotqa_with_data.load(split="validation", limit=1)
        problem = problems[0]  # answer is "yes"

        result = hotpotqa_with_data.evaluate_response(problem, "The answer is: yes")
        assert result["em"] == 1.0
        assert result["f1"] == 1.0

    def test_evaluate_response_incorrect(self, hotpotqa_with_data):
        problems = hotpotqa_with_data.load(split="validation", limit=1)
        problem = problems[0]  # answer is "yes"

        result = hotpotqa_with_data.evaluate_response(problem, "The answer is: no")
        assert result["em"] == 0.0
        assert result["f1"] == 0.0

    def test_evaluate_response_partial(self, hotpotqa_with_data):
        problems = hotpotqa_with_data.load(split="validation")
        problem = problems[2]  # answer is "I'm a Jayhawk"

        result = hotpotqa_with_data.evaluate_response(
            problem, "The answer is: Jayhawk"
        )
        # Partial match: "jayhawk" is one of the tokens in "im jayhawk"
        assert result["f1"] > 0.0

    # ---- Supporting facts ----

    def test_supporting_titles(self, hotpotqa_with_data):
        problems = hotpotqa_with_data.load(split="validation", limit=1)
        titles = problems[0].metadata["supporting_titles"]

        assert "Scott Derrickson" in titles
        assert "Ed Wood" in titles

    # ---- Task IDs ----

    def test_task_ids_sequential(self, hotpotqa_with_data):
        problems = hotpotqa_with_data.load(split="validation")
        for i, problem in enumerate(problems):
            assert problem.task_id == f"HotpotQA/{i}"

    # ---- Download (mocked) ----

    @patch("datasets.load_dataset")
    def test_download_success(self, mock_load_dataset, hotpotqa_dataset):
        mock_item = {
            "id": "test123",
            "question": "Test question?",
            "answer": "test answer",
            "type": "bridge",
            "level": "easy",
            "supporting_facts": {"title": ["A"], "sent_id": [0]},
            "context": {
                "title": ["A", "B"],
                "sentences": [["Sentence one. "], ["Sentence two. "]],
            },
        }
        # Return a list-like object that supports iteration and indexing
        mock_load_dataset.return_value = [mock_item]

        result = hotpotqa_dataset.download()

        assert result is True
        assert (hotpotqa_dataset.data_dir / "hotpotqa_validation.json").exists()
        assert (hotpotqa_dataset.data_dir / "hotpotqa_sample.json").exists()

    @patch("datasets.load_dataset")
    def test_download_failure(self, mock_load_dataset, hotpotqa_dataset):
        mock_load_dataset.side_effect = Exception("Network error")
        result = hotpotqa_dataset.download()
        assert result is False

    # ---- System prompt ----

    def test_system_prompt(self):
        assert "web_search" in HotpotQADataset.SYSTEM_PROMPT
        assert "tool_call" in HotpotQADataset.SYSTEM_PROMPT


# ============================================================================
# Registry Integration Tests
# ============================================================================


class TestHotpotQARegistration:
    """Tests for HotpotQA integration with the DatasetRegistry."""

    def test_is_registered(self):
        assert DatasetRegistry.is_registered("hotpotqa")

    def test_in_list(self):
        datasets = DatasetRegistry.list_datasets()
        assert "hotpotqa" in datasets

    def test_create_via_registry(self, temp_dir):
        dataset = DatasetRegistry.create("hotpotqa", temp_dir)
        assert isinstance(dataset, HotpotQADataset)
        assert dataset.data_dir == temp_dir

    def test_alongside_other_datasets(self):
        names = DatasetRegistry.list_names()
        assert "mbpp" in names
        assert "humaneval" in names
        assert "gsm8k" in names
        assert "hotpotqa" in names


# ============================================================================
# SimulatedWebSearch Tests
# ============================================================================


class TestSimulatedWebSearch:
    """Tests for the SimulatedWebSearch tool."""

    @pytest.fixture
    def paragraphs(self):
        """Sample paragraphs for search testing."""
        return [
            {
                "title": "Scott Derrickson",
                "text": "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
            },
            {
                "title": "Ed Wood",
                "text": "Edward Davis Wood Jr. was an American filmmaker, often cited as the worst director.",
            },
            {
                "title": "Adam Collis",
                "text": "Adam Collis is an American filmmaker who directed Funny Ha Ha.",
            },
            {
                "title": "Tyler Bates",
                "text": "Tyler Bates is an American musician who has composed film scores.",
            },
        ]

    def test_search_returns_results(self, paragraphs):
        ws = SimulatedWebSearch(paragraphs)
        results = ws.search("Scott Derrickson director")
        assert len(results) > 0
        assert "Scott Derrickson" in results

    def test_search_ranking(self, paragraphs):
        ws = SimulatedWebSearch(paragraphs)
        results = ws.search("Scott Derrickson American director")
        # Scott Derrickson should be the top result
        assert results.startswith("[Scott Derrickson]")

    def test_search_top_k(self, paragraphs):
        ws = SimulatedWebSearch(paragraphs)
        results = ws.search("American", top_k=2)
        # Should return exactly 2 paragraphs
        # Count the paragraph headers
        headers = [line for line in results.split("\n") if line.startswith("[")]
        assert len(headers) == 2

    def test_search_count(self, paragraphs):
        ws = SimulatedWebSearch(paragraphs)
        assert ws.search_count == 0

        ws.search("test query 1")
        assert ws.search_count == 1

        ws.search("test query 2")
        assert ws.search_count == 2

    def test_search_log(self, paragraphs):
        ws = SimulatedWebSearch(paragraphs)
        ws.search("Scott Derrickson")

        log = ws.search_log
        assert len(log) == 1
        assert log[0]["search_number"] == 1
        assert log[0]["query"] == "Scott Derrickson"
        assert log[0]["num_results"] > 0
        assert log[0]["total_network_delay_ms"] == 0  # No network sim

    def test_search_stats(self, paragraphs):
        ws = SimulatedWebSearch(paragraphs)
        ws.search("query 1")
        ws.search("query 2")

        stats = ws.get_search_stats()
        assert stats["search_count"] == 2
        assert stats["total_result_bytes"] > 0
        assert stats["total_network_delay_ms"] == 0

    def test_empty_paragraphs(self):
        ws = SimulatedWebSearch([])
        results = ws.search("anything")
        assert results == ""

    def test_empty_query(self, paragraphs):
        ws = SimulatedWebSearch(paragraphs)
        results = ws.search("")
        # Should return first top_k paragraphs as fallback
        assert len(results) > 0

    # ---- Network Simulation ----

    def test_with_network_latency(self, paragraphs):
        """Test that network simulation adds delay and logs it."""
        net_config = NetworkConfig(
            name="test_net",
            rtt_ms=50,
            jitter_ms=0,
            packet_loss_rate=0.0,
            bandwidth_mbps=100.0,
        )
        ws = SimulatedWebSearch(paragraphs, network_config=net_config)

        t0 = __import__("time").time()
        ws.search("Scott Derrickson")
        elapsed = __import__("time").time() - t0

        # Should have at least ~100ms delay (2 x 50ms RTT: upload + download)
        assert elapsed >= 0.08  # Allow some tolerance

        # Check log shows network delay
        log = ws.search_log[0]
        assert log["upload_delay_ms"] > 0
        assert log["download_delay_ms"] > 0
        assert log["total_network_delay_ms"] > 0

    def test_network_summary(self, paragraphs):
        net_config = NETWORK_PRESETS["wifi"]
        ws = SimulatedWebSearch(paragraphs, network_config=net_config)
        ws.search("test query")

        summary = ws.get_network_summary()
        assert summary is not None
        assert summary["total_transfers"] == 2  # 1 upload + 1 download
        assert summary["total_delay_ms"] > 0

    def test_no_network_summary_without_config(self, paragraphs):
        ws = SimulatedWebSearch(paragraphs)
        assert ws.get_network_summary() is None

    def test_network_delay_compounds(self, paragraphs):
        """Test that multiple searches compound network delay."""
        net_config = NetworkConfig(
            name="test_net",
            rtt_ms=50,
            jitter_ms=0,
            packet_loss_rate=0.0,
            bandwidth_mbps=1000.0,
        )
        ws = SimulatedWebSearch(paragraphs, network_config=net_config)

        ws.search("query 1")
        ws.search("query 2")
        ws.search("query 3")

        stats = ws.get_search_stats()
        # Each search has upload + download = ~100ms, so 3 searches = ~300ms
        assert stats["total_network_delay_ms"] >= 250  # Allow tolerance
        assert stats["search_count"] == 3

    def test_poor_cellular_high_latency(self, paragraphs):
        """Test that poor cellular adds significant latency."""
        net_config = NETWORK_PRESETS["poor_cellular"]
        ws = SimulatedWebSearch(paragraphs, network_config=net_config, seed=42)

        ws.search("test")

        stats = ws.get_search_stats()
        # poor_cellular: RTT=200ms, so upload+download >= ~400ms
        assert stats["total_network_delay_ms"] >= 200


# ============================================================================
# TF-IDF Ranking Tests
# ============================================================================


class TestTFIDFRanking:
    """Tests for the TF-IDF paragraph ranking in SimulatedWebSearch."""

    @pytest.fixture
    def search_tool(self):
        paragraphs = [
            {"title": "Python Language", "text": "Python is a programming language used for web development and data science."},
            {"title": "Java Language", "text": "Java is a programming language used for enterprise applications."},
            {"title": "French Cuisine", "text": "French cuisine is known for its refined techniques and flavors."},
        ]
        return SimulatedWebSearch(paragraphs)

    def test_relevant_results_first(self, search_tool):
        results = search_tool.search("Python programming language")
        assert results.startswith("[Python Language]")

    def test_irrelevant_ranked_lower(self, search_tool):
        results = search_tool.search("programming language")
        # Both Python and Java should appear before French Cuisine
        python_pos = results.find("Python Language")
        java_pos = results.find("Java Language")
        french_pos = results.find("French Cuisine")

        # At least one programming result should come before cuisine
        assert min(python_pos, java_pos) < french_pos

    def test_specific_query_matches_specific_doc(self, search_tool):
        results = search_tool.search("French cuisine refined techniques")
        assert results.startswith("[French Cuisine]")
