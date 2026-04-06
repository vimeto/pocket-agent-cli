"""HotpotQA dataset implementation for web search QA benchmark.

HotpotQA is a multi-hop question answering dataset that requires reasoning
over multiple Wikipedia paragraphs. Each example includes supporting context
paragraphs, making it ideal for evaluating simulated web search.

Dataset source: https://huggingface.co/datasets/hotpot_qa
"""

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import Dataset, Problem
from .registry import DatasetRegistry


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison.

    Applies lowercasing, article removal, punctuation removal,
    and whitespace normalization. Standard HotpotQA evaluation.

    Args:
        s: Raw answer string.

    Returns:
        Normalized answer string.
    """
    if not s:
        return ""

    # Lowercase
    s = s.lower()

    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)

    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))

    # Normalize whitespace
    s = " ".join(s.split())

    return s.strip()


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Compute exact match score between prediction and ground truth.

    Args:
        prediction: Predicted answer string.
        ground_truth: Gold answer string.

    Returns:
        1.0 if normalized strings match, 0.0 otherwise.
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth.

    Standard HotpotQA evaluation metric: treats both answer strings as
    bags of tokens and computes precision, recall, and F1.

    Args:
        prediction: Predicted answer string.
        ground_truth: Gold answer string.

    Returns:
        Token-level F1 score between 0.0 and 1.0.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(prediction_tokens)
    recall = num_common / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def extract_answer_from_response(response: str) -> str:
    """Extract the final answer from a model response.

    Tries several patterns to locate the answer in the model's output.

    Args:
        response: Full model response text.

    Returns:
        Extracted answer string, or the last non-empty line as fallback.
    """
    if not response:
        return ""

    # Remove thinking blocks
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = response

    # Pattern 1: "The answer is: X" or "Answer: X"
    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*[:\s]*(.+?)(?:\.|$)",
        r"(?:^|\n)\s*Answer\s*:\s*(.+?)(?:\.|$)",
        r"\*\*Answer\*\*\s*:\s*(.+?)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1).strip()
            # Clean up markdown/formatting
            answer = re.sub(r"\*\*(.+?)\*\*", r"\1", answer)
            answer = answer.strip('"\'')
            if answer:
                return answer

    # Fallback: last non-empty line
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    if lines:
        last = lines[-1]
        # Strip common prefixes
        last = re.sub(r"^(So|Therefore|Thus|Hence|In conclusion),?\s*", "", last, flags=re.IGNORECASE)
        return last.strip(".").strip()

    return cleaned.strip()


@DatasetRegistry.register("hotpotqa")
class HotpotQADataset(Dataset):
    """HotpotQA multi-hop question answering dataset.

    HotpotQA contains questions that require reasoning over multiple
    supporting paragraphs. The context field provides pre-retrieved
    paragraphs that a simulated web search tool can return.

    Dataset source: https://huggingface.co/datasets/hotpot_qa

    Splits:
        - validation: ~7,405 examples (we use first 200 for benchmark)
        - sample: First 5 examples (for quick testing)
    """

    name = "hotpotqa"
    description = "Multi-hop QA with web search — 200 questions from HotpotQA validation"
    url = "https://huggingface.co/datasets/hotpot_qa"

    SYSTEM_PROMPT = (
        "You are a question answering assistant with access to a web search tool.\n\n"
        "Available tool:\n"
        "- web_search(query) - Search the web and return relevant passages\n\n"
        "To use the tool, write:\n"
        "<tool_call>\n"
        '{"name": "web_search", "arguments": {"query": "your search query"}}\n'
        "</tool_call>\n\n"
        "To answer the question:\n"
        "1. Search for relevant information using web_search\n"
        "2. Read the results carefully\n"
        "3. Answer the question based on the search results\n\n"
        "Provide your final answer clearly, starting with 'The answer is: '"
    )

    VALIDATION_SPLIT_SIZE = 7405
    BENCHMARK_SIZE = 200  # default number of problems for benchmarking

    def __init__(self, data_dir: Path):
        """Initialize HotpotQA dataset.

        Args:
            data_dir: Directory where dataset files are stored.
        """
        super().__init__(data_dir)

    @property
    def problem_count(self) -> int:
        """Number of problems in the benchmark subset."""
        return self.BENCHMARK_SIZE

    @property
    def available_splits(self) -> List[str]:
        """Available dataset splits."""
        return ["validation", "sample"]

    def is_downloaded(self, data_dir: Optional[Path] = None) -> bool:
        """Check if HotpotQA dataset is downloaded.

        Args:
            data_dir: Directory to check (uses self.data_dir if None).

        Returns:
            True if HotpotQA data file exists.
        """
        data_dir = Path(data_dir) if data_dir else self.data_dir
        return (data_dir / "hotpotqa_validation.json").exists()

    def download(self, data_dir: Optional[Path] = None) -> bool:
        """Download HotpotQA dataset from HuggingFace.

        Uses the ``datasets`` library to fetch ``hotpot_qa`` (distractor config).

        Args:
            data_dir: Directory to download to (uses self.data_dir if None).

        Returns:
            True if download successful, False otherwise.
        """
        data_dir = Path(data_dir) if data_dir else self.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        try:
            from datasets import load_dataset

            print(f"Downloading HotpotQA dataset from HuggingFace ({self.url})...")
            ds = load_dataset("hotpot_qa", "distractor", split="validation")

            # Convert to list of dicts, take first BENCHMARK_SIZE
            data = []
            for i, item in enumerate(ds):
                if i >= self.BENCHMARK_SIZE:
                    break
                data.append({
                    "id": item["id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "type": item["type"],
                    "level": item["level"],
                    "supporting_facts": {
                        "title": item["supporting_facts"]["title"],
                        "sent_id": item["supporting_facts"]["sent_id"],
                    },
                    "context": {
                        "title": item["context"]["title"],
                        "sentences": item["context"]["sentences"],
                    },
                })

            # Save validation split
            val_path = data_dir / "hotpotqa_validation.json"
            with open(val_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved validation split ({len(data)} problems) to {val_path}")

            # Save sample (first 5)
            sample_path = data_dir / "hotpotqa_sample.json"
            with open(sample_path, "w") as f:
                json.dump(data[:5], f, indent=2)
            print(f"Saved sample (5 problems) to {sample_path}")

            return True

        except ImportError:
            print(
                "Error: 'datasets' library is required for HotpotQA download. "
                "Install with: pip install datasets"
            )
            return False
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def load(self, split: str = "validation", limit: Optional[int] = None) -> List[Problem]:
        """Load HotpotQA problems.

        Args:
            split: Dataset split ("validation" or "sample").
            limit: Maximum number of problems to load.

        Returns:
            List of Problem objects.

        Raises:
            FileNotFoundError: If dataset files don't exist.
            ValueError: If split is invalid.
        """
        if split not in self.available_splits:
            raise ValueError(
                f"Invalid split '{split}'. Available: {self.available_splits}"
            )

        raw_data = self._load_raw_data(split)

        if limit is not None:
            raw_data = raw_data[:limit]

        problems = []
        for idx, item in enumerate(raw_data):
            problem = self._convert_to_problem(item, idx)
            problems.append(problem)

        return problems

    def _load_raw_data(self, split: str) -> List[Dict[str, Any]]:
        """Load raw HotpotQA data from files.

        Args:
            split: Dataset split to load.

        Returns:
            List of raw problem dictionaries.
        """
        if split == "sample":
            sample_file = self.data_dir / "hotpotqa_sample.json"
            if sample_file.exists():
                with open(sample_file) as f:
                    return json.load(f)
            # Fall back to validation with limit
            val_file = self.data_dir / "hotpotqa_validation.json"
            if val_file.exists():
                with open(val_file) as f:
                    data = json.load(f)
                return data[:5]

        data_file = self.data_dir / f"hotpotqa_{split}.json"
        if data_file.exists():
            with open(data_file) as f:
                return json.load(f)

        raise FileNotFoundError(
            f"HotpotQA dataset not found in {self.data_dir}. "
            f"Run download() first or check the data directory."
        )

    def _convert_to_problem(self, item: Dict[str, Any], index: int) -> Problem:
        """Convert raw HotpotQA item to Problem object.

        Args:
            item: Raw HotpotQA problem dictionary.
            index: Zero-based index of the problem in the split.

        Returns:
            Problem object with context stored in metadata.
        """
        # Build context paragraphs: list of (title, full_text) tuples
        paragraphs = []
        context = item.get("context", {})
        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])

        for title, sents in zip(titles, sentences_list):
            full_text = "".join(sents)
            paragraphs.append({"title": title, "text": full_text, "sentences": sents})

        # Identify supporting fact titles
        supporting_titles = list(set(
            item.get("supporting_facts", {}).get("title", [])
        ))

        # Test case stores expected answer for evaluation
        test_case = f"EXPECTED_ANSWER: {item['answer']}"

        return Problem(
            task_id=f"HotpotQA/{index}",
            prompt=item["question"],
            canonical_solution=item["answer"],
            test_cases=[test_case],
            entry_point="answer",
            metadata={
                "source": "hotpotqa",
                "hotpotqa_id": item.get("id", ""),
                "answer": item["answer"],
                "type": item.get("type", ""),
                "level": item.get("level", ""),
                "paragraphs": paragraphs,
                "supporting_titles": supporting_titles,
            },
        )

    def evaluate_response(
        self, problem: Problem, model_response: str
    ) -> Dict[str, Any]:
        """Evaluate a model response against the ground truth.

        Args:
            problem: The HotpotQA problem.
            model_response: The model's full response text.

        Returns:
            Dict with 'em' (exact match), 'f1' (F1 score), and
            'predicted_answer' fields.
        """
        ground_truth = problem.metadata.get("answer", "")
        predicted = extract_answer_from_response(model_response)

        em = exact_match_score(predicted, ground_truth)
        f1 = f1_score(predicted, ground_truth)

        return {
            "em": em,
            "f1": f1,
            "predicted_answer": predicted,
            "ground_truth": ground_truth,
        }

    def get_paragraphs(self, problem: Problem) -> List[Dict[str, str]]:
        """Get context paragraphs for a problem.

        Args:
            problem: A HotpotQA Problem object.

        Returns:
            List of paragraph dicts with 'title' and 'text' keys.
        """
        return problem.metadata.get("paragraphs", [])
