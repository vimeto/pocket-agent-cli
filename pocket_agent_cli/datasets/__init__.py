"""Dataset abstraction layer for benchmark datasets.

This module provides a unified interface for loading and managing
benchmark datasets like MBPP and HumanEval.

Example usage:
    from pocket_agent_cli.datasets import DatasetRegistry, Problem

    # List available datasets
    datasets = DatasetRegistry.list_datasets()
    print(datasets)  # {'mbpp': '...', 'humaneval': '...'}

    # Create a dataset instance
    dataset = DatasetRegistry.create("mbpp", Path("~/.pocket-agent-cli/data"))

    # Download if needed
    if not dataset.is_downloaded():
        dataset.download()

    # Load problems
    problems = dataset.load(split="test", limit=10)

    for problem in problems:
        print(f"Problem {problem.task_id}: {problem.prompt[:50]}...")
"""

from .base import Dataset, Problem
from .registry import DatasetRegistry

# Import dataset implementations to register them
from . import mbpp
from . import humaneval
from . import gsm8k
from . import hotpotqa
from . import bfcl

__all__ = [
    "Dataset",
    "Problem",
    "DatasetRegistry",
]
