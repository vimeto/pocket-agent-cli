# Implementation Plan: Test Coverage, Dataset Abstraction, and HumanEval Integration

## Executive Summary

This plan addresses three key improvements to the pocket-agent-cli measurement framework:
1. Adding robust test coverage across all components
2. Creating an abstraction layer for easier dataset integration
3. Implementing the HumanEval benchmark dataset

Based on analysis of 20+ source files, this document provides detailed implementation steps.

---

## Progress Checklist

### Phase 1: Test Infrastructure
- [x] Create `tests/conftest.py` with shared fixtures
- [x] Add pytest configuration to `pyproject.toml`
- [x] Implement `tests/test_config.py`
- [x] Implement `tests/test_thinking_filter.py`
- [x] Implement `tests/test_result_export.py`

### Phase 2: Dataset Abstraction
- [x] Create `pocket_agent_cli/datasets/__init__.py`
- [x] Implement `base.py` with Dataset ABC and Problem dataclass
- [x] Implement `registry.py` with DatasetRegistry
- [x] Migrate MBPP to new `mbpp.py` implementation
- [x] Update `BenchmarkService` to use new abstraction
- [x] Add dataset CLI commands

### Phase 3: HumanEval Integration
- [x] Implement `humaneval.py` dataset
- [x] Update prompt generation for HumanEval format
- [x] Update test running for HumanEval check() format
- [x] Add HumanEval-specific tests

### Phase 4: Core Component Tests
- [ ] Implement `tests/test_benchmark_service.py`
- [ ] Implement `tests/test_inference_service.py`
- [ ] Implement `tests/test_tool_executor.py`
- [ ] Implement `tests/test_cli.py`

### Phase 5: Integration Tests
- [ ] Create `tests/integration/` directory structure
- [ ] Implement end-to-end benchmark tests
- [ ] Generate coverage reports and verify 80%+ coverage

---

## Part 1: Robust Test Coverage

### Current State Analysis

The project currently has minimal test coverage with only 3 test files:
- `tests/test_tool_extractor.py` - Tests for tool parsing (well-covered, 27 test cases)
- `tests/test_energy_monitoring.py` - Tests for power calculations (well-covered)
- `tests/test_gpu_inference.py` - Integration tests for GPU inference

### Test Coverage Goals

| Component | Current Coverage | Target Coverage |
|-----------|------------------|-----------------|
| Tool Extractor | ~80% | 95% |
| Energy Monitoring | ~70% | 90% |
| Benchmark Service | 0% | 85% |
| Inference Service | 0% | 80% |
| Config Module | 0% | 90% |
| CLI Commands | 0% | 70% |
| Dataset Loading | 0% | 90% |
| Result Export | 0% | 85% |
| Thinking Filter | 0% | 90% |

### 1.1 Unit Tests to Add

#### `tests/test_benchmark_service.py` (New)

```python
# Test cases needed:
- test_load_dataset_sample
- test_load_dataset_full
- test_load_dataset_fallback
- test_evaluate_problem_base_mode
- test_evaluate_problem_tool_submission_mode
- test_evaluate_problem_full_tool_mode
- test_extract_code_from_tool_calls
- test_extract_code_from_python_blocks
- test_extract_code_fallback
- test_run_tests_all_pass
- test_run_tests_some_fail
- test_run_tests_with_function_alias
- test_calculate_aggregate_stats
- test_calculate_pass_at_k
- test_benchmark_session_to_dict
- test_problem_result_to_dict
```

#### `tests/test_inference_service.py` (New)

```python
# Test cases needed:
- test_format_prompt_llama
- test_format_prompt_gemma
- test_format_prompt_qwen
- test_format_prompt_with_tools
- test_parse_tool_calls_json_block
- test_parse_tool_calls_native
- test_parse_tool_calls_none
- test_generate_streaming (mock llama-cpp)
- test_generate_with_tools (mock)
- test_temperature_override
- test_model_info_loaded
- test_model_info_not_loaded
```

#### `tests/test_tool_executor.py` (New)

```python
# Test cases needed:
- test_create_sandbox_docker
- test_create_sandbox_local
- test_run_python_code_docker
- test_run_python_code_local
- test_run_python_code_timeout
- test_upsert_file
- test_read_file_exists
- test_read_file_not_found
- test_submit_python_solution
- test_run_submission_tests_all_pass
- test_run_submission_tests_some_fail
- test_cleanup_sandbox
- test_execute_tools_max_iterations
```

#### `tests/test_config.py` (New)

```python
# Test cases needed:
- test_inference_config_defaults
- test_inference_config_validation
- test_benchmark_config_defaults
- test_benchmark_config_validation
- test_benchmark_modes_defined
- test_available_tools_schema
- test_model_version_management
- test_directory_creation
- test_environment_variable_override
```

#### `tests/test_thinking_filter.py` (New)

```python
# Test cases needed:
- test_filter_token_no_thinking
- test_filter_token_with_thinking_block
- test_filter_nested_thinking_blocks
- test_filter_partial_tags
- test_filter_multiple_blocks
- test_flush_remaining_buffer
- test_get_stats
- test_get_thinking_content
- test_remove_thinking_blocks_static
- test_various_thinking_patterns
```

#### `tests/test_result_export.py` (New)

```python
# Test cases needed:
- test_export_json
- test_export_csv
- test_export_markdown
- test_load_results
- test_compare_results
- test_export_metadata
```

#### `tests/test_cli.py` (New)

```python
# Test cases needed (using click.testing.CliRunner):
- test_model_list_command
- test_model_download_command
- test_benchmark_command_validation
- test_benchmark_command_invalid_mode
- test_chat_command_no_model
- test_info_command
- test_download_dataset_command
```

### 1.2 Integration Tests to Add

#### `tests/integration/test_benchmark_flow.py` (New)

```python
# End-to-end tests with mocked model:
- test_full_benchmark_base_mode
- test_full_benchmark_tool_submission_mode
- test_full_benchmark_full_tool_mode
- test_benchmark_with_pass_at_k
- test_benchmark_coordinator_multiple_models
- test_benchmark_result_persistence
```

### 1.3 Test Fixtures and Utilities

#### `tests/conftest.py` (New)

```python
# Shared fixtures:
- mock_model fixture
- mock_inference_service fixture
- mock_tool_executor fixture
- sample_dataset fixture
- temp_results_dir fixture
- mock_llama_cpp fixture
```

### 1.4 Testing Infrastructure Updates

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
test = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
    "pytest-mock>=3.10",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-v --cov=pocket_agent_cli --cov-report=html"
```

---

## Part 2: Dataset Abstraction Layer

### Current State

Datasets are currently hardcoded:
- MBPP dataset is loaded directly in `BenchmarkService._load_dataset()`
- Dataset path is hardcoded to `DATA_DIR / "mbpp_sample.json"`
- No abstraction for different dataset formats
- Download script is MBPP-specific

### 2.1 Dataset Interface Design

#### `pocket_agent_cli/datasets/base.py` (New)

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Problem:
    """Universal problem representation."""
    task_id: str
    prompt: str  # The problem description/function signature
    canonical_solution: str  # Reference solution
    test_cases: List[str]  # List of test assertions
    entry_point: str  # Function name to call
    metadata: Optional[Dict[str, Any]] = None


class Dataset(ABC):
    """Abstract base class for benchmark datasets."""

    name: str
    description: str
    url: str

    @abstractmethod
    def load(self, split: str = "test", limit: Optional[int] = None) -> List[Problem]:
        """Load problems from the dataset."""
        pass

    @abstractmethod
    def download(self, data_dir: Path) -> bool:
        """Download the dataset to the specified directory."""
        pass

    @abstractmethod
    def is_downloaded(self, data_dir: Path) -> bool:
        """Check if dataset is already downloaded."""
        pass

    @property
    @abstractmethod
    def problem_count(self) -> int:
        """Total number of problems in the dataset."""
        pass

    def get_sample(self, n: int = 3) -> List[Problem]:
        """Get a small sample for testing."""
        return self.load(limit=n)
```

### 2.2 Dataset Registry

#### `pocket_agent_cli/datasets/registry.py` (New)

```python
from typing import Dict, Type, Optional
from .base import Dataset


class DatasetRegistry:
    """Registry for available datasets."""

    _datasets: Dict[str, Type[Dataset]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a dataset."""
        def decorator(dataset_class: Type[Dataset]):
            cls._datasets[name] = dataset_class
            return dataset_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[Dataset]]:
        """Get a dataset by name."""
        return cls._datasets.get(name)

    @classmethod
    def list_datasets(cls) -> Dict[str, str]:
        """List all registered datasets with descriptions."""
        return {
            name: dataset_cls.description
            for name, dataset_cls in cls._datasets.items()
        }

    @classmethod
    def create(cls, name: str, data_dir: Path) -> Dataset:
        """Create and configure a dataset instance."""
        dataset_cls = cls.get(name)
        if not dataset_cls:
            raise ValueError(f"Unknown dataset: {name}")
        return dataset_cls(data_dir)
```

### 2.3 MBPP Dataset Implementation

#### `pocket_agent_cli/datasets/mbpp.py` (New)

```python
import json
import requests
from pathlib import Path
from typing import List, Optional
from .base import Dataset, Problem
from .registry import DatasetRegistry


@DatasetRegistry.register("mbpp")
class MBPPDataset(Dataset):
    """MBPP (Mostly Basic Python Problems) dataset."""

    name = "mbpp"
    description = "974 crowd-sourced Python programming problems"
    url = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._problems: Optional[List[Problem]] = None

    @property
    def problem_count(self) -> int:
        return 974

    def is_downloaded(self, data_dir: Path = None) -> bool:
        data_dir = data_dir or self.data_dir
        return (data_dir / "mbpp_full.json").exists()

    def download(self, data_dir: Path = None) -> bool:
        data_dir = data_dir or self.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(self.url)
            response.raise_for_status()

            # Parse JSONL
            raw_problems = []
            for line in response.text.strip().split('\n'):
                if line:
                    raw_problems.append(json.loads(line))

            # Save full dataset
            with open(data_dir / "mbpp_full.json", 'w') as f:
                json.dump(raw_problems, f, indent=2)

            # Save test split (problems 11-510)
            test_problems = [p for p in raw_problems if 11 <= p['task_id'] <= 510]
            with open(data_dir / "mbpp_test.json", 'w') as f:
                json.dump(test_problems, f, indent=2)

            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def load(self, split: str = "test", limit: Optional[int] = None) -> List[Problem]:
        if split == "test":
            path = self.data_dir / "mbpp_test.json"
        elif split == "full":
            path = self.data_dir / "mbpp_full.json"
        else:
            path = self.data_dir / "mbpp_sample.json"

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}. Run download first.")

        with open(path) as f:
            raw_data = json.load(f)

        problems = []
        for item in raw_data[:limit]:
            problems.append(Problem(
                task_id=str(item['task_id']),
                prompt=item['text'],
                canonical_solution=item['code'],
                test_cases=item['test_list'],
                entry_point=self._extract_function_name(item['code']),
            ))

        return problems

    def _extract_function_name(self, code: str) -> str:
        """Extract the main function name from code."""
        import re
        match = re.search(r'def\s+(\w+)\s*\(', code)
        return match.group(1) if match else "solution"
```

### 2.4 Update BenchmarkService

#### Changes to `pocket_agent_cli/benchmarks/benchmark_service.py`

```python
# Add import
from ..datasets import DatasetRegistry, Problem

class BenchmarkService:
    def __init__(self, inference_service, config=None, dataset_name: str = "mbpp"):
        self.inference_service = inference_service
        self.config = config
        self.tool_executor = ToolExecutor()
        self.system_monitor = UnifiedMonitor()

        # Load dataset through registry
        self.dataset = DatasetRegistry.create(dataset_name, DATA_DIR)
        if not self.dataset.is_downloaded():
            print(f"Downloading {dataset_name} dataset...")
            self.dataset.download()

        self.problems = self.dataset.load(
            split="test" if config and config.problems_limit else "sample"
        )

    async def _evaluate_problem(self, problem: Problem, mode: BenchmarkMode):
        # Use Problem dataclass instead of dict
        # problem.task_id, problem.prompt, problem.test_cases, problem.entry_point
        ...
```

### 2.5 Update CLI

#### Changes to `pocket_agent_cli/cli.py`

```python
@cli.command("download-dataset")
@click.option("--dataset", type=click.Choice(DatasetRegistry.list_datasets().keys()),
              default="mbpp", help="Dataset to download")
def download_dataset(dataset: str):
    """Download benchmark datasets."""
    from .datasets import DatasetRegistry

    ds = DatasetRegistry.create(dataset, DATA_DIR)
    console.print(f"[bold]Downloading {ds.name} dataset...[/bold]")
    console.print(f"Description: {ds.description}")

    if ds.download():
        console.print(f"[green]Downloaded {ds.problem_count} problems[/green]")
    else:
        console.print("[red]Download failed[/red]")


@cli.command("list-datasets")
def list_datasets():
    """List available benchmark datasets."""
    from .datasets import DatasetRegistry

    table = Table(title="Available Datasets")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Downloaded", style="green")

    for name, desc in DatasetRegistry.list_datasets().items():
        ds = DatasetRegistry.create(name, DATA_DIR)
        downloaded = "[green]Yes[/green]" if ds.is_downloaded() else "[red]No[/red]"
        table.add_row(name, desc, downloaded)

    console.print(table)
```

---

## Part 3: HumanEval Integration

### HumanEval Dataset Overview

Based on [OpenAI's HumanEval](https://github.com/openai/human-eval):
- 164 hand-written Python programming problems
- Each includes: function signature, docstring, canonical solution, unit tests
- Uses pass@k metric for evaluation

### 3.1 HumanEval Dataset Implementation

#### `pocket_agent_cli/datasets/humaneval.py` (New)

```python
import json
import gzip
from pathlib import Path
from typing import List, Optional, Dict, Any
from .base import Dataset, Problem
from .registry import DatasetRegistry


@DatasetRegistry.register("humaneval")
class HumanEvalDataset(Dataset):
    """HumanEval dataset from OpenAI."""

    name = "humaneval"
    description = "164 hand-written Python programming problems from OpenAI"
    url = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._problems: Optional[List[Problem]] = None

    @property
    def problem_count(self) -> int:
        return 164

    def is_downloaded(self, data_dir: Path = None) -> bool:
        data_dir = data_dir or self.data_dir
        return (data_dir / "humaneval.json").exists()

    def download(self, data_dir: Path = None) -> bool:
        import requests
        data_dir = data_dir or self.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Download gzipped file
            response = requests.get(self.url)
            response.raise_for_status()

            # Decompress and parse JSONL
            raw_problems = []
            decompressed = gzip.decompress(response.content).decode('utf-8')

            for line in decompressed.strip().split('\n'):
                if line:
                    raw_problems.append(json.loads(line))

            # Convert to our format
            converted = []
            for item in raw_problems:
                converted.append({
                    'task_id': item['task_id'],
                    'prompt': item['prompt'],
                    'canonical_solution': item['canonical_solution'],
                    'test': item['test'],
                    'entry_point': item['entry_point'],
                })

            # Save dataset
            with open(data_dir / "humaneval.json", 'w') as f:
                json.dump(converted, f, indent=2)

            # Create sample file (first 5 problems)
            with open(data_dir / "humaneval_sample.json", 'w') as f:
                json.dump(converted[:5], f, indent=2)

            print(f"Downloaded {len(converted)} HumanEval problems")
            return True

        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def load(self, split: str = "test", limit: Optional[int] = None) -> List[Problem]:
        if split == "sample":
            path = self.data_dir / "humaneval_sample.json"
        else:
            path = self.data_dir / "humaneval.json"

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}. Run download first.")

        with open(path) as f:
            raw_data = json.load(f)

        problems = []
        for item in raw_data[:limit]:
            # Convert HumanEval test format to our assertion format
            test_cases = self._convert_tests(item['test'], item['entry_point'])

            problems.append(Problem(
                task_id=item['task_id'],
                prompt=item['prompt'],
                canonical_solution=item['canonical_solution'],
                test_cases=test_cases,
                entry_point=item['entry_point'],
                metadata={
                    'original_test': item['test']
                }
            ))

        return problems

    def _convert_tests(self, test_code: str, entry_point: str) -> List[str]:
        """Convert HumanEval test format to individual assertions.

        HumanEval uses a check(candidate) function that contains assertions.
        We need to extract these or run the check function.
        """
        # HumanEval tests typically look like:
        # def check(candidate):
        #     assert candidate([1, 2, 3]) == 6
        #     assert candidate([]) == 0

        import re

        # Extract assert statements
        assertions = re.findall(r'assert\s+.+', test_code)

        # Replace 'candidate' with the actual function name
        converted = []
        for assertion in assertions:
            converted.append(assertion.replace('candidate', entry_point))

        return converted if converted else [f"# Run check function\ncheck({entry_point})"]

    def create_test_harness(self, problem: Problem, generated_code: str) -> str:
        """Create complete test code for HumanEval.

        HumanEval requires running the check() function from the test.
        """
        original_test = problem.metadata.get('original_test', '')

        return f'''
{generated_code}

{original_test}

check({problem.entry_point})
'''
```

### 3.2 Update BenchmarkService for HumanEval

The `_run_tests` method needs to handle HumanEval's `check()` function format:

```python
async def _run_tests(
    self,
    code: str,
    problem: Problem,  # Change from test_cases list to Problem
) -> List[TestResult]:
    """Run test cases against code."""
    results = []

    # Check if we need HumanEval-style testing
    if problem.metadata and 'original_test' in problem.metadata:
        # Use check() function harness
        from ..datasets.humaneval import HumanEvalDataset

        test_code = f'''
{code}

{problem.metadata['original_test']}

check({problem.entry_point})
'''
        try:
            output = await self.tool_executor._run_python_code(test_code)

            # No output = all assertions passed
            has_error = any(err in output.lower() for err in [
                "assertionerror", "error:", "traceback", "exception"
            ])

            results.append(TestResult(
                test_case="HumanEval check() function",
                passed=not has_error,
                output=output or "All assertions passed",
            ))
        except Exception as e:
            results.append(TestResult(
                test_case="HumanEval check() function",
                passed=False,
                error=str(e),
            ))
    else:
        # Standard MBPP-style individual assertions
        for test_case in problem.test_cases:
            # ... existing logic ...
            pass

    return results
```

### 3.3 Prompt Templates for HumanEval

HumanEval provides function signatures with docstrings. Update `model_prompts.py`:

```python
HUMANEVAL_PROMPTS = {
    "base": {
        "system_prompt": "Complete the given Python function. Output ONLY the function implementation (no signature, no docstring).",
        "user_suffix": "\nComplete this function:\n",
    },
    "tool_submission": {
        "system_prompt": "Complete the Python function and submit using submit_python_solution tool.",
        "user_suffix": "\nComplete and submit:\n",
    },
    "full_tool": {
        "system_prompt": """Complete the Python function. Available tools:
- run_python_code: Execute code to test
- submit_python_solution: Submit final solution

MUST submit complete solution with function signature and implementation.""",
        "user_suffix": "",
    }
}
```

### 3.4 Dataset-Specific Mode Configuration

Create flexible prompt generation based on dataset:

```python
# In benchmark_service.py
def _prepare_prompt(self, problem: Problem, mode: BenchmarkMode) -> Dict[str, str]:
    """Prepare prompt based on dataset and mode."""

    dataset_name = getattr(self.dataset, 'name', 'mbpp')

    if dataset_name == 'humaneval':
        # HumanEval: prompt contains function signature + docstring
        # Model should complete the function body
        user_content = f"{problem.prompt}\n\n# Complete the function above"
    else:
        # MBPP: prompt is just description, need to generate whole function
        test_examples = "\n".join(problem.test_cases[:2])
        user_content = f"{problem.prompt}\n\nExample test cases:\n{test_examples}"

    return {
        'system': mode.system_prompt,
        'user': user_content,
    }
```

---

## Part 4: Implementation Order and Dependencies

### Phase 1: Test Infrastructure (Week 1)

1. Create `tests/conftest.py` with shared fixtures
2. Add pytest configuration to `pyproject.toml`
3. Implement `tests/test_config.py`
4. Implement `tests/test_thinking_filter.py`
5. Implement `tests/test_result_export.py`

### Phase 2: Dataset Abstraction (Week 2)

1. Create `pocket_agent_cli/datasets/__init__.py`
2. Implement `base.py` with `Dataset` ABC and `Problem` dataclass
3. Implement `registry.py` with `DatasetRegistry`
4. Migrate MBPP to new `mbpp.py` implementation
5. Update `BenchmarkService` to use new abstraction
6. Add dataset CLI commands

### Phase 3: HumanEval Integration (Week 3)

1. Implement `humaneval.py` dataset
2. Update prompt generation for HumanEval format
3. Update test running for HumanEval check() format
4. Add HumanEval-specific tests

### Phase 4: Core Component Tests (Week 4)

1. Implement `tests/test_benchmark_service.py`
2. Implement `tests/test_inference_service.py`
3. Implement `tests/test_tool_executor.py`
4. Implement `tests/test_cli.py`

### Phase 5: Integration Tests (Week 5)

1. Create `tests/integration/` directory
2. Implement end-to-end benchmark tests
3. Add CI/CD pipeline configuration
4. Generate coverage reports

---

## File Structure Summary

```
pocket_agent_cli/
├── datasets/                    # NEW: Dataset abstraction
│   ├── __init__.py              # Export registry and base classes
│   ├── base.py                  # Dataset ABC, Problem dataclass
│   ├── registry.py              # DatasetRegistry singleton
│   ├── mbpp.py                  # MBPP implementation
│   └── humaneval.py             # HumanEval implementation
├── benchmarks/
│   ├── benchmark_service.py     # UPDATED: Use dataset abstraction
│   └── benchmark_coordinator.py # UPDATED: Dataset selection
├── cli.py                       # UPDATED: Dataset commands
└── ...

tests/
├── conftest.py                  # NEW: Shared fixtures
├── test_config.py               # NEW
├── test_thinking_filter.py      # NEW
├── test_benchmark_service.py    # NEW
├── test_inference_service.py    # NEW
├── test_tool_executor.py        # NEW
├── test_result_export.py        # NEW
├── test_cli.py                  # NEW
├── test_tool_extractor.py       # EXISTS
├── test_energy_monitoring.py    # EXISTS
├── test_gpu_inference.py        # EXISTS
└── integration/                 # NEW
    └── test_benchmark_flow.py   # NEW
```

---

## Success Criteria

1. **Test Coverage**: Achieve 80%+ code coverage across all modules
2. **Dataset Abstraction**:
   - Any new dataset can be added with <100 lines of code
   - No changes required to `BenchmarkService` core logic
3. **HumanEval Integration**:
   - Successfully download and load all 164 problems
   - Correct test execution using `check()` function format
   - Accurate pass@k computation

---

## References

- [OpenAI HumanEval GitHub](https://github.com/openai/human-eval)
- [HumanEval on Hugging Face](https://huggingface.co/datasets/openai/openai_humaneval)
- [MBPP Dataset](https://github.com/google-research/google-research/tree/master/mbpp)
