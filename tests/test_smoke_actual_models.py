"""Intensive smoke tests using ACTUAL models.

These tests verify that the full pipeline works with real models,
not mocks. They test actual inference and code generation capabilities.

Markers:
    @pytest.mark.slow - Tests that take significant time due to model loading
    @pytest.mark.smoke - Smoke tests for quick validation
"""

import pytest
import asyncio
import time
from pathlib import Path
from typing import Optional, Tuple

from pocket_agent_cli.services.inference_service import InferenceService
from pocket_agent_cli.models.model_service import ModelService
from pocket_agent_cli.benchmarks.benchmark_service import BenchmarkService, BenchmarkProblemResult
from pocket_agent_cli.config import BenchmarkConfig, BENCHMARK_MODES, InferenceConfig, Model
from pocket_agent_cli.datasets import DatasetRegistry, Problem


# ============================================================================
# Constants
# ============================================================================

# Models to test (in order of preference for testing)
# Note: Gemma models don't use thinking tokens, so they're faster for testing
# Qwen-3 models use thinking tokens which require more max_tokens
PREFERRED_MODELS = [
    "gemma-3n-e2b-it",    # 2B params - fast, no thinking tokens
    "qwen-3-4b",          # 4B params - capable but uses thinking
    "llama-3.2-3b-instruct",  # 3B params
    "qwen-3-0.6b",        # 0.6B params - smallest
]

# Datasets
DATASETS = ["mbpp", "humaneval"]


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def data_dir():
    """Get the actual package data directory."""
    return Path(__file__).parent.parent / "pocket_agent_cli" / "data"


@pytest.fixture(scope="module")
def model_service():
    """Create a real model service."""
    return ModelService()


@pytest.fixture(scope="module")
def inference_service():
    """Create a real inference service (no model loaded yet)."""
    service = InferenceService()
    yield service
    # Cleanup: unload any loaded model
    if service.current_model is not None:
        service.unload_model()


def get_available_model(model_service: ModelService) -> Optional[Tuple[Model, str]]:
    """Find the first available downloaded model from our preferred list.

    Returns tuple of (Model, version_name) or None if no models available.
    """
    downloaded = model_service.get_downloaded_models()
    downloaded_ids = {d["model"].id for d in downloaded}

    for model_id in PREFERRED_MODELS:
        if model_id in downloaded_ids:
            # Find the downloaded entry
            for d in downloaded:
                if d["model"].id == model_id:
                    return (d["model"], d["version"])

    return None


# ============================================================================
# Test: Model Loading Smoke Tests
# ============================================================================

class TestModelLoadingSmoke:
    """Smoke tests for model loading with actual models."""

    @pytest.mark.slow
    @pytest.mark.smoke
    def test_can_load_any_model(self, model_service, inference_service):
        """Test that we can load at least one model."""
        result = get_available_model(model_service)

        if result is None:
            pytest.skip("No models available for testing")

        model, version = result
        model_id = model.id

        print(f"\nLoading model: {model_id} (version: {version})")
        start = time.time()

        # Get model with specific version
        model = model_service.get_model(model_id, version)
        config = InferenceConfig()

        inference_service.load_model(model, config)
        load_time = time.time() - start

        print(f"Model loaded in {load_time:.2f}s")

        assert inference_service.current_model is not None

    @pytest.mark.slow
    @pytest.mark.smoke
    def test_model_can_generate_text(self, model_service, inference_service):
        """Test that a loaded model can generate text."""
        result = get_available_model(model_service)

        if result is None:
            pytest.skip("No models available for testing")

        model, version = result
        model_id = model.id

        # Load model if not already loaded
        if inference_service.current_model is None:
            model = model_service.get_model(model_id, version)
            config = InferenceConfig()
            inference_service.load_model(model, config)

        # Simple generation test using message format
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a Python function that returns 42:\n\ndef answer():"}
        ]

        print(f"\nGenerating with message: {messages[1]['content'][:50]}...")

        response_text = ""
        token_count = 0

        for chunk in inference_service.generate(messages=messages, stream=True):
            if "token" in chunk:
                response_text += chunk["token"]
                token_count += 1

        print(f"Generated {token_count} tokens: {response_text[:100]}...")

        assert len(response_text) > 0, "Model should generate some text"


# ============================================================================
# Test: Dataset Loading Smoke Tests
# ============================================================================

class TestDatasetLoadingSmoke:
    """Smoke tests for dataset loading."""

    @pytest.mark.smoke
    def test_mbpp_dataset_loads(self, data_dir):
        """Test MBPP dataset loads correctly."""
        registry = DatasetRegistry()
        dataset = registry.get("mbpp", data_dir=data_dir)

        problems = dataset.load_problems()

        assert len(problems) > 0, "MBPP should have problems"
        assert all(isinstance(p, Problem) for p in problems)

        print(f"\nMBPP loaded {len(problems)} problems")

    @pytest.mark.smoke
    def test_humaneval_dataset_loads(self, data_dir):
        """Test HumanEval dataset loads correctly."""
        registry = DatasetRegistry()
        dataset = registry.get("humaneval", data_dir=data_dir)

        problems = dataset.load_problems()

        assert len(problems) > 0, "HumanEval should have problems"
        assert all(isinstance(p, Problem) for p in problems)

        print(f"\nHumanEval loaded {len(problems)} problems")


# ============================================================================
# Test: Full Pipeline Smoke Tests with Actual Models
# ============================================================================

class TestFullPipelineSmoke:
    """Smoke tests for the full benchmark pipeline with actual models."""

    @pytest.mark.slow
    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_mbpp_single_problem_with_real_model(self, model_service, inference_service, data_dir):
        """Test running a single MBPP problem with a real model."""
        result = get_available_model(model_service)

        if result is None:
            pytest.skip("No models available for testing")

        model, version = result
        model_id = model.id

        # Load model
        if inference_service.current_model is None:
            print(f"\nLoading model: {model_id}")
            model = model_service.get_model(model_id, version)
            config = InferenceConfig()
            inference_service.load_model(model, config)

        # Create benchmark service
        bench_config = BenchmarkConfig(
            model_name=model_id,
            problems_limit=1,
            max_tokens=512,
            temperature=0.1,
        )

        service = BenchmarkService(
            inference_service=inference_service,
            config=bench_config,
            dataset_name='mbpp',
            data_dir=data_dir
        )

        # Run on first problem
        problem = service.problems[0]
        problem_dict = service._problem_to_dict(problem)
        mode = BENCHMARK_MODES['base']

        print(f"\nRunning problem: {problem.task_id}")
        print(f"Description: {problem.prompt[:100]}...")

        # Generate prompt and build messages
        user_content = service._prepare_problem_prompt(problem_dict, mode)
        messages = [
            {"role": "system", "content": mode.system_prompt},
            {"role": "user", "content": user_content},
        ]

        response_text = ""
        for chunk in inference_service.generate(messages=messages, stream=True):
            if "token" in chunk:
                response_text += chunk["token"]

        print(f"\nGenerated response ({len(response_text)} chars):")
        print(response_text[:500])

        # Extract code
        code = service._extract_code(response_text, [])

        print(f"\nExtracted code:")
        print(code[:500] if code else "No code extracted")

        assert len(response_text) > 0, "Should generate a response"


# ============================================================================
# Test: Benchmark Run with Test Execution (Most Intensive)
# ============================================================================

class TestBenchmarkWithTestExecution:
    """Tests that run actual benchmarks with test execution."""

    @pytest.mark.slow
    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_larger_model_passes_at_least_one_mbpp_task(self, model_service, inference_service, data_dir):
        """Test that a larger model (Qwen-3-4b or Gemma) can pass at least one MBPP task.

        This is the key validation test - at least one real model should pass
        at least one real task.
        """
        result = get_available_model(model_service)

        if result is None:
            pytest.skip("No models available for testing")

        model, version = result
        model_id = model.id

        print(f"\n{'='*60}")
        print(f"INTENSIVE SMOKE TEST: MBPP with {model_id}")
        print(f"{'='*60}")

        # Load model
        if inference_service.current_model is None:
            print(f"\nLoading model: {model_id}...")
            model = model_service.get_model(model_id, version)
            config = InferenceConfig()
            inference_service.load_model(model, config)
            print("Model loaded!")

        # Create benchmark service
        # Note: MBPP problems 1-5 are more complex, problems 6+ are simpler
        # We load more problems and skip to simpler ones for better success rate
        bench_config = BenchmarkConfig(
            model_name=model_id,
            problems_limit=15,  # Load enough to get to simpler problems
            max_tokens=1024,  # Enough for code generation
            temperature=0.1,  # Low temperature for deterministic output
        )

        service = BenchmarkService(
            inference_service=inference_service,
            config=bench_config,
            dataset_name='mbpp',
            data_dir=data_dir
        )

        # Create sandbox for test execution
        service.tool_executor._create_sandbox()

        passed_any = False
        results_summary = []

        # Skip first 5 complex problems, test 5 simpler ones starting from index 5
        problems_to_test = service.problems[5:10]
        for idx, problem in enumerate(problems_to_test):
            problem_dict = service._problem_to_dict(problem)
            mode = BENCHMARK_MODES['base']

            print(f"\n--- Problem {idx + 1}/{len(problems_to_test)}: {problem.task_id} ---")
            print(f"Description: {problem.prompt[:80]}...")

            # Generate prompt and build messages
            user_content = service._prepare_problem_prompt(problem_dict, mode)
            messages = [
                {"role": "system", "content": mode.system_prompt},
                {"role": "user", "content": user_content},
            ]

            response_text = ""
            start_time = time.time()

            for chunk in inference_service.generate(messages=messages, stream=True):
                if "token" in chunk:
                    response_text += chunk["token"]

            gen_time = time.time() - start_time

            # Extract code
            code = service._extract_code(response_text, [])

            print(f"Generated in {gen_time:.2f}s, {len(response_text)} chars")
            print(f"Code preview: {code[:100] if code else 'No code'}...")

            if not code:
                print("SKIP: No code extracted")
                results_summary.append((problem.task_id, "no_code", 0, 0))
                continue

            # Run tests
            try:
                test_results = await service._run_tests(
                    code,
                    problem_dict["test_list"],
                    problem_dict
                )

                passed = sum(1 for r in test_results if r.passed)
                total = len(test_results)

                print(f"Tests: {passed}/{total} passed")

                results_summary.append((problem.task_id, "tested", passed, total))

                if passed == total and total > 0:
                    passed_any = True
                    print("SUCCESS! All tests passed!")
                elif passed > 0:
                    print(f"PARTIAL: {passed}/{total} tests passed")
                else:
                    # Show first failure for debugging
                    for r in test_results:
                        if not r.passed:
                            print(f"First failure: {r.test_case[:50]}...")
                            if r.output:
                                print(f"Error: {r.output[:100]}...")
                            break

            except Exception as e:
                print(f"ERROR: {str(e)[:100]}")
                results_summary.append((problem.task_id, "error", 0, 0))

        # Summary
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        for task_id, status, passed, total in results_summary:
            if status == "tested":
                print(f"  {task_id}: {passed}/{total} tests passed")
            else:
                print(f"  {task_id}: {status}")

        # This is the key assertion
        assert passed_any, f"Model {model_id} should pass at least one MBPP task"

    @pytest.mark.slow
    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_larger_model_passes_at_least_one_humaneval_task(self, model_service, inference_service, data_dir):
        """Test that a larger model can pass at least one HumanEval task.

        Similar to MBPP test but for HumanEval format.
        """
        result = get_available_model(model_service)

        if result is None:
            pytest.skip("No models available for testing")

        model, version = result
        model_id = model.id

        print(f"\n{'='*60}")
        print(f"INTENSIVE SMOKE TEST: HumanEval with {model_id}")
        print(f"{'='*60}")

        # Load model
        if inference_service.current_model is None:
            print(f"\nLoading model: {model_id}...")
            model = model_service.get_model(model_id, version)
            config = InferenceConfig()
            inference_service.load_model(model, config)
            print("Model loaded!")

        # Create benchmark service
        # HumanEval has simpler function-level completions
        bench_config = BenchmarkConfig(
            model_name=model_id,
            problems_limit=10,  # Load enough problems
            max_tokens=1024,  # Enough for function completions
            temperature=0.1,
        )

        service = BenchmarkService(
            inference_service=inference_service,
            config=bench_config,
            dataset_name='humaneval',
            data_dir=data_dir
        )

        # Create sandbox for test execution
        service.tool_executor._create_sandbox()

        passed_any = False
        results_summary = []

        # Test first 5 HumanEval problems
        problems_to_test = service.problems[:5]
        for idx, problem in enumerate(problems_to_test):
            problem_dict = service._problem_to_dict(problem)
            mode = BENCHMARK_MODES['base']

            print(f"\n--- Problem {idx + 1}/{len(problems_to_test)}: {problem.task_id} ---")
            print(f"Entry point: {problem.entry_point}")

            # Generate prompt and build messages
            user_content = service._prepare_problem_prompt(problem_dict, mode)
            messages = [
                {"role": "system", "content": mode.system_prompt},
                {"role": "user", "content": user_content},
            ]

            response_text = ""
            start_time = time.time()

            for chunk in inference_service.generate(messages=messages, stream=True):
                if "token" in chunk:
                    response_text += chunk["token"]

            gen_time = time.time() - start_time

            # Extract code
            code = service._extract_code(response_text, [])

            print(f"Generated in {gen_time:.2f}s, {len(response_text)} chars")
            print(f"Code preview: {code[:100] if code else 'No code'}...")

            if not code:
                print("SKIP: No code extracted")
                results_summary.append((problem.task_id, "no_code", 0, 0))
                continue

            # For HumanEval, we need to combine prompt + generated code
            full_code = problem.prompt + code

            # Run tests
            try:
                test_results = await service._run_tests(
                    full_code,
                    problem_dict["test_list"],
                    problem_dict
                )

                passed = sum(1 for r in test_results if r.passed)
                total = len(test_results)

                print(f"Tests: {passed}/{total} passed")

                results_summary.append((problem.task_id, "tested", passed, total))

                if passed == total and total > 0:
                    passed_any = True
                    print("SUCCESS! All tests passed!")
                elif passed > 0:
                    print(f"PARTIAL: {passed}/{total} tests passed")
                else:
                    for r in test_results:
                        if not r.passed:
                            print(f"First failure: {r.test_case[:50]}...")
                            if r.output:
                                print(f"Error: {r.output[:100]}...")
                            break

            except Exception as e:
                print(f"ERROR: {str(e)[:100]}")
                results_summary.append((problem.task_id, "error", 0, 0))

        # Summary
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        for task_id, status, passed, total in results_summary:
            if status == "tested":
                print(f"  {task_id}: {passed}/{total} tests passed")
            else:
                print(f"  {task_id}: {status}")

        # This test passes if at least one task succeeds
        assert passed_any, f"Model {model_id} should pass at least one HumanEval task"


# ============================================================================
# Test: Tool Submission Mode Smoke Test
# ============================================================================

class TestToolSubmissionModeSmoke:
    """Smoke tests for tool submission benchmark mode."""

    @pytest.mark.slow
    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_tool_submission_mode_generates_response(self, model_service, inference_service, data_dir):
        """Test that tool submission mode can generate responses."""
        result = get_available_model(model_service)

        if result is None:
            pytest.skip("No models available for testing")

        model, version = result
        model_id = model.id

        # Load model
        if inference_service.current_model is None:
            print(f"\nLoading model: {model_id}")
            model = model_service.get_model(model_id, version)
            config = InferenceConfig()
            inference_service.load_model(model, config)

        # Create benchmark service
        bench_config = BenchmarkConfig(
            model_name=model_id,
            problems_limit=1,
            max_tokens=512,
            temperature=0.1,
        )

        service = BenchmarkService(
            inference_service=inference_service,
            config=bench_config,
            dataset_name='mbpp',
            data_dir=data_dir
        )

        # Use tool_submission mode
        problem = service.problems[0]
        problem_dict = service._problem_to_dict(problem)
        mode = BENCHMARK_MODES['tool_submission']

        print(f"\nTesting tool_submission mode with problem: {problem.task_id}")

        # Generate prompt and build messages
        user_content = service._prepare_problem_prompt(problem_dict, mode)
        messages = [
            {"role": "system", "content": mode.system_prompt},
            {"role": "user", "content": user_content},
        ]

        # The tool submission prompt should mention tools
        assert "submit_python_solution" in user_content or "tool" in user_content.lower() or "submit_python_solution" in mode.system_prompt

        print("Tool submission mode prompt generated correctly")

        # Generate response
        response_text = ""
        for chunk in inference_service.generate(messages=messages, stream=True):
            if "token" in chunk:
                response_text += chunk["token"]

        print(f"Response preview: {response_text[:300]}...")

        # Even if the model doesn't use tools perfectly, we just verify the mode works
        assert len(response_text) > 0, "Should generate a response in tool submission mode"
