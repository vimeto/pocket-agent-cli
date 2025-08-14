"""Benchmark evaluation service."""

import json
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import time
from ..config import DATA_DIR, BENCHMARK_MODES, AVAILABLE_TOOLS, SUBMIT_TOOL, BenchmarkMode, BenchmarkConfig
from ..services import InferenceService
from ..tools import ToolExecutor
from ..monitoring.unified_monitor import UnifiedMonitor
from ..utils.model_prompts import get_model_prompt
import os
DEBUG_BENCHMARK = os.environ.get("DEBUG_BENCHMARK", "").lower() == "true"
STREAM_OUTPUT = os.environ.get("STREAM_OUTPUT", "").lower() == "true"

# Import profile decorator
try:
    from line_profiler import profile
except ImportError:
    # Define a no-op decorator if line_profiler is not available
    def profile(func):
        return func


@dataclass
class TestResult:
    """Result of a single test case."""

    test_case: str
    passed: bool
    output: Optional[str] = None
    error: Optional[str] = None


@dataclass
class BenchmarkProblemResult:
    """Result for a single benchmark problem."""

    problem_id: int
    start_time: datetime
    end_time: datetime
    response: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    test_results: List[TestResult] = None
    success: bool = False
    metrics: Dict[str, Any] = None
    model_load_time_ms: Optional[float] = None
    cold_start: bool = False
    context_length_used: int = 0
    inter_token_latencies: List[float] = None
    run_id: int = 0  # For pass@k tracking
    temperature: float = 0.7  # Temperature used for this run

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["start_time"] = self.start_time.isoformat()
        data["end_time"] = self.end_time.isoformat()
        data["duration_seconds"] = (self.end_time - self.start_time).total_seconds()
        return data


@dataclass
class BenchmarkSession:
    """A complete benchmark session."""

    session_id: str
    model_id: str
    mode: str
    start_time: datetime
    end_time: Optional[datetime] = None
    problems: List[BenchmarkProblemResult] = None
    system_metrics: List[Dict[str, Any]] = None
    aggregate_stats: Optional[Dict[str, Any]] = None
    config: Optional[BenchmarkConfig] = None  # Configuration used
    pass_at_k: Optional[Dict[int, Dict[str, Any]]] = None  # Pass@k results per problem

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        # Handle config conversion with Path objects
        config_dict = None
        if self.config:
            config_dict = self.config.model_dump()
            # Convert Path objects to strings
            if 'output_dir' in config_dict and hasattr(config_dict['output_dir'], '__fspath__'):
                config_dict['output_dir'] = str(config_dict['output_dir'])

        data = {
            "session_id": self.session_id,
            "model_id": self.model_id,
            "mode": self.mode,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "problems": [p.to_dict() for p in (self.problems or [])],
            "system_metrics": self.system_metrics,
            "aggregate_stats": self.aggregate_stats,
            "config": config_dict,
            "pass_at_k": self.pass_at_k,
        }
        return data


class BenchmarkService:
    """Service for running and evaluating benchmarks."""

    def __init__(self, inference_service: InferenceService, config: Optional[BenchmarkConfig] = None):
        self.inference_service = inference_service
        self.config = config
        self.tool_executor = ToolExecutor()
        self.system_monitor = UnifiedMonitor()  # Using unified monitor for performance
        self.dataset_path = DATA_DIR / "mbpp_sample.json"
        self.problems = self._load_dataset()

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load MBPP dataset."""
        # Try to load from the data directory first
        if self.dataset_path.exists():
            with open(self.dataset_path, "r") as f:
                return json.load(f)

        # Check for full dataset
        full_dataset_path = Path(__file__).parent.parent / "data" / "mbpp_full.json"
        if full_dataset_path.exists():
            print(f"Loading full MBPP dataset ({full_dataset_path})")
            with open(full_dataset_path, "r") as f:
                return json.load(f)

        # Check for test dataset
        test_dataset_path = Path(__file__).parent.parent / "data" / "mbpp_test.json"
        if test_dataset_path.exists():
            print(f"Loading test MBPP dataset ({test_dataset_path})")
            with open(test_dataset_path, "r") as f:
                return json.load(f)

        # Fall back to sample dataset
        sample_path = Path(__file__).parent.parent / "data" / "mbpp_sample.json"
        if sample_path.exists():
            print(f"Warning: Using sample dataset with only 3 problems. Run download_mbpp.py for full dataset.")
            with open(sample_path, "r") as f:
                return json.load(f)

        print("Error: No MBPP dataset found!")
        return []

    async def run_benchmark(
        self,
        mode: str,
        problem_ids: Optional[List[int]] = None,
        progress_callback: Optional[callable] = None,
    ) -> BenchmarkSession:
        """Run a benchmark evaluation.

        Args:
            mode: Benchmark mode (base, tool_submission, full_tool)
            problem_ids: Specific problem IDs to run, or None for all
            progress_callback: Callback for progress updates

        Returns:
            Benchmark session with results
        """
        if mode not in BENCHMARK_MODES:
            raise ValueError(f"Invalid mode: {mode}")

        benchmark_mode = BENCHMARK_MODES[mode]

        # Create session
        session = BenchmarkSession(
            session_id=f"bench_{int(datetime.now().timestamp())}",
            model_id=self.inference_service.current_model.id,
            mode=mode,
            start_time=datetime.now(),
            problems=[],
        )

        # Store session ID for cold start detection
        self._current_session_id = session.session_id

        # Start system monitoring
        # self.system_monitor.start_monitoring()

        try:
            # Select problems to run
            if problem_ids:
                problems = [p for p in self.problems if p["task_id"] in problem_ids]
            else:
                problems = self.problems

            # Run each problem
            for i, problem in enumerate(problems):
                if progress_callback:
                    progress_callback(i, len(problems), f"Running problem {problem['task_id']}")

                result = await self._evaluate_problem(problem, benchmark_mode)
                session.problems.append(result)

            # Finalize session
            session.end_time = datetime.now()
            session.system_metrics = self.system_monitor.export_metrics()
            session.aggregate_stats = self._calculate_aggregate_stats(session)

        finally:
            self.system_monitor.stop_monitoring()

        return session

    @profile
    async def _evaluate_problem(
        self,
        problem: Dict[str, Any],
        mode: BenchmarkMode,
    ) -> BenchmarkProblemResult:
        """Evaluate a single problem.

        Args:
            problem: Problem definition
            mode: Benchmark mode

        Returns:
            Problem result
        """
        start_time = datetime.now()
        if DEBUG_BENCHMARK:
            print(f"\n[DEBUG] Starting evaluation of problem {problem['task_id']}")
            print(f"[DEBUG] Problem text: {problem['text'][:100]}...")

        # Create tool executor with test cases for this problem
        self.tool_executor = ToolExecutor(use_docker=True, test_cases=problem.get("test_list", []))
        # Ensure sandbox is created
        self.tool_executor._create_sandbox()

        # Check if this is a cold start (first problem)
        is_cold_start = False
        model_load_time_ms = None

        # Determine if this is the first problem (cold start)
        try:
            # Check if we're in a session context
            if hasattr(self, '_current_session_id'):
                is_cold_start = self._current_session_id != getattr(self, '_last_session_id', None)
                self._last_session_id = self._current_session_id
        except:
            pass

        # Get model-specific prompt
        if self.inference_service.current_model:
            model_prompts = get_model_prompt(self.inference_service.current_model.architecture, mode.name)
            system_prompt = model_prompts.get("system_prompt", mode.system_prompt)
            user_suffix = model_prompts.get("user_suffix", "")
        else:
            system_prompt = mode.system_prompt
            user_suffix = ""

        # Prepare messages with test cases included
        # Format test cases to show expected function signatures
        test_list = problem.get("test_list", [])
        # Limit test examples shown to avoid token overflow
        if mode.name == "full_tool":
            # For full_tool mode, show only 1-2 examples to save tokens
            test_examples = "\n".join(test_list[:2])
        else:
            # For other modes, show up to 3 examples
            test_examples = "\n".join(test_list[:3])
        problem_with_tests = f"{problem['text']}\n\nExample test cases:\n{test_examples}"
        
        user_content = mode.user_prompt_template.format(problem=problem_with_tests)
        
        # Add model-specific user suffix if provided
        if user_suffix:
            user_content += user_suffix
        
        # Add explicit instruction for Gemma models in full_tool mode (backward compatibility)
        elif mode.name == "full_tool" and self.inference_service.current_model and self.inference_service.current_model.architecture == "gemma":
            user_content += "\n\nREMEMBER: Output ONLY [submit_python_solution(code=\"...\")] - NO ```python blocks!"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        if DEBUG_BENCHMARK:
            print(f"[DEBUG] Messages prepared, system prompt length: {len(system_prompt)}")
            print(f"[DEBUG] User prompt length: {len(user_content)}")


        # Prepare tools if needed
        tools = None
        if mode.requires_tools:
            if mode.name == "tool_submission":
                tools = [SUBMIT_TOOL]
            else:
                tools = AVAILABLE_TOOLS

        # Track token generation for inter-token latencies
        inter_token_latencies = []
        last_token_time = None
        thinking_tokens_count = 0
        regular_tokens_count = 0

        # Generate response based on mode
        if mode.name == "base":
            # Base mode: simple generation
            if DEBUG_BENCHMARK:
                print(f"[DEBUG] Starting inference in base mode...")
            response = ""
            metrics = {}
            token_count = 0
            token_buffer = []  # Buffer for efficient printing
            last_print_time = time.time()

            for chunk in self.inference_service.generate(messages, stream=True):
                current_time = time.time()
                if last_token_time is not None:
                    inter_token_latencies.append((current_time - last_token_time) * 1000)
                last_token_time = current_time

                token = chunk["token"]  # This is already filtered
                response += token  # Only add non-thinking tokens to response
                metrics = chunk["metrics"]
                token_count += 1
                
                # Track thinking vs regular tokens
                if chunk.get("is_thinking", False):
                    thinking_tokens_count += 1
                else:
                    regular_tokens_count += 1

                # Show progress every 100 tokens (less frequent)
                if DEBUG_BENCHMARK and token_count % 100 == 0:
                    print(f"\n[DEBUG] Generated {token_count} tokens, TPS: {metrics.get('tps', 0):.1f}")

            # Print remaining buffer
            if STREAM_OUTPUT and token_buffer:
                print(''.join(token_buffer), flush=True)

            # Add thinking token stats to metrics
            if "thinking_stats" in metrics:
                thinking_stats = metrics["thinking_stats"]
            else:
                thinking_stats = {
                    "thinking_tokens": thinking_tokens_count,
                    "regular_tokens": regular_tokens_count,
                    "total_tokens": token_count,
                    "thinking_ratio": thinking_tokens_count / token_count if token_count > 0 else 0
                }
            metrics["thinking_stats"] = thinking_stats
            
            if DEBUG_BENCHMARK:
                print(f"\n[DEBUG] Generation complete. Total tokens: {token_count}")
                if thinking_tokens_count > 0:
                    print(f"[DEBUG] Thinking tokens: {thinking_tokens_count}, Regular tokens: {regular_tokens_count}")
            tool_calls = None

        elif mode.name == "tool_submission":
            # Tool submission mode: single tool call expected
            response, tool_calls, metrics = self.inference_service.generate_with_tools(
                messages=messages,
                tools=tools,
            )
            # Extract inter-token latencies from metrics if available
            if 'inter_token_latencies' in metrics:
                inter_token_latencies = metrics['inter_token_latencies']
            
            # Add thinking stats if not present
            if 'thinking_stats' not in metrics:
                metrics['thinking_stats'] = {
                    "thinking_tokens": 0,
                    "regular_tokens": metrics.get('tokens', 0),
                    "total_tokens": metrics.get('tokens', 0),
                    "thinking_ratio": 0.0
                }

        else:  # full_tool mode
            # Full tool mode: iterative tool usage
            all_responses = []
            all_tool_calls = []
            final_code = None
            iteration_count = 0
            submission_found = False
            submission_via_tool = False  # Track if submitted via tool call

            for iteration in range(mode.max_iterations):
                iteration_count += 1
                response, tool_calls, metrics = self.inference_service.generate_with_tools(
                    messages=messages,
                    tools=tools,
                )
                # Extract inter-token latencies from first response
                if iteration == 0 and 'inter_token_latencies' in metrics:
                    inter_token_latencies = metrics['inter_token_latencies']
                
                # Add thinking stats if not present
                if 'thinking_stats' not in metrics:
                    metrics['thinking_stats'] = {
                        "thinking_tokens": 0,
                        "regular_tokens": metrics.get('tokens', 0),
                        "total_tokens": metrics.get('tokens', 0),
                        "thinking_ratio": 0.0
                    }

                print(f"[DEBUG] Response: {response}")

                # Check if response contains Python blocks
                if "```python" in response:
                    # If we have Python blocks, this is NOT a proper tool call submission
                    submission_via_tool = False
                    print("⚠️  Model returned Python block instead of tool call format")

                all_responses.append(response)
                messages.append({"role": "assistant", "content": response})

                if tool_calls:
                    all_tool_calls.extend(tool_calls)

                    # First, execute ALL non-submission tools to ensure files are created
                    non_submission_tools = []
                    submission_tool = None

                    for call in tool_calls:
                        if call.get("name") == "submit_python_solution":
                            submission_tool = call
                            submission_found = True
                            # Only count as tool call if NOT from Python block
                            if "```python" not in response:
                                submission_via_tool = True
                                print(f"✓ Model submitted solution via tool call at iteration {iteration_count}")
                        else:
                            non_submission_tools.append(call)

                    # Execute all non-submission tools first
                    if non_submission_tools:
                        tool_results = await self.tool_executor.execute_tools(non_submission_tools)

                        # Add tool results to messages
                        tool_message = "Tool execution results:\n"
                        for call, result in zip(non_submission_tools, tool_results):
                            tool_message += f"\n{call['name']}:\n"
                            if result.get("success"):
                                tool_message += result.get("output", "Success")
                            else:
                                tool_message += f"Error: {result.get('error', 'Unknown error')}"

                        messages.append({"role": "user", "content": tool_message})

                    # Break if submission was found
                    if submission_found:
                        break
                else:
                    # No tool calls made in this iteration
                    # Check if we should prompt for submission
                    if iteration < mode.max_iterations - 1:  # Not the last iteration
                        print(f"⚠ No tool calls at iteration {iteration_count}, prompting for submission")
                        # Model-specific prompt
                        if self.inference_service.current_model and self.inference_service.current_model.architecture == "gemma":
                            submission_prompt = (
                                "No function calls found. Use ONE of these formats:\n"
                                "[submit_python_solution(code=\"your complete solution here\")]\n"
                                "OR {\"name\": \"submit_python_solution\", \"parameters\": {\"code\": \"solution\"}}"
                            )
                        else:
                            submission_prompt = (
                                "No tool calls parsed. Return tool calls in ```tool_call\n{...}``` blocks.\n"
                                "Example: ```tool_call\n{\"name\": \"submit_python_solution\", \"parameters\": {\"code\": \"def solution(): pass\"}}\n```"
                            )
                        messages.append({"role": "user", "content": submission_prompt})
                    else:
                        print(f"✗ Reached max iterations ({iteration_count}) without submission")
                        break

            response = "\n".join(all_responses)
            tool_calls = all_tool_calls

        # Extract code from response
        code = self._extract_code(response, tool_calls)

        # Run tests
        test_results = await self._run_tests(code, problem["test_list"])

        # Determine success
        success = all(tr.passed for tr in test_results)
        passed_tests = sum(1 for tr in test_results if tr.passed)
        total_tests = len(test_results)

        # Log final results for full_tool mode
        if mode.name == "full_tool" and 'iteration_count' in locals():
            print(f"Problem completed in {iteration_count} iterations (max: {mode.max_iterations}) - Tests: {passed_tests}/{total_tests}")

        end_time = datetime.now()

        # Calculate context length used
        context_length_used = len(prompt) if 'prompt' in locals() else len(str(messages))

        # Add iteration count to metrics for full_tool mode
        if mode.name == "full_tool" and 'iteration_count' in locals():
            metrics['iteration_count'] = iteration_count
            metrics['submission_found'] = submission_found if 'submission_found' in locals() else False
            metrics['submission_via_tool'] = submission_via_tool if 'submission_via_tool' in locals() else False

            # Log submission method
            if not submission_via_tool and submission_found:
                print("⚠️  Solution extracted from Python block, not tool call")

        # Cleanup sandbox after evaluation
        if hasattr(self.tool_executor, '_cleanup_sandbox'):
            self.tool_executor._cleanup_sandbox()
            
        return BenchmarkProblemResult(
            problem_id=problem["task_id"],
            start_time=start_time,
            end_time=end_time,
            response=response,
            tool_calls=tool_calls,
            test_results=test_results,
            success=success,
            metrics=metrics,
            model_load_time_ms=model_load_time_ms,
            cold_start=is_cold_start,
            context_length_used=context_length_used,
            inter_token_latencies=inter_token_latencies if inter_token_latencies else None,
        )

    def _extract_code(
        self,
        response: str,
        tool_calls: Optional[List[Dict[str, Any]]],
    ) -> str:
        """Extract code from response.

        Args:
            response: Model response
            tool_calls: Tool calls if any

        Returns:
            Extracted code
        """
        # First check tool calls
        if tool_calls:
            # Look for submit_python_solution calls first
            for call in tool_calls:
                if call.get("name") == "submit_python_solution":
                    params = call.get("parameters", {})
                    # Check for code parameter
                    if "code" in params:
                        return params["code"]
                    # Check for filename parameter
                    elif "filename" in params:
                        # Look for the corresponding upsert_file call
                        filename = params["filename"]
                        for tc in tool_calls:
                            if tc.get("name") == "upsert_file" and tc.get("parameters", {}).get("filename") == filename:
                                content = tc.get("parameters", {}).get("content", "")
                                # Clean up content - remove assert statements if they're included
                                lines = content.split('\n')
                                clean_lines = []
                                for line in lines:
                                    if not line.strip().startswith('assert '):
                                        clean_lines.append(line)
                                    else:
                                        break  # Stop at first assert
                                return '\n'.join(clean_lines).rstrip()
            
            # If no submit_python_solution, look for other code sources
            for call in tool_calls:
                if call.get("name") == "run_python_code":
                    return call.get("parameters", {}).get("code", "")
                elif call.get("name") == "upsert_file":
                    content = call.get("parameters", {}).get("content", "")
                    if content and 'def ' in content:
                        # Clean up content
                        lines = content.split('\n')
                        clean_lines = []
                        for line in lines:
                            if not line.strip().startswith('assert '):
                                clean_lines.append(line)
                            else:
                                break
                        return '\n'.join(clean_lines).rstrip()

        # Try to extract code blocks from response
        # Pattern 1: ```python ... ```
        code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0]

        # Pattern 2: ``` ... ```
        code_blocks = re.findall(r'```\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0]

        # Pattern 3: Indented code block
        lines = response.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            if line.startswith('def ') or line.startswith('class '):
                in_code = True

            if in_code:
                if line and not line[0].isspace() and not line.startswith('def') and not line.startswith('class'):
                    break
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines)

        # Fallback: return entire response
        return response.strip()

    async def _run_tests(
        self,
        code: str,
        test_cases: List[str],
    ) -> List[TestResult]:
        """Run test cases against code.

        Args:
            code: Code to test
            test_cases: List of test assertions

        Returns:
            List of test results
        """
        results = []
        
        if DEBUG_BENCHMARK:
            print(f"\n[DEBUG] Running {len(test_cases)} test cases")
            print(f"[DEBUG] Code to test (first 200 chars): {code[:200]}...")
        
        # Extract expected function name from test cases
        expected_function = None
        if test_cases:
            # Look for function calls in assert statements
            import re
            match = re.search(r'assert\s+(\w+)\(', test_cases[0])
            if match:
                expected_function = match.group(1)
                if DEBUG_BENCHMARK:
                    print(f"[DEBUG] Expected function name from tests: {expected_function}")
        
        # Check if we need to add a function alias
        if expected_function and expected_function not in code:
            # Look for function definitions in the code
            func_match = re.search(r'def\s+(\w+)\s*\([^)]*\):', code)
            if func_match:
                actual_function = func_match.group(1)
                if DEBUG_BENCHMARK:
                    print(f"[DEBUG] Found function: {actual_function}, adding alias for {expected_function}")
                # Add alias at the end of code
                code = code.rstrip() + f"\n\n# Alias for test compatibility\n{expected_function} = {actual_function}\n"

        for i, test_case in enumerate(test_cases, 1):
            # Combine code and test
            test_code = f"{code}\n\n{test_case}"
            
            if DEBUG_BENCHMARK:
                print(f"\n[DEBUG] Test case {i}: {test_case[:100]}...")

            # Execute test
            try:
                # Ensure sandbox exists before running code
                if not self.tool_executor.sandbox_dir:
                    self.tool_executor._create_sandbox()
                    
                output = await self.tool_executor._run_python_code(test_code)
                
                if DEBUG_BENCHMARK:
                    print(f"[DEBUG] Test output: {repr(output[:200])}...")

                # Check if test passed
                # Empty output or no errors means success for assertion tests
                output_lower = output.lower()
                has_error = any(err in output_lower for err in [
                    "assertionerror", "error:", "traceback", "exception",
                    "syntaxerror", "nameerror", "typeerror", "valueerror",
                    "indexerror", "keyerror", "attributeerror", "zerodivisionerror"
                ])
                
                # Also check for explicit failure indicators
                has_failure = any(fail in output_lower for fail in ["failed", "fail"])
                
                if has_error or has_failure:
                    results.append(TestResult(
                        test_case=test_case,
                        passed=False,
                        output=output,
                    ))
                else:
                    # No errors and no explicit failures = test passed
                    results.append(TestResult(
                        test_case=test_case,
                        passed=True,
                        output=output or "Test passed (no output)",
                    ))
                    
                if DEBUG_BENCHMARK:
                    print(f"[DEBUG] Test {i} result: {'PASSED' if results[-1].passed else 'FAILED'}")

            except Exception as e:
                if DEBUG_BENCHMARK:
                    print(f"[DEBUG] Test {i} exception: {str(e)}")
                results.append(TestResult(
                    test_case=test_case,
                    passed=False,
                    error=str(e),
                ))
        
        if DEBUG_BENCHMARK:
            passed_count = sum(1 for r in results if r.passed)
            print(f"\n[DEBUG] Test summary: {passed_count}/{len(results)} tests passed")

        return results

    def _calculate_aggregate_stats(
        self,
        session: BenchmarkSession,
    ) -> Dict[str, Any]:
        """Calculate comprehensive aggregate statistics for a session.

        Args:
            session: Benchmark session

        Returns:
            Aggregate statistics
        """
        if not session.problems:
            return {}

        import statistics as stat

        # Calculate pass rate
        passed = sum(1 for p in session.problems if p.success)
        total = len(session.problems)

        # Calculate timing metrics
        ttft_values = [p.metrics.get("ttft") for p in session.problems if p.metrics and p.metrics.get("ttft")]
        tps_values = [p.metrics.get("tps") for p in session.problems if p.metrics and p.metrics.get("tps")]

        stats = {
            "total_problems": total,
            "passed_problems": passed,
            "pass_rate": passed / total if total > 0 else 0,
            "total_duration_seconds": (session.end_time - session.start_time).total_seconds(),
        }

        # Performance metrics
        if ttft_values:
            stats["ttft"] = {
                "avg_ms": stat.mean(ttft_values),
                "min_ms": min(ttft_values),
                "max_ms": max(ttft_values),
                "stddev_ms": stat.stdev(ttft_values) if len(ttft_values) > 1 else 0,
            }

        if tps_values:
            stats["tps"] = {
                "avg": stat.mean(tps_values),
                "min": min(tps_values),
                "max": max(tps_values),
                "stddev": stat.stdev(tps_values) if len(tps_values) > 1 else 0,
            }

        # Inter-token latencies
        all_latencies = []
        for p in session.problems:
            if p.inter_token_latencies:
                all_latencies.extend(p.inter_token_latencies)

        if all_latencies:
            stats["inter_token_latency"] = {
                "avg_ms": stat.mean(all_latencies),
                "min_ms": min(all_latencies),
                "max_ms": max(all_latencies),
                "p50_ms": stat.median(all_latencies),
                "p95_ms": stat.quantiles(all_latencies, n=20)[18] if len(all_latencies) >= 20 else max(all_latencies),
                "p99_ms": stat.quantiles(all_latencies, n=100)[98] if len(all_latencies) >= 100 else max(all_latencies),
                "jitter_ms": stat.stdev(all_latencies) if len(all_latencies) > 1 else 0,
            }

        # Cold vs warm start analysis
        cold_start_problems = [p for p in session.problems if p.cold_start]
        warm_start_problems = [p for p in session.problems if not p.cold_start]

        if cold_start_problems:
            cold_durations = [(p.end_time - p.start_time).total_seconds() for p in cold_start_problems]
            stats["cold_start"] = {
                "count": len(cold_start_problems),
                "avg_duration_s": stat.mean(cold_durations),
                "model_load_times_ms": [p.model_load_time_ms for p in cold_start_problems if p.model_load_time_ms],
            }

        if warm_start_problems:
            warm_durations = [(p.end_time - p.start_time).total_seconds() for p in warm_start_problems]
            stats["warm_start"] = {
                "count": len(warm_start_problems),
                "avg_duration_s": stat.mean(warm_durations),
            }

        # Context length impact
        context_lengths = [p.context_length_used for p in session.problems if p.context_length_used > 0]
        if context_lengths:
            stats["context_length"] = {
                "avg": stat.mean(context_lengths),
                "max": max(context_lengths),
                "performance_correlation": self._calculate_context_performance_correlation(session.problems),
            }

        # Token generation metrics
        total_tokens = sum(p.metrics.get("tokens", 0) for p in session.problems if p.metrics)
        stats["total_tokens_generated"] = total_tokens

        # Full tool mode iteration statistics
        iteration_counts = [p.metrics.get("iteration_count", 0) for p in session.problems
                           if p.metrics and "iteration_count" in p.metrics]
        if iteration_counts:
            stats["full_tool_iterations"] = {
                "avg": stat.mean(iteration_counts),
                "min": min(iteration_counts),
                "max": max(iteration_counts),
                "submission_rate": sum(1 for p in session.problems
                                     if p.metrics and p.metrics.get("submission_found", False)) / len(session.problems)
            }

        # Add system metrics summary
        if session.system_metrics:
            system_summary = self.system_monitor.get_summary()
            stats["system_metrics"] = system_summary

        return stats

    def _calculate_context_performance_correlation(self, problems: List[BenchmarkProblemResult]) -> float:
        """Calculate correlation between context length and performance."""
        try:
            contexts = []
            tps_values = []

            for p in problems:
                if p.context_length_used > 0 and p.metrics and p.metrics.get("tps"):
                    contexts.append(p.context_length_used)
                    tps_values.append(p.metrics["tps"])

            if len(contexts) < 2:
                return 0.0

            # Calculate Pearson correlation coefficient
            import statistics as stat
            mean_x = stat.mean(contexts)
            mean_y = stat.mean(tps_values)

            numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(contexts, tps_values))
            denominator = (sum((x - mean_x)**2 for x in contexts) * sum((y - mean_y)**2 for y in tps_values))**0.5

            if denominator == 0:
                return 0.0

            return numerator / denominator
        except:
            return 0.0

    def list_problems(self) -> List[Dict[str, Any]]:
        """List all available problems.

        Returns:
            List of problem summaries
        """
        return [
            {
                "task_id": p["task_id"],
                "text": p["text"][:100] + "..." if len(p["text"]) > 100 else p["text"],
            }
            for p in self.problems
        ]

    def _calculate_pass_at_k(self, n: int, c: int, k: int) -> float:
        """Calculate pass@k metric.

        Args:
            n: total number of samples
            c: number of correct samples
            k: k in pass@k

        Returns:
            pass@k score
        """
        if n - c < k:
            return 1.0

        import math
        return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))

    async def run_benchmark_with_config(
        self,
        config: BenchmarkConfig,
        progress_callback: Optional[callable] = None,
    ) -> BenchmarkSession:
        """Run benchmark with pass@k support using configuration.

        Args:
            config: Benchmark configuration
            progress_callback: Callback for progress updates

        Returns:
            Benchmark session with pass@k results
        """
        if config.mode not in BENCHMARK_MODES:
            raise ValueError(f"Invalid mode: {config.mode}")

        benchmark_mode = BENCHMARK_MODES[config.mode]

        # Create session
        session = BenchmarkSession(
            session_id=f"bench_{config.model_name}_{config.mode}_{int(datetime.now().timestamp())}",
            model_id=self.inference_service.current_model.id,
            mode=config.mode,
            start_time=datetime.now(),
            problems=[],
            config=config,
            pass_at_k={},
        )

        # Start monitoring if enabled
        if config.system_monitoring:
            self.system_monitor.start_monitoring()

        try:
            # Select problems
            if config.problem_ids:
                problems = [p for p in self.problems if p["task_id"] in config.problem_ids]
            elif config.problems_limit:
                problems = self.problems[:config.problems_limit]
            else:
                problems = self.problems

            # Run each problem multiple times
            for i, problem in enumerate(problems):
                problem_results = []

                # Run num_samples times for pass@k
                actual_runs = 0
                for run_id in range(config.num_samples):
                    if progress_callback:
                        progress_callback(
                            f"Problem {problem['task_id']} - Run {run_id + 1}/{config.num_samples}"
                        )

                    # Run with temperature sampling
                    result = await self._evaluate_problem_with_temperature(
                        problem, benchmark_mode, config.temperature, run_id
                    )
                    problem_results.append(result)
                    session.problems.append(result)
                    actual_runs += 1
                    
                    # Early stopping: if problem passes and we're not in exhaustive mode, skip remaining samples
                    if result.success and not config.exhaustive_passes and run_id < config.num_samples - 1:
                        if DEBUG_BENCHMARK:
                            print(f"✓ Problem {problem['task_id']} passed on run {run_id + 1}, skipping remaining {config.num_samples - run_id - 1} runs")
                        break

                # Calculate pass@k for this problem
                successful_runs = sum(1 for r in problem_results if r.success)
                
                # When early stopping is enabled, use actual runs for Pass@k calculation
                if actual_runs < config.num_samples and successful_runs > 0 and not config.exhaustive_passes:
                    # Early stopped after success - use actual runs for Pass@k
                    # This gives accurate metrics based on what we actually tested
                    n_for_passk = actual_runs
                else:
                    # Either exhaustive mode or all samples were run
                    n_for_passk = config.num_samples
                
                session.pass_at_k[problem["task_id"]] = {
                    "total_runs": config.num_samples,
                    "actual_runs": actual_runs,
                    "successful_runs": successful_runs,
                    "early_stopped": actual_runs < config.num_samples,
                    "pass_at_1": self._calculate_pass_at_k(n_for_passk, successful_runs, 1),
                    "pass_at_3": self._calculate_pass_at_k(n_for_passk, successful_runs, 3) if n_for_passk >= 3 else None,
                    "pass_at_5": self._calculate_pass_at_k(n_for_passk, successful_runs, 5) if n_for_passk >= 5 else None,
                    "pass_at_10": self._calculate_pass_at_k(n_for_passk, successful_runs, 10) if n_for_passk >= 10 else None,
                }

            # Finalize session
            session.end_time = datetime.now()

            if config.system_monitoring:
                session.system_metrics = self.system_monitor.export_metrics()
                self.system_monitor.stop_monitoring()

            session.aggregate_stats = self._calculate_aggregate_stats_with_pass_k(session)

        finally:
            if config.system_monitoring:
                self.system_monitor.stop_monitoring()

        return session

    async def _evaluate_problem_with_temperature(
        self,
        problem: Dict[str, Any],
        mode: BenchmarkMode,
        temperature: float,
        run_id: int = 0,
    ) -> BenchmarkProblemResult:
        """Evaluate a problem with specified temperature.

        This is a wrapper around _evaluate_problem that adds temperature control.
        """
        # Temporarily set the temperature
        original_temp = getattr(self.inference_service, 'temperature', 0.7)
        self.inference_service.temperature = temperature

        try:
            result = await self._evaluate_problem(problem, mode)
            result.run_id = run_id
            result.temperature = temperature
            return result
        finally:
            # Restore original temperature
            self.inference_service.temperature = original_temp

    def _calculate_aggregate_stats_with_pass_k(
        self,
        session: BenchmarkSession,
    ) -> Dict[str, Any]:
        """Calculate aggregate stats including pass@k metrics."""
        # Get base stats
        stats = self._calculate_aggregate_stats(session)

        # Add pass@k aggregates
        if session.pass_at_k:
            # Calculate averages only for problems with enough samples
            pass_at_1_values = [p["pass_at_1"] for p in session.pass_at_k.values()]
            pass_at_3_values = [p["pass_at_3"] for p in session.pass_at_k.values() if p["pass_at_3"] is not None]
            pass_at_5_values = [p["pass_at_5"] for p in session.pass_at_k.values() if p["pass_at_5"] is not None]
            pass_at_10_values = [p["pass_at_10"] for p in session.pass_at_k.values() if p["pass_at_10"] is not None]

            stats["pass_at_k"] = {
                "overall_pass_at_1": sum(pass_at_1_values) / len(pass_at_1_values) if pass_at_1_values else 0.0,
                "overall_pass_at_3": sum(pass_at_3_values) / len(pass_at_3_values) if pass_at_3_values else None,
                "overall_pass_at_5": sum(pass_at_5_values) / len(pass_at_5_values) if pass_at_5_values else None,
                "overall_pass_at_10": sum(pass_at_10_values) / len(pass_at_10_values) if pass_at_10_values else None,
            }

        # Temperature metrics
        if session.problems:
            temps = [p.temperature for p in session.problems]
            stats["temperature"] = {
                "mean": sum(temps) / len(temps),
                "values": list(set(temps)),
            }

        return stats
