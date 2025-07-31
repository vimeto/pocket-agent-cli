"""Benchmark evaluation service."""

import json
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import time
from ..config import DATA_DIR, BENCHMARK_MODES, AVAILABLE_TOOLS, SUBMIT_TOOL, BenchmarkMode
from ..services import InferenceService
from ..tools import ToolExecutor
# from ..monitoring import SystemMonitor
from ..monitoring.simple_monitor import SimpleMonitor


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "session_id": self.session_id,
            "model_id": self.model_id,
            "mode": self.mode,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "problems": [p.to_dict() for p in (self.problems or [])],
            "system_metrics": self.system_metrics,
            "aggregate_stats": self.aggregate_stats,
        }
        return data


class BenchmarkService:
    """Service for running and evaluating benchmarks."""
    
    def __init__(self, inference_service: InferenceService):
        self.inference_service = inference_service
        self.tool_executor = ToolExecutor()
        self.system_monitor = SimpleMonitor()  # Using simple monitor for now
        self.dataset_path = DATA_DIR / "mbpp_sample.json"
        self.problems = self._load_dataset()
    
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load MBPP dataset."""
        if not self.dataset_path.exists():
            # Use the sample dataset we created
            sample_path = Path(__file__).parent.parent / "data" / "mbpp_sample.json"
            if sample_path.exists():
                with open(sample_path, "r") as f:
                    return json.load(f)
            else:
                return []
        
        with open(self.dataset_path, "r") as f:
            return json.load(f)
    
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
        self.system_monitor.start_monitoring()
        
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
        
        # Prepare messages
        messages = [
            {"role": "system", "content": mode.system_prompt},
            {"role": "user", "content": mode.user_prompt_template.format(
                problem_description=problem["text"]
            )},
        ]
        
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
        
        # Generate response based on mode
        if mode.name == "base":
            # Base mode: simple generation
            response = ""
            metrics = {}
            token_count = 0
            
            for chunk in self.inference_service.generate(messages, stream=True):
                current_time = time.time()
                if last_token_time is not None:
                    inter_token_latencies.append((current_time - last_token_time) * 1000)
                last_token_time = current_time
                
                response += chunk["token"]
                metrics = chunk["metrics"]
                token_count += 1
            
            tool_calls = None
            
        elif mode.name == "tool_submission":
            # Tool submission mode: single tool call expected
            response, tool_calls, metrics = self.inference_service.generate_with_tools(
                messages=messages,
                tools=tools,
            )
            
        else:  # full_tool mode
            # Full tool mode: iterative tool usage
            all_responses = []
            all_tool_calls = []
            final_code = None
            
            for iteration in range(mode.max_iterations):
                response, tool_calls, metrics = self.inference_service.generate_with_tools(
                    messages=messages,
                    tools=tools,
                )
                
                all_responses.append(response)
                messages.append({"role": "assistant", "content": response})
                
                if tool_calls:
                    all_tool_calls.extend(tool_calls)
                    
                    # Check for submit_python_solution
                    for call in tool_calls:
                        if call.get("name") == "submit_python_solution":
                            final_code = call.get("parameters", {}).get("code", "")
                            break
                    
                    if final_code:
                        break
                    
                    # Execute tools
                    tool_results = await self.tool_executor.execute_tools(tool_calls)
                    
                    # Add tool results to messages
                    tool_message = "Tool execution results:\n"
                    for call, result in zip(tool_calls, tool_results):
                        tool_message += f"\n{call['name']}:\n"
                        if result.get("success"):
                            tool_message += result.get("output", "Success")
                        else:
                            tool_message += f"Error: {result.get('error', 'Unknown error')}"
                    
                    messages.append({"role": "user", "content": tool_message})
                else:
                    # No tool calls, stop iteration
                    break
            
            response = "\n".join(all_responses)
            tool_calls = all_tool_calls
        
        # Extract code from response
        code = self._extract_code(response, tool_calls)
        
        # Run tests
        test_results = await self._run_tests(code, problem["test_list"])
        
        # Determine success
        success = all(tr.passed for tr in test_results)
        
        end_time = datetime.now()
        
        # Calculate context length used
        context_length_used = len(prompt) if 'prompt' in locals() else len(str(messages))
        
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
            for call in tool_calls:
                if call.get("name") == "submit_python_solution":
                    params = call.get("parameters", {})
                    # Check for code parameter
                    if "code" in params:
                        return params["code"]
                    # Check for filename parameter
                    elif "filename" in params:
                        # In full_tool mode, the file should have been created
                        # Look for the last upsert_file call with that filename
                        filename = params["filename"]
                        for tc in reversed(tool_calls):
                            if tc.get("name") == "upsert_file" and tc.get("parameters", {}).get("filename") == filename:
                                return tc.get("parameters", {}).get("content", "")
                elif call.get("name") == "run_python_code":
                    return call.get("parameters", {}).get("code", "")
                elif call.get("name") == "upsert_file":
                    content = call.get("parameters", {}).get("content", "")
                    if content:
                        return content
        
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
        
        for test_case in test_cases:
            # Combine code and test
            test_code = f"{code}\n\n{test_case}"
            
            # Execute test
            try:
                output = await self.tool_executor._run_python_code(test_code)
                
                # Check if test passed (no output usually means success)
                if "AssertionError" in output or "Error" in output:
                    results.append(TestResult(
                        test_case=test_case,
                        passed=False,
                        output=output,
                    ))
                else:
                    results.append(TestResult(
                        test_case=test_case,
                        passed=True,
                        output=output,
                    ))
                    
            except Exception as e:
                results.append(TestResult(
                    test_case=test_case,
                    passed=False,
                    error=str(e),
                ))
        
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