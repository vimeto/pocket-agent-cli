"""Benchmark coordinator for running multiple model/mode combinations."""

import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime
import os

from ..config import BenchmarkConfig, InferenceConfig, DEFAULT_MODELS, BENCHMARK_MODES, RESULTS_DIR
from ..models import ModelService
from ..services import InferenceService
from .benchmark_service import BenchmarkService, BenchmarkSession


class BenchmarkCoordinator:
    """Coordinates benchmark runs across multiple models and modes."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model_service = ModelService()
        self.results = []

    async def run_all_benchmarks(
        self,
        progress_callback: Optional[callable] = None
    ) -> List[BenchmarkSession]:
        """Run benchmarks for all configured model/mode combinations."""
        # Determine models to run
        models = self._get_models_to_run()
        modes = self._get_modes_to_run()

        if not models:
            raise ValueError("No models available for benchmarking")

        total_combinations = len(models) * len(modes)
        current = 0

        all_sessions = []

        for model_id in models:
            for mode in modes:
                # prune docker containers with label pocket_agent_cli_sandbox
                os.system("docker ps -a --filter 'label=pocket_agent_cli_sandbox' -q | xargs -r docker rm -f")

                current += 1
                if progress_callback:
                    progress_callback(
                        f"Running {model_id} - {mode} ({current}/{total_combinations})"
                    )

                try:
                    session = await self._run_single_benchmark(model_id, mode, progress_callback)
                    all_sessions.append(session)

                    # Save session immediately
                    self._save_session(session)

                except Exception as e:
                    import traceback
                    print(f"Error running {model_id} - {mode}: {e}")
                    traceback.print_exc()
                    continue

        # Save summary
        self._save_summary(all_sessions)

        return all_sessions

    def _get_models_to_run(self) -> List[str]:
        """Get list of models to benchmark."""
        if self.config.model_name == "all":
            # Get all downloaded models
            models = []
            for model_config in DEFAULT_MODELS:
                model = self.model_service.get_model(model_config["id"])
                if model and model.downloaded:
                    models.append(model.id)
            return models
        else:
            # Single model
            model = self.model_service.get_model(self.config.model_name)
            if not model or not model.downloaded:
                raise ValueError(f"Model {self.config.model_name} not found or not downloaded")
            return [self.config.model_name]

    def _get_modes_to_run(self) -> List[str]:
        """Get list of modes to benchmark."""
        if self.config.mode == "all":
            return list(BENCHMARK_MODES.keys())
        else:
            if self.config.mode not in BENCHMARK_MODES:
                raise ValueError(f"Invalid mode: {self.config.mode}")
            return [self.config.mode]

    async def _run_single_benchmark(
        self,
        model_id: str,
        mode: str,
        progress_callback: Optional[callable] = None
    ) -> BenchmarkSession:
        """Run a single model/mode benchmark."""
        # Load model
        model = self.model_service.get_model(model_id)
        inference_service = InferenceService()

        # Create config for this specific run
        run_config = BenchmarkConfig(
            model_name=model_id,
            mode=mode,
            problem_ids=self.config.problem_ids,
            problems_limit=self.config.problems_limit,
            num_samples=self.config.num_samples,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            enable_tools=self.config.enable_tools,
            system_monitoring=self.config.system_monitoring,
            output_dir=self.config.output_dir,
            save_individual_runs=self.config.save_individual_runs,
            compute_pass_at_k=self.config.compute_pass_at_k,
            parallel_runs=self.config.parallel_runs,
        )

        # Load model with InferenceConfig
        inference_config = InferenceConfig(
            temperature=run_config.temperature,
            max_tokens=run_config.max_tokens,
        )
        inference_service.load_model(model, inference_config)

        # Create benchmark service
        benchmark_service = BenchmarkService(inference_service, run_config)

        # Run benchmark
        try:
            session = await benchmark_service.run_benchmark_with_config(
                run_config, progress_callback
            )
            return session
        finally:
            # Cleanup
            inference_service.unload_model()

    def _save_session(self, session: BenchmarkSession):
        """Save individual session results."""
        output_dir = self.config.output_dir / session.model_id / session.mode
        output_dir.mkdir(parents=True, exist_ok=True)

        # Main session file
        session_file = output_dir / f"{session.session_id}.json"
        with open(session_file, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

        # Individual runs if configured
        if self.config.save_individual_runs and session.problems:
            runs_dir = output_dir / "runs" / session.session_id
            runs_dir.mkdir(parents=True, exist_ok=True)

            for result in session.problems:
                run_file = runs_dir / f"problem_{result.problem_id}_run_{result.run_id}.json"
                with open(run_file, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)

    def _save_summary(self, sessions: List[BenchmarkSession]):
        """Save summary of all sessions."""
        # Convert config to dict and handle Path objects
        config_dict = self.config.model_dump()
        # Convert Path objects to strings
        if 'output_dir' in config_dict and hasattr(config_dict['output_dir'], '__fspath__'):
            config_dict['output_dir'] = str(config_dict['output_dir'])

        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": config_dict,
            "sessions": [
                {
                    "model_id": s.model_id,
                    "mode": s.mode,
                    "session_id": s.session_id,
                    "aggregate_stats": s.aggregate_stats,
                }
                for s in sessions
            ],
        }

        summary_file = self.config.output_dir / "benchmark_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
