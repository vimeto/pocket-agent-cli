"""Command-line interface for pocket-agent-cli."""

import os
import click
import asyncio
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn
from rich.prompt import Prompt, Confirm
from .config import InferenceConfig, BENCHMARK_MODES, AVAILABLE_TOOLS
from .models import ModelService
from .services import InferenceService
from .benchmarks import BenchmarkService
from .monitoring import SystemMonitor
from .utils.result_export import export_results
from .tools import ToolExecutor


console = Console()


@click.group()
def cli():
    """Pocket Agent CLI - Local LLM inference and benchmarking tool."""
    pass


@cli.group()
def model():
    """Model management commands."""
    pass


@model.command("list")
def list_models():
    """List all available models."""
    service = ModelService()
    models = service.list_models()
    
    table = Table(title="Available Models")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Versions", style="yellow")
    table.add_column("Downloaded", style="green")
    
    for model in models:
        # Build versions string showing which are downloaded
        versions_info = []
        for v_name, v_data in model.versions.items():
            size_gb = v_data.size / (1024*1024*1024)
            if v_data.downloaded:
                versions_info.append(f"[green]✓ {v_name}[/green] ({size_gb:.1f}GB)")
            else:
                versions_info.append(f"[dim]✗ {v_name}[/dim] ({size_gb:.1f}GB)")
        
        versions_str = "\n".join(versions_info)
        
        # Overall downloaded status (any version)
        any_downloaded = any(v.downloaded for v in model.versions.values())
        downloaded_str = "[green]Yes[/green]" if any_downloaded else "[red]No[/red]"
        
        table.add_row(
            model.id,
            model.name,
            versions_str,
            downloaded_str
        )
    
    console.print(table)


@model.command("download")
@click.argument("model_id")
@click.option("--version", "-v", help="Model version to download (Q4_K_M, F16, BF16, etc)")
@click.option("--token", envvar="HF_TOKEN", help="Hugging Face token for gated models")
def download_model(model_id: str, version: Optional[str], token: Optional[str]):
    """Download a model from Hugging Face."""
    service = ModelService()
    
    async def download():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            version_str = f" ({version})" if version else ""
            task = progress.add_task(f"Downloading {model_id}{version_str}...", total=100)
            
            def update_progress(downloaded, total):
                if total > 0:
                    progress.update(task, completed=downloaded/total * 100)
            
            try:
                model = await service.download_model(
                    model_id,
                    version=version,
                    hf_token=token,
                    progress_callback=update_progress
                )
                console.print(f"[green]✓[/green] Model {model.name}{version_str} downloaded successfully!")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to download model: {e}")
    
    asyncio.run(download())


@model.command("delete")
@click.argument("model_id")
def delete_model(model_id: str):
    """Delete a downloaded model."""
    service = ModelService()
    
    if Confirm.ask(f"Are you sure you want to delete {model_id}?"):
        if service.delete_model(model_id):
            console.print(f"[green]✓[/green] Model {model_id} deleted.")
        else:
            console.print(f"[red]✗[/red] Model {model_id} not found or not downloaded.")


@model.command("refresh")
def refresh_models():
    """Refresh model metadata from config defaults.
    
    Updates the models.json file with any changes from DEFAULT_MODELS in config.py,
    while preserving download status and paths for existing versions.
    """
    service = ModelService()
    
    # The service already does this on init, but we'll make it explicit
    service.refresh_from_defaults()
    
    console.print("[green]✓[/green] Model metadata refreshed from defaults")
    console.print("This command syncs any changes from config.py to your models.json file.")


@cli.command("chat")
@click.option("--model", "-m", required=True, help="Model ID to use")
@click.option("--tools/--no-tools", default=False, help="Enable tool usage")
@click.option("--temperature", "-t", default=0.7, help="Temperature for generation")
@click.option("--max-tokens", default=2048, help="Maximum tokens to generate")
@click.option("--message", "-M", help="Single message for non-interactive mode")
def chat(model: str, tools: bool, temperature: float, max_tokens: int, message: Optional[str]):
    """Interactive chat mode."""
    # Initialize services
    model_service = ModelService()
    inference_service = InferenceService()
    tool_executor = ToolExecutor(use_docker=True) if tools else None
    
    # Load model
    model_obj = model_service.get_model(model)
    if not model_obj or not model_obj.downloaded:
        console.print(f"[red]Error:[/red] Model {model} not found or not downloaded.")
        console.print("Use 'pocket-agent model list' to see available models.")
        return
    
    config = InferenceConfig(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    try:
        console.print(f"Loading model {model_obj.name}...")
        inference_service.load_model(model_obj, config)
        console.print("[green]Model loaded successfully![/green]")
    except Exception as e:
        console.print(f"[red]Failed to load model:[/red] {e}")
        return
    
    # Chat loop
    messages = []
    
    # Add system message for tools mode
    if tools:
        # Different system messages for different architectures
        if model_obj.architecture == "qwen":
            system_message = """You are a helpful assistant with tools. When asked to write code or files, use these tools:
- upsert_file: Create/update files
- run_python_file: Run a Python file
- run_python_code: Execute Python code directly

Respond with JSON tool calls like: {"name": "upsert_file", "parameters": {"filename": "test.py", "content": "print('hello')"}}"""
        else:
            system_message = """You are a helpful assistant with access to tools for file management and code execution.

When you need to:
- Create or modify a file: use the upsert_file tool
- List files: use the list_files tool
- Run Python code: use the run_python_file or run_python_code tool
- Read a file: use the read_file tool
- Delete a file: use the delete_file tool

Always use tools when appropriate instead of just showing code examples.
When using run_python_code, make sure your code is complete and will produce output."""
        messages.append({"role": "system", "content": system_message})
    
    # Handle single message mode
    if message:
        messages.append({"role": "user", "content": message})
    else:
        console.print("\n[bold]Chat mode[/bold] (type 'exit' to quit, 'clear' to reset)")
        console.print("─" * console.width)
    
    while True:
        # Get user input
        if not message:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
            
            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "clear":
                messages = []
                console.print("[dim]Conversation cleared[/dim]")
                continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
        
        # Generate response
        console.print("\n[bold green]Assistant[/bold green]", end="")
        
        if tools:
            # With tools - agentic loop
            max_iterations = 5  # Maximum tool iterations
            iteration = 0
            
            while iteration < max_iterations:
                # Stream response with tool support
                response = ""
                if iteration == 0:
                    console.print(": ", end="")
                else:
                    console.print(f"\n[bold green]Assistant[/bold green]: ", end="")
                
                # Stream and collect response
                metrics = {}
                for chunk in inference_service.generate(messages, stream=True, tools=AVAILABLE_TOOLS):
                    token = chunk["token"]
                    response += token
                    console.print(token, end="")
                    metrics = chunk["metrics"]
                
                console.print()  # New line after response
                
                # Parse tool calls from the complete response
                tool_calls = inference_service._parse_tool_calls(response)
                
                # Debug: Show what we tried to parse if no tools found
                if not tool_calls and response.strip():
                    console.print(f"\n[dim yellow]No tool calls detected in response.[/dim yellow]")
                
                # Add assistant message
                messages.append({"role": "assistant", "content": response})
                
                # If no tool calls, we're done
                if not tool_calls:
                    break
                
                # Execute tools
                console.print("\n[dim]Executing tools:[/dim]")
                tool_results = asyncio.run(tool_executor.execute_tools(tool_calls))
                
                # Display tool results
                for call, result in zip(tool_calls, tool_results):
                    console.print(f"\n[bold cyan]Tool: {call['name']}[/bold cyan]")
                    params_str = json.dumps(call.get('parameters', {}), indent=2)
                    console.print(f"[dim]Parameters:[/dim]\n{params_str}")
                    
                    if result.get("success"):
                        output = result.get("output", "")
                        # Truncate very long outputs
                        if len(output) > 500:
                            output = output[:500] + "\n... (truncated)"
                        console.print(f"[dim]Result:[/dim]\n{output}")
                    else:
                        console.print(f"[red]Error:[/red] {result.get('error', 'Unknown error')}")
                
                # Add tool results to messages for next iteration
                tool_message = "Tool execution results:\n"
                for call, result in zip(tool_calls, tool_results):
                    tool_message += f"\n{call['name']}:\n"
                    if result.get("success"):
                        tool_message += result.get("output", "No output")
                    else:
                        tool_message += f"Error: {result.get('error', 'Unknown error')}"
                
                messages.append({"role": "user", "content": tool_message})
                iteration += 1
            
            # Don't double-add the response, it's already added in the loop
            response = messages[-2]["content"] if messages[-2]["role"] == "assistant" else response
        else:
            # Without tools
            response = ""
            console.print(": ", end="")
            
            # Stream response
            metrics = {}
            for chunk in inference_service.generate(messages, stream=True):
                token = chunk["token"]
                response += token
                console.print(token, end="")
                metrics = chunk["metrics"]
            
            console.print()  # New line after response
            
            # Show metrics
            if metrics.get("ttft"):
                console.print(f"\n[dim]TTFT: {metrics['ttft']:.0f}ms, TPS: {metrics['tps']:.1f}[/dim]")
            
            # Add assistant message (only for non-tools mode, tools mode adds it in the loop)
            messages.append({"role": "assistant", "content": response})
        
        # Exit if single message mode
        if message:
            break
    
    # Cleanup
    inference_service.unload_model()
    if tool_executor:
        tool_executor._cleanup_sandbox()
    if not message:
        console.print("\n[dim]Goodbye![/dim]")


@cli.command("benchmark")
@click.option("--model", "-m", required=True, help="Model ID to use or 'all' for all models")
@click.option("--model-version", "-v", help="Model version (Q4_K_M, F16, BF16, etc)")
@click.option("--mode", default="base", help="Benchmark mode or 'all' for all modes")
@click.option("--problems", "-p", help="Comma-separated problem IDs (e.g., 1,2,3)")
@click.option("--problems-limit", "-l", type=int, help="Number of problems to run")
@click.option("--num-samples", "-n", type=int, default=10, help="Number of samples per problem for pass@k")
@click.option("--temperature", "-t", type=float, default=0.7, help="Temperature for sampling")
@click.option("--output-dir", "-o", help="Output directory for results")
@click.option("--no-monitoring", is_flag=True, help="Disable system monitoring")
@click.option("--parallel", type=int, default=1, help="Number of parallel runs")
@click.option("--line-profiler", is_flag=True, help="Enable line profiling")
@click.option("--exhaustive-passes", is_flag=True, help="Run all samples even if problem already passes")
def benchmark(
    model: str, 
    model_version: Optional[str],
    mode: str, 
    problems: Optional[str], 
    problems_limit: Optional[int],
    num_samples: int,
    temperature: float,
    output_dir: Optional[str],
    no_monitoring: bool,
    parallel: int,
    line_profiler: bool,
    exhaustive_passes: bool
):
    """Run enhanced benchmark evaluation with pass@k support."""
    from .benchmarks.benchmark_coordinator import BenchmarkCoordinator
    from .config import BenchmarkConfig, RESULTS_DIR
    
    # Check if we need to restart with profiler
    if line_profiler and '__wrapped_by_profiler__' not in os.environ:
        # Re-run the command with kernprof
        import sys
        from .utils.profiling import run_with_profiler
        
        # Mark that we're wrapped to avoid infinite recursion
        os.environ['__wrapped_by_profiler__'] = '1'
        
        # Build args without --line-profiler flag
        args = ['benchmark']
        args.extend(['--model', model])
        args.extend(['--mode', mode])
        if problems:
            args.extend(['--problems', problems])
        if problems_limit:
            args.extend(['--problems-limit', str(problems_limit)])
        args.extend(['--num-samples', str(num_samples)])
        args.extend(['--temperature', str(temperature)])
        if output_dir:
            args.extend(['--output-dir', output_dir])
        if no_monitoring:
            args.append('--no-monitoring')
        args.extend(['--parallel', str(parallel)])
        
        # Run with profiler
        sys.exit(run_with_profiler(args))
    
    # Parse problem IDs
    problem_ids = None
    if problems:
        try:
            problem_ids = [int(p.strip()) for p in problems.split(",")]
        except ValueError:
            console.print("[red]Error:[/red] Invalid problem IDs format")
            return
    
    # Create benchmark configuration
    output_path = Path(output_dir) if output_dir else RESULTS_DIR / "benchmarks"
    
    benchmark_config = BenchmarkConfig(
        model_name=model,
        model_version=model_version,
        mode=mode,
        problem_ids=problem_ids,
        problems_limit=problems_limit,
        num_samples=num_samples,
        temperature=temperature,
        enable_tools=True,
        system_monitoring=not no_monitoring,
        output_dir=output_path,
        save_individual_runs=True,
        compute_pass_at_k=[1, 3, 5, 10],
        parallel_runs=parallel,
        exhaustive_passes=exhaustive_passes
    )
    
    # Validate models if not "all"
    if model != "all":
        model_service = ModelService()
        model_obj = model_service.get_model(model, version=model_version)
        if not model_obj:
            console.print(f"[red]Error:[/red] Model {model} not found.")
            return
        if model_version and not model_obj.is_downloaded(model_version):
            console.print(f"[red]Error:[/red] Model {model} version {model_version} not downloaded.")
            return
        elif not model_version and not model_obj.is_downloaded():
            console.print(f"[red]Error:[/red] Model {model} not downloaded.")
            return
    
    # Validate mode if not "all"
    if mode != "all" and mode not in BENCHMARK_MODES:
        console.print(f"[red]Error:[/red] Invalid mode: {mode}")
        console.print(f"Available modes: {', '.join(BENCHMARK_MODES.keys())}")
        return
    
    # Create benchmark coordinator
    coordinator = BenchmarkCoordinator(benchmark_config)
    
    async def run():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            overall_task = progress.add_task("Running benchmarks...", total=None)
            
            def update_progress(desc):
                progress.update(overall_task, description=desc)
            
            sessions = await coordinator.run_all_benchmarks(
                progress_callback=update_progress
            )
            
            progress.update(overall_task, description="Benchmarks complete!")
        
        return sessions
    
    # Display configuration
    console.print(f"\n[bold]Benchmark Configuration[/bold]")
    console.print("─" * console.width)
    console.print(f"  Models: {model}")
    console.print(f"  Modes: {mode}")
    console.print(f"  Samples per problem: {num_samples}")
    console.print(f"  Temperature: {temperature}")
    console.print(f"  Output directory: {output_path}")
    console.print(f"  System monitoring: {'Enabled' if not no_monitoring else 'Disabled'}")
    
    # Run benchmarks
    sessions = asyncio.run(run())
    
    # Display summary results
    console.print(f"\n[bold]Summary Results[/bold]")
    console.print("─" * console.width)
    
    for session in sessions:
        console.print(f"\n[bold]{session.model_id} - {session.mode}[/bold]")
        if session.aggregate_stats:
            stats = session.aggregate_stats
            console.print(f"  Problems: {stats['total_problems']}")
            
            # Display pass@k if available
            if 'pass_at_k' in stats:
                pass_k = stats['pass_at_k']
                console.print(f"  Pass@1: {pass_k['overall_pass_at_1']:.1%}")
                if pass_k['overall_pass_at_3'] is not None:
                    console.print(f"  Pass@3: {pass_k['overall_pass_at_3']:.1%}")
                if pass_k['overall_pass_at_5'] is not None:
                    console.print(f"  Pass@5: {pass_k['overall_pass_at_5']:.1%}")
                if pass_k['overall_pass_at_10'] is not None:
                    console.print(f"  Pass@10: {pass_k['overall_pass_at_10']:.1%}")
            else:
                # Fallback to simple pass rate
                console.print(f"  Pass rate: {stats.get('pass_rate', 0):.1%}")
            
            # Performance metrics
            if 'ttft' in stats and stats['ttft']:
                console.print(f"  Avg TTFT: {stats['ttft']['avg_ms']:.0f}ms")
            if 'tps' in stats and stats['tps']:
                console.print(f"  Avg TPS: {stats['tps']['avg']:.1f}")
            
            # System metrics if available
            if 'system_metrics' in stats:
                sys_metrics = stats['system_metrics']
                if sys_metrics.get('cpu_temperature_avg') is not None:
                    console.print(f"  Avg CPU Temp: {sys_metrics['cpu_temperature_avg']:.1f}°C")
                if sys_metrics.get('gpu_temperature_avg') is not None:
                    console.print(f"  Avg GPU Temp: {sys_metrics['gpu_temperature_avg']:.1f}°C")
    
    console.print(f"\n[green]✓[/green] Results saved to {output_path}")


@cli.command("info")
def info():
    """Show system information."""
    monitor = SystemMonitor()
    metrics = monitor.get_current_metrics()
    
    console.print("[bold]System Information[/bold]")
    console.print("─" * console.width)
    console.print(f"CPU Usage: {metrics.cpu_percent:.1f}%")
    console.print(f"Memory: {metrics.memory_used_mb:.0f}MB / {metrics.memory_used_mb + metrics.memory_available_mb:.0f}MB ({metrics.memory_percent:.1f}%)")
    
    if metrics.temperature:
        console.print(f"Temperature: {metrics.temperature:.1f}°C")
    
    if metrics.power_consumption_ma:
        console.print(f"Power: {metrics.power_consumption_ma:.0f}mA")


@cli.command("download-dataset")
@click.option("--dataset", type=click.Choice(["mbpp"]), default="mbpp", help="Dataset to download")
def download_dataset(dataset: str):
    """Download benchmark datasets."""
    if dataset == "mbpp":
        from .data.download_mbpp import main as download_mbpp
        console.print("[bold]Downloading MBPP dataset...[/bold]")
        download_mbpp()
    else:
        console.print(f"[red]Unknown dataset: {dataset}[/red]")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()