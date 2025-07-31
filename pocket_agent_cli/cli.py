"""Command-line interface for pocket-agent-cli."""

import click
import asyncio
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
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
    table.add_column("Size", style="green")
    table.add_column("Quantization", style="yellow")
    table.add_column("Downloaded", style="blue")
    
    for model in models:
        size_mb = f"{model.size / (1024*1024):.0f} MB"
        downloaded = "✓" if model.downloaded else "✗"
        table.add_row(
            model.id,
            model.name,
            size_mb,
            model.quantization or "-",
            downloaded
        )
    
    console.print(table)


@model.command("download")
@click.argument("model_id")
@click.option("--token", envvar="HF_TOKEN", help="Hugging Face token for gated models")
def download_model(model_id: str, token: Optional[str]):
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
            task = progress.add_task(f"Downloading {model_id}...", total=100)
            
            def update_progress(downloaded, total):
                if total > 0:
                    progress.update(task, completed=downloaded/total * 100)
            
            try:
                model = await service.download_model(
                    model_id,
                    hf_token=token,
                    progress_callback=update_progress
                )
                console.print(f"[green]✓[/green] Model {model.name} downloaded successfully!")
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
@click.option("--model", "-m", required=True, help="Model ID to use")
@click.option("--mode", type=click.Choice(list(BENCHMARK_MODES.keys())), default="base", help="Benchmark mode")
@click.option("--problems", "-p", help="Comma-separated problem IDs (e.g., 1,2,3)")
@click.option("--output", "-o", help="Output file for results")
def benchmark(model: str, mode: str, problems: Optional[str], output: Optional[str]):
    """Run benchmark evaluation."""
    # Initialize services
    model_service = ModelService()
    inference_service = InferenceService()
    
    # Load model
    model_obj = model_service.get_model(model)
    if not model_obj or not model_obj.downloaded:
        console.print(f"[red]Error:[/red] Model {model} not found or not downloaded.")
        return
    
    config = InferenceConfig()
    
    try:
        console.print(f"Loading model {model_obj.name}...")
        inference_service.load_model(model_obj, config)
    except Exception as e:
        console.print(f"[red]Failed to load model:[/red] {e}")
        return
    
    # Parse problem IDs
    problem_ids = None
    if problems:
        try:
            problem_ids = [int(p.strip()) for p in problems.split(",")]
        except ValueError:
            console.print("[red]Error:[/red] Invalid problem IDs format")
            return
    
    # Run benchmark
    benchmark_service = BenchmarkService(inference_service)
    
    async def run():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Running {mode} benchmark...", total=None)
            
            def update_progress(current, total, desc):
                progress.update(task, description=f"{desc} ({current}/{total})")
            
            session = await benchmark_service.run_benchmark(
                mode=mode,
                problem_ids=problem_ids,
                progress_callback=update_progress
            )
            
            progress.update(task, description="Benchmark complete!")
        
        return session
    
    console.print(f"\n[bold]Running {mode} benchmark[/bold]")
    console.print("─" * console.width)
    
    session = asyncio.run(run())
    
    # Display results
    stats = session.aggregate_stats
    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Total problems: {stats['total_problems']}")
    console.print(f"  Passed: {stats['passed_problems']} ({stats['pass_rate']*100:.1f}%)")
    console.print(f"  Duration: {stats['total_duration_seconds']:.1f}s")
    
    if "avg_ttft_ms" in stats:
        console.print(f"  Avg TTFT: {stats['avg_ttft_ms']:.0f}ms")
    if "avg_tps" in stats:
        console.print(f"  Avg TPS: {stats['avg_tps']:.1f}")
    
    # Save results if requested
    if output:
        export_results(session, Path(output))
        console.print(f"\n[green]✓[/green] Results saved to {output}")
    
    # Cleanup
    inference_service.unload_model()


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


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()