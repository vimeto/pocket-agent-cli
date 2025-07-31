"""Tool executor for handling LLM tool calls."""

import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import docker
import tempfile
import shutil
from ..config import SANDBOX_DIR


class ToolExecutor:
    """Executes tool calls in a sandboxed environment."""
    
    def __init__(self, use_docker: bool = True):
        """Initialize the tool executor.
        
        Args:
            use_docker: Whether to use Docker for sandboxing
        """
        self.use_docker = use_docker
        self.docker_client = None
        self.sandbox_dir = None
        self.container = None
        
        if use_docker:
            try:
                self.docker_client = docker.from_env()
                # Pull Python image if not present
                try:
                    self.docker_client.images.get("python:3.11-slim")
                except docker.errors.ImageNotFound:
                    print("Pulling Python Docker image...")
                    self.docker_client.images.pull("python:3.11-slim")
            except Exception as e:
                print(f"Docker not available: {e}")
                self.use_docker = False
    
    async def execute_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        max_iterations: int = 10,
    ) -> List[Dict[str, Any]]:
        """Execute a list of tool calls sequentially.
        
        Args:
            tool_calls: List of tool calls to execute
            max_iterations: Maximum number of iterations
            
        Returns:
            List of tool results
        """
        results = []
        
        # Create sandbox
        self._create_sandbox()
        
        try:
            for i, call in enumerate(tool_calls):
                if i >= max_iterations:
                    results.append({
                        "error": f"Exceeded maximum iterations ({max_iterations})"
                    })
                    break
                
                result = await self._execute_single_tool(call)
                results.append(result)
                
                # Stop on critical errors
                if result.get("error") and "critical" in result.get("error", "").lower():
                    break
            
        finally:
            # Cleanup sandbox
            self._cleanup_sandbox()
        
        return results
    
    async def _execute_single_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool call.
        
        Args:
            tool_call: Tool call with name and parameters
            
        Returns:
            Tool execution result
        """
        tool_name = tool_call.get("name")
        parameters = tool_call.get("parameters", {})
        
        # Route to appropriate handler
        handlers = {
            "run_python_code": self._run_python_code,
            "run_python_file": self._run_python_file,
            "upsert_file": self._upsert_file,
            "delete_file": self._delete_file,
            "list_files": self._list_files,
            "read_file": self._read_file,
            "submit_python_solution": self._submit_python_solution,
        }
        
        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            result = await handler(**parameters)
            return {"success": True, "output": result}
        except Exception as e:
            return {"error": str(e)}
    
    def _create_sandbox(self) -> None:
        """Create a sandbox directory for code execution."""
        if self.use_docker and self.docker_client:
            # Create temporary directory for Docker volume
            self.sandbox_dir = tempfile.mkdtemp(prefix="pocket_agent_sandbox_")
        else:
            # Use local sandbox directory
            self.sandbox_dir = SANDBOX_DIR / f"session_{int(asyncio.get_event_loop().time())}"
            self.sandbox_dir.mkdir(parents=True, exist_ok=True)
    
    def _cleanup_sandbox(self) -> None:
        """Clean up the sandbox directory."""
        if self.container:
            try:
                self.container.stop()
                self.container.remove()
            except:
                pass
            self.container = None
        
        if self.sandbox_dir:
            if isinstance(self.sandbox_dir, str):
                shutil.rmtree(self.sandbox_dir, ignore_errors=True)
            elif isinstance(self.sandbox_dir, Path) and self.sandbox_dir.exists():
                shutil.rmtree(self.sandbox_dir, ignore_errors=True)
            self.sandbox_dir = None
    
    async def _run_python_code(self, code: str) -> str:
        """Execute Python code in sandbox.
        
        Args:
            code: Python code to execute
            
        Returns:
            Execution output
        """
        if self.use_docker and self.docker_client:
            return await self._run_python_docker(code)
        else:
            return await self._run_python_local(code)
    
    async def _run_python_docker(self, code: str) -> str:
        """Execute Python code in Docker container.
        
        Args:
            code: Python code to execute
            
        Returns:
            Execution output
        """
        # Write code to file
        code_file = Path(self.sandbox_dir) / "code.py"
        with open(code_file, "w") as f:
            f.write(code)
        
        # Run in container
        try:
            # Create container without starting it
            container = self.docker_client.containers.create(
                "python:3.11-slim",
                command=["python", "/sandbox/code.py"],
                volumes={str(self.sandbox_dir): {"bind": "/sandbox", "mode": "rw"}},
                working_dir="/sandbox",
                mem_limit="512m",
                cpu_quota=50000,  # 50% CPU
            )
            
            # Start the container
            container.start()
            
            # Wait for completion with timeout
            exit_code = container.wait(timeout=30)
            
            # Get logs
            logs = container.logs(stdout=True, stderr=True)
            output = logs.decode("utf-8")
            
            # Remove container
            container.remove()
            
            return output
            
        except docker.errors.ContainerError as e:
            return f"Error: {e.stderr.decode('utf-8')}"
        except docker.errors.APIError as e:
            if "timeout" in str(e).lower():
                try:
                    container.kill()
                    container.remove()
                except:
                    pass
                return "Error: Code execution timed out (30s limit)"
            return f"Docker API error: {str(e)}"
        except Exception as e:
            return f"Docker execution error: {str(e)}"
    
    async def _run_python_local(self, code: str) -> str:
        """Execute Python code locally (less secure).
        
        Args:
            code: Python code to execute
            
        Returns:
            Execution output
        """
        import subprocess
        
        # Write code to file
        code_file = self.sandbox_dir / "code.py"
        with open(code_file, "w") as f:
            f.write(code)
        
        # Run with subprocess
        try:
            result = subprocess.run(
                ["python", str(code_file)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.sandbox_dir),
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\nErrors:\n{result.stderr}"
            
            return output
            
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out (30s limit)"
        except Exception as e:
            return f"Execution error: {str(e)}"
    
    async def _run_python_file(self, filename: str) -> str:
        """Execute a Python file in sandbox.
        
        Args:
            filename: Name of the file to execute
            
        Returns:
            Execution output
        """
        file_path = Path(self.sandbox_dir) / filename
        if not file_path.exists():
            return f"Error: File '{filename}' not found"
        
        code = file_path.read_text()
        return await self._run_python_code(code)
    
    async def _upsert_file(self, filename: str, content: str) -> str:
        """Create or update a file in sandbox.
        
        Args:
            filename: Name of the file
            content: File content
            
        Returns:
            Success message
        """
        file_path = Path(self.sandbox_dir) / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        
        return f"File '{filename}' created/updated successfully"
    
    async def _delete_file(self, filename: str) -> str:
        """Delete a file from sandbox.
        
        Args:
            filename: Name of the file to delete
            
        Returns:
            Success message or error
        """
        file_path = Path(self.sandbox_dir) / filename
        if not file_path.exists():
            return f"Error: File '{filename}' not found"
        
        file_path.unlink()
        return f"File '{filename}' deleted successfully"
    
    async def _list_files(self) -> str:
        """List all files in sandbox.
        
        Returns:
            List of files as string
        """
        files = []
        sandbox_path = Path(self.sandbox_dir)
        
        for file_path in sandbox_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(sandbox_path)
                files.append(str(relative_path))
        
        if not files:
            return "No files in sandbox"
        
        return "Files in sandbox:\n" + "\n".join(f"- {f}" for f in sorted(files))
    
    async def _read_file(self, filename: str) -> str:
        """Read a file from sandbox.
        
        Args:
            filename: Name of the file to read
            
        Returns:
            File content or error
        """
        file_path = Path(self.sandbox_dir) / filename
        if not file_path.exists():
            return f"Error: File '{filename}' not found"
        
        try:
            content = file_path.read_text()
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    async def _submit_python_solution(self, code: Optional[str] = None, filename: Optional[str] = None) -> str:
        """Submit a Python solution for benchmarking.
        
        Args:
            code: Python code to submit
            filename: Path to Python file to submit
            
        Returns:
            Submission confirmation
        """
        if filename:
            # Read code from file
            file_path = Path(self.sandbox_dir) / filename
            if not file_path.exists():
                return f"Error: File '{filename}' not found"
            code = file_path.read_text()
        elif not code:
            return "Error: Either 'code' or 'filename' must be provided"
        
        # Store the code for benchmark evaluation
        # In a real implementation, this would trigger the benchmark tests
        return f"Solution submitted successfully ({len(code)} characters)"
    
    def get_sandbox_path(self) -> Optional[Path]:
        """Get the current sandbox directory path.
        
        Returns:
            Path to sandbox directory or None
        """
        if self.sandbox_dir:
            return Path(self.sandbox_dir)
        return None