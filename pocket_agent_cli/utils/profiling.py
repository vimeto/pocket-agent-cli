"""Line profiling utilities for performance analysis."""

import os
import sys
import subprocess
from pathlib import Path


def run_with_profiler(args):
    """Run the command with line profiler enabled."""
    # Try to find kernprof in the virtual environment first
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path:
        kernprof_path = os.path.join(venv_path, 'bin', 'kernprof')
        if not os.path.exists(kernprof_path):
            kernprof_path = None
    else:
        # Fallback to system kernprof
        kernprof_path = subprocess.run(
            ["which", "kernprof"],
            capture_output=True,
            text=True
        ).stdout.strip()
    
    if not kernprof_path:
        print("Error: kernprof not found. Please install line_profiler.")
        return 1
    
    # Build the command
    cmd = [
        kernprof_path,
        "-l",  # Line-by-line profiling
        "-v",  # Verbose output (show results immediately)
        sys.argv[0],  # The pocket-agent script
    ] + args
    
    # Run with profiler
    print(f"Running with line profiler: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode