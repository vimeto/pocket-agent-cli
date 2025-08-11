#!/usr/bin/env python
"""
Direct runner for pocket-agent CLI that bypasses Singularity wrapper issues.
Use this if the normal pocket-agent command causes bus errors.
"""

import sys
import os

# Ensure Docker is disabled
os.environ['DISABLE_DOCKER'] = '1'

# Add pocket-agent-cli to path if needed
project_dir = os.environ.get('PROJECT_DIR', '/projappl/project_2013932/vtoivone/pocket-agent-cli')
sys.path.insert(0, project_dir)

# Import and run the CLI
try:
    from pocket_agent_cli.cli import cli
    
    # Run the CLI with the provided arguments
    cli()
except ImportError as e:
    print(f"Error importing pocket-agent-cli: {e}")
    print(f"Make sure you've activated the environment:")
    print(f"  source {project_dir}/slurm/activate_env.sh")
    sys.exit(1)
except Exception as e:
    print(f"Error running pocket-agent-cli: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)