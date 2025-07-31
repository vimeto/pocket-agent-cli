#!/usr/bin/env python3
"""Download the full MBPP dataset from Hugging Face."""

import json
import requests
from pathlib import Path
from typing import List, Dict, Any


def download_mbpp_dataset() -> List[Dict[str, Any]]:
    """Download MBPP dataset from Hugging Face datasets."""
    print("Downloading MBPP dataset from Hugging Face...")
    
    # MBPP dataset URL
    url = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse JSONL format (one JSON object per line)
        problems = []
        for line in response.text.strip().split('\n'):
            if line:
                problem = json.loads(line)
                problems.append(problem)
        
        print(f"Downloaded {len(problems)} problems")
        return problems
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        
        # Try alternative source
        print("Trying alternative source...")
        try:
            # Use Hugging Face datasets API
            import subprocess
            result = subprocess.run([
                "curl", "-s",
                "https://datasets-server.huggingface.co/first-rows?dataset=mbpp&config=full&split=train"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                rows = data.get("rows", [])
                problems = [row["row"] for row in rows]
                print(f"Downloaded {len(problems)} problems from alternative source")
                return problems
                
        except Exception as e2:
            print(f"Alternative source also failed: {e2}")
    
    return []


def convert_to_benchmark_format(problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert MBPP format to our benchmark format."""
    converted = []
    
    for problem in problems:
        # MBPP format has: task_id, text, code, test_list
        # Some datasets might have slightly different field names
        converted_problem = {
            "task_id": problem.get("task_id", len(converted)),
            "text": problem.get("text", problem.get("prompt", "")),
            "code": problem.get("code", problem.get("canonical_solution", "")),
            "test_list": problem.get("test_list", problem.get("test", []))
        }
        
        # Ensure test_list is a list
        if isinstance(converted_problem["test_list"], str):
            # If it's a single string, split by newlines
            converted_problem["test_list"] = [
                t.strip() for t in converted_problem["test_list"].split('\n')
                if t.strip() and t.strip().startswith("assert")
            ]
        
        converted.append(converted_problem)
    
    return converted


def save_dataset(problems: List[Dict[str, Any]], output_path: Path):
    """Save dataset to file."""
    with open(output_path, 'w') as f:
        json.dump(problems, f, indent=2)
    print(f"Saved {len(problems)} problems to {output_path}")


def create_sample_dataset(problems: List[Dict[str, Any]], sample_size: int = 10) -> List[Dict[str, Any]]:
    """Create a smaller sample dataset for testing."""
    return problems[:sample_size]


def main():
    """Main function to download and save MBPP dataset."""
    # Define paths
    data_dir = Path(__file__).parent
    full_dataset_path = data_dir / "mbpp_full.json"
    test_dataset_path = data_dir / "mbpp_test.json"
    
    # Download dataset
    problems = download_mbpp_dataset()
    
    if not problems:
        print("Failed to download dataset. Using manual download instructions:")
        print("\nManual download instructions:")
        print("1. Visit: https://github.com/google-research/google-research/tree/master/mbpp")
        print("2. Download mbpp.jsonl")
        print("3. Place it in:", data_dir)
        return
    
    # Convert to our format
    problems = convert_to_benchmark_format(problems)
    
    # Save full dataset
    save_dataset(problems, full_dataset_path)
    
    # Create test dataset (first 100 problems)
    test_problems = create_sample_dataset(problems, 100)
    save_dataset(test_problems, test_dataset_path)
    
    # Show statistics
    print("\nDataset statistics:")
    print(f"Total problems: {len(problems)}")
    print(f"Test dataset: {len(test_problems)} problems")
    
    # Show sample problem
    if problems:
        print("\nSample problem:")
        print(f"Task ID: {problems[0]['task_id']}")
        print(f"Text: {problems[0]['text'][:100]}...")
        print(f"Tests: {len(problems[0]['test_list'])} test cases")


if __name__ == "__main__":
    main()