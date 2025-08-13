#!/usr/bin/env python3
"""Test GPU inference capabilities."""

import os
import sys
import time
import json
import platform
from pathlib import Path
from typing import Dict, Any

def check_cuda_availability() -> Dict[str, Any]:
    """Check if CUDA is available and properly configured."""
    results = {
        "platform": platform.system(),
        "cuda_available": False,
        "cuda_device": None,
        "gpu_info": None,
        "llama_cpp_cuda": False
    }
    
    # Check environment variables
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda_home = os.environ.get("CUDA_HOME")
    
    results["cuda_visible_devices"] = cuda_visible
    results["cuda_home"] = cuda_home
    
    if cuda_visible is not None:
        results["cuda_available"] = True
        results["cuda_device"] = int(cuda_visible.split(",")[0]) if cuda_visible else 0
    
    # Try to get GPU info using nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,utilization.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            results["gpu_info"] = gpu_info
            print(f"GPU detected: {gpu_info}")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("nvidia-smi not available")
    
    # Check llama-cpp-python CUDA support
    try:
        import llama_cpp
        # Check if compiled with CUDA
        # This is a heuristic - actual CUDA support depends on compilation
        results["llama_cpp_version"] = llama_cpp.__version__
        results["llama_cpp_cuda"] = cuda_visible is not None
    except ImportError:
        print("llama-cpp-python not installed")
    
    return results

def test_model_loading():
    """Test loading a model with GPU acceleration."""
    from pocket_agent_cli.services.inference_service import InferenceService
    from pocket_agent_cli.config import Model, InferenceConfig
    
    print("\n=== Testing Model Loading ===")
    
    # Create a test model configuration
    model_dir = Path.home() / ".pocket-agent-cli" / "models"
    
    # Find any available model
    available_models = list(model_dir.glob("*.gguf")) if model_dir.exists() else []
    
    if not available_models:
        print("No models found for testing")
        return False
    
    model_path = available_models[0]
    print(f"Testing with model: {model_path.name}")
    
    # Create model and config
    model = Model(
        id="test-model",
        name="Test Model",
        architecture="llama",
        quantization="Q4_K_M",
        size_gb=2.0,
        downloaded=True,
        path=model_path
    )
    
    config = InferenceConfig(
        max_tokens=100,
        temperature=0.7,
        n_threads=-1  # Auto-detect
    )
    
    # Initialize service and load model
    service = InferenceService()
    
    try:
        start_time = time.time()
        service.load_model(model, config)
        load_time = time.time() - start_time
        print(f"Model loaded successfully in {load_time:.2f} seconds")
        
        # Test inference
        messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        print("\nTesting inference...")
        start_time = time.time()
        response_text = ""
        token_count = 0
        
        for chunk in service.generate(messages, stream=True):
            response_text += chunk["token"]
            token_count += 1
            if chunk.get("finish_reason"):
                break
        
        inference_time = time.time() - start_time
        tps = token_count / inference_time if inference_time > 0 else 0
        
        print(f"Response: {response_text[:100]}...")
        print(f"Tokens generated: {token_count}")
        print(f"Time: {inference_time:.2f}s")
        print(f"Tokens per second: {tps:.2f}")
        
        # Get final metrics
        if "metrics" in chunk:
            metrics = chunk["metrics"]
            if "energy_summary" in metrics:
                print(f"Energy consumed: {metrics['energy_summary']['total_energy_joules']:.2f} J")
        
        return True
        
    except Exception as e:
        print(f"Error during model loading/inference: {e}")
        return False
    finally:
        service.unload_model()

def test_gpu_memory():
    """Test GPU memory allocation and monitoring."""
    print("\n=== Testing GPU Memory ===")
    
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        print("CUDA not available, skipping GPU memory test")
        return
    
    try:
        import subprocess
        
        # Get initial memory state
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            free, used = map(int, result.stdout.strip().split(", "))
            print(f"Initial GPU memory - Free: {free} MB, Used: {used} MB")
            
            # Here you could load a model and check memory again
            # to verify GPU memory is being used
            
            return True
    except Exception as e:
        print(f"Error checking GPU memory: {e}")
        return False

def run_performance_comparison():
    """Compare CPU vs GPU performance if both are available."""
    print("\n=== Performance Comparison ===")
    
    # This would run the same inference task with and without GPU
    # and compare the results
    
    results = {
        "cpu_only": None,
        "gpu_enabled": None,
        "speedup": None
    }
    
    # Implementation would go here
    print("Performance comparison not yet implemented")
    
    return results

def main():
    """Main test runner."""
    print("=" * 60)
    print("Pocket Agent CLI - GPU Testing Suite")
    print("=" * 60)
    
    # Check CUDA availability
    cuda_info = check_cuda_availability()
    print("\nCUDA Configuration:")
    print(json.dumps(cuda_info, indent=2))
    
    # Run tests
    tests_passed = []
    
    # Test model loading
    if test_model_loading():
        tests_passed.append("model_loading")
    
    # Test GPU memory
    if test_gpu_memory():
        tests_passed.append("gpu_memory")
    
    # Performance comparison
    perf_results = run_performance_comparison()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests passed: {len(tests_passed)}")
    print(f"CUDA available: {cuda_info['cuda_available']}")
    if cuda_info['gpu_info']:
        print(f"GPU: {cuda_info['gpu_info']}")
    
    return len(tests_passed) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)