"""Simplified system monitoring that actually works."""

import time
import psutil
import platform
import subprocess
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import threading


@dataclass
class BasicMetrics:
    """Basic system metrics that work reliably."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    
    # Optional metrics added carefully
    cpu_per_core: List[float] = None
    gpu_metal_active: bool = False
    gpu_utilization_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    gpu_process_count: int = 0
    power_draw_watts: Optional[float] = None
    cpu_temperature_c: Optional[float] = None
    gpu_temperature_c: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'cpu_per_core': self.cpu_per_core,
            'gpu_metal_active': self.gpu_metal_active,
            'gpu_utilization_percent': self.gpu_utilization_percent,
            'gpu_memory_mb': self.gpu_memory_mb,
            'gpu_process_count': self.gpu_process_count,
            'power_draw_watts': self.power_draw_watts,
            'cpu_temperature_c': self.cpu_temperature_c,
            'gpu_temperature_c': self.gpu_temperature_c,
        }


class SimpleMonitor:
    """Simple, working system monitor."""
    
    def __init__(self):
        self.metrics_history: List[BasicMetrics] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.interval = 1.0
        self._errors = []
        
    def start_monitoring(self) -> None:
        """Start monitoring in background thread."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.metrics_history = []
        self._errors = []
        
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            self.monitor_thread = None
            
    def _monitor_loop(self) -> None:
        """Simple monitoring loop that won't hang."""
        while self.is_monitoring:
            try:
                metrics = self._get_basic_metrics()
                if metrics:
                    self.metrics_history.append(metrics)
            except Exception as e:
                self._errors.append(f"Monitor error: {e}")
            
            time.sleep(self.interval)
            
    def _get_basic_metrics(self) -> Optional[BasicMetrics]:
        """Get basic metrics that work reliably."""
        try:
            # Basic metrics that always work
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            metrics = BasicMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
            )
            
            # Try to add per-core CPU
            try:
                metrics.cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            except:
                pass
                
            # Check for Metal/GPU usage (macOS)
            if platform.system() == "Darwin":
                gpu_info = self._get_gpu_metrics()
                metrics.gpu_metal_active = gpu_info['active']
                metrics.gpu_utilization_percent = gpu_info.get('utilization')
                metrics.gpu_memory_mb = gpu_info.get('memory_mb')
                metrics.gpu_process_count = gpu_info.get('process_count', 0)
                
            # Try to get power (very simple)
            metrics.power_draw_watts = self._estimate_power()
            
            # Try to get temperatures
            cpu_temp, gpu_temp = self._get_temperatures()
            metrics.cpu_temperature_c = cpu_temp
            metrics.gpu_temperature_c = gpu_temp
            
            return metrics
            
        except Exception as e:
            self._errors.append(f"Metrics error: {e}")
            return None
            
    def _check_metal_active(self) -> bool:
        """Check if Metal is being used by llama.cpp."""
        try:
            # Method 1: Check for llama processes with GPU activity
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    name = proc.info.get('name', '').lower()
                    cmdline = ' '.join(proc.info.get('cmdline', []))
                    
                    # Check if it's a llama process
                    if 'python' in name and ('llama' in cmdline or 'pocket-agent' in cmdline):
                        # Check if process has significant CPU usage (likely using GPU)
                        if proc.cpu_percent(interval=0.1) > 10:
                            return True
                except:
                    continue
                    
            # Method 2: Quick check for Metal framework activity
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True,
                timeout=0.5
            )
            if result.returncode == 0:
                # Look for Metal-related processes
                output = result.stdout
                if 'Metal' in output or 'AGXMetal' in output:
                    return True
                    
            return False
        except:
            return False
            
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get comprehensive GPU metrics on macOS."""
        gpu_info = {
            'active': False,
            'utilization': None,
            'memory_mb': None,
            'process_count': 0
        }
        
        try:
            # Method 1: Use ioreg to check GPU state
            gpu_util = self._get_gpu_utilization_ioreg()
            if gpu_util is not None:
                gpu_info['utilization'] = gpu_util
                gpu_info['active'] = gpu_util > 0
            
            # Method 2: Check activity monitor data
            if gpu_info['utilization'] is None:
                gpu_util = self._get_gpu_utilization_activity_monitor()
                if gpu_util is not None:
                    gpu_info['utilization'] = gpu_util
                    gpu_info['active'] = gpu_util > 0
            
            # Count GPU processes
            gpu_process_count = 0
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    name = proc.info.get('name', '').lower()
                    if 'python' in name:
                        cpu_percent = proc.cpu_percent(interval=0.1)
                        if cpu_percent > 20:  # Significant usage
                            cmdline = ' '.join(proc.info.get('cmdline', []))
                            if 'llama' in cmdline or 'pocket-agent' in cmdline:
                                gpu_process_count += 1
                                gpu_info['active'] = True
                except:
                    continue
            
            gpu_info['process_count'] = gpu_process_count
            
            # Fallback: check if Metal is active
            if not gpu_info['active']:
                gpu_info['active'] = self._check_metal_active()
                
        except Exception as e:
            self._errors.append(f"GPU metrics error: {e}")
            
        return gpu_info
    
    def _get_gpu_utilization_ioreg(self) -> Optional[float]:
        """Get REAL GPU utilization from macOS ioreg. Returns None if not available."""
        try:
            result = subprocess.run(
                ['ioreg', '-r', '-d', '1', '-w', '0', '-c', 'IOAccelerator'],
                capture_output=True,
                text=True,
                timeout=0.5
            )
            
            if result.returncode == 0 and result.stdout:
                import re
                # Look for Device Utilization % in PerformanceStatistics
                # Format: "Device Utilization %"=9
                match = re.search(r'"Device Utilization %"=(\d+)', result.stdout)
                if match:
                    return float(match.group(1))
                
                # Alternative format with spaces
                match = re.search(r'"Device Utilization %"\s*=\s*(\d+)', result.stdout)
                if match:
                    return float(match.group(1))
                    
        except:
            pass
            
        # No data available - return None
        return None
    
    def _get_gpu_utilization_activity_monitor(self) -> Optional[float]:
        """Try to get GPU utilization from Activity Monitor data. Returns None if not available."""
        try:
            # Try to get GPU statistics from powermetrics WITHOUT sudo
            # This won't work without sudo, so it will return None
            result = subprocess.run(
                ['powermetrics', '--samplers', 'gpu_power', '--sample-count', '1', '--sample-rate', '100'],
                capture_output=True,
                text=True,
                timeout=0.5
            )
            
            if result.returncode == 0 and result.stdout:
                import re
                # Look for GPU active residency
                gpu_match = re.search(r'GPU active residency:\s*([\d.]+)%', result.stdout)
                if gpu_match:
                    return float(gpu_match.group(1))
                    
        except:
            pass
            
        # NO ESTIMATIONS - return None
        return None
    
    def _get_temperatures(self) -> tuple[Optional[float], Optional[float]]:
        """Get CPU and GPU temperatures if available. Returns (cpu_temp, gpu_temp)."""
        cpu_temp = None
        gpu_temp = None
        
        # Use MacOSMonitor for temperature readings
        if platform.system() == "Darwin":
            try:
                from .macos_helpers import MacOSMonitor
                mac_monitor = MacOSMonitor()
                temps = mac_monitor.get_temperature()
                
                if 'cpu' in temps:
                    cpu_temp = temps['cpu']
                if 'gpu' in temps:
                    gpu_temp = temps['gpu']
                    
                # If we got valid temperatures, return them
                if cpu_temp is not None or gpu_temp is not None:
                    return cpu_temp, gpu_temp
            except Exception as e:
                self._errors.append(f"MacOSMonitor error: {e}")
        
        # Fallback: Try osx-cpu-temp directly if MacOSMonitor didn't work
        if cpu_temp is None:
            try:
                result = subprocess.run(
                    ['osx-cpu-temp'],
                    capture_output=True,
                    text=True,
                    timeout=0.5
                )
                if result.returncode == 0 and result.stdout:
                    # Parse output like "58.2°C"
                    import re
                    match = re.search(r'([0-9.]+)', result.stdout)
                    if match:
                        temp = float(match.group(1))
                        # osx-cpu-temp sometimes returns 0.0 which is invalid
                        if temp > 0:
                            cpu_temp = temp
            except:
                pass
        
        # Method 2: Try powermetrics (requires sudo, so likely won't work)
        if cpu_temp is None:
            try:
                result = subprocess.run(
                    ['sudo', '-n', 'powermetrics', '--samplers', 'thermal', '-n', '1', '-i', '1000'],
                    capture_output=True,
                    text=True,
                    timeout=2.0
                )
                if result.returncode == 0 and result.stdout:
                    import re
                    # Look for CPU die temperature
                    cpu_match = re.search(r'CPU die temperature:\s*([0-9.]+)', result.stdout)
                    if cpu_match:
                        cpu_temp = float(cpu_match.group(1))
                    # Look for GPU die temperature  
                    gpu_match = re.search(r'GPU die temperature:\s*([0-9.]+)', result.stdout)
                    if gpu_match:
                        gpu_temp = float(gpu_match.group(1))
            except:
                pass
                
        # Method 3: Check thermal state from ioreg (not temperature but thermal info)
        # This gives us thermal pressure but not actual temperature
        # We won't use this for temperature values
        
        return cpu_temp, gpu_temp
            
    def _estimate_power(self) -> Optional[float]:
        """Simple power estimation based on CPU."""
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            # Simple estimation: 5W base + up to 25W under load
            return 5.0 + (cpu / 100.0) * 25.0
        except:
            return None
            
    def export_metrics(self) -> List[Dict[str, Any]]:
        """Export metrics as list of dicts."""
        return [m.to_dict() for m in self.metrics_history]
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.metrics_history:
            return {"samples": 0}
            
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_percent for m in self.metrics_history]
        
        return {
            "samples": len(self.metrics_history),
            "duration_seconds": (
                self.metrics_history[-1].timestamp - 
                self.metrics_history[0].timestamp
            ).total_seconds(),
            "cpu_avg": sum(cpu_values) / len(cpu_values),
            "cpu_max": max(cpu_values),
            "memory_avg": sum(memory_values) / len(memory_values),
            "memory_max": max(memory_values),
            "gpu_metal_active_percent": sum(
                1 for m in self.metrics_history if m.gpu_metal_active
            ) / len(self.metrics_history) * 100 if self.metrics_history else 0,
            "gpu_utilization_avg": sum(
                m.gpu_utilization_percent for m in self.metrics_history 
                if m.gpu_utilization_percent is not None
            ) / len([m for m in self.metrics_history if m.gpu_utilization_percent is not None]) 
            if any(m.gpu_utilization_percent is not None for m in self.metrics_history) else None,
            "gpu_utilization_max": max(
                (m.gpu_utilization_percent for m in self.metrics_history 
                 if m.gpu_utilization_percent is not None), default=None
            ),
            "gpu_process_count_avg": sum(
                m.gpu_process_count for m in self.metrics_history
            ) / len(self.metrics_history) if self.metrics_history else 0,
            "cpu_temperature_avg": sum(
                m.cpu_temperature_c for m in self.metrics_history 
                if m.cpu_temperature_c is not None
            ) / len([m for m in self.metrics_history if m.cpu_temperature_c is not None]) 
            if any(m.cpu_temperature_c is not None for m in self.metrics_history) else None,
            "gpu_temperature_avg": sum(
                m.gpu_temperature_c for m in self.metrics_history 
                if m.gpu_temperature_c is not None
            ) / len([m for m in self.metrics_history if m.gpu_temperature_c is not None]) 
            if any(m.gpu_temperature_c is not None for m in self.metrics_history) else None,
            "errors": self._errors[:5],  # First 5 errors
        }