"""Unified high-performance monitoring for inference metrics."""

import time
import threading
import subprocess
import platform
import psutil
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import json
import os

# Debug flag for monitoring
DEBUG_MONITOR = os.environ.get("DEBUG_MONITOR", "").lower() == "true"


@dataclass
class UnifiedMetrics:
    """All system metrics collected in one pass."""
    timestamp: datetime
    # Power metrics
    power_watts: float
    # CPU metrics
    cpu_percent: float
    # Memory metrics
    memory_percent: float
    memory_used_mb: float
    # Optional metrics
    battery_drain_mah: Optional[float] = None
    cpu_per_core: Optional[List[float]] = None
    cpu_temperature_c: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    gpu_temperature_c: Optional[float] = None
    gpu_power_watts: Optional[float] = None
    cpu_power_watts: Optional[float] = None
    cpu_freq_mhz: Optional[float] = None  # Average CPU frequency
    gpu_freq_mhz: Optional[float] = None  # GPU frequency
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "power_watts": self.power_watts,
            "battery_drain_mah": self.battery_drain_mah,
            "cpu_percent": self.cpu_percent,
            "cpu_per_core": self.cpu_per_core,
            "cpu_temperature_c": self.cpu_temperature_c,
            "cpu_power_watts": self.cpu_power_watts,
            "cpu_freq_mhz": self.cpu_freq_mhz,
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "gpu_temperature_c": self.gpu_temperature_c,
            "gpu_power_watts": self.gpu_power_watts,
            "gpu_freq_mhz": self.gpu_freq_mhz,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
        }


class UnifiedMonitor:
    """High-performance unified monitor that collects all metrics in one pass."""
    
    def __init__(self, sample_interval: float = 2.0):
        """Initialize monitor.
        
        Args:
            sample_interval: Time between samples in seconds (default 2.0s)
        """
        self.sample_interval = sample_interval
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Use deque for efficient metric storage with automatic size limit
        self.metrics_buffer = deque(maxlen=1000)  # Keep last 1000 samples
        
        # Cached values for fast access
        self._latest_metrics: Optional[UnifiedMetrics] = None
        self._metrics_lock = threading.Lock()
        
        # Total energy tracking
        self._start_time: Optional[float] = None
        self._total_energy_joules: float = 0.0
        self._last_energy_update: Optional[float] = None
        
        # Platform detection
        self.platform = platform.system()
        self.is_apple_silicon = (
            self.platform == "Darwin" and 
            platform.processor() == "arm"
        )
        
        # Check for available tools once
        self._has_powermetrics = self._check_command("powermetrics")
        self._has_ioreg = self._check_command("ioreg")
        self._has_osx_cpu_temp = self._check_command("osx-cpu-temp")
        self._has_smctemp = self._check_command("smctemp")
        self._powermetrics_sudo = self._check_sudo_powermetrics()
        
    def _check_command(self, command: str) -> bool:
        """Check if command is available."""
        try:
            result = subprocess.run(
                ["which", command], 
                capture_output=True,
                timeout=1.0
            )
            return result.returncode == 0
        except:
            return False
    
    def _check_sudo_powermetrics(self) -> bool:
        """Check if we can run powermetrics with sudo."""
        if not self._has_powermetrics:
            return False
        try:
            result = subprocess.run(
                ["sudo", "-n", "powermetrics", "--help"],
                capture_output=True,
                timeout=2.0
            )
            return result.returncode == 0
        except:
            return False
    
    def start_monitoring(self) -> None:
        """Start monitoring in background thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.metrics_buffer.clear()
        self._latest_metrics = None
        self._start_time = time.time()
        self._total_energy_joules = 0.0
        self._last_energy_update = self._start_time
        
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return summary."""
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=3.0)
            self.monitor_thread = None
        
        return self._calculate_summary()
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop - collects all metrics efficiently."""
        while self.is_monitoring:
            start_time = time.time()
            
            try:
                if self._powermetrics_sudo:
                    # Use powermetrics for comprehensive data
                    metrics = self._collect_with_powermetrics()
                else:
                    # Fallback to multiple sources
                    metrics = self._collect_fallback()
                
                if metrics:
                    # Update energy tracking
                    current_time = time.time()
                    if self._last_energy_update:
                        time_delta = current_time - self._last_energy_update
                        self._total_energy_joules += metrics.power_watts * time_delta
                    self._last_energy_update = current_time
                    
                    # Store metrics
                    with self._metrics_lock:
                        self._latest_metrics = metrics
                        self.metrics_buffer.append(metrics)
                
            except Exception as e:
                if DEBUG_MONITOR:
                    print(f"[DEBUG UnifiedMonitor] Monitor error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Sleep for remaining interval
            elapsed = time.time() - start_time
            sleep_time = self.sample_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _collect_with_powermetrics(self) -> Optional[UnifiedMetrics]:
        """Collect all metrics using powermetrics (most efficient)."""
        try:
            if DEBUG_MONITOR:
                print("[DEBUG] Using _collect_with_powermetrics method")
            # Run powermetrics once for all data (text format to get power values)
            cmd = [
                "sudo", "-n", "powermetrics",
                "--samplers", "cpu_power,gpu_power",
                "--sample-count", "1",
                "--sample-rate", "100",  # 100ms sample
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=2.0  # Increased timeout for powermetrics
            )
            
            if result.returncode != 0:
                return self._collect_fallback()
            
            # Get psutil metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            memory = psutil.virtual_memory()
            
            # Parse power values from text output
            cpu_power = 0.0
            gpu_power = 0.0
            gpu_utilization = 0.0
            cpu_freq = None
            gpu_freq = None
            
            # Track CPU frequencies across all cores
            cpu_freqs = []
            
            for line in result.stdout.split('\n'):
                if 'CPU Power:' in line:
                    match = re.search(r'CPU Power:\s*(\d+)\s*mW', line)
                    if match:
                        cpu_power = float(match.group(1)) / 1000.0  # Convert mW to W
                elif 'GPU Power:' in line and 'CPU + GPU' not in line:
                    match = re.search(r'GPU Power:\s*(\d+)\s*mW', line)
                    if match:
                        gpu_power = float(match.group(1)) / 1000.0  # Convert mW to W
                elif 'GPU HW active residency:' in line:
                    match = re.search(r'GPU HW active residency:\s*([\d.]+)%', line)
                    if match:
                        gpu_utilization = float(match.group(1))
                elif line.strip().startswith('CPU') and 'frequency:' in line:
                    # Parse CPU frequency lines like "CPU 0 frequency: 2424 MHz"
                    match = re.search(r'CPU \d+ frequency:\s*(\d+)\s*MHz', line)
                    if match:
                        cpu_freqs.append(float(match.group(1)))
                elif 'GPU HW active frequency:' in line:
                    # Parse GPU frequency line like "GPU HW active frequency: 444 MHz"
                    match = re.search(r'GPU HW active frequency:\s*([\d.]+)\s*MHz', line)
                    if match:
                        gpu_freq = float(match.group(1))
            
            # Calculate average CPU frequency
            if cpu_freqs:
                cpu_freq = sum(cpu_freqs) / len(cpu_freqs)
                if DEBUG_MONITOR:
                    print(f"[DEBUG] Found {len(cpu_freqs)} CPU frequencies, avg: {cpu_freq:.1f} MHz")
                    print(f"[DEBUG] GPU frequency: {gpu_freq} MHz")
            
            # Get temperature from ioreg/smctemp if available
            cpu_temp = None
            if self._has_smctemp or self._has_osx_cpu_temp:
                cpu_temp = self._get_cpu_temp()
            
            return UnifiedMetrics(
                timestamp=datetime.now(),
                # Power
                power_watts=cpu_power + gpu_power,
                cpu_power_watts=cpu_power,
                gpu_power_watts=gpu_power,
                # CPU
                cpu_percent=cpu_percent,
                cpu_per_core=cpu_per_core,
                cpu_temperature_c=cpu_temp,
                cpu_freq_mhz=cpu_freq,
                # GPU
                gpu_utilization_percent=gpu_utilization,
                gpu_freq_mhz=gpu_freq,
                # Memory
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
            )
            
        except Exception as e:
            if DEBUG_MONITOR:
                print(f"[DEBUG] _collect_with_powermetrics failed: {e}")
                import traceback
                traceback.print_exc()
            return self._collect_fallback()
    
    def _collect_fallback(self) -> Optional[UnifiedMetrics]:
        """Collect metrics using multiple tools (less efficient but works without sudo)."""
        try:
            # Get all psutil metrics first (fast)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            memory = psutil.virtual_memory()
            
            # Initialize metrics
            metrics = UnifiedMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                cpu_per_core=cpu_per_core,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                power_watts=0.0,
            )
            
            # Get power from ioreg (single call)
            if self._has_ioreg:
                power_data = self._get_ioreg_power()
                metrics.power_watts = power_data.get("power_watts", 0.0)
                metrics.battery_drain_mah = power_data.get("battery_ma")
            
            # Estimate power if not available
            if metrics.power_watts == 0.0:
                metrics.power_watts = self._estimate_power(cpu_percent)
            
            # Get temperature (single call)
            if self._has_smctemp or self._has_osx_cpu_temp:
                metrics.cpu_temperature_c = self._get_cpu_temp()
            
            # Get GPU metrics from ioreg
            gpu_data = self._get_gpu_metrics_ioreg()
            metrics.gpu_utilization_percent = gpu_data.get("utilization")
            
            # Get GPU temperature if available
            if self._has_smctemp:
                metrics.gpu_temperature_c = self._get_gpu_temp()
            
            return metrics
            
        except Exception:
            return None
    
    def _get_ioreg_power(self) -> Dict[str, Any]:
        """Get power metrics from IORegistry in one call."""
        try:
            result = subprocess.run(
                ["ioreg", "-r", "-c", "AppleSmartBattery"],
                capture_output=True,
                text=True,
                timeout=0.5
            )
            
            if result.returncode != 0:
                return {"power_watts": 0.0}
            
            # Parse battery data
            voltage_v = None
            amperage_ma = None
            
            for line in result.stdout.split('\n'):
                if 'Voltage' in line and voltage_v is None:
                    match = re.search(r'= (\d+)', line)
                    if match:
                        voltage_v = int(match.group(1)) / 1000.0
                elif 'InstantAmperage' in line:
                    match = re.search(r'= (-?\d+)', line)
                    if match:
                        raw_amperage = int(match.group(1))
                        # Handle 64-bit unsigned to signed conversion
                        if raw_amperage > 2**63:
                            raw_amperage = raw_amperage - 2**64
                        amperage_ma = abs(raw_amperage)
            
            power_watts = 0.0
            if voltage_v and amperage_ma:
                power_watts = (voltage_v * amperage_ma) / 1000.0
                # Validate reasonable power range (0.1W to 200W for laptops)
                if power_watts < 0.1 or power_watts > 200.0:
                    if DEBUG_MONITOR:
                        print(f"[DEBUG] Invalid power calculated: {power_watts}W, falling back to estimation")
                    power_watts = 0.0  # Fall back to estimation
            
            return {
                "power_watts": power_watts,
                "battery_ma": amperage_ma
            }
            
        except Exception:
            return {"power_watts": 0.0}
    
    def _get_cpu_temp(self) -> Optional[float]:
        """Get CPU temperature quickly."""
        # Try smctemp first (more reliable)
        if self._has_smctemp:
            try:
                result = subprocess.run(
                    ["smctemp", "-c", "-i25", "-n180", "-f"],
                    capture_output=True,
                    text=True,
                    timeout=0.5
                )
                if result.returncode == 0 and result.stdout.strip():
                    try:
                        temp = float(result.stdout.strip())
                        if temp > 0:  # Valid temperature
                            return temp
                    except ValueError:
                        pass
            except Exception:
                pass
        
        # Fallback to osx-cpu-temp
        if self._has_osx_cpu_temp:
            try:
                result = subprocess.run(
                    ["osx-cpu-temp"],
                    capture_output=True,
                    text=True,
                    timeout=0.5
                )
                if result.returncode == 0:
                    match = re.search(r'([0-9.]+)', result.stdout)
                    if match:
                        temp = float(match.group(1))
                        if temp > 0:  # Valid temperature
                            return temp
            except Exception:
                pass
        
        return None
    
    def _get_gpu_temp(self) -> Optional[float]:
        """Get GPU temperature using smctemp."""
        if self._has_smctemp:
            try:
                result = subprocess.run(
                    ["smctemp", "-g", "-i25", "-n180", "-f"],
                    capture_output=True,
                    text=True,
                    timeout=0.5
                )
                if result.returncode == 0 and result.stdout.strip():
                    try:
                        temp = float(result.stdout.strip())
                        if temp > 0:  # Valid temperature
                            return temp
                    except ValueError:
                        pass
            except Exception:
                pass
        return None
    
    def _get_gpu_metrics_ioreg(self) -> Dict[str, Any]:
        """Get GPU metrics from ioreg quickly."""
        try:
            result = subprocess.run(
                ['ioreg', '-r', '-d', '1', '-w', '0', '-c', 'IOAccelerator'],
                capture_output=True,
                text=True,
                timeout=0.5
            )
            
            if result.returncode == 0 and result.stdout:
                match = re.search(r'"Device Utilization %"\s*=\s*(\d+)', result.stdout)
                if match:
                    return {"utilization": float(match.group(1))}
        except Exception:
            pass
        
        return {"utilization": None}
    
    def _estimate_power(self, cpu_percent: float) -> float:
        """Estimate power based on CPU usage."""
        if self.is_apple_silicon:
            # M-series: ~5W idle, ~30W max
            return 5.0 + (cpu_percent / 100.0) * 25.0
        else:
            # Intel: ~10W idle, ~60W max
            return 10.0 + (cpu_percent / 100.0) * 50.0
    
    def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """Get latest metrics for real-time display."""
        with self._metrics_lock:
            if self._latest_metrics:
                return {
                    "power_watts": self._latest_metrics.power_watts,
                    "cpu_percent": self._latest_metrics.cpu_percent,
                    "gpu_utilization_percent": self._latest_metrics.gpu_utilization_percent,
                    "temperature_c": self._latest_metrics.cpu_temperature_c,
                    "total_energy_joules": self._total_energy_joules,
                    "cpu_power_watts": self._latest_metrics.cpu_power_watts,
                    "gpu_power_watts": self._latest_metrics.gpu_power_watts,
                    "cpu_freq_mhz": self._latest_metrics.cpu_freq_mhz,
                    "gpu_freq_mhz": self._latest_metrics.gpu_freq_mhz,
                }
        return None
    
    def get_energy_summary(self) -> Dict[str, Any]:
        """Get energy consumption summary."""
        if not self.metrics_buffer:
            return {
                "total_energy_joules": 0.0,
                "total_energy_wh": 0.0,
                "avg_power_watts": 0.0,
                "duration_seconds": 0.0,
            }
        
        duration = time.time() - self._start_time if self._start_time else 0
        power_values = [m.power_watts for m in self.metrics_buffer]
        
        return {
            "total_energy_joules": self._total_energy_joules,
            "total_energy_wh": self._total_energy_joules / 3600.0,
            "avg_power_watts": sum(power_values) / len(power_values),
            "max_power_watts": max(power_values),
            "min_power_watts": min(power_values),
            "duration_seconds": duration,
            "samples": len(self.metrics_buffer),
        }
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics."""
        if not self.metrics_buffer:
            return {"samples": 0}
        
        metrics_list = list(self.metrics_buffer)
        
        # Basic stats
        summary = self.get_energy_summary()
        
        # CPU stats
        cpu_values = [m.cpu_percent for m in metrics_list]
        summary["cpu_avg_percent"] = sum(cpu_values) / len(cpu_values)
        summary["cpu_max_percent"] = max(cpu_values)
        
        # Memory stats
        memory_values = [m.memory_percent for m in metrics_list]
        summary["memory_avg_percent"] = sum(memory_values) / len(memory_values)
        summary["memory_max_percent"] = max(memory_values)
        
        # Temperature stats
        cpu_temps = [m.cpu_temperature_c for m in metrics_list if m.cpu_temperature_c]
        if cpu_temps:
            summary["cpu_temp_avg_c"] = sum(cpu_temps) / len(cpu_temps)
            summary["cpu_temp_max_c"] = max(cpu_temps)
        
        # GPU stats
        gpu_utils = [m.gpu_utilization_percent for m in metrics_list if m.gpu_utilization_percent is not None]
        if gpu_utils:
            summary["gpu_utilization_avg_percent"] = sum(gpu_utils) / len(gpu_utils)
            summary["gpu_utilization_max_percent"] = max(gpu_utils)
        
        return summary
    
    def export_metrics(self) -> List[Dict[str, Any]]:
        """Export all collected metrics."""
        return [m.to_dict() for m in self.metrics_buffer]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics (alias for _calculate_summary)."""
        return self._calculate_summary()