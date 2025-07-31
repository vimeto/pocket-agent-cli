"""System monitoring for performance metrics."""

import time
import psutil
import platform
import subprocess
import os
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import threading
import statistics

# Lazy load macOS helpers to avoid initialization issues
_MACOS_MONITOR = None

def _get_macos_monitor():
    """Lazy load MacOSMonitor."""
    global _MACOS_MONITOR
    if _MACOS_MONITOR is None and platform.system() == "Darwin":
        try:
            from .macos_helpers import MacOSMonitor
            _MACOS_MONITOR = MacOSMonitor()
        except ImportError:
            pass
    return _MACOS_MONITOR


@dataclass
class SystemMetrics:
    """Comprehensive system performance metrics."""
    
    # Required fields first
    timestamp: datetime
    cpu_percent: float
    memory_used_mb: float
    memory_available_mb: float
    memory_percent: float
    
    # Optional fields with defaults
    # CPU Metrics
    cpu_percent_per_core: List[float] = field(default_factory=list)
    cpu_frequency_current: Optional[float] = None
    cpu_frequency_min: Optional[float] = None
    cpu_frequency_max: Optional[float] = None
    cpu_efficiency_cores_percent: Optional[float] = None
    cpu_performance_cores_percent: Optional[float] = None
    cpu_throttled: bool = False
    
    # Memory Metrics
    memory_wired_mb: Optional[float] = None
    memory_compressed_mb: Optional[float] = None
    memory_swap_used_mb: Optional[float] = None
    memory_pressure: Optional[str] = None  # "green", "yellow", "red"
    
    # GPU/Metal Metrics
    gpu_utilization_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    metal_in_use: bool = False
    
    # Thermal Metrics
    temperature_cpu: Optional[float] = None
    temperature_gpu: Optional[float] = None
    temperature_ambient: Optional[float] = None
    fan_speed_rpm: Optional[int] = None
    thermal_pressure: Optional[str] = None  # "nominal", "fair", "serious", "critical"
    
    # Power Metrics
    power_consumption_watts: Optional[float] = None
    power_consumption_ma: Optional[float] = None
    battery_level_percent: Optional[float] = None
    battery_drain_rate_watts: Optional[float] = None
    battery_time_remaining_minutes: Optional[int] = None
    power_adapter_connected: bool = False
    
    # I/O Metrics
    disk_read_mb_per_sec: Optional[float] = None
    disk_write_mb_per_sec: Optional[float] = None
    disk_io_queue_depth: Optional[int] = None
    
    # Network Metrics (for hybrid setups)
    network_bytes_sent_per_sec: Optional[float] = None
    network_bytes_recv_per_sec: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class SystemMonitor:
    """Monitor system performance metrics."""
    
    def __init__(self):
        self.is_monitoring = False
        self.metrics_history: List[SystemMetrics] = []
        self.monitor_thread: Optional[threading.Thread] = None
        self.interval = 1.0  # Sample every second
        self._last_disk_io = None
        self._last_network_io = None
        self._last_io_time = None
        self._has_powermetrics = self._check_powermetrics()
        self._has_ioreg = self._check_ioreg()
        self._metal_process_names = ['llama-cpp', 'python', 'llama']  # Processes to monitor for Metal usage
        self._collection_errors = []  # Track any collection errors
        
    def start_monitoring(self) -> None:
        """Start continuous monitoring in background thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.metrics_history = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            self.monitor_thread = None
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)
            except Exception as e:
                # Log error but continue monitoring
                self._collection_errors.append(str(e))
            time.sleep(self.interval)
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get comprehensive current system metrics."""
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_percent_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        
        # Safe cpu_freq access
        try:
            cpu_freq = psutil.cpu_freq()
        except:
            cpu_freq = None
            
        # Disable advanced features that might cause issues
        efficiency_percent, performance_percent = None, None  # self._get_core_type_usage()
        cpu_throttled = False  # self._is_cpu_throttled()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        memory_percent = memory.percent
        
        # macOS specific memory metrics
        wired_mb, compressed_mb = self._get_macos_memory_stats()
        swap = psutil.swap_memory()
        swap_used_mb = swap.used / (1024 * 1024) if swap else None
        memory_pressure = self._get_memory_pressure()
        
        # GPU/Metal metrics
        gpu_util, gpu_mem_used, gpu_mem_total, metal_in_use = self._get_gpu_metrics()
        
        # Thermal metrics
        temps = self._get_all_temperatures()
        fan_speed = self._get_fan_speed()
        thermal_pressure = self._get_thermal_pressure()
        
        # Power metrics
        power_metrics = self._get_power_metrics()
        battery_info = self._get_battery_info()
        
        # I/O metrics
        disk_read, disk_write, io_queue = self._get_disk_io_metrics()
        
        # Network metrics
        net_sent, net_recv = self._get_network_io_metrics()
        
        return SystemMetrics(
            timestamp=timestamp,
            # CPU
            cpu_percent=cpu_percent,
            cpu_percent_per_core=cpu_percent_per_core,
            cpu_frequency_current=cpu_freq.current if cpu_freq else None,
            cpu_frequency_min=cpu_freq.min if cpu_freq else None,
            cpu_frequency_max=cpu_freq.max if cpu_freq else None,
            cpu_efficiency_cores_percent=efficiency_percent,
            cpu_performance_cores_percent=performance_percent,
            cpu_throttled=cpu_throttled,
            # Memory
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            memory_percent=memory_percent,
            memory_wired_mb=wired_mb,
            memory_compressed_mb=compressed_mb,
            memory_swap_used_mb=swap_used_mb,
            memory_pressure=memory_pressure,
            # GPU
            gpu_utilization_percent=gpu_util,
            gpu_memory_used_mb=gpu_mem_used,
            gpu_memory_total_mb=gpu_mem_total,
            metal_in_use=metal_in_use,
            # Thermal
            temperature_cpu=temps.get('cpu'),
            temperature_gpu=temps.get('gpu'),
            temperature_ambient=temps.get('ambient'),
            fan_speed_rpm=fan_speed,
            thermal_pressure=thermal_pressure,
            # Power
            power_consumption_watts=power_metrics.get('watts'),
            power_consumption_ma=power_metrics.get('milliamps'),
            battery_level_percent=battery_info.get('level'),
            battery_drain_rate_watts=battery_info.get('drain_rate'),
            battery_time_remaining_minutes=battery_info.get('time_remaining'),
            power_adapter_connected=battery_info.get('plugged', False),
            # I/O
            disk_read_mb_per_sec=disk_read,
            disk_write_mb_per_sec=disk_write,
            disk_io_queue_depth=io_queue,
            # Network
            network_bytes_sent_per_sec=net_sent,
            network_bytes_recv_per_sec=net_recv,
        )
    
    def _check_powermetrics(self) -> bool:
        """Check if powermetrics is available (macOS)."""
        if platform.system() != "Darwin":
            return False
        try:
            subprocess.run(['which', 'powermetrics'], capture_output=True, check=True)
            return True
        except:
            return False
    
    def _check_ioreg(self) -> bool:
        """Check if ioreg is available (macOS)."""
        if platform.system() != "Darwin":
            return False
        try:
            subprocess.run(['which', 'ioreg'], capture_output=True, check=True)
            return True
        except:
            return False
    
    def _get_all_temperatures(self) -> Dict[str, float]:
        """Get all available temperatures."""
        temps = {}
        
        if platform.system() == "Darwin" and _get_macos_monitor():
            # Use macOS helper for temperature monitoring
            temps = _get_macos_monitor().get_temperature()
        elif platform.system() == "Darwin":
            # Fallback to basic method
            try:
                result = subprocess.run(['osx-cpu-temp'], capture_output=True, text=True)
                if result.returncode == 0:
                    match = re.search(r'([0-9.]+)', result.stdout)
                    if match:
                        temps['cpu'] = float(match.group(1))
            except:
                pass
        else:
            # Linux - use psutil
            try:
                sensor_temps = psutil.sensors_temperatures()
                for name, entries in sensor_temps.items():
                    for entry in entries:
                        if 'cpu' in entry.label.lower():
                            temps['cpu'] = entry.current
                        elif 'gpu' in entry.label.lower():
                            temps['gpu'] = entry.current
            except:
                pass
        
        return temps
    
    def _get_core_type_usage(self) -> Tuple[Optional[float], Optional[float]]:
        """Get usage for efficiency and performance cores (Apple Silicon)."""
        if platform.system() != "Darwin" or platform.processor() != 'arm':
            return None, None
        
        try:
            # Get per-core usage
            per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # On Apple Silicon, efficiency cores are typically the first 4
            # and performance cores are the rest (this varies by chip)
            # M1: 4 efficiency, 4 performance
            # M1 Pro/Max: 2 efficiency, 8-10 performance
            # This is a heuristic - actual core detection would require system_profiler
            
            num_cores = len(per_core)
            if num_cores >= 8:
                # Assume first 2-4 are efficiency cores
                efficiency_cores = per_core[:4] if num_cores > 8 else per_core[:2]
                performance_cores = per_core[4:] if num_cores > 8 else per_core[2:]
                
                efficiency_percent = statistics.mean(efficiency_cores) if efficiency_cores else None
                performance_percent = statistics.mean(performance_cores) if performance_cores else None
                
                return efficiency_percent, performance_percent
        except:
            pass
        
        return None, None
    
    def _is_cpu_throttled(self) -> bool:
        """Check if CPU is being throttled."""
        if platform.system() == "Darwin":
            try:
                # Check thermal state
                result = subprocess.run(
                    ['pmset', '-g', 'thermlog'],
                    capture_output=True, text=True
                )
                if result.returncode == 0 and 'CPU Power' in result.stdout:
                    # Look for reduced CPU power indicators
                    return 'reduced' in result.stdout.lower()
            except:
                pass
        
        # Check if current frequency is significantly below max
        try:
            freq = psutil.cpu_freq()
            if freq and freq.max > 0:
                # If running at less than 80% of max frequency, might be throttled
                return (freq.current / freq.max) < 0.8
        except:
            pass
        
        return False
    
    def _get_macos_memory_stats(self) -> Tuple[Optional[float], Optional[float]]:
        """Get macOS-specific memory stats (wired, compressed)."""
        if platform.system() != "Darwin":
            return None, None
        
        if _get_macos_monitor():
            # Use macOS helper for detailed memory stats
            stats = _get_macos_monitor().get_memory_details()
            return stats.get('wired_mb'), stats.get('compressed_mb')
        
        # Fallback to basic method
        try:
            result = subprocess.run(['vm_stat'], capture_output=True, text=True)
            if result.returncode == 0:
                wired_mb = None
                compressed_mb = None
                
                for line in result.stdout.split('\n'):
                    if 'Pages wired down' in line:
                        match = re.search(r'([0-9]+)', line)
                        if match:
                            wired_mb = int(match.group(1)) * 4096 / (1024 * 1024)
                    elif 'Pages occupied by compressor' in line:
                        match = re.search(r'([0-9]+)', line)
                        if match:
                            compressed_mb = int(match.group(1)) * 4096 / (1024 * 1024)
                
                return wired_mb, compressed_mb
        except:
            pass
        
        return None, None
    
    def _get_memory_pressure(self) -> Optional[str]:
        """Get memory pressure state."""
        if platform.system() == "Darwin":
            try:
                # Check memory pressure
                memory = psutil.virtual_memory()
                if memory.percent > 90:
                    return "red"
                elif memory.percent > 75:
                    return "yellow"
                else:
                    return "green"
            except:
                pass
        
        return None
    
    def _get_gpu_metrics(self) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
        """Get GPU utilization and Metal usage."""
        gpu_util = None
        gpu_mem_used = None
        gpu_mem_total = None
        metal_in_use = False
        
        if platform.system() == "Darwin" and _get_macos_monitor():
            # Use macOS helper for GPU metrics
            gpu_info = _get_macos_monitor().get_gpu_metal_usage()
            if gpu_info:
                gpu_util = gpu_info.get('utilization_percent')
                gpu_mem_used = gpu_info.get('vram_used_mb')
                metal_in_use = gpu_info.get('metal_active', False)
        
        if platform.system() == "Darwin" and not metal_in_use:
            # Fallback: Check if Metal is being used by our processes
            try:
                for proc in psutil.process_iter(['pid', 'name']):
                    if any(name in proc.info['name'].lower() for name in self._metal_process_names):
                        # Simple check for GPU activity
                        result = subprocess.run(
                            ['ioreg', '-r', '-c', 'IOAccelerator'],
                            capture_output=True, text=True
                        )
                        if result.returncode == 0 and 'PerformanceStatistics' in result.stdout:
                            metal_in_use = True
                            break
            except:
                pass
        
        return gpu_util, gpu_mem_used, gpu_mem_total, metal_in_use
    
    def _get_fan_speed(self) -> Optional[int]:
        """Get fan speed in RPM."""
        if platform.system() == "Darwin" and self._has_ioreg:
            try:
                result = subprocess.run(
                    ['ioreg', '-c', 'IOPlatformExpertDevice', '-r'],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    # This would require SMC access for accurate fan speeds
                    # Placeholder for now
                    pass
            except:
                pass
        
        return None
    
    def _get_thermal_pressure(self) -> Optional[str]:
        """Get thermal pressure state."""
        if platform.system() == "Darwin" and _get_macos_monitor():
            # Use macOS helper for thermal pressure
            pressure = _get_macos_monitor().get_thermal_pressure()
            if pressure:
                return pressure
        
        # Fallback to basic check
        if platform.system() == "Darwin":
            try:
                # Check if CPU is throttled based on frequency
                freq = psutil.cpu_freq()
                if freq and freq.max > 0:
                    ratio = freq.current / freq.max
                    if ratio > 0.9:
                        return "nominal"
                    elif ratio > 0.7:
                        return "fair"
                    elif ratio > 0.5:
                        return "serious"
                    else:
                        return "critical"
            except:
                pass
        
        return None
    
    def _get_power_metrics(self) -> Dict[str, float]:
        """Get detailed power consumption metrics."""
        metrics = {}
        
        if platform.system() == "Darwin" and _get_macos_monitor():
            # Try to get power metrics without sudo first
            power_data = _get_macos_monitor().get_power_metrics_without_sudo()
            if 'power_draw_watts' in power_data:
                metrics['watts'] = power_data['power_draw_watts']
                metrics['milliamps'] = power_data.get('battery_amperage_ma', metrics['watts'] * 1000 / 12)
            elif 'estimated_power_watts' in power_data:
                metrics['watts'] = power_data['estimated_power_watts']
                metrics['milliamps'] = metrics['watts'] * 1000 / 12
        
        # Fallback to estimation if no real metrics
        if 'watts' not in metrics:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            # Rough estimation for M-series Macs
            base_watts = 5.0  # Base consumption
            max_additional_watts = 25.0  # Max additional under load
            watts = base_watts + (cpu_percent / 100.0) * max_additional_watts
            metrics['watts'] = watts
            metrics['milliamps'] = watts * 1000 / 12
        
        return metrics
    
    def _get_battery_info(self) -> Dict[str, Any]:
        """Get battery information."""
        info = {}
        
        try:
            battery = psutil.sensors_battery()
            if battery:
                info['level'] = battery.percent
                info['plugged'] = battery.power_plugged
                info['time_remaining'] = battery.secsleft // 60 if battery.secsleft > 0 else None
                
                # Calculate drain rate (rough estimation)
                if not battery.power_plugged and info['time_remaining']:
                    # Estimate based on remaining time and current level
                    hours_remaining = info['time_remaining'] / 60
                    if hours_remaining > 0:
                        # Assume 50Wh battery capacity (typical for laptops)
                        battery_capacity_wh = 50
                        current_wh = battery_capacity_wh * (battery.percent / 100)
                        info['drain_rate'] = current_wh / hours_remaining
        except:
            pass
        
        return info
    
    def _get_disk_io_metrics(self) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        """Get disk I/O metrics."""
        try:
            current_io = psutil.disk_io_counters()
            current_time = time.time()
            
            if self._last_disk_io and self._last_io_time:
                time_delta = current_time - self._last_io_time
                if time_delta > 0:
                    read_bytes_delta = current_io.read_bytes - self._last_disk_io.read_bytes
                    write_bytes_delta = current_io.write_bytes - self._last_disk_io.write_bytes
                    
                    read_mb_per_sec = (read_bytes_delta / (1024 * 1024)) / time_delta
                    write_mb_per_sec = (write_bytes_delta / (1024 * 1024)) / time_delta
                    
                    # Estimate queue depth based on busy time
                    queue_depth = None
                    if hasattr(current_io, 'busy_time') and hasattr(self._last_disk_io, 'busy_time'):
                        busy_delta = current_io.busy_time - self._last_disk_io.busy_time
                        queue_depth = int(busy_delta / time_delta * 10)  # Rough estimate
                    
                    self._last_disk_io = current_io
                    self._last_io_time = current_time
                    
                    return read_mb_per_sec, write_mb_per_sec, queue_depth
            
            self._last_disk_io = current_io
            self._last_io_time = current_time
        except:
            pass
        
        return None, None, None
    
    def _get_network_io_metrics(self) -> Tuple[Optional[float], Optional[float]]:
        """Get network I/O metrics."""
        try:
            current_io = psutil.net_io_counters()
            current_time = time.time()
            
            if self._last_network_io and self._last_io_time:
                time_delta = current_time - self._last_io_time
                if time_delta > 0:
                    sent_delta = current_io.bytes_sent - self._last_network_io.bytes_sent
                    recv_delta = current_io.bytes_recv - self._last_network_io.bytes_recv
                    
                    sent_per_sec = sent_delta / time_delta
                    recv_per_sec = recv_delta / time_delta
                    
                    self._last_network_io = current_io
                    
                    return sent_per_sec, recv_per_sec
            
            self._last_network_io = current_io
        except:
            pass
        
        return None, None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of collected metrics.
        
        Returns:
            Dictionary with aggregated metrics
        """
        if not self.metrics_history:
            return {
                "samples": 0,
                "duration_seconds": 0,
            }
        
        duration = (
            self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp
        ).total_seconds()
        
        summary = {
            "samples": len(self.metrics_history),
            "duration_seconds": duration,
        }
        
        # Helper function to calculate stats
        def calc_stats(values: List[float], name: str) -> Dict[str, float]:
            if not values:
                return {}
            return {
                f"avg_{name}": statistics.mean(values),
                f"max_{name}": max(values),
                f"min_{name}": min(values),
                f"stddev_{name}": statistics.stdev(values) if len(values) > 1 else 0,
            }
        
        # CPU metrics
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        summary["cpu"] = calc_stats(cpu_values, "percent")
        
        # Per-core CPU
        if self.metrics_history[0].cpu_percent_per_core:
            num_cores = len(self.metrics_history[0].cpu_percent_per_core)
            for core_idx in range(num_cores):
                core_values = [m.cpu_percent_per_core[core_idx] for m in self.metrics_history 
                              if len(m.cpu_percent_per_core) > core_idx]
                summary[f"cpu_core_{core_idx}"] = calc_stats(core_values, "percent")
        
        # Efficiency vs Performance cores
        eff_values = [m.cpu_efficiency_cores_percent for m in self.metrics_history 
                     if m.cpu_efficiency_cores_percent is not None]
        perf_values = [m.cpu_performance_cores_percent for m in self.metrics_history 
                      if m.cpu_performance_cores_percent is not None]
        if eff_values:
            summary["cpu_efficiency_cores"] = calc_stats(eff_values, "percent")
        if perf_values:
            summary["cpu_performance_cores"] = calc_stats(perf_values, "percent")
        
        # CPU frequency and throttling
        freq_values = [m.cpu_frequency_current for m in self.metrics_history 
                      if m.cpu_frequency_current is not None]
        if freq_values:
            summary["cpu_frequency"] = calc_stats(freq_values, "mhz")
        
        throttled_count = sum(1 for m in self.metrics_history if m.cpu_throttled)
        summary["cpu_throttled_percent"] = (throttled_count / len(self.metrics_history)) * 100
        
        # Memory metrics
        memory_values = [m.memory_used_mb for m in self.metrics_history]
        summary["memory"] = calc_stats(memory_values, "used_mb")
        
        # Memory pressure
        pressure_counts = {"green": 0, "yellow": 0, "red": 0}
        for m in self.metrics_history:
            if m.memory_pressure:
                pressure_counts[m.memory_pressure] += 1
        summary["memory_pressure"] = pressure_counts
        
        # GPU metrics
        gpu_util_values = [m.gpu_utilization_percent for m in self.metrics_history 
                          if m.gpu_utilization_percent is not None]
        if gpu_util_values:
            summary["gpu"] = calc_stats(gpu_util_values, "utilization_percent")
        
        metal_usage_count = sum(1 for m in self.metrics_history if m.metal_in_use)
        summary["metal_usage_percent"] = (metal_usage_count / len(self.metrics_history)) * 100
        
        # Temperature metrics
        cpu_temp_values = [m.temperature_cpu for m in self.metrics_history 
                          if m.temperature_cpu is not None]
        gpu_temp_values = [m.temperature_gpu for m in self.metrics_history 
                          if m.temperature_gpu is not None]
        if cpu_temp_values:
            summary["temperature_cpu"] = calc_stats(cpu_temp_values, "celsius")
        if gpu_temp_values:
            summary["temperature_gpu"] = calc_stats(gpu_temp_values, "celsius")
        
        # Thermal pressure
        thermal_counts = {"nominal": 0, "fair": 0, "serious": 0, "critical": 0}
        for m in self.metrics_history:
            if m.thermal_pressure:
                thermal_counts[m.thermal_pressure] += 1
        summary["thermal_pressure"] = thermal_counts
        
        # Power metrics
        power_w_values = [m.power_consumption_watts for m in self.metrics_history 
                         if m.power_consumption_watts is not None]
        if power_w_values:
            summary["power"] = calc_stats(power_w_values, "watts")
            summary["power"]["total_wh"] = sum(power_w_values) * self.interval / 3600
            summary["power"]["avg_watts_per_token"] = None  # Will be calculated with token data
        
        # Battery metrics
        battery_values = [m.battery_level_percent for m in self.metrics_history 
                         if m.battery_level_percent is not None]
        drain_values = [m.battery_drain_rate_watts for m in self.metrics_history 
                       if m.battery_drain_rate_watts is not None]
        if battery_values:
            summary["battery"] = {
                "start_percent": battery_values[0],
                "end_percent": battery_values[-1],
                "drain_percent": battery_values[0] - battery_values[-1],
            }
        if drain_values:
            summary["battery"]["avg_drain_watts"] = statistics.mean(drain_values)
        
        # I/O metrics
        disk_read_values = [m.disk_read_mb_per_sec for m in self.metrics_history 
                           if m.disk_read_mb_per_sec is not None]
        disk_write_values = [m.disk_write_mb_per_sec for m in self.metrics_history 
                            if m.disk_write_mb_per_sec is not None]
        if disk_read_values:
            summary["disk_io"] = {
                **calc_stats(disk_read_values, "read_mb_per_sec"),
                **calc_stats(disk_write_values, "write_mb_per_sec"),
            }
        
        return summary
    
    def get_latest_metrics(self, n: int = 10) -> List[SystemMetrics]:
        """Get the latest n metrics.
        
        Args:
            n: Number of metrics to return
            
        Returns:
            List of recent metrics
        """
        return self.metrics_history[-n:] if self.metrics_history else []
    
    def clear_history(self) -> None:
        """Clear metrics history."""
        self.metrics_history = []
    
    def export_metrics(self) -> List[Dict[str, Any]]:
        """Export all metrics as list of dictionaries.
        
        Returns:
            List of metric dictionaries
        """
        return [m.to_dict() for m in self.metrics_history]