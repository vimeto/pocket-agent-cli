"""Energy monitoring specifically for inference measurements."""

import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import platform
import subprocess
import re
import statistics


@dataclass
class EnergyMetrics:
    """Energy consumption metrics during inference."""
    timestamp: datetime
    power_watts: float
    energy_joules: float
    battery_drain_mah: Optional[float] = None
    gpu_power_watts: Optional[float] = None
    cpu_power_watts: Optional[float] = None
    dram_power_watts: Optional[float] = None
    other_power_watts: Optional[float] = None
    temperature_c: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "power_watts": self.power_watts,
            "energy_joules": self.energy_joules,
            "battery_drain_mah": self.battery_drain_mah,
            "gpu_power_watts": self.gpu_power_watts,
            "cpu_power_watts": self.cpu_power_watts,
            "dram_power_watts": self.dram_power_watts,
            "other_power_watts": self.other_power_watts,
            "temperature_c": self.temperature_c,
        }


class EnergyMonitor:
    """Monitor energy consumption during inference."""
    
    def __init__(self, sample_interval: float = 0.1):
        """Initialize energy monitor.
        
        Args:
            sample_interval: Sampling interval in seconds (default 0.1s for 10Hz)
        """
        self.sample_interval = sample_interval
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.energy_samples: List[EnergyMetrics] = []
        self._start_time: Optional[float] = None
        self._total_energy: float = 0.0
        
        # Platform detection
        self.platform = platform.system()
        self.is_apple_silicon = (
            self.platform == "Darwin" and 
            platform.processor() == "arm"
        )
        
        # Check available tools
        self._has_powermetrics = self._check_command("powermetrics")
        self._has_ioreg = self._check_command("ioreg")
        self._powermetrics_sudo = self._check_sudo_powermetrics()
        
    def _check_command(self, command: str) -> bool:
        """Check if command is available."""
        try:
            result = subprocess.run(
                ["which", command], 
                capture_output=True
            )
            return result.returncode == 0
        except:
            return False
    
    def _check_sudo_powermetrics(self) -> bool:
        """Check if we can run powermetrics with sudo."""
        if not self._has_powermetrics:
            return False
        try:
            # Check if we have passwordless sudo for powermetrics
            result = subprocess.run(
                ["sudo", "-n", "powermetrics", "--help"],
                capture_output=True,
                timeout=2.0
            )
            return result.returncode == 0
        except:
            return False
    
    def start_monitoring(self) -> None:
        """Start energy monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.energy_samples = []
        self._start_time = time.time()
        self._total_energy = 0.0
        
        if self._powermetrics_sudo:
            # Use powermetrics for accurate measurement
            self.monitor_thread = threading.Thread(
                target=self._monitor_with_powermetrics,
                daemon=True
            )
        else:
            # Fallback to IORegistry-based monitoring
            self.monitor_thread = threading.Thread(
                target=self._monitor_with_ioreg,
                daemon=True
            )
        
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return energy summary."""
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=3.0)
            self.monitor_thread = None
        
        return self._calculate_summary()
    
    def _monitor_with_powermetrics(self) -> None:
        """Monitor using powermetrics (most accurate)."""
        cmd = [
            "sudo", "-n", "powermetrics",
            "--samplers", "cpu_power,gpu_power,thermal",
            "--sample-rate", str(int(1000 * self.sample_interval)),  # Convert to ms
            "--format", "plist"
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            import plistlib
            buffer = ""
            
            while self.is_monitoring and process.poll() is None:
                line = process.stdout.readline()
                buffer += line
                
                # Check if we have a complete plist
                if "</plist>" in buffer:
                    try:
                        # Parse the plist data
                        plist_data = buffer.encode('utf-8')
                        data = plistlib.loads(plist_data)
                        
                        # Extract power metrics
                        metrics = self._parse_powermetrics_data(data)
                        if metrics:
                            self.energy_samples.append(metrics)
                            self._total_energy += metrics.power_watts * self.sample_interval
                        
                        buffer = ""
                    except:
                        # Continue collecting
                        pass
            
            process.terminate()
            
        except Exception as e:
            print(f"Powermetrics monitoring failed: {e}")
            # Fallback to ioreg method
            self._monitor_with_ioreg()
    
    def _parse_powermetrics_data(self, data: Dict) -> Optional[EnergyMetrics]:
        """Parse powermetrics plist data."""
        try:
            timestamp = datetime.now()
            
            # Extract processor metrics
            processor = data.get("processor", {})
            cpu_power = processor.get("package_watts", 0.0)
            gpu_power = processor.get("gpu_watts", 0.0)
            dram_power = processor.get("dram_watts", 0.0)
            
            # Total system power
            total_power = cpu_power + gpu_power + dram_power
            
            # Temperature from thermal data
            thermal = data.get("thermal_pressure", {})
            temperature = thermal.get("die_temperature_c")
            
            return EnergyMetrics(
                timestamp=timestamp,
                power_watts=total_power,
                energy_joules=self._total_energy,
                cpu_power_watts=cpu_power,
                gpu_power_watts=gpu_power,
                dram_power_watts=dram_power,
                temperature_c=temperature
            )
        except:
            return None
    
    def _monitor_with_ioreg(self) -> None:
        """Monitor using IORegistry (no sudo required)."""
        last_sample_time = time.time()
        
        while self.is_monitoring:
            current_time = time.time()
            
            # Get power metrics from IORegistry
            metrics = self._get_ioreg_power_metrics()
            
            if metrics:
                # Calculate energy since last sample
                time_delta = current_time - last_sample_time
                energy_delta = metrics.power_watts * time_delta
                self._total_energy += energy_delta
                
                metrics.energy_joules = self._total_energy
                self.energy_samples.append(metrics)
            
            last_sample_time = current_time
            
            # Sleep until next sample
            sleep_time = self.sample_interval - (time.time() - current_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _get_ioreg_power_metrics(self) -> Optional[EnergyMetrics]:
        """Get power metrics from IORegistry."""
        try:
            timestamp = datetime.now()
            power_watts = 0.0
            battery_ma = None
            temperature = None
            
            # Get battery information
            result = subprocess.run(
                ["ioreg", "-r", "-c", "AppleSmartBattery"],
                capture_output=True,
                text=True,
                timeout=1.0
            )
            
            if result.returncode == 0:
                # Parse battery metrics
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
                            battery_ma = amperage_ma
                
                # Calculate power from V * I
                if voltage_v and amperage_ma:
                    power_watts = (voltage_v * amperage_ma) / 1000.0
                    # Validate reasonable power range (0.1W to 200W for laptops)
                    if power_watts < 0.1 or power_watts > 200.0:
                        print(f"[WARNING] Invalid power calculated: {power_watts}W (V={voltage_v}, I={amperage_ma}mA)")
                        power_watts = 0.0  # Fall back to estimation
            
            # Try to get temperature
            temp_result = subprocess.run(
                ["ioreg", "-r", "-n", "PMU tdie"],
                capture_output=True,
                text=True,
                timeout=1.0
            )
            
            if temp_result.returncode == 0:
                match = re.search(r'"temperature" = (\d+)', temp_result.stdout)
                if match:
                    # Temperature is usually in centi-celsius
                    temperature = int(match.group(1)) / 100.0
            
            # If we couldn't get power from battery, estimate from CPU
            if power_watts == 0.0:
                power_watts = self._estimate_power_from_cpu()
            
            return EnergyMetrics(
                timestamp=timestamp,
                power_watts=power_watts,
                energy_joules=self._total_energy,
                battery_drain_mah=battery_ma,
                temperature_c=temperature
            )
            
        except:
            return None
    
    def _estimate_power_from_cpu(self) -> float:
        """Estimate power from CPU usage as last resort."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Rough estimation for Apple Silicon
            if self.is_apple_silicon:
                # M1/M2: ~5W idle, ~30W max
                base_power = 5.0
                max_additional = 25.0
            else:
                # Intel Mac: ~10W idle, ~60W max
                base_power = 10.0
                max_additional = 50.0
            
            return base_power + (cpu_percent / 100.0) * max_additional
            
        except:
            return 10.0  # Default fallback
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate energy consumption summary."""
        if not self.energy_samples:
            return {
                "total_energy_joules": 0.0,
                "total_energy_wh": 0.0,
                "avg_power_watts": 0.0,
                "duration_seconds": 0.0,
                "samples": 0
            }
        
        duration = time.time() - self._start_time
        
        # Calculate statistics
        power_values = [s.power_watts for s in self.energy_samples]
        avg_power = statistics.mean(power_values)
        max_power = max(power_values)
        min_power = min(power_values)
        
        # Battery drain
        battery_samples = [s.battery_drain_mah for s in self.energy_samples 
                          if s.battery_drain_mah is not None]
        avg_battery_ma = statistics.mean(battery_samples) if battery_samples else None
        
        # Temperature
        temp_samples = [s.temperature_c for s in self.energy_samples 
                       if s.temperature_c is not None]
        avg_temp = statistics.mean(temp_samples) if temp_samples else None
        max_temp = max(temp_samples) if temp_samples else None
        
        # Component power breakdown (if available)
        cpu_samples = [s.cpu_power_watts for s in self.energy_samples 
                      if s.cpu_power_watts is not None]
        gpu_samples = [s.gpu_power_watts for s in self.energy_samples 
                      if s.gpu_power_watts is not None]
        
        summary = {
            "total_energy_joules": self._total_energy,
            "total_energy_wh": self._total_energy / 3600.0,
            "avg_power_watts": avg_power,
            "max_power_watts": max_power,
            "min_power_watts": min_power,
            "duration_seconds": duration,
            "samples": len(self.energy_samples),
            "sample_rate_hz": len(self.energy_samples) / duration if duration > 0 else 0,
        }
        
        if avg_battery_ma:
            summary["avg_battery_drain_ma"] = avg_battery_ma
            summary["total_battery_drain_mah"] = avg_battery_ma * duration / 3600.0
        
        if avg_temp:
            summary["avg_temperature_c"] = avg_temp
            summary["max_temperature_c"] = max_temp
        
        if cpu_samples:
            summary["avg_cpu_power_watts"] = statistics.mean(cpu_samples)
        
        if gpu_samples:
            summary["avg_gpu_power_watts"] = statistics.mean(gpu_samples)
        
        return summary
    
    def get_current_power(self) -> Optional[float]:
        """Get current power consumption."""
        if self.energy_samples:
            return self.energy_samples[-1].power_watts
        return None
    
    def export_samples(self) -> List[Dict[str, Any]]:
        """Export all energy samples."""
        return [s.to_dict() for s in self.energy_samples]