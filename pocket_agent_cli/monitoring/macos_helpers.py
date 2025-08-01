"""macOS-specific monitoring helpers."""

import subprocess
import re
import json
from typing import Dict, Any, Optional, Tuple
import os


class MacOSMonitor:
    """Helper class for macOS-specific monitoring."""
    
    def __init__(self):
        self._has_osx_cpu_temp = self._check_command('osx-cpu-temp')
        self._has_powermetrics = self._check_command('powermetrics')
        self._has_ioreg = self._check_command('ioreg')
        self._has_istats = self._check_command('istats')
        self._has_smctemp = self._check_command('smctemp')
        
    def _check_command(self, command: str) -> bool:
        """Check if a command is available."""
        try:
            result = subprocess.run(['which', command], capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    def get_temperature(self) -> Dict[str, float]:
        """Get temperature readings using available methods."""
        temps = {}
        
        # Method 1: smctemp (best for M1/M2 Macs)
        if self._has_smctemp:
            try:
                # Get CPU temperature with M2 Mac options
                result = subprocess.run(['smctemp', '-c', '-i25', '-n180', '-f'], 
                                      capture_output=True, text=True, timeout=5.0)
                if result.returncode == 0 and result.stdout.strip():
                    try:
                        # The output should be a single temperature value
                        temp = float(result.stdout.strip())
                        if temp > 0:  # Valid temperature
                            temps['cpu'] = temp
                    except ValueError:
                        # If not a simple float, try regex
                        match = re.search(r'([0-9.]+)', result.stdout.strip())
                        if match:
                            temp = float(match.group(1))
                            if temp > 0:
                                temps['cpu'] = temp
                
                # Get GPU temperature with M2 Mac options
                result = subprocess.run(['smctemp', '-g', '-i25', '-n180', '-f'], 
                                      capture_output=True, text=True, timeout=5.0)
                if result.returncode == 0 and result.stdout.strip():
                    try:
                        # The output should be a single temperature value
                        temp = float(result.stdout.strip())
                        if temp > 0:  # Valid temperature
                            temps['gpu'] = temp
                    except ValueError:
                        # If not a simple float, try regex
                        match = re.search(r'([0-9.]+)', result.stdout.strip())
                        if match:
                            temp = float(match.group(1))
                            if temp > 0:
                                temps['gpu'] = temp
            except:
                pass
        
        # Method 2: osx-cpu-temp (fallback)
        if self._has_osx_cpu_temp and 'cpu' not in temps:
            try:
                result = subprocess.run(['osx-cpu-temp'], capture_output=True, text=True)
                if result.returncode == 0:
                    # Parse "58.2°C"
                    match = re.search(r'([0-9.]+)', result.stdout)
                    if match:
                        temp = float(match.group(1))
                        if temp > 0:  # Ignore 0.0°C readings
                            temps['cpu'] = temp
            except:
                pass
        
        # Method 3: iStats (if installed via gem)
        if self._has_istats:
            try:
                result = subprocess.run(['istats'], capture_output=True, text=True)
                if result.returncode == 0:
                    # Parse iStats output
                    for line in result.stdout.split('\n'):
                        if 'CPU temp' in line:
                            match = re.search(r'([0-9.]+)', line)
                            if match:
                                temps['cpu'] = float(match.group(1))
                        elif 'GPU temp' in line:
                            match = re.search(r'([0-9.]+)', line)
                            if match:
                                temps['gpu'] = float(match.group(1))
            except:
                pass
        
        # Method 3: SMC reading via ioreg (more complex but no extra tools needed)
        if self._has_ioreg and not temps:
            temps.update(self._read_smc_temps())
        
        return temps
    
    def _read_smc_temps(self) -> Dict[str, float]:
        """Read temperatures from SMC using ioreg."""
        temps = {}
        
        # This is a simplified approach - full SMC reading requires more complex code
        # For now, we'll try to get basic thermal zone info
        try:
            result = subprocess.run(
                ['ioreg', '-r', '-n', 'AppleSMC'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                # Parse SMC data - this is platform specific
                # Would need proper SMC key decoding
                pass
        except:
            pass
        
        return temps
    
    def get_memory_details(self) -> Dict[str, float]:
        """Get detailed memory statistics using vm_stat."""
        stats = {}
        
        try:
            result = subprocess.run(['vm_stat'], capture_output=True, text=True)
            if result.returncode == 0:
                # Page size is 4096 bytes on macOS
                page_size = 4096
                
                for line in result.stdout.split('\n'):
                    if ':' in line:
                        parts = line.split(':')
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value_str = parts[1].strip().rstrip('.')
                            
                            # Extract number
                            match = re.search(r'(\d+)', value_str)
                            if match:
                                value = int(match.group(1))
                                
                                # Convert pages to MB
                                value_mb = (value * page_size) / (1024 * 1024)
                                
                                if 'wired down' in key:
                                    stats['wired_mb'] = value_mb
                                elif 'occupied by compressor' in key:
                                    stats['compressed_mb'] = value_mb
                                elif 'File-backed pages' in key:
                                    stats['file_backed_mb'] = value_mb
                                elif 'Anonymous pages' in key:
                                    stats['anonymous_mb'] = value_mb
                                elif 'Purgeable pages' in key:
                                    stats['purgeable_mb'] = value_mb
                                elif 'Swapins' in key:
                                    stats['swapins'] = value
                                elif 'Swapouts' in key:
                                    stats['swapouts'] = value
        except:
            pass
        
        return stats
    
    def get_power_metrics_without_sudo(self) -> Dict[str, Any]:
        """Get power metrics without requiring sudo."""
        metrics = {}
        
        # Method 1: Read from IORegistry (no sudo required)
        if self._has_ioreg:
            try:
                # Get battery info
                result = subprocess.run(
                    ['ioreg', '-r', '-c', 'AppleSmartBattery'],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    # Parse battery data
                    for line in result.stdout.split('\n'):
                        if 'CurrentCapacity' in line:
                            match = re.search(r'= (\d+)', line)
                            if match:
                                metrics['battery_current_capacity'] = int(match.group(1))
                        elif 'MaxCapacity' in line:
                            match = re.search(r'= (\d+)', line)
                            if match:
                                metrics['battery_max_capacity'] = int(match.group(1))
                        elif 'InstantAmperage' in line:
                            match = re.search(r'= (-?\d+)', line)
                            if match:
                                raw_amperage = int(match.group(1))
                                # Handle 64-bit unsigned to signed conversion
                                if raw_amperage > 2**63:
                                    raw_amperage = raw_amperage - 2**64
                                # Convert to positive for drain rate
                                amperage = abs(raw_amperage)
                                metrics['battery_amperage_ma'] = amperage
                        elif 'Voltage' in line and 'Voltage' not in metrics:
                            match = re.search(r'= (\d+)', line)
                            if match:
                                voltage_mv = int(match.group(1))
                                metrics['battery_voltage_v'] = voltage_mv / 1000.0
                
                # Calculate power from amperage and voltage
                if 'battery_amperage_ma' in metrics and 'battery_voltage_v' in metrics:
                    watts = (metrics['battery_amperage_ma'] / 1000.0) * metrics['battery_voltage_v']
                    metrics['power_draw_watts'] = watts
            except:
                pass
        
        # Method 2: Activity Monitor data (less accurate but available)
        try:
            # Get process info for rough power estimation
            result = subprocess.run(
                ['ps', '-A', '-o', 'pid,%cpu,comm'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                total_cpu = 0
                for line in result.stdout.split('\n')[1:]:  # Skip header
                    parts = line.split(None, 2)
                    if len(parts) >= 2:
                        try:
                            cpu_percent = float(parts[1])
                            total_cpu += cpu_percent
                        except:
                            pass
                
                # Very rough power estimation based on CPU usage
                # Assuming 5W idle, 30W max for M-series chips
                base_power = 5.0
                max_additional = 25.0
                estimated_power = base_power + (total_cpu / 100.0) * max_additional
                
                if 'power_draw_watts' not in metrics:
                    metrics['estimated_power_watts'] = estimated_power
        except:
            pass
        
        return metrics
    
    def get_thermal_pressure(self) -> Optional[str]:
        """Get thermal pressure without sudo."""
        # Check if system is thermally throttled
        try:
            # Use sysctl to check thermal state
            result = subprocess.run(
                ['sysctl', 'machdep.xcpm.cpu_thermal_level'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                match = re.search(r'= (\d+)', result.stdout)
                if match:
                    level = int(match.group(1))
                    if level == 0:
                        return "nominal"
                    elif level <= 30:
                        return "fair"
                    elif level <= 70:
                        return "serious"
                    else:
                        return "critical"
        except:
            pass
        
        # Alternative: check CPU frequency throttling
        try:
            result = subprocess.run(
                ['sysctl', 'hw.cpufrequency', 'hw.cpufrequency_max'],
                capture_output=True, text=True
            )
            # Parse and compare frequencies
        except:
            pass
        
        return None
    
    def get_gpu_metal_usage(self) -> Dict[str, Any]:
        """Check Metal GPU usage without sudo."""
        gpu_info = {}
        
        try:
            # Check GPU usage via ioreg
            result = subprocess.run(
                ['ioreg', '-r', '-c', 'IOAccelerator'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                # Look for performance statistics
                if 'PerformanceStatistics' in result.stdout:
                    gpu_info['metal_active'] = True
                    
                    # Try to extract utilization if available
                    lines = result.stdout.split('\n')
                    for i, line in enumerate(lines):
                        if 'Device Utilization' in line and i + 1 < len(lines):
                            match = re.search(r'(\d+)%', lines[i + 1])
                            if match:
                                gpu_info['utilization_percent'] = int(match.group(1))
                        elif 'VRAM' in line:
                            # Extract VRAM usage if available
                            match = re.search(r'(\d+) MB', line)
                            if match:
                                gpu_info['vram_used_mb'] = int(match.group(1))
        except:
            pass
        
        return gpu_info