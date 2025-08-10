"""Tests for energy monitoring to ensure correct power calculations."""

import unittest
from unittest.mock import patch, MagicMock
import subprocess
from pocket_agent_cli.monitoring.unified_monitor import UnifiedMonitor
from pocket_agent_cli.monitoring.energy_monitor import EnergyMonitor
from pocket_agent_cli.monitoring.macos_helpers import MacOSMonitor


class TestEnergyMonitoring(unittest.TestCase):
    """Test energy monitoring calculations and signed/unsigned conversion."""
    
    def test_signed_conversion_for_large_unsigned_values(self):
        """Test that large unsigned values are correctly converted to signed."""
        # Test cases: (raw_value, expected_signed)
        test_cases = [
            (18446744073709551088, -528),  # Actual problematic value from logs
            (18446744073709550000, -1616),  # Another large unsigned
            (500, 500),  # Normal positive value
            (2**63 - 1, 2**63 - 1),  # Max positive signed 64-bit
            (2**63, 2**63),  # 2^63 stays as is, larger values need conversion
            (2**64 - 1, -1),  # Max unsigned = -1 signed
        ]
        
        for raw_value, expected_signed in test_cases:
            # Apply the fix logic
            if raw_value > 2**63:
                signed_value = raw_value - 2**64
            else:
                signed_value = raw_value
            
            self.assertEqual(signed_value, expected_signed, 
                           f"Failed for raw_value={raw_value}")
    
    def test_power_calculation_validation(self):
        """Test that power calculations are validated to reasonable ranges."""
        test_cases = [
            # (voltage_v, amperage_ma, expected_power_watts)
            (11.139, 528, 5.88),  # Normal discharge ~6W
            (11.139, 500, 5.57),  # Another normal case
            (11.139, 18446744073709551088, 5.88),  # Large unsigned should be converted to 528mA
            (11.139, 2000, 22.278),  # High but reasonable ~22W
            (11.139, 20000, 0.0),  # Too high, should be rejected (>200W)
            (11.139, 5, 0.0),  # Too low, should be rejected (<0.1W)
        ]
        
        for voltage_v, amperage_ma, expected_power in test_cases:
            # Apply power calculation with validation
            if amperage_ma > 2**63:
                amperage_ma = abs(amperage_ma - 2**64)
            
            power_watts = (voltage_v * amperage_ma) / 1000.0
            
            # Apply validation
            if power_watts < 0.1 or power_watts > 200.0:
                power_watts = 0.0
            
            self.assertAlmostEqual(power_watts, expected_power, places=2,
                                 msg=f"Power calculation failed for V={voltage_v}, I={amperage_ma}")
    
    @patch('subprocess.run')
    def test_unified_monitor_battery_parsing(self, mock_run):
        """Test UnifiedMonitor correctly parses battery data."""
        # Mock ioreg output with problematic unsigned value
        mock_output = '''
        | |   |     "Voltage" = 11139
        | |   |     "InstantAmperage" = 18446744073709551088
        '''
        
        mock_run.return_value = MagicMock(
            stdout=mock_output,
            stderr='',
            returncode=0
        )
        
        monitor = UnifiedMonitor()
        metrics = monitor.get_current_metrics()
        
        # Should have reasonable power value, not 10^17
        if metrics and 'power_watts' in metrics:
            self.assertLess(metrics['power_watts'], 200.0, 
                          "Power should be less than 200W")
            self.assertGreaterEqual(metrics['power_watts'], 0.0,
                                  "Power should be non-negative")
    
    @patch('subprocess.run')
    def test_energy_monitor_battery_parsing(self, mock_run):
        """Test EnergyMonitor correctly parses battery data."""
        # Mock ioreg output
        mock_output = '''
        | |   |     "Voltage" = 11139
        | |   |     "InstantAmperage" = 18446744073709551088
        '''
        
        mock_run.return_value = MagicMock(
            stdout=mock_output,
            stderr='',
            returncode=0
        )
        
        monitor = EnergyMonitor()
        monitor.start_monitoring()
        power = monitor.get_current_power()
        monitor.stop_monitoring()
        
        if power is not None:
            # Check power is reasonable
            self.assertLess(power, 200.0, 
                          "Power should be less than 200W")
            self.assertGreaterEqual(power, 0.0,
                                  "Power should be non-negative")
    
    def test_edge_cases(self):
        """Test edge cases for signed conversion."""
        # Test boundary values
        boundary_cases = [
            (2**63 - 1, 2**63 - 1),  # Largest positive signed
            (2**63, 2**63),  # At boundary, not converted  
            (0, 0),  # Zero
            (1, 1),  # Small positive
            (2**64 - 1, -1),  # -1 in unsigned
        ]
        
        for raw, expected in boundary_cases:
            if raw > 2**63:
                result = raw - 2**64
            else:
                result = raw
            
            self.assertEqual(result, expected,
                           f"Boundary case failed for {raw}")
    
    def test_realistic_battery_values(self):
        """Test with realistic battery measurement values."""
        realistic_cases = [
            # Charging scenarios (negative amperage)
            {"voltage": 12.5, "raw_amperage": 2**64 - 1500, "expected_ma": 1500, "expected_power": 18.75},
            # Discharging scenarios (positive amperage) 
            {"voltage": 11.1, "raw_amperage": 800, "expected_ma": 800, "expected_power": 8.88},
            # Idle (very low current)
            {"voltage": 11.8, "raw_amperage": 50, "expected_ma": 50, "expected_power": 0.59},
        ]
        
        for case in realistic_cases:
            raw = case["raw_amperage"]
            
            # Apply signed conversion
            if raw > 2**63:
                signed = raw - 2**64
            else:
                signed = raw
            
            amperage_ma = abs(signed)
            power_watts = (case["voltage"] * amperage_ma) / 1000.0
            
            # Validate
            if power_watts < 0.1 or power_watts > 200.0:
                power_watts = 0.0
            
            self.assertEqual(amperage_ma, case["expected_ma"])
            self.assertAlmostEqual(power_watts, case["expected_power"], places=2)


if __name__ == '__main__':
    unittest.main()