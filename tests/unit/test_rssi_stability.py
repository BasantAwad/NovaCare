"""
RSSI Stability and Reliability Test Suite
=======================================
Validates the Kalman filter and trend detection of the RSSI Tracker
against simulated noisy RSSI data, imitating real-world BLE variations.
"""

import sys
import os
import unittest
import math
import random

# Add project root to path so we can import submodules directly
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_summon_path = os.path.join(_project_root, 'optimized_runtime', 'summon')
sys.path.insert(0, _summon_path)

from rssi_tracker import RSSITracker

class TestRSSIStability(unittest.TestCase):
    def setUp(self):
        self.tracker = RSSITracker()
        # Ensure repeatable tests
        random.seed(42)

    def generate_noisy_signal(self, base_rssi, noise_variance=5.0, count=20):
        """Generate a simulated noisy RSSI signal around a base value."""
        return [base_rssi + random.uniform(-noise_variance, noise_variance) for _ in range(count)]

    def test_kalman_noise_rejection(self):
        """Test that the Kalman filter correctly rejects high-frequency noise."""
        base = -60
        noisy_signals = self.generate_noisy_signal(base, noise_variance=10.0, count=50)
        
        for rssi in noisy_signals:
            self.tracker.update(int(rssi))
            
        # The filtered value should be very close to the base despite 10dBm noise
        self.assertAlmostEqual(self.tracker.current_rssi, base, delta=3.0)

    def test_trend_detection_with_noise(self):
        """Test trend detection works even with noisy signals."""
        # Simulate moving closer (RSSI increasing) with noise
        rssi_values = []
        for i in range(20):
            base = -80 + i  # Approaching: -80 to -61
            rssi_values.append(base + random.uniform(-4, 4))
            
        for rssi in rssi_values:
            self.tracker.update(int(rssi))
            
        # Should detect 'improving' trend
        self.assertEqual(self.tracker.get_trend(), "improving")

        self.tracker.reset()
        
        # Simulate moving away (RSSI decreasing) with noise
        rssi_values = []
        for i in range(20):
            base = -60 - i  # Leaving: -60 to -79
            rssi_values.append(base + random.uniform(-4, 4))
            
        for rssi in rssi_values:
            self.tracker.update(int(rssi))
            
        # Should detect 'degrading' trend
        self.assertEqual(self.tracker.get_trend(), "degrading")

    def test_spurious_drop_rejection(self):
        """Test that a single severe signal drop (common in BLE) doesn't ruin the estimate."""
        # Stable signal at -50
        for _ in range(15):
            self.tracker.update(-50)
            
        # Spurious drop to -90
        self.tracker.update(-90)
        
        # Filtered should NOT drop all the way to -90, should stay > -65
        self.assertGreater(self.tracker.current_rssi, -65)
        
        # Followed by return to -50
        for _ in range(10):
            self.tracker.update(-50)
            
        self.assertAlmostEqual(self.tracker.current_rssi, -50, delta=2.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
