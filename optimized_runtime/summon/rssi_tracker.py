"""
RSSI Tracker — BLE Signal Processing for User Localization
==========================================================

Processes raw BLE RSSI readings from the user's phone to:
1. Smooth noisy readings with a Kalman filter
2. Estimate relative direction (gradient sampling)
3. Detect signal trends (improving/degrading/stable)
4. Estimate approximate distance from RSSI

This is a pure-Python module with ZERO external dependencies
to ensure fast startup and minimal resource usage.

Typical RSSI ranges:
  -30 dBm : Very close (< 1m)
  -50 dBm : Close (1-3m)
  -70 dBm : Medium (3-8m)
  -85 dBm : Far (8-15m)
  -100 dBm: Very far / signal lost
"""

import time
import math
import logging
from typing import Optional, Tuple, List, Dict, Any
from collections import deque

logger = logging.getLogger(__name__)

# RSSI thresholds (dBm)
RSSI_ARRIVAL_THRESHOLD = -40      # Close enough to declare arrival
RSSI_NEAR_THRESHOLD = -55         # Entering near-target phase
RSSI_SIGNAL_LOST_THRESHOLD = -95  # Signal effectively lost
RSSI_MIN_SAMPLES_FOR_DIRECTION = 3

# Distance estimation constants (log-distance path loss model)
# RSSI = -(10 * n * log10(d) + A)
# where n = path loss exponent, A = RSSI at 1 meter
RSSI_AT_1M = -55       # Calibration: RSSI at exactly 1 meter
PATH_LOSS_EXPONENT = 2.7  # Indoor environment (2.0=free space, 3.0=obstacles)


class KalmanFilter1D:
    """
    Simple 1D Kalman filter for RSSI smoothing.

    Designed for single-variable filtering with minimal compute.
    ~0.01ms per update, <1KB RAM.
    """

    def __init__(
        self,
        process_variance: float = 1.0,
        measurement_variance: float = 10.0,
        initial_estimate: float = -70.0,
    ):
        self.process_variance = process_variance      # Q: how much we expect state to change
        self.measurement_variance = measurement_variance  # R: noise in measurements
        self.estimate = initial_estimate
        self.error_covariance = 1.0

    def update(self, measurement: float) -> float:
        """Process a new measurement and return filtered estimate."""
        # Prediction step
        predicted_estimate = self.estimate
        predicted_error = self.error_covariance + self.process_variance

        # Update step
        kalman_gain = predicted_error / (predicted_error + self.measurement_variance)
        self.estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        self.error_covariance = (1 - kalman_gain) * predicted_error

        return self.estimate

    def reset(self, initial_estimate: float = -70.0):
        """Reset filter state."""
        self.estimate = initial_estimate
        self.error_covariance = 1.0


class RSSITracker:
    """
    Tracks and processes BLE RSSI readings from the user's phone.

    Features:
    - Kalman-filtered RSSI for smooth readings
    - Trend detection (improving/degrading/stable)
    - Distance estimation from RSSI
    - Directional sampling (compare RSSI at different headings)
    - Signal loss detection
    """

    def __init__(self, window_size: int = 20):
        self._kalman = KalmanFilter1D()
        self._raw_history: deque = deque(maxlen=window_size)
        self._filtered_history: deque = deque(maxlen=window_size)
        self._directional_samples: Dict[float, List[float]] = {}
        self._last_update_time: float = 0.0
        self._trend_window: int = 5  # Samples to average for trend

    @property
    def current_rssi(self) -> int:
        """Latest filtered RSSI value."""
        if not self._filtered_history:
            return -100
        return int(round(self._filtered_history[-1]))

    @property
    def raw_rssi(self) -> int:
        """Latest raw RSSI value."""
        if not self._raw_history:
            return -100
        return int(self._raw_history[-1])

    @property
    def signal_lost(self) -> bool:
        """Whether signal appears to be lost."""
        if not self._filtered_history:
            return True
        return self.current_rssi <= RSSI_SIGNAL_LOST_THRESHOLD

    @property
    def is_near_target(self) -> bool:
        """Whether RSSI indicates proximity to target."""
        return self.current_rssi >= RSSI_NEAR_THRESHOLD

    @property
    def is_at_target(self) -> bool:
        """Whether RSSI indicates arrival at target."""
        return self.current_rssi >= RSSI_ARRIVAL_THRESHOLD

    @property
    def sample_count(self) -> int:
        """Number of samples collected."""
        return len(self._filtered_history)

    def update(self, raw_rssi: int, heading_deg: float = -1) -> float:
        """
        Process a new RSSI reading.

        Args:
            raw_rssi: Raw RSSI in dBm (negative integer)
            heading_deg: Current robot heading when this sample was taken.
                         Pass -1 if heading unknown.

        Returns:
            Filtered RSSI value.
        """
        now = time.monotonic()
        self._last_update_time = now

        self._raw_history.append(raw_rssi)
        filtered = self._kalman.update(float(raw_rssi))
        
        # Secondary moving average filter (last 5 samples) for high-frequency noise rejection
        window = list(self._raw_history)[-5:]
        moving_avg = sum(window) / len(window)
        
        # Optimal blend: 70% Kalman Filter, 30% Moving Average
        blended = 0.7 * filtered + 0.3 * moving_avg
        self._filtered_history.append(blended)

        # Store directional sample if heading is known
        if heading_deg >= 0:
            # Bucket headings into 45-degree sectors
            sector = round(heading_deg / 45) * 45 % 360
            if sector not in self._directional_samples:
                self._directional_samples[sector] = []
            self._directional_samples[sector].append(blended)
            # Keep only recent samples per sector
            if len(self._directional_samples[sector]) > 10:
                self._directional_samples[sector] = self._directional_samples[sector][-10:]

        return blended

    def get_trend(self) -> str:
        """
        Determine signal trend.

        Returns: "improving", "degrading", "stable", or "unknown"
        """
        if len(self._filtered_history) < self._trend_window * 2:
            return "unknown"

        history = list(self._filtered_history)
        recent = history[-self._trend_window:]
        older = history[-self._trend_window * 2:-self._trend_window]

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        delta = recent_avg - older_avg

        if delta > 3.0:
            return "improving"
        elif delta < -3.0:
            return "degrading"
        return "stable"

    def estimate_distance_m(self) -> float:
        """
        Estimate distance to phone from RSSI using log-distance path loss model.

        Returns approximate distance in meters, or -1 if unknown.
        Note: Indoor RSSI-to-distance is inherently imprecise (±30-50%).
        """
        rssi = self.current_rssi
        if rssi <= RSSI_SIGNAL_LOST_THRESHOLD:
            return -1.0

        try:
            distance = 10 ** ((RSSI_AT_1M - rssi) / (10 * PATH_LOSS_EXPONENT))
            return round(max(0.1, min(distance, 50.0)), 1)
        except (ValueError, OverflowError):
            return -1.0

    def get_best_heading(self) -> Optional[float]:
        """
        Determine the heading with the strongest RSSI.

        Returns the best heading in degrees (0-360), or None if
        insufficient directional data.
        """
        if len(self._directional_samples) < RSSI_MIN_SAMPLES_FOR_DIRECTION:
            return None

        best_sector = None
        best_rssi = -200.0

        for sector, samples in self._directional_samples.items():
            if len(samples) < 2:
                continue
            avg = sum(samples[-5:]) / min(len(samples), 5)
            if avg > best_rssi:
                best_rssi = avg
                best_sector = sector

        return best_sector

    def get_gradient_direction(self) -> Optional[float]:
        """
        Estimate the direction of increasing RSSI by comparing
        recent directional samples.

        More sophisticated than get_best_heading — computes a
        weighted vector sum of all sector averages.

        Returns heading in degrees (0-360), or None.
        """
        if len(self._directional_samples) < RSSI_MIN_SAMPLES_FOR_DIRECTION:
            return None

        # Compute weighted vector sum
        x_sum = 0.0
        y_sum = 0.0

        for sector_deg, samples in self._directional_samples.items():
            if len(samples) < 1:
                continue
            avg_rssi = sum(samples[-5:]) / min(len(samples), 5)
            # Convert RSSI to a positive weight (stronger signal = higher weight)
            weight = max(0, avg_rssi + 100)  # -100dBm → 0, -30dBm → 70
            rad = math.radians(sector_deg)
            x_sum += weight * math.cos(rad)
            y_sum += weight * math.sin(rad)

        if abs(x_sum) < 0.01 and abs(y_sum) < 0.01:
            return None

        angle = math.degrees(math.atan2(y_sum, x_sum))
        return angle % 360

    def reset(self):
        """Reset all tracking data."""
        self._kalman.reset()
        self._raw_history.clear()
        self._filtered_history.clear()
        self._directional_samples.clear()
        self._last_update_time = 0.0

    def get_snapshot(self) -> Dict[str, Any]:
        """Get current tracker state for status reporting."""
        return {
            "current_rssi": self.current_rssi,
            "raw_rssi": self.raw_rssi,
            "trend": self.get_trend(),
            "distance_estimate_m": self.estimate_distance_m(),
            "signal_lost": self.signal_lost,
            "is_near_target": self.is_near_target,
            "is_at_target": self.is_at_target,
            "sample_count": self.sample_count,
            "best_heading": self.get_best_heading(),
        }
