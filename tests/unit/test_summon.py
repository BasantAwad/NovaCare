"""
Tests for the Summon module — State Machine, Protocol, RSSI Tracker, Stuck Detector
"""

import sys
import os
import time
import unittest
from unittest.mock import MagicMock

# Add project root to path so we can import submodules directly
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_summon_path = os.path.join(_project_root, 'optimized_runtime', 'summon')
sys.path.insert(0, _summon_path)

from summon_state import SummonState, SummonStateMachine
from protocol import (
    SummonProtocol, SummonMessageType, SummonRequest, SummonStatus, SummonResult,
)
from rssi_tracker import RSSITracker, KalmanFilter1D
from stuck_detector import StuckDetector, StuckType


# ============================================================================
# State Machine Tests
# ============================================================================

class TestSummonStateMachine(unittest.TestCase):
    def setUp(self):
        self.sm = SummonStateMachine()

    def test_initial_state_is_idle(self):
        self.assertEqual(self.sm.state, SummonState.IDLE)
        self.assertFalse(self.sm.is_active)

    def test_valid_transition_idle_to_requested(self):
        result = self.sm.transition(SummonState.REQUESTED, "test")
        self.assertTrue(result)
        self.assertEqual(self.sm.state, SummonState.REQUESTED)
        self.assertTrue(self.sm.is_active)

    def test_invalid_transition_idle_to_navigating(self):
        result = self.sm.transition(SummonState.NAVIGATING, "invalid")
        self.assertFalse(result)
        self.assertEqual(self.sm.state, SummonState.IDLE)

    def test_full_success_path(self):
        transitions = [
            SummonState.REQUESTED,
            SummonState.INITIALIZING,
            SummonState.SCANNING_RSSI,
            SummonState.NAVIGATING,
            SummonState.ARRIVING,
            SummonState.ARRIVED,
        ]
        for state in transitions:
            result = self.sm.transition(state)
            self.assertTrue(result, f"Failed transition to {state.value}")
        self.assertTrue(self.sm.is_terminal)

    def test_cancellation_from_navigating(self):
        self.sm.transition(SummonState.REQUESTED)
        self.sm.transition(SummonState.INITIALIZING)
        self.sm.transition(SummonState.SCANNING_RSSI)
        self.sm.transition(SummonState.NAVIGATING)
        result = self.sm.transition(SummonState.CANCELLED, "user")
        self.assertTrue(result)
        self.assertTrue(self.sm.is_terminal)

    def test_recovery_path(self):
        self.sm.transition(SummonState.REQUESTED)
        self.sm.transition(SummonState.INITIALIZING)
        self.sm.transition(SummonState.SCANNING_RSSI)
        self.sm.transition(SummonState.NAVIGATING)
        self.sm.transition(SummonState.RECOVERING, "stuck")
        self.sm.transition(SummonState.SCANNING_RSSI, "rescan")
        self.assertEqual(self.sm.state, SummonState.SCANNING_RSSI)

    def test_wall_following_path(self):
        self.sm.transition(SummonState.REQUESTED)
        self.sm.transition(SummonState.INITIALIZING)
        self.sm.transition(SummonState.SCANNING_RSSI)
        self.sm.transition(SummonState.NAVIGATING)
        self.sm.transition(SummonState.WALL_FOLLOWING, "obstacle")
        self.sm.transition(SummonState.NAVIGATING, "cleared")
        self.assertEqual(self.sm.state, SummonState.NAVIGATING)

    def test_same_state_noop(self):
        self.sm.transition(SummonState.REQUESTED)
        result = self.sm.transition(SummonState.REQUESTED)
        self.assertTrue(result)

    def test_reset(self):
        self.sm.transition(SummonState.REQUESTED)
        self.sm.reset()
        self.assertEqual(self.sm.state, SummonState.IDLE)

    def test_listener_called(self):
        listener = MagicMock()
        self.sm.add_listener(listener)
        self.sm.transition(SummonState.REQUESTED, "test")
        listener.assert_called_once_with(SummonState.IDLE, SummonState.REQUESTED, "test")

    def test_history_recorded(self):
        self.sm.transition(SummonState.REQUESTED)
        self.sm.transition(SummonState.INITIALIZING)
        self.assertEqual(len(self.sm.history), 2)
        self.assertEqual(self.sm.history[0]["from"], "idle")
        self.assertEqual(self.sm.history[0]["to"], "requested")

    def test_elapsed_seconds(self):
        self.assertEqual(self.sm.elapsed_seconds, 0.0)
        self.sm.transition(SummonState.REQUESTED)
        time.sleep(0.1)
        self.assertGreater(self.sm.elapsed_seconds, 0.05)

    def test_terminal_states_can_reset_to_idle(self):
        # ARRIVED requires going through ARRIVING
        sm = SummonStateMachine()
        sm.transition(SummonState.REQUESTED)
        sm.transition(SummonState.INITIALIZING)
        sm.transition(SummonState.SCANNING_RSSI)
        sm.transition(SummonState.NAVIGATING)
        sm.transition(SummonState.ARRIVING)
        sm.transition(SummonState.ARRIVED)
        result = sm.transition(SummonState.IDLE)
        self.assertTrue(result, "Cannot reset from arrived")

        # FAILED and CANCELLED can come directly from NAVIGATING
        for terminal in [SummonState.FAILED, SummonState.CANCELLED]:
            sm = SummonStateMachine()
            sm.transition(SummonState.REQUESTED)
            sm.transition(SummonState.INITIALIZING)
            sm.transition(SummonState.SCANNING_RSSI)
            sm.transition(SummonState.NAVIGATING)
            sm.transition(terminal)
            result = sm.transition(SummonState.IDLE)
            self.assertTrue(result, f"Cannot reset from {terminal.value}")


# ============================================================================
# Protocol Tests
# ============================================================================

class TestSummonProtocol(unittest.TestCase):
    def test_build_request(self):
        msg = SummonProtocol.build_request("user1", "AA:BB:CC:DD:EE:FF")
        self.assertEqual(msg["type"], "summon_request")
        self.assertEqual(msg["payload"]["user_id"], "user1")
        self.assertEqual(msg["payload"]["ble_mac"], "AA:BB:CC:DD:EE:FF")

    def test_build_cancel(self):
        msg = SummonProtocol.build_cancel("user1")
        self.assertEqual(msg["type"], "summon_cancel")
        self.assertIn("timestamp", msg["payload"])

    def test_build_status(self):
        status = SummonStatus(state="navigating", rssi_current=-65)
        msg = SummonProtocol.build_status(status)
        self.assertEqual(msg["type"], "summon_status")
        self.assertEqual(msg["payload"]["state"], "navigating")

    def test_parse_valid(self):
        msg = {"type": "summon_request", "payload": {"user_id": "u1", "ble_mac": "XX"}}
        msg_type, payload = SummonProtocol.parse(msg)
        self.assertEqual(msg_type, SummonMessageType.SUMMON_REQUEST)

    def test_parse_unknown(self):
        msg = {"type": "unknown_type", "payload": {}}
        msg_type, payload = SummonProtocol.parse(msg)
        self.assertIsNone(msg_type)

    def test_summon_request_roundtrip(self):
        req = SummonRequest(user_id="u1", ble_mac="AA:BB")
        d = req.to_dict()
        req2 = SummonRequest.from_dict(d)
        self.assertEqual(req.user_id, req2.user_id)
        self.assertEqual(req.ble_mac, req2.ble_mac)

    def test_build_ack(self):
        msg = SummonProtocol.build_ack("summon_request", True, "ok")
        self.assertEqual(msg["type"], "summon_ack")
        self.assertTrue(msg["payload"]["success"])

    def test_build_arrived(self):
        result = SummonResult(success=True, reason="arrived", total_time_seconds=45.2)
        msg = SummonProtocol.build_arrived(result)
        self.assertEqual(msg["type"], "summon_arrived")
        self.assertTrue(msg["payload"]["success"])

    def test_build_failed(self):
        result = SummonResult(success=False, reason="timeout")
        msg = SummonProtocol.build_failed(result)
        self.assertEqual(msg["type"], "summon_failed")
        self.assertFalse(msg["payload"]["success"])


# ============================================================================
# RSSI Tracker Tests
# ============================================================================

class TestKalmanFilter1D(unittest.TestCase):
    def test_converges_to_stable_signal(self):
        kf = KalmanFilter1D(initial_estimate=-80)
        for _ in range(20):
            kf.update(-60.0)
        self.assertAlmostEqual(kf.estimate, -60.0, delta=2.0)

    def test_smooths_noisy_signal(self):
        kf = KalmanFilter1D()
        readings = [-70, -65, -75, -68, -72, -70, -66, -74, -69, -71]
        results = [kf.update(r) for r in readings]
        # Filtered should have lower variance than raw
        raw_var = sum((r - (-70))**2 for r in readings) / len(readings)
        filt_var = sum((r - (-70))**2 for r in results[-5:]) / 5
        self.assertLess(filt_var, raw_var)

    def test_reset(self):
        kf = KalmanFilter1D()
        kf.update(-50)
        kf.reset(-80)
        self.assertEqual(kf.estimate, -80)


class TestRSSITracker(unittest.TestCase):
    def setUp(self):
        self.tracker = RSSITracker()

    def test_initial_state(self):
        self.assertEqual(self.tracker.current_rssi, -100)
        self.assertTrue(self.tracker.signal_lost)
        self.assertEqual(self.tracker.sample_count, 0)

    def test_update_returns_filtered(self):
        result = self.tracker.update(-65)
        self.assertIsInstance(result, float)
        self.assertEqual(self.tracker.sample_count, 1)

    def test_arrival_detection(self):
        for _ in range(10):
            self.tracker.update(-35)
        self.assertTrue(self.tracker.is_at_target)

    def test_near_target_detection(self):
        for _ in range(10):
            self.tracker.update(-50)
        self.assertTrue(self.tracker.is_near_target)

    def test_signal_lost_detection(self):
        for _ in range(10):
            self.tracker.update(-98)
        self.assertTrue(self.tracker.signal_lost)

    def test_trend_improving(self):
        for rssi in range(-90, -50, 2):
            self.tracker.update(rssi)
        self.assertEqual(self.tracker.get_trend(), "improving")

    def test_trend_degrading(self):
        for rssi in range(-50, -90, -2):
            self.tracker.update(rssi)
        self.assertEqual(self.tracker.get_trend(), "degrading")

    def test_distance_estimation(self):
        for _ in range(5):
            self.tracker.update(-55)
        dist = self.tracker.estimate_distance_m()
        self.assertAlmostEqual(dist, 1.0, delta=0.5)

    def test_directional_sampling(self):
        for _ in range(5):
            self.tracker.update(-60, heading_deg=0)
        for _ in range(5):
            self.tracker.update(-80, heading_deg=180)
        for _ in range(5):
            self.tracker.update(-70, heading_deg=90)
        best = self.tracker.get_best_heading()
        self.assertEqual(best, 0)  # Forward had strongest signal

    def test_reset(self):
        self.tracker.update(-60)
        self.tracker.reset()
        self.assertEqual(self.tracker.sample_count, 0)
        self.assertEqual(self.tracker.current_rssi, -100)

    def test_snapshot(self):
        self.tracker.update(-65)
        snap = self.tracker.get_snapshot()
        self.assertIn("current_rssi", snap)
        self.assertIn("trend", snap)
        self.assertIn("distance_estimate_m", snap)


# ============================================================================
# Stuck Detector Tests
# ============================================================================

class TestStuckDetector(unittest.TestCase):
    def setUp(self):
        self.detector = StuckDetector()

    def test_initial_state_not_stuck(self):
        self.assertFalse(self.detector.is_stuck)
        self.assertEqual(self.detector.stuck_type, StuckType.NONE)

    def test_progress_clears_stuck(self):
        self.detector.update(rssi=-70, heading=0, is_moving=True)
        self.detector.update(rssi=-60, heading=0, is_moving=True)  # Improved
        self.assertFalse(self.detector.is_stuck)

    def test_oscillation_detection(self):
        """Rapid direction changes should trigger stuck (spinning or oscillating)."""
        for i in range(12):
            heading = 0 if i % 2 == 0 else 180
            self.detector.update(rssi=-70, heading=heading, is_moving=True)
        self.assertTrue(self.detector.is_stuck)
        # With alternating 0/180 headings and no RSSI improvement,
        # spinning detection triggers (consecutive turns without progress)
        self.assertIn(self.detector.stuck_type, [StuckType.SPINNING, StuckType.OSCILLATING])

    def test_recovery_attempt_clears_stuck(self):
        # Force a stuck condition
        self.detector._stuck_type = StuckType.NO_PROGRESS
        self.detector.record_recovery_attempt()
        self.assertFalse(self.detector.is_stuck)
        self.assertEqual(self.detector.recovery_attempts, 1)

    def test_should_abort_after_max_recovery(self):
        for _ in range(5):
            self.detector.record_recovery_attempt()
        self.assertTrue(self.detector.should_abort)

    def test_reset(self):
        self.detector.record_recovery_attempt()
        self.detector.reset()
        self.assertEqual(self.detector.recovery_attempts, 0)
        self.assertFalse(self.detector.is_stuck)

    def test_snapshot(self):
        snap = self.detector.get_snapshot()
        self.assertIn("is_stuck", snap)
        self.assertIn("stuck_type", snap)
        self.assertIn("recovery_attempts", snap)

    def test_heading_delta_wraparound(self):
        delta = StuckDetector._heading_delta(10, 350)
        self.assertAlmostEqual(delta, 20, delta=1)
        delta = StuckDetector._heading_delta(350, 10)
        self.assertAlmostEqual(delta, -20, delta=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
