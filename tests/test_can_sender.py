"""
Unit tests for CANSenderThread.
"""
import time
import unittest
from unittest.mock import MagicMock

from src.payload.can_payload import CANPayload, build_can_payload
from src.can_bus.can_sender import CANSenderThread


class TestCANSender(unittest.TestCase):
    def test_post_latest_wins(self):
        sender = CANSenderThread(enabled=False)

        p1 = build_can_payload(level_id=1, face_detected=True)
        p2 = build_can_payload(level_id=2, face_detected=True)
        p3 = build_can_payload(level_id=0, face_detected=True)

        sender.post(p1)
        sender.post(p2)
        sender.post(p3)

        self.assertEqual(len(sender._payload_slot), 1)
        latest = sender._payload_slot.pop()
        self.assertEqual(latest.driver_state, 2)

    def test_alive_counter_increments_and_wraps(self):
        sender = CANSenderThread(enabled=False)
        sender._bus = MagicMock()
        sender.is_connected

        for expected_counter in range(16):
            payload = build_can_payload(level_id=1, face_detected=True)
            sender._send_frame(payload)
            self.assertEqual(payload.alive_counter, expected_counter)

        payload_wrap = build_can_payload(level_id=1, face_detected=True)
        sender._send_frame(payload_wrap)
        self.assertEqual(payload_wrap.alive_counter, 0)

    def test_thread_start_and_stop_lifecycle(self):
        sender = CANSenderThread(send_interval=0.05, enabled=False)
        sender.start()
        self.assertTrue(sender.is_alive())

        sender.post(build_can_payload(level_id=1, face_detected=True))
        time.sleep(0.1)

        sender.stop()
        sender.join(timeout=1.0)
        self.assertFalse(sender.is_alive())


if __name__ == '__main__':
    unittest.main()
