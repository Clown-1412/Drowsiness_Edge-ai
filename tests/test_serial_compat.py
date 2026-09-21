"""
Regression tests for SerialDebugger with 8-byte CANPayload.
"""
import unittest
from unittest.mock import MagicMock

from src.debug.serial_debugger import SerialDebugger
from src.payload.can_payload import build_can_payload


class TestSerialCompatibility(unittest.TestCase):
    def test_serial_debugger_accepts_8byte_payload(self):
        debugger = SerialDebugger(enabled=False)
        debugger._sender = MagicMock()
        debugger.enabled = True

        p1 = build_can_payload(level_id=1, face_detected=True)
        raw_bytes = p1.to_bytes()
        self.assertEqual(len(raw_bytes), 8)

        sent = debugger.update(
            level_id=1,
            face_detected=True,
            payload_bytes=raw_bytes,
            current_time=100.0,
        )
        self.assertTrue(sent)
        debugger._sender.send.assert_called_once_with(raw_bytes)

        sent_same = debugger.update(
            level_id=1,
            face_detected=True,
            payload_bytes=raw_bytes,
            current_time=101.0,
        )
        self.assertFalse(sent_same)

        sent_hb = debugger.update(
            level_id=1,
            face_detected=True,
            payload_bytes=raw_bytes,
            current_time=103.5,
        )
        self.assertTrue(sent_hb)


if __name__ == '__main__':
    unittest.main()
