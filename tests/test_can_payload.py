"""
Unit tests for CANPayload dataclass and build_can_payload factory.
"""
import unittest
from src.payload.can_payload import CANPayload, build_can_payload


class TestCANPayload(unittest.TestCase):
    def test_default_payload_bytes_length(self):
        payload = CANPayload(
            driver_state=0,
            face_detected=True,
            alert_level=0,
            alive_counter=0,
            buffer_ready=True,
            can_bus_ok=True,
        )
        raw = payload.to_bytes()
        self.assertEqual(len(raw), 8, "CAN payload must be exactly 8 bytes.")

    def test_byte_0_state_id(self):
        # Level 1 (Alert) -> State 0
        p_alert = build_can_payload(level_id=1, face_detected=True, buffer_ready=True)
        self.assertEqual(p_alert.driver_state, 0)
        self.assertEqual(p_alert.to_bytes()[0], 0x00)

        # Level 2 (Drowsy) -> State 1
        p_drowsy = build_can_payload(level_id=2, face_detected=True, buffer_ready=True)
        self.assertEqual(p_drowsy.driver_state, 1)
        self.assertEqual(p_drowsy.to_bytes()[0], 0x01)

        # Level 0 (Microsleep) -> State 2
        p_sleep = build_can_payload(level_id=0, face_detected=True, buffer_ready=True)
        self.assertEqual(p_sleep.driver_state, 2)
        self.assertEqual(p_sleep.to_bytes()[0], 0x02)

        # Buffer not ready -> State 3 (INIT)
        p_init = build_can_payload(level_id=1, face_detected=True, buffer_ready=False)
        self.assertEqual(p_init.driver_state, 3)
        self.assertEqual(p_init.to_bytes()[0], 0x03)

    def test_byte_1_tracking_face(self):
        p_face_ok = build_can_payload(level_id=1, face_detected=True)
        self.assertEqual(p_face_ok.to_bytes()[1], 0x01)

        p_no_face = build_can_payload(level_id=1, face_detected=False)
        self.assertEqual(p_no_face.to_bytes()[1], 0x00)

    def test_byte_2_alert_level(self):
        p_alert = build_can_payload(level_id=1, face_detected=True)
        self.assertEqual(p_alert.alert_level, 0)
        self.assertEqual(p_alert.to_bytes()[2], 0x00)

        p_drowsy = build_can_payload(level_id=2, face_detected=True)
        self.assertEqual(p_drowsy.alert_level, 1)
        self.assertEqual(p_drowsy.to_bytes()[2], 0x01)

        p_sleep = build_can_payload(level_id=0, face_detected=True)
        self.assertEqual(p_sleep.alert_level, 2)
        self.assertEqual(p_sleep.to_bytes()[2], 0x02)

    def test_byte_3_alive_counter(self):
        for ac in range(16):
            p = CANPayload(
                driver_state=0,
                face_detected=True,
                alert_level=0,
                alive_counter=ac,
                buffer_ready=True,
            )
            self.assertEqual(p.to_bytes()[3], ac)

        p_overflow = CANPayload(
            driver_state=0,
            face_detected=True,
            alert_level=0,
            alive_counter=16,
            buffer_ready=True,
        )
        self.assertEqual(p_overflow.to_bytes()[3], 0x00)

    def test_byte_4_system_status(self):
        p1 = CANPayload(
            driver_state=0,
            face_detected=True,
            alert_level=0,
            buffer_ready=True,
            can_bus_ok=True,
        )
        self.assertEqual(p1.to_bytes()[4], 0x03)

        p2 = CANPayload(
            driver_state=0,
            face_detected=True,
            alert_level=0,
            buffer_ready=False,
            can_bus_ok=True,
        )
        self.assertEqual(p2.to_bytes()[4], 0x02)

    def test_bytes_5_to_7_reserved(self):
        p = build_can_payload(level_id=0, face_detected=True)
        raw = p.to_bytes()
        self.assertEqual(raw[5], 0x00)
        self.assertEqual(raw[6], 0x00)
        self.assertEqual(raw[7], 0x00)

    def test_hex_and_display_dict(self):
        p = CANPayload(
            driver_state=2,
            face_detected=True,
            alert_level=2,
            alive_counter=7,
            buffer_ready=True,
            can_bus_ok=True,
        )
        expected_hex = "02 01 02 07 03 00 00 00"
        self.assertEqual(p.to_hex_string(), expected_hex)

        d = p.to_display_dict()
        self.assertEqual(d['state'], "MICROSLEEP")
        self.assertEqual(d['tracking'], "FACE_OK")
        self.assertEqual(d['alert'], "CRITICAL")
        self.assertEqual(d['alive'], 7)
        self.assertEqual(d['hex'], expected_hex)


if __name__ == '__main__':
    unittest.main()
