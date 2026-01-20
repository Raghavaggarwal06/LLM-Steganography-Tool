import base64
import unittest

from itext2bin import _frame_with_1byte_len_header


class TestFraming(unittest.TestCase):
    def test_header_is_payload_length(self) -> None:
        payload = b"\x01\x02\x03\xff"
        framed = _frame_with_1byte_len_header(payload)
        self.assertEqual(framed[0], len(payload))
        self.assertEqual(framed[1:], payload)

    def test_header_limit_255(self) -> None:
        payload = b"x" * 255
        framed = _frame_with_1byte_len_header(payload)
        self.assertEqual(framed[0], 255)

        with self.assertRaises(ValueError):
            _frame_with_1byte_len_header(b"x" * 256)

    def test_base64_roundtrip_of_framed_bytes(self) -> None:
        payload = b"hello\x00world"
        framed = _frame_with_1byte_len_header(payload)
        encoded = base64.b64encode(framed)
        decoded = base64.b64decode(encoded, validate=True)
        self.assertEqual(decoded, framed)


if __name__ == "__main__":
    unittest.main()

