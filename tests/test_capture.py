import unittest
from threading import Event

from capture import LatestCapture


class CaptureTests(unittest.TestCase):
    def test_latest_frame_replaces_old_frames_instead_of_queuing_them(self):
        done = Event()

        class Source:
            def __init__(self):
                self.index = 0

            def read(self):
                self.index += 1
                if self.index > 20:
                    done.set()
                    return False, None
                return True, self.index

            def release(self):
                pass

        capture = LatestCapture(Source())
        try:
            self.assertTrue(done.wait(2))
            ok, frame = capture.read()
            self.assertTrue(ok)
            self.assertEqual(frame, 20)
            self.assertEqual(capture.read(), (False, None))
        finally:
            capture.close()
