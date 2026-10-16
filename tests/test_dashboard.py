import json
import unittest
from urllib.request import urlopen

from dashboard import Dashboard


class DashboardTests(unittest.TestCase):
    def test_mjpeg_stream_sends_latest_frame_without_polling(self):
        dashboard = Dashboard(port=0)
        try:
            dashboard.publish({"status": "running"}, b"frame-one")
            with urlopen(dashboard.url + "/stream.mjpg", timeout=3) as response:
                self.assertIn(
                    "multipart/x-mixed-replace", response.headers["Content-Type"]
                )
                self.assertEqual(response.readline(), b"--frame\r\n")
                self.assertEqual(response.readline(), b"Content-Type: image/jpeg\r\n")
                self.assertEqual(response.readline(), b"Content-Length: 9\r\n")
                self.assertEqual(response.readline(), b"\r\n")
                self.assertEqual(response.read(9), b"frame-one")
        finally:
            dashboard.close()

    def test_http_serves_page_status_and_latest_frame(self):
        dashboard = Dashboard(port=0)
        try:
            state = {"status": "running", "objects": [], "alerts": [], "timestamp": 1}
            dashboard.publish(state, b"test-jpeg")
            state["timestamp"] = 999
            with urlopen(dashboard.url + "/api/status") as response:
                self.assertEqual(json.load(response)["timestamp"], 1)
            with urlopen(dashboard.url + "/") as response:
                self.assertIn("showModal", response.read().decode())
            with urlopen(dashboard.url + "/frame.jpg") as response:
                self.assertEqual(response.read(), b"test-jpeg")
        finally:
            dashboard.close()
