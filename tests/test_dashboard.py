import json
import unittest
from urllib.request import urlopen

from dashboard import Dashboard


class DashboardTests(unittest.TestCase):
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
