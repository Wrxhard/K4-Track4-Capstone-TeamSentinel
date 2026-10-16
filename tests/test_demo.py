import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from demo import display_tracks, parser, run, update_relations
from sentinel import Detection, Track


class FakeModels:
    def __init__(self):
        self.calls = 0
        self.relation_calls = 0

    def detect(self, frame):
        self.calls += 1
        person = Detection((10, 10, 70, 180), 0.9, "person")
        bag = Detection((90, 90, 140, 150), 0.9, "backpack")
        return [person] if self.calls in (3, 4) else [person, bag]

    def relate(self, frame, tracks):
        self.relation_calls += 1
        assert all(t.observed for t in tracks)
        subject = next(i for i, t in enumerate(tracks) if t.label == "person")
        obj = next(i for i, t in enumerate(tracks) if t.label == "backpack")
        return [
            SimpleNamespace(
                subject_idx=subject, object_idx=obj, predicate="carrying", score=0.8
            )
        ]


class DemoTests(unittest.TestCase):
    def test_detector_keeps_weak_boxes_for_tracking_association(self):
        from demo import Models

        model = Models.__new__(Models)
        model.is_onnx_detector = False
        model.device = "cpu"
        model.confidence = 0.35
        model.max_boxes = 16
        model.detector_size = 640

        def predict(frame, **kwargs):
            self.assertLessEqual(kwargs["conf"], 0.1)
            return [SimpleNamespace(boxes=None)]

        model.detector = SimpleNamespace(predict=predict)
        self.assertEqual(model.detect(np.zeros((10, 10, 3), dtype=np.uint8)), [])

    def test_rf_detr_cuda_request_cannot_silently_use_cpu(self):
        from demo import RFDetrDetector

        session = SimpleNamespace(
            get_providers=lambda: ["CPUExecutionProvider"],
            get_inputs=lambda: [SimpleNamespace(name="images", shape=[1, 3, 384, 384])],
        )
        with (
            patch("onnxruntime.InferenceSession", return_value=session),
            self.assertRaisesRegex(RuntimeError, "CUDA"),
        ):
            RFDetrDetector(model_path="models/rf-detr.onnx", device="cuda")

    def test_display_does_not_stack_old_phone_or_person_on_live_boxes(self):
        tracks = [
            Track(1, "cell phone", (10, 10, 60, 100), 0.9, 0, False),
            Track(2, "cell phone", (15, 15, 65, 105), 0.9, 1),
            Track(3, "person", (0, 0, 100, 200), 0.9, 0, False),
            Track(4, "backpack", (200, 100, 240, 140), 0.9, 0, False),
        ]
        shown = display_tracks(tracks, 1)
        self.assertEqual({t.track_id for t in shown}, {2, 4})

    def test_webcam_index_is_passed_to_capture_and_released(self):
        frame = np.zeros((200, 240, 3), dtype=np.uint8)
        capture = SimpleNamespace(
            isOpened=lambda: True,
            read=lambda: (True, frame.copy()),
            get=lambda key: 30,
            set=lambda key, value: True,
            released=False,
        )
        capture.release = lambda: setattr(capture, "released", True)
        with (
            TemporaryDirectory() as tmp,
            patch("cv2.VideoCapture", return_value=capture) as open_camera,
        ):
            args = parser().parse_args(
                [
                    "--source",
                    "0",
                    "--max-frames",
                    "3",
                    "--no-preview",
                    "--output",
                    str(Path(tmp) / "camera"),
                ]
            )
            result = run(args, SimpleNamespace(detect=lambda frame: []))
        open_camera.assert_called_once_with(0)
        self.assertEqual(result["frames"], 3)
        self.assertTrue(capture.released)

    def test_two_person_video_alert_contains_saved_evidence(self):
        from simulate import Scenario

        with TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "scenario.mp4", Path(tmp) / "output"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360)
            )
            self.assertTrue(writer.isOpened())
            for _ in range(70):
                writer.write(np.zeros((360, 640, 3), dtype=np.uint8))
            writer.release()
            args = parser().parse_args(
                [
                    "--source",
                    str(source),
                    "--output",
                    str(output),
                    "--relation-every",
                    "1",
                    "--no-preview",
                ]
            )
            run(args, Scenario())
            events = [
                json.loads(line)
                for line in (output / "alerts.jsonl").read_text().splitlines()
            ]
            alerts = [event for event in events if event["type"] == "ALERT"]
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0]["reason"], "OWNER_LEFT_CAMERA")
            self.assertTrue((output / alerts[0]["evidence"]["owner"]).is_file())
            self.assertTrue((output / alerts[0]["evidence"]["scene"]).is_file())

    def test_video_writes_every_frame_and_marks_occluded_relation_stale(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (240, 200)
            )
            self.assertTrue(writer.isOpened())
            for _ in range(6):
                writer.write(np.zeros((200, 240, 3), dtype=np.uint8))
            writer.release()
            output = Path(tmp) / "output"
            args = parser().parse_args(
                [
                    "--source",
                    str(source),
                    "--output",
                    str(output),
                    "--relation-every",
                    "1",
                    "--no-preview",
                ]
            )
            result = run(args, FakeModels())
            self.assertEqual(result["frames"], 6)
            rows = [
                json.loads(line)
                for line in (output / "events.jsonl").read_text().splitlines()
            ]
            bag_ids = {
                track["track_id"]
                for row in rows
                for track in row["tracks"]
                if track["label"] == "backpack"
            }
            self.assertEqual(len(bag_ids), 1)
            self.assertFalse(
                next(t for t in rows[2]["tracks"] if t["label"] == "backpack")[
                    "observed"
                ]
            )
            self.assertTrue(rows[2]["relations"][0]["stale"])
            self.assertFalse(rows[4]["relations"][0]["stale"])
            capture = cv2.VideoCapture(str(output / "annotated.mp4"))
            count = 0
            while capture.read()[0]:
                count += 1
            capture.release()
            self.assertEqual(count, 6)

    def test_no_relation_does_not_turn_into_zero_score(self):
        tracks = [
            Track(1, "person", (0, 0, 10, 20), 0.9, 0),
            Track(2, "backpack", (20, 20, 30, 30), 0.9, 0),
        ]
        old = {(1, 2): {"predicate": "holding"}}
        self.assertEqual(update_relations(old, [], tracks, 1), {})

    def test_image_with_no_objects_still_saves_output(self):
        with TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "input.jpg", Path(tmp) / "output"
            cv2.imwrite(str(source), np.zeros((200, 240, 3), dtype=np.uint8))
            args = parser().parse_args(
                ["--source", str(source), "--output", str(output), "--no-preview"]
            )
            run(args, SimpleNamespace(detect=lambda frame: []))
            self.assertTrue((output / "annotated.jpg").is_file())

    def test_rf_detr_detector_dummy_frame(self):
        from demo import RFDetrDetector

        if Path("models/rf-detr.onnx").is_file():
            det = RFDetrDetector("models/rf-detr.onnx", "models/rf-detr.json")
            res = det.detect(np.zeros((384, 384, 3), dtype=np.uint8))
            self.assertIsInstance(res, list)


if __name__ == "__main__":
    unittest.main()
