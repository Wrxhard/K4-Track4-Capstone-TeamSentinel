import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import cv2
import numpy as np

from demo import parser, run, update_relations
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


if __name__ == "__main__":
    unittest.main()
