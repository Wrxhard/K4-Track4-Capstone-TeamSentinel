"""Synthetic two-person scenario to test alerts/modal without downloading models."""

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from demo import parser, run
from sentinel import Detection


class Scenario:
    synthetic = True

    def __init__(self):
        self.frame = 0

    def detect(self, frame):
        self.frame += 1
        objects = [
            Detection((210, 190, 270, 250), 0.95, "backpack"),
            Detection((450, 65, 520, 270), 0.95, "person"),
        ]
        if self.frame <= 20:
            objects.append(Detection((155, 65, 225, 270), 0.95, "person"))
        return objects

    def relate(self, frame, tracks):
        owner = next(
            (
                i
                for i, t in enumerate(tracks)
                if t.label == "person" and t.bbox[0] < 300
            ),
            None,
        )
        obj = next((i for i, t in enumerate(tracks) if t.label == "backpack"), None)
        if owner is None or obj is None:
            return []
        return [
            SimpleNamespace(
                subject_idx=owner, object_idx=obj, predicate="carrying", score=0.9
            )
        ]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--port", type=int, default=8080)
    cli.add_argument("--no-web", action="store_true")
    args = cli.parse_args()
    root = Path("outputs") / f"synthetic-{time.time_ns()}"
    root.mkdir(parents=True)
    source = root / "scenario.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360)
    )
    if not writer.isOpened():
        raise RuntimeError("Cannot create synthetic video")
    for index in range(90):
        frame = np.full((360, 640, 3), (31, 23, 15), dtype=np.uint8)
        cv2.rectangle(frame, (210, 190), (270, 250), (170, 130, 65), -1)
        cv2.rectangle(frame, (450, 65), (520, 270), (95, 140, 180), -1)
        if index < 20:
            cv2.rectangle(frame, (155, 65), (225, 270), (125, 170, 110), -1)
        cv2.putText(
            frame,
            "SYNTHETIC DEMO - NOT REAL MODEL INFERENCE",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (235, 235, 235),
            1,
        )
        writer.write(frame)
    writer.release()
    options = [
        "--source",
        str(source),
        "--output",
        str(root / "result"),
        "--no-preview",
        "--relation-every",
        "1",
        "--port",
        str(args.port),
    ]
    if not args.no_web:
        options.append("--web")
    print(
        "SYNTHETIC DEMO: owner leaves after 2s; second person remains. Alert after about 5s.",
        flush=True,
    )
    print(run(parser().parse_args(options), Scenario()))


if __name__ == "__main__":
    main()
