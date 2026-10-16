"""RelateAnything demo with class-aware ByteTrack/Kalman occlusion memory."""

from dataclasses import dataclass, replace
from math import hypot, isfinite

import numpy as np
import supervision as sv


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    label: str


@dataclass
class Track:
    track_id: int
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float
    last_seen: float
    observed: bool = True
    stationary: bool = True


def relation_inputs(tracks):
    """Never treat a Kalman prediction as a newly observed relation endpoint."""
    return [track for track in tracks if track.observed]


class ObjectTracker:
    def __init__(self, fps=30, hold_seconds=10):
        if (
            not isfinite(fps)
            or fps <= 0
            or not isfinite(hold_seconds)
            or hold_seconds <= 0
        ):
            raise ValueError("fps and hold_seconds must be finite and positive")
        self.fps, self.hold_seconds = fps, hold_seconds
        self.trackers = {}
        self.memory = {}
        self.next_id = 1
        self.previous_time = -1

    def update(self, detections, timestamp):
        if not isfinite(timestamp) or timestamp < 0 or timestamp <= self.previous_time:
            raise ValueError("timestamp must be finite, nonnegative and increasing")
        self.previous_time = timestamp
        self.memory = {
            key: value
            for key, value in self.memory.items()
            if timestamp - value.last_seen <= self.hold_seconds
        }
        classes = set(self.trackers) | {d.label for d in detections}
        output = []
        for label in sorted(classes):
            if label not in self.trackers:
                self.trackers[label] = sv.ByteTrack(
                    frame_rate=self.fps,
                    lost_track_buffer=max(1, round(self.hold_seconds * 30)),
                    track_activation_threshold=0.25,
                )
            tracker = self.trackers[label]
            tensors = np.array(
                [(*d.bbox, d.confidence) for d in detections if d.label == label],
                dtype=np.float32,
            ).reshape(-1, 5)
            observed = tracker.update_with_tensors(tensors)
            visible_ids = set()
            for item in observed:
                local_id = int(item.external_track_id)
                visible_ids.add(local_id)
                key = (label, local_id)
                box = tuple(float(v) for v in item.tlbr)
                old = self.memory.get(key)
                stationary = True
                if old is not None:
                    delta = timestamp - old.last_seen
                    displacement = hypot(
                        (box[0] + box[2] - old.bbox[0] - old.bbox[2]) / 2,
                        (box[1] + box[3] - old.bbox[1] - old.bbox[3]) / 2,
                    )
                    height = max(1, box[3] - box[1])
                    stationary = displacement / height / delta < 0.15
                track_id = old.track_id if old else self.next_id
                if old is None:
                    self.next_id += 1
                track = Track(
                    track_id,
                    label,
                    box,
                    float(item.score),
                    timestamp,
                    stationary=stationary,
                )
                self.memory[key] = track
                output.append(track)
            for item in tracker.lost_tracks:
                local_id = int(item.external_track_id)
                old = self.memory.get((label, local_id))
                if old is not None and local_id not in visible_ids:
                    box = (
                        old.bbox
                        if old.stationary
                        else tuple(float(v) for v in item.tlbr)
                    )
                    output.append(replace(old, bbox=box, observed=False))
            # Do not accumulate an unbounded removed-track history on long streams.
            tracker.removed_tracks = tracker.removed_tracks[-100:]
        return sorted(output, key=lambda track: track.track_id)
