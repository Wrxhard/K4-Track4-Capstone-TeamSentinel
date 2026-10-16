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
    def __init__(self, fps=30, hold_seconds=10, new_track_confidence=0.25):
        if (
            not isfinite(fps)
            or fps <= 0
            or not isfinite(hold_seconds)
            or hold_seconds <= 0
        ):
            raise ValueError("fps and hold_seconds must be finite and positive")
        self.fps, self.hold_seconds = fps, hold_seconds
        if not isfinite(new_track_confidence) or not 0.2 <= new_track_confidence <= 1:
            raise ValueError("new_track_confidence must be in [.2, 1]")
        self.new_track_confidence = new_track_confidence
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
                    # ByteTrack adds 0.1 to this threshold when creating new tracks.
                    track_activation_threshold=self.new_track_confidence - 0.1,
                )
            tracker = self.trackers[label]
            tensors = np.array(
                [(*d.bbox, d.confidence) for d in detections if d.label == label],
                dtype=np.float32,
            ).reshape(-1, 5)
            self._freeze_stationary(tracker, label)
            tracker.update_with_tensors(tensors)
            self._freeze_stationary(tracker, label)
            # A newly seen box is still an observation before track confirmation.
            # Internal IDs exist immediately; external IDs do not until confirmation.
            observed = [
                item
                for item in tracker.tracked_tracks
                if item.frame_id == tracker.frame_id
            ]
            visible_ids = {int(item.internal_track_id) for item in observed}
            recovered = self._recover_unique(
                tracker, label, observed, visible_ids, timestamp
            )
            for item in observed:
                local_id = int(item.internal_track_id)
                key = (label, local_id)
                box = tuple(float(v) for v in item.tlbr)
                old = self.memory.get(key) or recovered.get(key)
                stationary = True
                if old is not None:
                    delta = max(0.001, timestamp - old.last_seen)
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
                local_id = int(item.internal_track_id)
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

    def _freeze_stationary(self, tracker, label):
        for item in tracker.lost_tracks:
            old = self.memory.get((label, int(item.internal_track_id)))
            if (
                old is None
                or not old.stationary
                or label == "person"
                or item.mean is None
            ):
                continue
            x1, y1, x2, y2 = old.bbox
            height = max(0.001, y2 - y1)
            item.mean[:4] = ((x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / height, height)
            item.mean[4:] = 0

    def _recover_unique(self, tracker, label, observed, visible_ids, timestamp):
        """Recover only mutually unique, overlapping lost tracks; never steal a visible ID."""
        candidates = {}
        for item in observed:
            key = (label, int(item.internal_track_id))
            if key in self.memory:
                continue
            a = item.tlbr
            matches = []
            for old_key, old in self.memory.items():
                if old_key[0] != label or old_key[1] in visible_ids:
                    continue
                if label == "person" and timestamp - old.last_seen > 1:
                    continue
                b = old.bbox
                intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                    0, min(a[3], b[3]) - max(a[1], b[1])
                )
                area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
                area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
                union = area_a + area_b - intersection
                if (
                    union > 0
                    and area_b > 0
                    and intersection / union >= 0.5
                    and 0.5 <= area_a / area_b <= 2
                ):
                    matches.append(old_key)
            candidates[key] = matches
        recovered = {}
        retired = set()
        for key, matches in candidates.items():
            if len(matches) != 1:
                continue
            old_key = matches[0]
            if sum(old_key in others for others in candidates.values()) != 1:
                continue
            recovered[key] = self.memory.pop(old_key)
            retired.add(old_key[1])
        tracker.lost_tracks = [
            item
            for item in tracker.lost_tracks
            if int(item.internal_track_id) not in retired
        ]
        return recovered
