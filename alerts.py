import math
from dataclasses import asdict, dataclass
from math import hypot, isfinite
from uuid import uuid4

PERSONAL_ITEMS = {
    "backpack",
    "handbag",
    "suitcase",
    "laptop",
    "cell phone",
    "phone",
    "wallet",
    "bag",
    "tablet",
    "keys",
}
OWNERSHIP_RELATIONS = {"holding", "carrying", "wearing", "using"}


@dataclass
class ItemState:
    object_id: int
    label: str
    owner_id: int | None = None
    owner_score: float | None = None
    fused_score: float | None = None
    state: str = "UNASSIGNED"
    missing_since: float | None = None
    unattended_since: float | None = None
    candidate_id: int | None = None
    candidate_samples: int = 0
    last_relation_at: float = -1
    alert_id: str | None = None
    reason: str | None = None


class AbandonmentMonitor:
    def __init__(
        self,
        exit_seconds=3,
        unattended_seconds=30,
        owner_threshold=0.5,
        owner_samples=2,
        safe_distance=1.5,
    ):
        values = (exit_seconds, unattended_seconds, safe_distance)
        if any(not isfinite(v) or v <= 0 for v in values):
            raise ValueError(
                "Timing and distance thresholds must be finite and positive"
            )
        if (
            not isfinite(owner_threshold)
            or not 0 <= owner_threshold <= 1
            or owner_samples < 1
        ):
            raise ValueError("Invalid ownership thresholds")
        self.exit_seconds, self.unattended_seconds = exit_seconds, unattended_seconds
        self.owner_threshold, self.owner_samples = owner_threshold, owner_samples
        self.safe_distance = safe_distance
        self.items = {}
        self.previous_time = -1

    def update(self, tracks, relations, timestamp):
        if not isfinite(timestamp) or timestamp <= self.previous_time or timestamp < 0:
            raise ValueError("Timestamp must be finite, nonnegative and increasing")
        self.previous_time = timestamp
        by_id = {t.track_id: t for t in tracks}
        self.items = {id: state for id, state in self.items.items() if id in by_id}
        events = []
        for obj in tracks:
            if obj.label not in PERSONAL_ITEMS:
                continue
            item = self.items.setdefault(
                obj.track_id, ItemState(obj.track_id, obj.label)
            )
            if item.owner_id is None and obj.observed:
                self._register_owner(item, obj, by_id, relations, timestamp)
            if item.owner_id is None:
                continue
            owner = by_id.get(item.owner_id)
            visible_owner = owner is not None and owner.observed
            if visible_owner:
                item.missing_since = None
            elif item.missing_since is None:
                item.missing_since = timestamp
            if not obj.observed:
                # Keep timers/history, but never fire a new alert on a predicted box.
                continue
            near_owner = visible_owner and self._near(owner, obj)
            if near_owner:
                if item.alert_id:
                    events.append(self._event(item, timestamp, "RESOLVED", obj))
                item.state, item.alert_id, item.reason = "ATTACHED", None, None
                item.unattended_since = None
                continue
            if not obj.stationary:
                item.unattended_since = None
                if not item.alert_id:
                    item.state = "MOVING"
                continue
            if item.unattended_since is None:
                item.unattended_since = timestamp
            if item.alert_id:
                continue
            item.state = "UNATTENDED"
            if (
                item.missing_since is not None
                and timestamp - item.missing_since >= self.exit_seconds
            ):
                item.reason = "OWNER_LEFT_CAMERA"
            elif timestamp - item.unattended_since >= self.unattended_seconds:
                item.reason = "UNATTENDED_TIMEOUT"
            else:
                continue
            item.state, item.alert_id = "ABANDONED", str(uuid4())
            events.append(self._event(item, timestamp, "ALERT", obj))
        return events

    def _register_owner(self, item, obj, by_id, relations, timestamp):
        candidates = []
        for edge in relations:
            owner = by_id.get(edge["subject_id"])
            if (
                edge["object_id"] == obj.track_id
                and owner is not None
                and owner.observed
                and owner.label == "person"
                and edge["predicate"] in OWNERSHIP_RELATIONS
                and edge["evaluated_at"] == timestamp
                and edge["score"] >= self.owner_threshold
            ):
                # Multimodal Confidence Fusion:
                # Fuses object detector confidence & person detector confidence with relationship score
                obj_conf = max(0.01, min(1.0, float(getattr(obj, "confidence", 1.0))))
                owner_conf = max(0.01, min(1.0, float(getattr(owner, "confidence", 1.0))))
                fused_score = float(edge["score"]) * math.sqrt(obj_conf * owner_conf)
                edge_copy = dict(edge)
                edge_copy["fused_score"] = fused_score
                candidates.append(edge_copy)
        candidates.sort(key=lambda edge: edge["fused_score"], reverse=True)
        if not candidates:
            return
        best = candidates[0]
        if len(candidates) > 1 and best["fused_score"] - candidates[1]["fused_score"] < 0.15:
            item.candidate_id, item.candidate_samples = None, 0
            return
        if best["evaluated_at"] <= item.last_relation_at:
            return
        if item.candidate_id != best["subject_id"]:
            item.candidate_id, item.candidate_samples = best["subject_id"], 0
        item.candidate_samples += 1
        item.last_relation_at = best["evaluated_at"]
        if item.candidate_samples >= self.owner_samples:
            item.owner_id = item.candidate_id
            item.owner_score = best["score"]
            item.fused_score = best["fused_score"]
            item.state = "ATTACHED"

    def _near(self, owner, obj):
        a, b = owner.bbox, obj.bbox
        distance = hypot(
            (a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2
        )
        return distance <= self.safe_distance * max(1, a[3] - a[1])

    @staticmethod
    def _event(item, timestamp, kind, obj):
        return {
            "type": kind,
            "alert_id": item.alert_id,
            "timestamp": timestamp,
            "object_id": item.object_id,
            "class_name": item.label,
            "owner_id": item.owner_id,
            "owner_score": item.owner_score,
            "fused_score": getattr(item, "fused_score", item.owner_score),
            "reason": item.reason,
            "bbox": list(obj.bbox),
        }

    def snapshot(self):
        return [asdict(item) for item in self.items.values()]
