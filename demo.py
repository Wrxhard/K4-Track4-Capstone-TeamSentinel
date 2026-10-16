"""Run RelateAnything on an image, video or webcam with short occlusion memory."""

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from alerts import AbandonmentMonitor
from dashboard import Dashboard
from sentinel import Detection, ObjectTracker, relation_inputs


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--source", default="0", help="Image, video path or webcam index (default: 0)"
    )
    cli.add_argument(
        "--output", type=Path, default=Path("outputs") / time.strftime("%Y%m%d-%H%M%S")
    )
    cli.add_argument(
        "--detector",
        default="yolo26n.pt",
        help="Detector used by the upstream-style demo",
    )
    cli.add_argument("--relation-model", default="maelic/relsgg-vits16plus")
    cli.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    cli.add_argument("--hold-seconds", type=float, default=10)
    cli.add_argument(
        "--relation-every",
        type=int,
        default=5,
        help="Run relation inference every N frames",
    )
    cli.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Display score cutoff; tune on your data",
    )
    cli.add_argument("--det-confidence", type=float, default=0.25)
    cli.add_argument("--max-boxes", type=int, default=32)
    cli.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    cli.add_argument("--no-preview", action="store_true")
    cli.add_argument(
        "--web", action="store_true", help="Serve local dashboard and alert modal"
    )
    cli.add_argument("--port", type=int, default=8080)
    cli.add_argument(
        "--open-browser",
        action="store_true",
        help="Open dashboard after models have loaded",
    )
    cli.add_argument("--exit-seconds", type=float, default=3)
    cli.add_argument("--unattended-seconds", type=float, default=30)
    cli.add_argument("--owner-threshold", type=float, default=0.5)
    cli.add_argument("--owner-samples", type=int, default=2)
    return cli


def validate(args):
    if not math.isfinite(args.hold_seconds) or args.hold_seconds <= 0:
        raise ValueError("--hold-seconds must be finite and positive")
    if args.relation_every < 1 or args.max_boxes < 2 or args.max_frames < 0:
        raise ValueError(
            "relation-every >= 1, max-boxes >= 2 and max-frames >= 0 required"
        )
    for name in ("threshold", "det_confidence"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if args.output.exists():
        raise FileExistsError(f"Use a new output directory: {args.output}")
    if not 0 <= args.port <= 65535:
        raise ValueError("--port must be in [0,65535]")
    AbandonmentMonitor(
        args.exit_seconds,
        args.unattended_seconds,
        args.owner_threshold,
        args.owner_samples,
    )


class Models:
    def __init__(self, args):
        import torch
        from relsgg import RelateAnything
        from ultralytics import YOLO

        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if args.device == "auto"
            else args.device
        )
        self.detector = YOLO(args.detector)
        load_relation = (
            RelateAnything.from_checkpoint
            if Path(args.relation_model).is_file()
            else RelateAnything.from_pretrained
        )
        self.relation = load_relation(
            args.relation_model,
            device=self.device,
            predicates=[
                "holding",
                "carrying",
                "wearing",
                "using",
                "has",
                "near",
                "beside",
            ],
        )
        self.confidence = args.det_confidence
        self.max_boxes = args.max_boxes

    def detect(self, frame):
        result = self.detector.predict(
            frame,
            device=self.device,
            verbose=False,
            conf=self.confidence,
            max_det=self.max_boxes,
        )[0]
        if result.boxes is None:
            return []
        return [
            Detection(
                tuple(float(v) for v in box), float(score), result.names[int(label)]
            )
            for box, score, label in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            )
        ]

    def relate(self, frame, tracks):
        import numpy as np

        return self.relation.predict(
            frame,
            np.array([t.bbox for t in tracks], dtype=np.float32),
            box_labels=[t.label for t in tracks],
            max_boxes=self.max_boxes,
            topk=20,
        )


def update_relations(cache, triplets, inputs, timestamp):
    """Replace edges for evaluated endpoints; missing pairs are unknown, not zero."""
    evaluated = {track.track_id for track in inputs}
    cache = {
        key: value
        for key, value in cache.items()
        if not (key[0] in evaluated and key[1] in evaluated)
    }
    for triplet in triplets:
        subject = inputs[triplet.subject_idx]
        obj = inputs[triplet.object_idx]
        key = (subject.track_id, obj.track_id)
        cache[key] = {
            "subject_id": subject.track_id,
            "object_id": obj.track_id,
            "predicate": triplet.predicate,
            "score": float(triplet.score),
            "evaluated_at": timestamp,
        }
    return cache


def draw(frame, tracks, relations, timestamp):
    import cv2

    output = frame.copy()
    by_id = {t.track_id: t for t in tracks}
    for track in tracks:
        color = (80, 210, 80) if track.observed else (150, 150, 150)
        x1, y1, x2, y2 = map(int, track.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        status = (
            f"{track.confidence:.2f}"
            if track.observed
            else f"OCCLUDED {timestamp - track.last_seen:.1f}s"
        )
        cv2.putText(
            output,
            f"#{track.track_id} {track.label} {status}",
            (max(0, x1), max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    for edge in relations:
        subject, obj = by_id[edge["subject_id"]], by_id[edge["object_id"]]
        start = (
            int((subject.bbox[0] + subject.bbox[2]) / 2),
            int((subject.bbox[1] + subject.bbox[3]) / 2),
        )
        end = (
            int((obj.bbox[0] + obj.bbox[2]) / 2),
            int((obj.bbox[1] + obj.bbox[3]) / 2),
        )
        color = (0, 190, 255) if edge["stale"] else (255, 210, 90)
        cv2.arrowedLine(output, start, end, color, 1, tipLength=0.04)
        text = f"{edge['predicate']} {edge['score']:.2f}"
        if edge["stale"]:
            text += f" [stale {edge['age_seconds']:.1f}s]"
        cv2.putText(
            output,
            text,
            (max(0, (start[0] + end[0]) // 2), max(16, (start[1] + end[1]) // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
    cv2.putText(
        output,
        "Based on RelateAnything by Maelic Neau - github.com/Maelic/RelateAnything",
        (8, output.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (240, 240, 240),
        1,
    )
    return output


def run(args, models=None):
    import cv2

    validate(args)
    image_mode = Path(args.source).suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }
    webcam = args.source.isdecimal()
    capture = writer = dashboard = None
    try:
        if image_mode:
            frame = cv2.imread(args.source)
            if frame is None:
                raise ValueError(f"Cannot read image: {args.source}")
            fps = 1.0
        else:
            capture = cv2.VideoCapture(int(args.source) if webcam else args.source)
            if not capture.isOpened():
                raise ValueError(f"Cannot open source: {args.source}")
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Source has no decodable frame")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                fps = 30.0
        models = models or Models(args)
        tracker = ObjectTracker(fps, args.hold_seconds)
        monitor = AbandonmentMonitor(
            args.exit_seconds,
            args.unattended_seconds,
            args.owner_threshold,
            args.owner_samples,
        )
        args.output.mkdir(parents=True, exist_ok=False)
        evidence = args.output / "evidence"
        evidence.mkdir()
        if args.web:
            dashboard = Dashboard(args.port)
            print(f"Web dashboard: {dashboard.url}", flush=True)
            if args.open_browser:
                import webbrowser

                webbrowser.open(dashboard.url)
        if not image_mode:
            writer = cv2.VideoWriter(
                str(args.output / "annotated.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (frame.shape[1], frame.shape[0]),
            )
            if not writer.isOpened():
                raise RuntimeError("Cannot open video writer")
        cache = {}
        alert_history = []
        owner_images = {}
        index, previous = 0, -1.0
        started = time.monotonic()
        with (
            (args.output / "events.jsonl").open("x", encoding="utf-8") as log,
            (args.output / "alerts.jsonl").open("x", encoding="utf-8") as alert_log,
        ):
            while True:
                timestamp = time.monotonic() - started if webcam else index / fps
                if capture is not None and not webcam:
                    decoded = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    if math.isfinite(decoded) and decoded > previous:
                        timestamp = decoded
                timestamp = (
                    max(timestamp, previous + 1 / fps)
                    if timestamp <= previous
                    else timestamp
                )
                previous = timestamp
                tracks = tracker.update(models.detect(frame), timestamp)
                inputs = relation_inputs(tracks)[: args.max_boxes]
                if index % args.relation_every == 0 and len(inputs) >= 2:
                    cache = update_relations(
                        cache, models.relate(frame, inputs), inputs, timestamp
                    )
                active = {t.track_id for t in tracks}
                cache = {
                    key: value
                    for key, value in cache.items()
                    if key[0] in active
                    and key[1] in active
                    and timestamp - value["evaluated_at"] <= args.hold_seconds
                }
                edges = [
                    {
                        **value,
                        "stale": timestamp > value["evaluated_at"],
                        "age_seconds": timestamp - value["evaluated_at"],
                    }
                    for value in cache.values()
                    if value["score"] >= args.threshold
                ]
                # Ownership consumes all fresh model results, independent of display cutoff.
                events = monitor.update(tracks, list(cache.values()), timestamp)
                objects = monitor.snapshot()
                by_id = {track.track_id: track for track in tracks}
                active_object_ids = {obj["object_id"] for obj in objects}
                owner_images = {
                    key: value
                    for key, value in owner_images.items()
                    if key in active_object_ids
                }
                for obj in objects:
                    owner = by_id.get(obj["owner_id"])
                    if (
                        owner is not None
                        and owner.observed
                        and obj["object_id"] not in owner_images
                    ):
                        x1, y1, x2, y2 = map(int, owner.bbox)
                        crop = frame[
                            max(0, y1) : min(frame.shape[0], y2),
                            max(0, x1) : min(frame.shape[1], x2),
                        ]
                        path = evidence / f"owner-object-{obj['object_id']}.jpg"
                        if crop.size and cv2.imwrite(str(path), crop):
                            owner_images[obj["object_id"]] = str(
                                path.relative_to(args.output)
                            )
                annotated = draw(frame, tracks, edges, timestamp)
                for obj in objects:
                    if obj["state"] == "ABANDONED":
                        box = by_id[obj["object_id"]].bbox
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 40, 240), 3)
                        cv2.putText(
                            annotated,
                            "ABANDONED",
                            (max(0, x1), max(30, y1 - 22)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (40, 40, 240),
                            2,
                        )
                for event in events:
                    if event["type"] == "ALERT":
                        path = evidence / f"alert-{event['alert_id']}.jpg"
                        event["evidence"] = {
                            "owner": owner_images.get(event["object_id"]),
                            "scene": str(path.relative_to(args.output))
                            if cv2.imwrite(str(path), annotated)
                            else None,
                        }
                        alert_history.append(dict(event))
                        alert_history = alert_history[-100:]
                    alert_log.write(json.dumps(event, allow_nan=False) + "\n")
                    alert_log.flush()
                log.write(
                    json.dumps(
                        {
                            "frame": index,
                            "timestamp": timestamp,
                            "tracks": [asdict(t) for t in tracks],
                            "relations": edges,
                            "objects": objects,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                if dashboard is not None:
                    ok_jpeg, jpeg = cv2.imencode(".jpg", annotated)
                    dashboard.publish(
                        {
                            "status": "running",
                            "synthetic": bool(getattr(models, "synthetic", False)),
                            "objects": objects,
                            "alerts": alert_history,
                            "timestamp": timestamp,
                        },
                        jpeg.tobytes() if ok_jpeg else None,
                    )
                if image_mode:
                    if not cv2.imwrite(str(args.output / "annotated.jpg"), annotated):
                        raise RuntimeError("Cannot save annotated image")
                else:
                    writer.write(annotated)
                index += 1
                if not args.no_preview:
                    cv2.imshow("TeamSentinel - RelateAnything | Q to quit", annotated)
                    if cv2.waitKey(0 if image_mode else 1) & 0xFF in (27, ord("q")):
                        break
                if image_mode or (args.max_frames and index >= args.max_frames):
                    break
                if dashboard is not None and not webcam:
                    # Keep offline video readable in the live web dashboard.
                    time.sleep(max(0, started + index / fps - time.monotonic()))
                ok, frame = capture.read()
                if not ok:
                    break
        summary = {
            "frames": index,
            "source": args.source,
            "output": str(args.output.resolve()),
            "elapsed_seconds": time.monotonic() - started,
            "hold_seconds": args.hold_seconds,
            "alerts": len(alert_history),
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        if dashboard is not None:
            dashboard.publish(
                {
                    "status": "finished",
                    "synthetic": bool(getattr(models, "synthetic", False)),
                    "objects": objects,
                    "alerts": alert_history,
                    "timestamp": timestamp,
                }
            )
            if writer is not None:
                writer.release()
            print(
                "Video finished. Dashboard stays open; press Ctrl+C to stop.",
                flush=True,
            )
            dashboard.wait()
        return summary
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if dashboard is not None:
            dashboard.close()
        if not args.no_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    cli = parser()
    print(
        "Based on RelateAnything by Ma\u00eblic Neau - https://github.com/Maelic/RelateAnything"
    )
    try:
        result = run(cli.parse_args())
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        cli.exit(1, f"Error: {error}\n")
