"""Run RelateAnything on an image, video or webcam with short occlusion memory."""

import argparse
import json
import math
import os
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

from alerts import PERSONAL_ITEMS, AbandonmentMonitor
from capture import LatestCapture
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
        "--latest-camera",
        action="store_true",
        help="Read webcam continuously and drop old buffered frames",
    )
    cli.add_argument("--camera-fps", type=float, default=30)
    cli.add_argument("--detector-size", type=int, default=640)
    cli.add_argument(
        "--no-record",
        action="store_true",
        help="Avoid MP4 encoding during live camera tests",
    )
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
    if (
        not math.isfinite(args.camera_fps)
        or args.camera_fps <= 0
        or args.detector_size < 32
    ):
        raise ValueError("Invalid camera FPS or detector size")
    AbandonmentMonitor(
        args.exit_seconds,
        args.unattended_seconds,
        args.owner_threshold,
        args.owner_samples,
    )


class RFDetrDetector:
    def __init__(
        self,
        model_path="models/rf-detr.onnx",
        config_path="models/rf-detr.json",
        conf=0.35,
        max_det=32,
        device="cpu",
    ):
        import json
        import onnxruntime as ort

        self.conf = conf
        self.max_det = max_det
        cfg_file = (
            Path(config_path)
            if Path(config_path).is_file()
            else Path(model_path).with_suffix(".json")
        )
        if cfg_file.is_file():
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.class_names = cfg.get("class_names", [])
        else:
            self.class_names = [
                "person",
                "phone",
                "wallet",
                "keys",
                "bag",
                "laptop",
                "dog",
                "cat",
            ]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = os.cpu_count() or 8
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if device.startswith("cuda"):
            try:
                import torch
                torch_lib = Path(torch.__file__).parent / "lib"
                if torch_lib.is_dir():
                    os.add_dll_directory(str(torch_lib))
                    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
            except Exception:
                pass

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        try:
            self.session = ort.InferenceSession(
                str(model_path), opts, providers=providers
            )
        except Exception:
            self.session = ort.InferenceSession(
                str(model_path), opts, providers=["CPUExecutionProvider"]
            )
        active_provider = self.session.get_providers()[0]
        print(f"RF-DETR ONNX provider: {active_provider}", flush=True)

        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.input_h = shape[2] if len(shape) > 2 and isinstance(shape[2], int) else 384
        self.input_w = shape[3] if len(shape) > 3 and isinstance(shape[3], int) else 384
        import numpy as np

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def detect(self, frame):
        import cv2
        import numpy as np

        h_orig, w_orig = frame.shape[:2]
        resized = cv2.resize(frame, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        inp = ((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None] - self.mean) / self.std
        dets, labels = self.session.run(
            None, {self.input_name: inp.astype(np.float32)}
        )
        scores = 1.0 / (1.0 + np.exp(-labels[0]))
        cls_scores = scores[:, : len(self.class_names)]
        best_cls = cls_scores.argmax(axis=1)
        best_conf = cls_scores.max(axis=1)
        valid = np.where(best_conf >= self.conf)[0]
        if len(valid) == 0:
            return []
        valid = valid[np.argsort(-best_conf[valid])][: self.max_det]
        out = []
        for idx in valid:
            c = best_cls[idx]
            conf = float(best_conf[idx])
            cx, cy, w, h = dets[0, idx]
            x1 = max(0.0, float((cx - w / 2) * w_orig))
            y1 = max(0.0, float((cy - h / 2) * h_orig))
            x2 = min(float(w_orig), float((cx + w / 2) * w_orig))
            y2 = min(float(h_orig), float((cy + h / 2) * h_orig))
            out.append(Detection((x1, y1, x2, y2), conf, self.class_names[c]))
        return out


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
        if self.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable. Run setup_gpu.ps1 or select --device cpu."
            )
        print(
            f"Inference device: {self.device}"
            + (f" ({torch.cuda.get_device_name(0)})" if self.device != "cpu" else ""),
            flush=True,
        )
        torch.set_num_threads(os.cpu_count() or 8)
        if self.device != "cpu":
            torch.backends.cudnn.benchmark = True
        detector_str = str(args.detector).lower()
        self.is_onnx_detector = (
            detector_str.endswith(".onnx")
            or detector_str in ("rf-detr", "rfdetr", "r-detr")
        )
        if self.is_onnx_detector:
            model_path = (
                Path("models/rf-detr.onnx")
                if detector_str in ("rf-detr", "rfdetr", "r-detr")
                else Path(args.detector)
            )
            self.detector = RFDetrDetector(
                model_path=model_path,
                conf=args.det_confidence,
                max_det=args.max_boxes,
                device=self.device,
            )
        else:
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
        self.detector_size = args.detector_size

    def detect(self, frame):
        if self.is_onnx_detector:
            return self.detector.detect(frame)
        result = self.detector.predict(
            frame,
            device=self.device,
            verbose=False,
            conf=self.confidence,
            max_det=self.max_boxes,
            imgsz=self.detector_size,
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


def display_tracks(tracks, timestamp):
    """Keep lost tracks in logic, but avoid drawing old boxes over new observations."""
    visible = [track for track in tracks if track.observed]
    shown = list(visible)
    for track in tracks:
        if track.observed or track.label == "person":
            continue
        if not track.stationary and timestamp - track.last_seen > 1:
            continue
        a = track.bbox
        obscured = False
        for current in visible:
            if current.label != track.label and current.label != "person":
                continue
            b = current.bbox
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0, min(a[3], b[3]) - max(a[1], b[1])
            )
            area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
            smaller = min(
                area_a,
                max(0, b[2] - b[0]) * max(0, b[3] - b[1]),
            )
            if (smaller and intersection / smaller >= 0.5) or (
                area_a > 0 and intersection / area_a >= 0.35
            ):
                obscured = True
                break
        if not obscured:
            shown.append(track)
    return shown


def draw(frame, tracks, relations, timestamp):
    import cv2

    output = frame.copy()
    tracks = display_tracks(tracks, timestamp)
    by_id = {t.track_id: t for t in tracks}
    for track in tracks:
        color = (80, 230, 110) if track.observed else (0, 210, 255)
        x1, y1, x2, y2 = map(int, track.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        status = (
            f"{track.confidence:.2f}"
            if track.observed
            else f"LOST {timestamp - track.last_seen:.1f}s"
        )
        label = f"#{track.track_id} {track.label} {status}"
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = max(0, min(x1, frame.shape[1] - width - 8))
        text_y = max(height + 6, min(y1 - 8, frame.shape[0] - 8))
        cv2.rectangle(
            output,
            (text_x, text_y - height - 4),
            (text_x + width + 5, text_y + 4),
            (15, 20, 30),
            -1,
        )
        cv2.putText(
            output,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    for edge in relations:
        if edge["subject_id"] not in by_id or edge["object_id"] not in by_id:
            continue
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
    capture = writer = dashboard = latest_camera = None
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
            if webcam:
                capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Source has no decodable frame")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                fps = 30.0
            if webcam and args.latest_camera:
                latest_camera = LatestCapture(capture)
        models = models or Models(args)
        if latest_camera is not None:
            ok, frame = latest_camera.read()
            if not ok:
                raise RuntimeError("Camera stopped while loading model")
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
        if not image_mode and not args.no_record:
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
        previous_completed = started
        frame_times = deque(maxlen=30)
        metrics = {}
        logged_states = {}
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
                detections = [
                    d
                    for d in models.detect(frame)
                    if d.label == "person" or d.label in PERSONAL_ITEMS
                ]
                tracks = tracker.update(detections, timestamp)
                inputs = relation_inputs(tracks)[: args.max_boxes]
                relation_ms = 0.0
                if (
                    index % args.relation_every == 0
                    and len(inputs) >= 2
                    and any(t.label == "person" for t in inputs)
                    and any(t.label in PERSONAL_ITEMS for t in inputs)
                ):
                    relation_start = time.monotonic()
                    cache = update_relations(
                        cache, models.relate(frame, inputs), inputs, timestamp
                    )
                    relation_ms = (time.monotonic() - relation_start) * 1000
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
                for obj in objects:
                    oid = obj["object_id"]
                    curr_state = (
                        obj.get("state"),
                        obj.get("owner_id"),
                        obj.get("predicate"),
                        round(obj.get("owner_score") or 0.0, 1),
                    )
                    if logged_states.get(oid) != curr_state:
                        logged_states[oid] = curr_state
                        if obj.get("owner_id") is not None:
                            pred = obj.get("predicate") or "near"
                            sc = obj.get("owner_score") or 0.0
                            fused = obj.get("fused_score") or sc
                            print(
                                f"[TRACK] #{oid} {obj['label']} -> Owner #{obj['owner_id']} | Relation: {pred} (score: {sc:.2f}, fused: {fused:.2f}) | {obj['state']}",
                                flush=True,
                            )
                        elif obj.get("candidate_id") is not None:
                            c_pred = obj.get("candidate_predicate") or "interacting"
                            c_sc = obj.get("candidate_score") or 0.0
                            print(
                                f"[TRACK] #{oid} {obj['label']} | Candidate: Owner #{obj['candidate_id']} ({c_pred} {c_sc:.2f}) | Samples: {obj.get('candidate_samples', 0)}",
                                flush=True,
                            )
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
                        print(
                            f"[ALERT] #{event['object_id']} {event['class_name']} ABANDONED (Reason: {event['reason']}, Owner: #{event['owner_id']}, Relation: {event.get('predicate', 'unknown')})",
                            flush=True,
                        )
                    elif event["type"] == "RESOLVED":
                        print(
                            f"[RESOLVED] #{event['object_id']} {event['class_name']} re-attached to Owner #{event['owner_id']}",
                            flush=True,
                        )
                    alert_log.write(json.dumps(event, allow_nan=False) + "\n")
                    alert_log.flush()
                completed = time.monotonic()
                frame_times.append(max(0.000001, completed - previous_completed))
                previous_completed = completed
                metrics = {
                    "inference_fps": round(len(frame_times) / sum(frame_times), 1),
                    "camera_requested_fps": args.camera_fps if webcam else fps,
                    "relation_ms": round(relation_ms, 1),
                    "device": str(getattr(models, "device", "synthetic")),
                }
                cv2.rectangle(
                    annotated,
                    (0, annotated.shape[0] - 48),
                    (annotated.shape[1], annotated.shape[0] - 22),
                    (15, 20, 30),
                    -1,
                )
                cv2.putText(
                    annotated,
                    f"{metrics['device']} | inference {metrics['inference_fps']:.1f} FPS | relation {relation_ms:.0f} ms",
                    (8, annotated.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (110, 240, 180),
                    1,
                )
                log.write(
                    json.dumps(
                        {
                            "frame": index,
                            "timestamp": timestamp,
                            "tracks": [asdict(t) for t in tracks],
                            "relations": edges,
                            "objects": objects,
                            "metrics": metrics,
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
                            "relations": edges,
                            "alerts": alert_history,
                            "timestamp": timestamp,
                            "metrics": metrics,
                        },
                        jpeg.tobytes() if ok_jpeg else None,
                    )
                if image_mode:
                    if not cv2.imwrite(str(args.output / "annotated.jpg"), annotated):
                        raise RuntimeError("Cannot save annotated image")
                elif writer is not None:
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
                ok, frame = (
                    latest_camera.read()
                    if latest_camera is not None
                    else capture.read()
                )
                if not ok:
                    break
        summary = {
            "frames": index,
            "source": args.source,
            "output": str(args.output.resolve()),
            "elapsed_seconds": time.monotonic() - started,
            "hold_seconds": args.hold_seconds,
            "alerts": len(alert_history),
            "metrics": metrics,
            "video_saved": writer is not None,
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
                    "relations": edges,
                    "alerts": alert_history,
                    "timestamp": timestamp,
                    "metrics": metrics,
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
        if latest_camera is not None:
            latest_camera.close()
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
