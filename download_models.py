"""Download RelateAnything sidecars and detector weights for local inference."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/relsgg-vits16plus"))
    args = parser.parse_args()
    # Set before importing Hugging Face: avoid Xet stalls observed on this machine.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    from huggingface_hub import snapshot_download
    from ultralytics import YOLO

    args.output.mkdir(parents=True, exist_ok=True)
    print(
        "Downloading relation model and sidecars; rerun this command to resume.",
        flush=True,
    )
    snapshot_download(
        "maelic/relsgg-vits16plus",
        revision="2db90096be5217bdc7a9003c042950f45723d105",
        local_dir=str(args.output),
        max_workers=1,
        allow_patterns=[
            "model.pth",
            "text_student.pt",
            "tokenizer*",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "calibration.json",
            "predicate_embeddings.npz",
        ],
    )
    YOLO("models/yolo26n.pt")
    print(f'Ready: --relation-model "{args.output / "model.pth"}"')


if __name__ == "__main__":
    main()
