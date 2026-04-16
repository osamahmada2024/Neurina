from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def detect_bbox(img_bgr, face_mesh):
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    xs = np.array([p.x * w for p in lm], dtype=np.float32)
    ys = np.array([p.y * h for p in lm], dtype=np.float32)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--max-images", type=int, default=800)
    ap.add_argument("--single-image", default="")
    ap.add_argument("--out-json", default="C:/Users/osama/Downloads/neurina_style_transfer/celebahq_mediapipe_stats.json")
    args = ap.parse_args()

    folder = Path(args.dir)
    files = sorted([p for p in folder.glob("*.jpg")])[: args.max_images]
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    )

    margins = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        bbox = detect_bbox(img, face_mesh)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        m = {
            "file": p.name,
            "left": x1 / bw,
            "right": (img.shape[1] - x2) / bw,
            "top": y1 / bh,
            "bottom": (img.shape[0] - y2) / bh,
            "eye_line_y_ratio": None,
            "mouth_y_ratio": None,
        }
        margins.append(m)

    if not margins:
        raise SystemExit("No faces detected.")

    arr = {k: np.array([m[k] for m in margins], dtype=np.float32) for k in ["left", "right", "top", "bottom"]}
    stats = {
        "count": len(margins),
        "median": {k: float(np.median(v)) for k, v in arr.items()},
        "p25": {k: float(np.percentile(v, 25)) for k, v in arr.items()},
        "p75": {k: float(np.percentile(v, 75)) for k, v in arr.items()},
    }

    single = {}
    if args.single_image:
        p = Path(args.single_image)
        img = cv2.imread(str(p))
        if img is not None:
            bbox = detect_bbox(img, face_mesh)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                single = {
                    "file": p.name,
                    "left": float(x1 / bw),
                    "right": float((img.shape[1] - x2) / bw),
                    "top": float(y1 / bh),
                    "bottom": float((img.shape[0] - y2) / bh),
                }

    out = {"dataset_stats": stats, "single_image": single}
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

