from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    from ..wing import FaceAligner as WingFaceAligner


def _letterbox_rgb(img_rgb: np.ndarray, size: int = 256) -> Tuple[np.ndarray, dict]:
    h, w = img_rgb.shape[:2]
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (size - new_w) // 2
    pad_h = (size - new_h) // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
    meta = {"scale": scale, "pad_w": pad_w, "pad_h": pad_h, "orig_w": w, "orig_h": h}
    return canvas, meta


def _landmarks_canvas_to_original(landmarks: np.ndarray, meta: dict) -> np.ndarray:
    out = np.empty_like(landmarks, dtype=np.float64)
    s = meta["scale"]
    out[:, 0] = (landmarks[:, 0].astype(np.float64) - meta["pad_w"]) / s
    out[:, 1] = (landmarks[:, 1].astype(np.float64) - meta["pad_h"]) / s
    return out


def letterbox_bgr_for_fan(image_bgr: np.ndarray, size: int = 256) -> Tuple[np.ndarray, dict]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return _letterbox_rgb(rgb, size=size)


def map_fan_landmarks_to_original(landmarks: np.ndarray, meta: dict) -> np.ndarray:
    return _landmarks_canvas_to_original(landmarks, meta)


def get_eye_centers_from_landmarks(landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    left_idx = np.array(list(range(60, 68)) + [96])
    right_idx = np.array(list(range(68, 76)) + [97])
    left_eye = landmarks[left_idx].mean(axis=0)
    right_eye = landmarks[right_idx].mean(axis=0)
    return left_eye.astype(np.float32), right_eye.astype(np.float32)


def get_mouth_center_from_landmarks(landmarks: np.ndarray) -> np.ndarray:
    mouth_center = landmarks[[76, 82]].mean(axis=0)
    return mouth_center.astype(np.float32)


def get_alignment_anchor_points(landmarks: np.ndarray) -> np.ndarray:
    left_eye, right_eye = get_eye_centers_from_landmarks(landmarks)
    mouth_center = get_mouth_center_from_landmarks(landmarks)
    return np.stack([left_eye, right_eye, mouth_center]).astype(np.float32)


def _default_reference_anchors(output_size: int) -> np.ndarray:
    fractions = np.array(
        [
            [0.38065471, 0.47963243],
            [0.62844727, 0.47854397],
            [0.50063965, 0.72117228],
        ],
        dtype=np.float32,
    )
    return fractions * float(output_size)


def get_reference_anchor_points(
    output_size: int,
    celeb_ref: Optional[np.ndarray] = None,
    reference_size: int = 256,
) -> np.ndarray:
    if celeb_ref is None:
        return _default_reference_anchors(output_size)
    scale = float(output_size) / float(reference_size) if reference_size else 1.0
    return get_alignment_anchor_points(np.asarray(celeb_ref, dtype=np.float32) * scale)


def apply_affine_to_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([pts, ones], axis=1)
    return hom @ matrix.T


def estimate_similarity_transform(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> np.ndarray:
    matrix, _ = cv2.estimateAffinePartial2D(
        np.asarray(src_points, dtype=np.float32),
        np.asarray(dst_points, dtype=np.float32),
        method=cv2.LMEDS,
    )
    if matrix is not None:
        return matrix.astype(np.float32)

    src_left, src_right, _ = np.asarray(src_points, dtype=np.float32)
    dst_left, dst_right, _ = np.asarray(dst_points, dtype=np.float32)
    src_vec = src_right - src_left
    dst_vec = dst_right - dst_left
    src_dist = float(np.linalg.norm(src_vec))
    dst_dist = float(np.linalg.norm(dst_vec))
    if src_dist < 1e-6 or dst_dist < 1e-6:
        raise ValueError("Could not estimate face alignment transform")

    src_angle = float(np.degrees(np.arctan2(src_vec[1], src_vec[0])))
    dst_angle = float(np.degrees(np.arctan2(dst_vec[1], dst_vec[0])))
    scale = dst_dist / src_dist
    src_mid = (src_left + src_right) * 0.5
    dst_mid = (dst_left + dst_right) * 0.5

    matrix = cv2.getRotationMatrix2D(tuple(src_mid), dst_angle - src_angle, scale)
    transformed_mid = apply_affine_to_points(src_mid.reshape(1, 2), matrix)[0]
    matrix[:, 2] += dst_mid - transformed_mid
    return matrix.astype(np.float32)


def _detect_wing_landmarks_on_original(
    wing_aligner: "WingFaceAligner",
    image_bgr: np.ndarray,
) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    lb, meta = _letterbox_rgb(rgb, size=256)
    tensor_lb = torch.from_numpy(lb).float().permute(2, 0, 1).unsqueeze(0).to(wing_aligner.device)
    tensor_lb = (tensor_lb / 255.0) * 2 - 1
    with torch.no_grad():
        landmarks_lb = wing_aligner.fan.get_landmark(tensor_lb).cpu().numpy()[0]
    return _landmarks_canvas_to_original(landmarks_lb, meta).astype(np.float32)


def get_tight_face_crop_bgr(image_bgr: np.ndarray, landmarks: np.ndarray, margin: float = 0.0) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x_min = float(np.min(landmarks[:, 0]))
    x_max = float(np.max(landmarks[:, 0]))
    y_min = float(np.min(landmarks[:, 1]))
    y_max = float(np.max(landmarks[:, 1]))
    bw = x_max - x_min
    bh = y_max - y_min
    x1 = max(0, int(np.floor(x_min - bw * margin)))
    y1 = max(0, int(np.floor(y_min - bh * margin)))
    x2 = min(rgb.shape[1], int(np.ceil(x_max + bw * margin)))
    y2 = min(rgb.shape[0], int(np.ceil(y_max + bh * margin)))
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return image_bgr
    return cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)


def get_tight_face_crop_with_wing(
    wing_aligner: "WingFaceAligner", image_bgr: np.ndarray, margin: float = 0.0
) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    lb, meta = _letterbox_rgb(rgb, size=256)
    tensor_lb = torch.from_numpy(lb).float().permute(2, 0, 1).unsqueeze(0).to(wing_aligner.device)
    tensor_lb = (tensor_lb / 255.0) * 2 - 1
    with torch.no_grad():
        landmarks_lb = wing_aligner.fan.get_landmark(tensor_lb).cpu().numpy()[0]
    lm = _landmarks_canvas_to_original(landmarks_lb, meta)
    n = len(lm)

    if n >= 27:
        y_min = float(np.min(lm[17:27, 1]))
    else:
        y_min = float(np.min(lm[:, 1]))
    y_max = float(lm[8, 1]) if n > 8 else float(np.max(lm[:, 1]))
    x_min = float(lm[0, 0]) if n > 0 else float(np.min(lm[:, 0]))
    x_max = float(lm[16, 0]) if n > 16 else float(np.max(lm[:, 0]))

    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min

    x1 = max(0, int(np.floor(x_min)))
    y1 = max(0, int(np.floor(y_min)))
    x2 = min(w, int(np.ceil(x_max)))
    y2 = min(h, int(np.ceil(y_max)))
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return image_bgr
    return cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)


def align_bgr_with_wing(
    wing_aligner: "WingFaceAligner",
    image_bgr: np.ndarray,
    output_size: int = 256,
    landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Geometry-preserving face alignment:
    rotate + uniform scale + translation using eyes and mouth only.
    This keeps facial proportions intact while making the eyes horizontal.
    """
    if landmarks is None:
        landmarks = _detect_wing_landmarks_on_original(wing_aligner, image_bgr)
    else:
        landmarks = np.asarray(landmarks, dtype=np.float32)

    src_points = get_alignment_anchor_points(landmarks)
    celeb_ref = getattr(wing_aligner, "CELEB_REF", None)
    reference_size = int(getattr(wing_aligner, "output_size", output_size) or output_size)
    dst_points = get_reference_anchor_points(output_size, celeb_ref=celeb_ref, reference_size=reference_size)
    matrix = estimate_similarity_transform(src_points, dst_points)
    return cv2.warpAffine(
        image_bgr,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
