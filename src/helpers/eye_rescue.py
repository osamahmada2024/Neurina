from __future__ import annotations

import cv2
import numpy as np

LEFT_EYE_IDX = np.array(list(range(60, 68)) + [96])
RIGHT_EYE_IDX = np.array(list(range(68, 76)) + [97])


def _eye_soft_mask(
    image_shape: tuple[int, int, int] | tuple[int, int],
    eye_points: np.ndarray,
    expand_x: float = 1.15,
    expand_y: float = 1.55,
    shift_y: float = 0.12,
) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.float32)
    eye = np.asarray(eye_points, dtype=np.float32)
    x, y, w, h = cv2.boundingRect(eye.astype(np.int32))
    cx = x + w / 2.0
    cy = y + h / 2.0 + h * shift_y
    rx = max(5, int(round((w / 2.0) * expand_x)))
    ry = max(4, int(round((h / 2.0) * expand_y)))
    cv2.ellipse(
        mask,
        (int(round(cx)), int(round(cy))),
        (rx, ry),
        0,
        0,
        360,
        1.0,
        -1,
    )
    blur = max(rx, ry) * 2 + 1
    if blur % 2 == 0:
        blur += 1
    return np.clip(cv2.GaussianBlur(mask, (blur, blur), 0), 0.0, 1.0)


def build_eye_preservation_mask(image_shape, landmarks: np.ndarray) -> np.ndarray:
    landmarks = np.asarray(landmarks, dtype=np.float32)
    mask = _eye_soft_mask(image_shape, landmarks[LEFT_EYE_IDX])
    mask += _eye_soft_mask(image_shape, landmarks[RIGHT_EYE_IDX])
    return np.clip(mask, 0.0, 1.0)


def should_rescue_dark_eyes(
    translated_bgr: np.ndarray,
    source_bgr: np.ndarray,
    landmarks: np.ndarray,
    darkness_ratio: float = 0.85,
    absolute_value_threshold: float = 72.0,
) -> bool:
    mask = build_eye_preservation_mask(translated_bgr.shape, landmarks)
    if float(mask.sum()) < 1e-3:
        return False

    translated_v = cv2.cvtColor(translated_bgr, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    source_v = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)

    translated_eye_mean = float((translated_v * mask).sum() / (mask.sum() + 1e-6))
    source_eye_mean = float((source_v * mask).sum() / (mask.sum() + 1e-6))

    return (
        translated_eye_mean < absolute_value_threshold
        and translated_eye_mean < (source_eye_mean * darkness_ratio)
    )


def blend_source_eyes(
    translated_bgr: np.ndarray,
    source_bgr: np.ndarray,
    landmarks: np.ndarray,
    alpha: float = 0.88,
) -> np.ndarray:
    mask = build_eye_preservation_mask(translated_bgr.shape, landmarks)[:, :, None]
    blended = (
        translated_bgr.astype(np.float32) * (1.0 - mask * alpha)
        + source_bgr.astype(np.float32) * (mask * alpha)
    )
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)
