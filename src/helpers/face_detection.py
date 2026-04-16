from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from .celeba_face_align import letterbox_bgr_for_fan, map_fan_landmarks_to_original


class FaceDetector:
    """FAN-based facial landmarks with lazy FAN import to avoid startup crashes."""

    def __init__(
        self,
        wing_model_path: str = None,
        fan_model: Optional[torch.nn.Module] = None,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.wing_model_path = wing_model_path or ""
        self.model = fan_model
        if self.model is None:
            self._load_model()
        else:
            self.model.eval()

    def _load_model(self):
        from ..wing import FAN

        if self.wing_model_path and os.path.exists(self.wing_model_path):
            self.model = FAN(fname_pretrained=self.wing_model_path).to(self.device).eval()
        else:
            self.model = FAN().to(self.device).eval()

    @torch.no_grad()
    def detect_landmarks(self, image: np.ndarray) -> Tuple[int, Optional[np.ndarray]]:
        try:
            if self.model is None:
                return (0, None)

            canvas_rgb, meta = letterbox_bgr_for_fan(image, size=256)
            image_tensor = torch.from_numpy(canvas_rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            image_tensor = image_tensor / 255.0 * 2 - 1
            landmarks_tensor = self.model.get_landmark(image_tensor)
            landmarks_np = landmarks_tensor.cpu().numpy()
            if len(landmarks_np.shape) == 3:
                landmarks_np = landmarks_np[0]
            landmarks_orig = map_fan_landmarks_to_original(landmarks_np, meta)
            return (1, landmarks_orig.astype(np.float32))
        except Exception:
            return (0, None)


class RoughFaceCropper:
    @staticmethod
    def align_face_region(image: np.ndarray, landmarks: np.ndarray, output_size: int = 256) -> np.ndarray:
        return cv2.resize(image, (output_size, output_size))
