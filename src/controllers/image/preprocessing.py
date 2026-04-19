from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
import torch
import httpx

from ...config import settings
from ...helpers.celeba_face_align import (
    align_bgr_with_wing,
    get_eye_centers_from_landmarks,
)
from ...helpers.eye_rescue import blend_source_eyes, should_rescue_dark_eyes
from ...models import database


class ImagePreprocessingMixin:
    """Face preparation, quality checks, and domain inference."""

    def _prepare_face_bgr(
        self,
        image_bgr: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        target_size: int = 256,
        strict_face_detection: bool = False,
    ) -> np.ndarray:
        if len(image_bgr.shape) == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

        if strict_face_detection:
            remote_crop, remote_status = self._crop_with_remote_service(
                image_bgr,
                target_size=target_size,
            )
            if remote_crop is not None:
                remote_landmarks = self._detect_landmarks(remote_crop)
                if self.wing_face_aligner is not None:
                    try:
                        return align_bgr_with_wing(
                            self.wing_face_aligner,
                            remote_crop,
                            output_size=target_size,
                            landmarks=remote_landmarks,
                        )
                    except Exception as exc:
                        print(
                            "Wing face alignment on detector crop failed, "
                            f"keeping detector crop: {exc}"
                        )
                return self._ensure_model_input_size(
                    remote_crop,
                    target_size=target_size,
                )

            opencv_crop = self._crop_with_opencv_detector(
                image_bgr,
                target_size=target_size,
            )
            if opencv_crop is not None:
                opencv_landmarks = self._detect_landmarks(opencv_crop)
                if self.wing_face_aligner is not None:
                    try:
                        return align_bgr_with_wing(
                            self.wing_face_aligner,
                            opencv_crop,
                            output_size=target_size,
                            landmarks=opencv_landmarks,
                        )
                    except Exception as exc:
                        print(
                            "Wing face alignment on OpenCV crop failed, "
                            f"keeping OpenCV crop: {exc}"
                        )
                return self._ensure_model_input_size(
                    opencv_crop,
                    target_size=target_size,
                )

            if remote_status == "no_face":
                raise ValueError(
                    "No usable face detected. Move closer to the camera or upload "
                    "a photo where the face is larger."
                )

        if self.wing_face_aligner is not None:
            try:
                return align_bgr_with_wing(
                    self.wing_face_aligner,
                    image_bgr,
                    output_size=target_size,
                    landmarks=landmarks,
                )
            except Exception as exc:
                print(f"Wing face alignment failed, falling back to crop service: {exc}")

        remote_crop, _ = self._crop_with_remote_service(
            image_bgr,
            target_size=target_size,
        )
        if remote_crop is not None:
            return remote_crop

        raise ValueError("Face crop service unavailable or no face detected")

    def _resize_with_aspect_and_pad(
        self,
        image_bgr: np.ndarray,
        target_size: int = 256,
    ) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        if height <= 0 or width <= 0:
            return cv2.resize(
                image_bgr,
                (target_size, target_size),
                interpolation=cv2.INTER_LANCZOS4,
            )

        scale = min(target_size / float(width), target_size / float(height))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            image_bgr,
            (new_width, new_height),
            interpolation=cv2.INTER_LANCZOS4,
        )
        pad_width = target_size - new_width
        pad_height = target_size - new_height
        left = pad_width // 2
        right = pad_width - left
        top = pad_height // 2
        bottom = pad_height - top
        return cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_REFLECT_101,
        )

    def _crop_with_remote_service(
        self,
        image_bgr: np.ndarray,
        target_size: int = 256,
    ) -> Tuple[Optional[np.ndarray], str]:
        urls = [settings.FACE_CROP_SERVICE_URL]
        if (
            "localhost" not in settings.FACE_CROP_SERVICE_URL
            and "127.0.0.1" not in settings.FACE_CROP_SERVICE_URL
        ):
            urls.append("http://localhost:8010/crop")

        saw_reachable_service = False
        saw_no_face = False
        try:
            ok, encoded = cv2.imencode(".png", image_bgr)
            if not ok:
                return None, "encode_failed"

            for url in urls:
                files = {"file": ("frame.png", encoded.tobytes(), "image/png")}
                data = {
                    "size": str(target_size),
                    "padding_left": str(settings.FACE_MARGIN_LEFT),
                    "padding_right": str(settings.FACE_MARGIN_RIGHT),
                    "padding_top": str(settings.FACE_MARGIN_TOP),
                    "padding_bottom": str(settings.FACE_MARGIN_BOTTOM),
                }
                try:
                    response = requests.post(
                        url,
                        files=files,
                        data=data,
                        timeout=settings.FACE_CROP_TIMEOUT_SECONDS,
                    )
                except requests.exceptions.Timeout:
                    continue
                except requests.exceptions.RequestException:
                    continue

                saw_reachable_service = True
                if response.status_code == 422:
                    saw_no_face = True
                    continue
                if response.status_code != 200:
                    continue

                decoded = np.frombuffer(response.content, np.uint8)
                cropped = cv2.imdecode(decoded, cv2.IMREAD_COLOR)
                if cropped is None:
                    continue
                return self._resize_with_aspect_and_pad(
                    cropped,
                    target_size=target_size,
                ), "ok"

            if saw_no_face:
                return None, "no_face"
            if saw_reachable_service:
                return None, "service_error"
            return None, "service_unavailable"
        except Exception:
            return None, "service_unavailable"

    def _get_opencv_face_cascade(self):
        if self.opencv_face_cascade is None:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(str(cascade_path))
            if cascade.empty():
                return None
            self.opencv_face_cascade = cascade
        return self.opencv_face_cascade

    def _detect_face_bbox_with_opencv(
        self,
        image_bgr: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        cascade = self._get_opencv_face_cascade()
        if cascade is None:
            return None

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        min_side = min(gray.shape[:2])
        min_face = max(20, int(round(min_side * 0.06)))
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(min_face, min_face),
        )
        if faces is None or len(faces) == 0:
            return None

        height, width = gray.shape[:2]
        image_center = np.array([width * 0.5, height * 0.5], dtype=np.float32)
        best_face = None
        best_score = None
        norm = float(max(height, width))
        for x, y, box_width, box_height in faces:
            center = np.array(
                [x + box_width * 0.5, y + box_height * 0.5],
                dtype=np.float32,
            )
            center_dist = float(np.linalg.norm(center - image_center) / max(norm, 1.0))
            area = float(box_width * box_height)
            score = area / (1.0 + 8.0 * center_dist * center_dist)
            if best_score is None or score > best_score:
                best_score = score
                best_face = (int(x), int(y), int(box_width), int(box_height))
        return best_face

    def _count_faces_in_image(self, image_bgr: np.ndarray) -> int:
        """Count the number of faces detected in an image."""
        cascade = self._get_opencv_face_cascade()
        if cascade is None:
            return 0

        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            min_side = min(gray.shape[:2])
            min_face = max(20, int(round(min_side * 0.06)))
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(min_face, min_face),
            )
            return len(faces) if faces is not None else 0
        except Exception as e:
            print(f"Face counting warning: {e}")
            return 0

    def _crop_with_opencv_detector(
        self,
        image_bgr: np.ndarray,
        target_size: int = 256,
    ) -> Optional[np.ndarray]:
        bbox = self._detect_face_bbox_with_opencv(image_bgr)
        if bbox is None:
            return None

        x, y, box_width, box_height = bbox
        height, width = image_bgr.shape[:2]
        x1 = max(0, int(np.floor(x - box_width * float(settings.FACE_MARGIN_LEFT))))
        x2 = min(
            width,
            int(np.ceil(x + box_width * (1.0 + float(settings.FACE_MARGIN_RIGHT)))),
        )
        y1 = max(0, int(np.floor(y - box_height * float(settings.FACE_MARGIN_TOP))))
        y2 = min(
            height,
            int(np.ceil(y + box_height * (1.0 + float(settings.FACE_MARGIN_BOTTOM)))),
        )
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return self._resize_with_aspect_and_pad(crop, target_size=target_size)

    def _preprocess_image(
        self,
        image_bgr: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        target_size: int = 256,
        strict_face_detection: bool = False,
    ) -> np.ndarray:
        return self._prepare_face_bgr(
            image_bgr,
            landmarks=landmarks,
            target_size=target_size,
            strict_face_detection=strict_face_detection,
        )

    def _ensure_model_input_size(
        self,
        image_bgr: np.ndarray,
        target_size: int = 256,
    ) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        if height == target_size and width == target_size:
            return image_bgr
        return self._resize_with_aspect_and_pad(image_bgr, target_size=target_size)

    @staticmethod
    def _resize_exact(image_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
        if image_bgr.shape[1] == width and image_bgr.shape[0] == height:
            return image_bgr
        return cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LANCZOS4)

    def _compute_face_quality_metrics(self, image_bgr: np.ndarray) -> dict:
        face_bgr = self._ensure_model_input_size(image_bgr, target_size=256)
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
        gradient_p90 = float(np.percentile(gradient_mag, 90))
        contrast_std = float(gray.std())
        return {
            "laplacian_var": laplacian_var,
            "gradient_p90": gradient_p90,
            "contrast_std": contrast_std,
        }

    def _is_translation_face_quality_too_low(
        self,
        image_bgr: np.ndarray,
    ) -> Tuple[bool, dict]:
        metrics = self._compute_face_quality_metrics(image_bgr)
        hard_blur = metrics["laplacian_var"] < float(settings.TRANSLATION_MIN_LAPLACIAN_VAR)
        soft_blur = (
            metrics["laplacian_var"] < float(settings.TRANSLATION_SOFT_MIN_LAPLACIAN_VAR)
            and metrics["gradient_p90"] < float(settings.TRANSLATION_MIN_GRADIENT_P90)
            and metrics["contrast_std"] < float(settings.TRANSLATION_MIN_CONTRAST_STD)
        )
        return hard_blur or soft_blur, metrics

    def _get_translation_face_quality_issue(
        self,
        image_bgr: np.ndarray,
        image_role: str,
        trace_context: Optional[str] = None,
    ) -> Optional[str]:
        if not bool(getattr(settings, "TRANSLATION_QUALITY_GATE_ENABLED", True)):
            return None

        rejected, metrics = self._is_translation_face_quality_too_low(image_bgr)
        print(
            f"{image_role} face quality metrics: "
            f"lap_var={metrics['laplacian_var']:.2f}, "
            f"grad_p90={metrics['gradient_p90']:.2f}, "
            f"contrast_std={metrics['contrast_std']:.2f}"
        )
        if not rejected:
            return None

        self._trace_image(
            image_bgr,
            f"quality_gate_rejected_{image_role}",
            image_role,
            trace_context=trace_context,
        )
        return image_role

    def _assert_translation_pair_quality(
        self,
        source_bgr: np.ndarray,
        reference_bgr: np.ndarray,
        trace_context: Optional[str] = None,
    ) -> None:
        source_issue = self._get_translation_face_quality_issue(
            source_bgr,
            "source",
            trace_context=trace_context,
        )
        reference_issue = self._get_translation_face_quality_issue(
            reference_bgr,
            "reference",
            trace_context=trace_context,
        )

        if source_issue and reference_issue:
            raise ValueError(
                "Both source and reference faces are too small or blurry for reliable "
                "translation. Please upload closer, sharper source and reference faces."
            )
        if source_issue:
            raise ValueError(
                "Source face is too small or blurry for reliable translation. "
                "Please upload a closer, sharper source face."
            )
        if reference_issue:
            raise ValueError(
                "Reference face is too small or blurry for reliable translation. "
                "Please upload a closer, sharper reference face."
            )

    def _detect_landmarks(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self.face_detector is not None:
            try:
                _, landmarks = self.face_detector.detect_landmarks(image_bgr)
                if landmarks is not None:
                    return landmarks
            except Exception as exc:
                print(f"FaceDetector landmarks failed: {exc}")

        if self.wing_face_aligner is not None:
            try:
                from ...helpers.face_detection import FaceDetector

                temp_detector = FaceDetector(fan_model=self.wing_face_aligner.fan)
                _, landmarks = temp_detector.detect_landmarks(image_bgr)
                return landmarks
            except Exception as exc:
                print(f"Wing FAN landmarks failed: {exc}")
        return None

    def _should_realign_face(
        self,
        image_bgr: np.ndarray,
        landmarks: Optional[np.ndarray],
    ) -> bool:
        if landmarks is None:
            return True

        height, width = image_bgr.shape[:2]
        if height <= 0 or width <= 0:
            return True

        left_eye, right_eye = get_eye_centers_from_landmarks(landmarks)
        eye_dist_frac = float(np.linalg.norm(right_eye - left_eye)) / float(width)
        roll_deg = abs(
            float(
                np.degrees(
                    np.arctan2(
                        right_eye[1] - left_eye[1],
                        right_eye[0] - left_eye[0],
                    )
                )
            )
        )
        face_w_frac = float(np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])) / float(width)
        return roll_deg > 5.0 or eye_dist_frac < 0.18 or face_w_frac < 0.38

    def _normalize_stored_face_for_model(
        self,
        image_bgr: np.ndarray,
        target_size: int = 256,
    ) -> np.ndarray:
        landmarks = self._detect_landmarks(image_bgr)
        if self._should_realign_face(image_bgr, landmarks):
            # Only realign if necessary - skip verbose logging for routine operations
            return self._prepare_face_bgr(
                image_bgr,
                landmarks=landmarks,
                target_size=target_size,
            )
        return self._ensure_model_input_size(image_bgr, target_size=target_size)

    def _get_inference_fan_model(self):
        if self.face_detector is not None and getattr(self.face_detector, "model", None) is not None:
            return self.face_detector.model
        return self.fan_model

    def _rescue_translated_eyes(
        self,
        translated_bgr: np.ndarray,
        source_doc: dict,
        trace_context: Optional[str] = None,
    ) -> np.ndarray:
        if not bool(getattr(settings, "TRANSLATION_EYE_RESCUE_ENABLED", True)):
            return translated_bgr

        try:
            source_display_bgr = self._decode_stored_image(
                source_doc,
                prefer_model_variant=False,
            )
            source_display_bgr = self._resize_exact(
                source_display_bgr,
                translated_bgr.shape[1],
                translated_bgr.shape[0],
            )
            landmarks = self._detect_landmarks(source_display_bgr)
            if landmarks is None:
                return translated_bgr

            if should_rescue_dark_eyes(translated_bgr, source_display_bgr, landmarks):
                print("Rescuing dark translated eyes from source canonical face")
                rescued = blend_source_eyes(
                    translated_bgr,
                    source_display_bgr,
                    landmarks,
                    alpha=float(getattr(settings, "TRANSLATION_EYE_RESCUE_ALPHA", 0.88)),
                )
                self._trace_image(
                    rescued,
                    "translated_eye_rescued_output",
                    "translated",
                    trace_context=trace_context,
                )
                return rescued
        except Exception as exc:
            print(f"Eye rescue skipped: {exc}")
        return translated_bgr

    def _infer_domain_label_from_face(self, image_bgr: np.ndarray) -> Optional[int]:
        if self.generator is None or self.style_encoder is None:
            return None

        face_bgr = self._ensure_model_input_size(image_bgr, target_size=256)
        image_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        device = next(self.generator.parameters()).device
        x = torch.from_numpy(image_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device)
        x = (x / 127.5) - 1.0

        fan_model = self._get_inference_fan_model()
        masks = None
        if fan_model is not None and float(settings.W_HPF) > 0:
            try:
                masks = fan_model.get_heatmap(x)
            except Exception as exc:
                print(f"Domain inference FAN heatmap failed: {exc}")

        best_label = None
        best_score = None
        with torch.no_grad():
            for label in sorted(self.label_to_domain):
                y = torch.tensor([label], dtype=torch.long, device=device)
                style = self.style_encoder(x, y)
                if masks is not None:
                    recon = self.generator(x, style, masks=masks)
                else:
                    recon = self.generator(x, style)
                score = float(torch.mean(torch.abs(recon - x)).item())
                if best_score is None or score < best_score:
                    best_score = score
                    best_label = label
        return best_label

    async def _ensure_image_domain_metadata(
        self,
        image_doc: dict,
        image_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[str], Optional[int]]:
        image_domain = image_doc.get("image_domain")
        domain_label = image_doc.get("domain_label")

        if image_domain in self.domain_to_label and domain_label is None:
            domain_label = self.domain_to_label[image_domain]
        elif domain_label in self.label_to_domain and image_domain is None:
            image_domain = self.label_to_domain[domain_label]

        if domain_label is None:
            try:
                if image_bgr is None and (
                    image_doc.get("model_image_data") or image_doc.get("image_data")
                ):
                    image_bgr = self._decode_stored_image(
                        image_doc,
                        prefer_model_variant=True,
                    )
                if image_bgr is not None:
                    domain_label = self._infer_domain_label_from_face(image_bgr)
                    if domain_label in self.label_to_domain:
                        image_domain = self.label_to_domain[domain_label]
            except Exception as exc:
                print(f"Image domain inference failed: {exc}")

        if image_doc.get("_id") and (
            image_doc.get("image_domain") != image_domain
            or image_doc.get("domain_label") != domain_label
        ):
            await database["images"].update_one(
                {"_id": image_doc["_id"]},
                {
                    "$set": {
                        "image_domain": image_domain,
                        "domain_label": domain_label,
                        "updated_at": datetime.utcnow(),
                    }
                },
            )
            image_doc["image_domain"] = image_domain
            image_doc["domain_label"] = domain_label

        return image_domain, domain_label
