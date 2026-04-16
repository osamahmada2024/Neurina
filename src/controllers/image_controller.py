from __future__ import annotations

import asyncio
import base64
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
import torch
from bson import ObjectId

from ..helpers.image_helpers import convert_base64_to_image, validate_image_file
from ..helpers.celeba_face_align import align_bgr_with_wing, get_eye_centers_from_landmarks
from ..helpers.eye_rescue import blend_source_eyes, should_rescue_dark_eyes
from ..config import settings
from ..models import database
from ..services.image_translation_service import translate_using_reference
from ..services.face_restoration_service import FaceRestorationService
from ..models.Enums import (
    ImageErrorMessage,
    ImageStatus,
    ImageType,
    TaskStatus,
    ValidationErrorMessage,
)
from ..schemes.image_schema import ImageUploadResponseSchema

PUBLIC_LIBRARY_USER_ID = ObjectId("000000000000000000000001")


class ImageController:
    """Handle image upload, tight face preprocessing, and translation."""

    def __init__(self):
        self.face_detector = None
        self.opencv_face_cascade = None
        self.wing_face_aligner = None
        self.fan_model = None
        self.generator = None
        self.style_encoder = None
        self.face_restoration_service = None
        self.trace_output_dir = Path(settings.TRACE_OUTPUT_DIR)
        self.trace_output_dir.mkdir(parents=True, exist_ok=True)
        self.domain_to_label = {"female": 0, "male": 1}
        self.label_to_domain = {label: domain for domain, label in self.domain_to_label.items()}
        self.supported_translation_modes = {
            "auto",
            "male_to_female",
            "female_to_male",
            "male_to_male",
            "female_to_female",
        }

    def initialize_models(
        self,
        generator,
        style_encoder,
        fan_model=None,
        wing_model_path: str = None,
        celeba_lm_path: str = None,
    ) -> None:
        self.generator = generator
        self.style_encoder = style_encoder
        self.fan_model = fan_model
        self.wing_face_aligner = None
        self.face_detector = None
        if wing_model_path and celeba_lm_path:
            try:
                from ..wing import FaceAligner as WingFaceAligner

                self.wing_face_aligner = WingFaceAligner(
                    wing_model_path,
                    celeba_lm_path,
                    256,
                )
            except Exception as exc:
                print(f"WingFaceAligner init skipped: {exc}")
        if wing_model_path:
            try:
                from ..helpers.face_detection import FaceDetector

                if fan_model is not None:
                    self.face_detector = FaceDetector(fan_model=fan_model)
                else:
                    self.face_detector = FaceDetector(wing_model_path=wing_model_path)
            except Exception as exc:
                print(f"FaceDetector init skipped: {exc}")

    def initialize_postprocessors(self, base_path: str) -> None:
        self.face_restoration_service = None
        if not bool(settings.SR_ENABLED):
            return
        try:
            self.face_restoration_service = FaceRestorationService(
                base_path=base_path,
                model_name=settings.SR_MODEL_NAME,
                outscale=float(settings.SR_OUTSCALE),
                tile=int(settings.SR_TILE),
                face_weight=float(settings.SR_FACE_WEIGHT),
                codeformer_fidelity=float(settings.SR_CODEFORMER_FIDELITY),
            )
            print(
                f"✓ Super-resolution ready: model={settings.SR_MODEL_NAME}, outscale={settings.SR_OUTSCALE}"
            )
        except Exception as exc:
            print(f"Super-resolution init skipped: {exc}")

    @staticmethod
    def _coerce_object_id(value, field_name: str) -> ObjectId:
        if isinstance(value, ObjectId):
            return value
        try:
            return ObjectId(str(value))
        except Exception as exc:
            raise ValueError(f"Invalid {field_name}") from exc

    @staticmethod
    def _sanitize_trace_context(trace_context: Optional[str]) -> str:
        if not trace_context:
            return ""
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(trace_context))

    def _trace_image(
        self,
        image_bgr: np.ndarray,
        stage: str,
        image_type: str = "unknown",
        trace_context: Optional[str] = None,
    ) -> None:
        """Save tracing frames and optional landmarks overlay."""
        if image_bgr is None or not isinstance(image_bgr, np.ndarray):
            return
        try:
            self.trace_output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            safe_stage = stage.replace(" ", "_")
            safe_type = image_type.replace(" ", "_")
            safe_context = self._sanitize_trace_context(trace_context)
            name_parts = [ts]
            if safe_context:
                name_parts.append(safe_context)
            name_parts.extend([safe_type, safe_stage])
            stem = "__".join(name_parts)

            raw_path = self.trace_output_dir / f"{stem}.png"
            cv2.imwrite(str(raw_path), image_bgr)

            marker = self.trace_output_dir / f"{stem}__landmarks_none.txt"
            marker.write_text("Landmarks disabled in core API (using face-crop microservice).", encoding="utf-8")
        except Exception as exc:
            print(f"Trace save warning ({stage}): {exc}")

    @staticmethod
    def _tensor_to_bgr_image(tensor: torch.Tensor) -> np.ndarray:
        image_rgb = tensor.detach().squeeze(0).clamp(-1, 1)
        image_rgb = ((image_rgb + 1.0) * 127.5).round().to(torch.uint8)
        image_rgb = image_rgb.permute(1, 2, 0).cpu().numpy()
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _encode_png_base64(image_bgr: np.ndarray) -> str:
        ok, buffer = cv2.imencode(".png", image_bgr)
        if not ok:
            raise ValueError("Failed to encode image as PNG")
        return base64.b64encode(buffer).decode()

    def _decode_stored_image(self, image_doc: dict, prefer_model_variant: bool = False) -> np.ndarray:
        if prefer_model_variant and image_doc.get("model_image_data"):
            return convert_base64_to_image(image_doc["model_image_data"])
        return convert_base64_to_image(image_doc["image_data"])

    @staticmethod
    def _serialize_image_doc(image_doc: dict) -> dict:
        doc = dict(image_doc)
        doc.pop("model_image_data", None)
        if doc.get("_id") is not None:
            doc["_id"] = str(doc["_id"])
        if doc.get("user_id") is not None:
            doc["user_id"] = str(doc["user_id"])
        return doc

    @staticmethod
    def _relative_library_key(image_path: Path, root_dir: Path) -> str:
        try:
            relative = image_path.resolve().relative_to(root_dir.resolve())
        except Exception:
            relative = image_path.name
        return str(relative).replace("\\", "/").lower()

    async def _get_accessible_reference_image(self, image_id: str, user_id: ObjectId) -> Optional[dict]:
        return await database["images"].find_one(
            {
                "_id": self._coerce_object_id(image_id, "reference_image_id"),
                "image_type": ImageType.REFERENCE.value,
                "$or": [
                    {"user_id": user_id},
                    {"is_public": True},
                ],
            }
        )

    def _prepare_face_bgr(
        self,
        image_bgr: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        target_size: int = 256,
        strict_face_detection: bool = False,
    ) -> np.ndarray:
        """Prepare face like CelebA-HQ alignment (Wing), with fallbacks."""
        if len(image_bgr.shape) == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

        if strict_face_detection:
            remote_crop, remote_status = self._crop_with_remote_service(image_bgr, target_size=target_size)
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
                        print(f"Wing face alignment on detector crop failed, keeping detector crop: {exc}")
                return self._ensure_model_input_size(remote_crop, target_size=target_size)

            opencv_crop = self._crop_with_opencv_detector(image_bgr, target_size=target_size)
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
                        print(f"Wing face alignment on OpenCV crop failed, keeping OpenCV crop: {exc}")
                return self._ensure_model_input_size(opencv_crop, target_size=target_size)

            if remote_status == "no_face":
                raise ValueError(
                    "No usable face detected. Move closer to the camera or upload a photo where the face is larger."
                )

        if self.wing_face_aligner is not None:
            try:
                aligned = align_bgr_with_wing(
                    self.wing_face_aligner,
                    image_bgr,
                    output_size=target_size,
                    landmarks=landmarks,
                )
                return aligned
            except Exception as exc:
                print(f"Wing face alignment failed, falling back to crop service: {exc}")
        remote_crop, _ = self._crop_with_remote_service(image_bgr, target_size=target_size)
        if remote_crop is not None:
            return remote_crop
        raise ValueError("Face crop service unavailable or no face detected")

    def _resize_with_aspect_and_pad(self, image_bgr: np.ndarray, target_size: int = 256) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return cv2.resize(image_bgr, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        scale = min(target_size / float(w), target_size / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        pad_w = target_size - new_w
        pad_h = target_size - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        return cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_REFLECT_101,
        )

    def _crop_with_remote_service(
        self, image_bgr: np.ndarray, target_size: int = 256
    ) -> Tuple[Optional[np.ndarray], str]:
        urls = [settings.FACE_CROP_SERVICE_URL]
        if "localhost" not in settings.FACE_CROP_SERVICE_URL and "127.0.0.1" not in settings.FACE_CROP_SERVICE_URL:
            urls.append("http://localhost:8010/crop")

        saw_reachable_service = False
        saw_no_face = False
        try:
            # Use lossless PNG to avoid pre-crop quality degradation.
            ok, enc = cv2.imencode(".png", image_bgr)
            if not ok:
                return None, "encode_failed"
            for url in urls:
                files = {"file": ("frame.png", enc.tobytes(), "image/png")}
                data = {
                    "size": str(target_size),
                    "padding_left": str(settings.FACE_MARGIN_LEFT),
                    "padding_right": str(settings.FACE_MARGIN_RIGHT),
                    "padding_top": str(settings.FACE_MARGIN_TOP),
                    "padding_bottom": str(settings.FACE_MARGIN_BOTTOM),
                }
                resp = requests.post(url, files=files, data=data, timeout=settings.FACE_CROP_TIMEOUT_SECONDS)
                saw_reachable_service = True
                if resp.status_code == 422:
                    saw_no_face = True
                    continue
                if resp.status_code != 200:
                    continue
                arr = np.frombuffer(resp.content, np.uint8)
                cropped = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if cropped is None:
                    continue
                return self._resize_with_aspect_and_pad(cropped, target_size=target_size), "ok"
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

    def _detect_face_bbox_with_opencv(self, image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
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

        h, w = gray.shape[:2]
        image_center = np.array([w * 0.5, h * 0.5], dtype=np.float32)
        best_face = None
        best_score = None
        norm = float(max(h, w))
        for x, y, bw, bh in faces:
            center = np.array([x + bw * 0.5, y + bh * 0.5], dtype=np.float32)
            center_dist = float(np.linalg.norm(center - image_center) / max(norm, 1.0))
            area = float(bw * bh)
            score = area / (1.0 + 8.0 * center_dist * center_dist)
            if best_score is None or score > best_score:
                best_score = score
                best_face = (int(x), int(y), int(bw), int(bh))
        return best_face

    def _crop_with_opencv_detector(
        self,
        image_bgr: np.ndarray,
        target_size: int = 256,
    ) -> Optional[np.ndarray]:
        bbox = self._detect_face_bbox_with_opencv(image_bgr)
        if bbox is None:
            return None

        x, y, bw, bh = bbox
        h, w = image_bgr.shape[:2]
        x1 = max(0, int(np.floor(x - bw * float(settings.FACE_MARGIN_LEFT))))
        x2 = min(w, int(np.ceil(x + bw * (1.0 + float(settings.FACE_MARGIN_RIGHT)))))
        y1 = max(0, int(np.floor(y - bh * float(settings.FACE_MARGIN_TOP))))
        y2 = min(h, int(np.ceil(y + bh * (1.0 + float(settings.FACE_MARGIN_BOTTOM)))))
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
        """Kept for compatibility. Uses strict face-only path."""
        return self._prepare_face_bgr(
            image_bgr,
            landmarks=landmarks,
            target_size=target_size,
            strict_face_detection=strict_face_detection,
        )

    def _ensure_model_input_size(self, image_bgr: np.ndarray, target_size: int = 256) -> np.ndarray:
        """Keep already-cropped images intact; only resize when shape is off-target."""
        h, w = image_bgr.shape[:2]
        if h == target_size and w == target_size:
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

    def _is_translation_face_quality_too_low(self, image_bgr: np.ndarray) -> Tuple[bool, dict]:
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
                "Both source and reference faces are too small or blurry for reliable translation. "
                "Please upload closer, sharper source and reference faces."
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
                from ..helpers.face_detection import FaceDetector

                temp_detector = FaceDetector(fan_model=self.wing_face_aligner.fan)
                _, landmarks = temp_detector.detect_landmarks(image_bgr)
                return landmarks
            except Exception as exc:
                print(f"Wing FAN landmarks failed: {exc}")
        return None

    def _should_realign_face(self, image_bgr: np.ndarray, landmarks: Optional[np.ndarray]) -> bool:
        if landmarks is None:
            return True

        h, w = image_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return True

        left_eye, right_eye = get_eye_centers_from_landmarks(landmarks)
        eye_dist_frac = float(np.linalg.norm(right_eye - left_eye)) / float(w)
        roll_deg = abs(float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))))
        face_w_frac = float(np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])) / float(w)

        return roll_deg > 5.0 or eye_dist_frac < 0.18 or face_w_frac < 0.38

    def _normalize_stored_face_for_model(self, image_bgr: np.ndarray, target_size: int = 256) -> np.ndarray:
        """
        Heal older DB images that were saved before alignment fixes.
        Re-align only when the face is too small or the roll angle is visibly off.
        """
        landmarks = self._detect_landmarks(image_bgr)
        if self._should_realign_face(image_bgr, landmarks):
            print("Re-aligning stored face before inference to recover old non-canonical crops")
            return self._prepare_face_bgr(image_bgr, landmarks=landmarks, target_size=target_size)
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
            source_display_bgr = self._decode_stored_image(source_doc, prefer_model_variant=False)
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
        """
        Infer the StarGAN-v2 domain by checking which style branch best reconstructs the same face.
        This avoids the bad default of always assuming label 0 when the client omitted image_domain.
        """
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
                if image_bgr is None and (image_doc.get("model_image_data") or image_doc.get("image_data")):
                    image_bgr = self._decode_stored_image(image_doc, prefer_model_variant=True)
                if image_bgr is not None:
                    domain_label = self._infer_domain_label_from_face(image_bgr)
                    if domain_label in self.label_to_domain:
                        image_domain = self.label_to_domain[domain_label]
            except Exception as exc:
                print(f"Image domain inference failed: {exc}")

        if image_doc.get("_id") and (image_doc.get("image_domain") != image_domain or image_doc.get("domain_label") != domain_label):
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

    def _build_processed_image_doc(
        self,
        image_array: np.ndarray,
        user_id: ObjectId,
        image_type: str,
        filename: str,
        image_domain: Optional[str] = None,
        trace_context: Optional[str] = None,
        extra_fields: Optional[dict] = None,
        display_image_bgr: Optional[np.ndarray] = None,
    ) -> dict:
        if image_type not in {ImageType.SOURCE.value, ImageType.REFERENCE.value}:
            raise ValueError(ValidationErrorMessage.INVALID_IMAGE_TYPE.value)

        domain_label = None
        if image_domain is not None:
            if image_domain not in self.domain_to_label:
                raise ValueError("image_domain must be one of: male, female")
            domain_label = int(self.domain_to_label[image_domain])

        self._trace_image(image_array, "upload_raw_in", image_type, trace_context=trace_context)

        landmarks = None
        if self.face_detector is not None:
            _, lm = self.face_detector.detect_landmarks(image_array)
            landmarks = lm

        preprocessed_image = self._preprocess_image(
            image_array,
            landmarks,
            target_size=256,
            strict_face_detection=True,
        )
        detected_faces = 1
        self._trace_image(preprocessed_image, "upload_face_crop_out", image_type, trace_context=trace_context)

        model_image_bgr = self._normalize_stored_face_for_model(preprocessed_image, target_size=256)
        self._trace_image(model_image_bgr, "upload_model_face_final", image_type, trace_context=trace_context)

        if domain_label is None:
            inferred_label = self._infer_domain_label_from_face(model_image_bgr)
            if inferred_label is not None:
                domain_label = int(inferred_label)
                image_domain = self.label_to_domain.get(domain_label)

        image_for_storage_bgr = display_image_bgr.copy() if display_image_bgr is not None else model_image_bgr
        if display_image_bgr is None:
            if self.face_restoration_service is not None and bool(settings.UPLOAD_SR_ENABLED):
                try:
                    image_for_storage_bgr = self.face_restoration_service.enhance(
                        model_image_bgr,
                        outscale=float(settings.UPLOAD_SR_OUTSCALE),
                    )
                    self._trace_image(
                        image_for_storage_bgr,
                        "upload_face_upscaled_out",
                        image_type,
                        trace_context=trace_context,
                    )
                except Exception as exc:
                    print(f"Upload super-resolution skipped: {exc}")
        else:
            self._trace_image(
                image_for_storage_bgr,
                "upload_display_image_preserved",
                image_type,
                trace_context=trace_context,
            )

        image_base64 = self._encode_png_base64(image_for_storage_bgr)
        model_image_base64 = self._encode_png_base64(model_image_bgr)
        self._trace_image(image_for_storage_bgr, "db_save_image", image_type, trace_context=trace_context)

        image_doc = {
            "user_id": user_id,
            "image_type": image_type,
            "image_data": image_base64,
            "model_image_data": model_image_base64,
            "original_filename": filename,
            "status": ImageStatus.PREPROCESSED.value,
            "faces_detected": int(detected_faces),
            "landmarks": landmarks.tolist() if landmarks is not None else None,
            "image_domain": image_domain,
            "domain_label": domain_label,
            "display_resolution": {
                "width": int(image_for_storage_bgr.shape[1]),
                "height": int(image_for_storage_bgr.shape[0]),
            },
            "model_resolution": {
                "width": int(model_image_bgr.shape[1]),
                "height": int(model_image_bgr.shape[0]),
            },
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        if extra_fields:
            image_doc.update(extra_fields)
        return image_doc

    async def upload_and_process_image(
        self,
        file,
        user_id: ObjectId,
        image_type: str,
        filename: str,
        image_domain: Optional[str] = None,
    ) -> ImageUploadResponseSchema:
        try:
            if not validate_image_file(filename):
                raise ValueError(ImageErrorMessage.INVALID_FORMAT.value)

            contents = await file.read()
            image_array = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
            if image_array is None:
                raise ValueError(ImageErrorMessage.READ_ERROR.value)
            image_doc = self._build_processed_image_doc(
                image_array=image_array,
                user_id=user_id,
                image_type=image_type,
                filename=filename,
                image_domain=image_domain,
            )
            result = await database["images"].insert_one(image_doc)
            return ImageUploadResponseSchema(
                image_id=str(result.inserted_id),
                status=ImageStatus.PREPROCESSED.value,
                message=f"Image uploaded and processed successfully. {int(image_doc['faces_detected'])} face(s) detected.",
                faces_detected=int(image_doc["faces_detected"]),
            )
        except ValueError:
            raise
        except Exception as exc:
            raise Exception(f"Error processing image: {exc}")

    async def import_public_reference_image(
        self,
        image_path: str | Path,
        root_dir: str | Path,
        image_domain: Optional[str] = None,
        public_collection: str = "ref_database",
    ) -> dict:
        path = Path(image_path)
        root = Path(root_dir)
        if not path.is_file():
            raise ValueError(f"Public reference file not found: {path}")
        if not validate_image_file(path.name):
            raise ValueError(f"Unsupported image format for public reference: {path.name}")

        library_key = self._relative_library_key(path, root)
        if image_domain is None:
            leading_dir = library_key.split("/", 1)[0].strip().lower()
            if leading_dir in self.domain_to_label:
                image_domain = leading_dir
        if image_domain not in self.domain_to_label:
            raise ValueError(
                f"Could not infer image_domain from {path}. Put files under male/ or female/, or pass image_domain explicitly."
            )

        image_array = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image_array is None:
            raise ValueError(f"Could not read image file: {path}")

        trace_context = f"public_ref_{library_key.replace('/', '_')}"
        image_doc = self._build_processed_image_doc(
            image_array=image_array,
            user_id=PUBLIC_LIBRARY_USER_ID,
            image_type=ImageType.REFERENCE.value,
            filename=path.name,
            image_domain=image_domain,
            trace_context=trace_context,
            display_image_bgr=image_array,
            extra_fields={
                "is_public": True,
                "public_collection": public_collection,
                "library_key": library_key,
                "library_source_path": str(path.resolve()),
            },
        )

        existing = await database["images"].find_one(
            {"is_public": True, "public_collection": public_collection, "library_key": library_key}
        )
        if existing:
            image_doc["created_at"] = existing.get("created_at", image_doc["created_at"])
            await database["images"].update_one(
                {"_id": existing["_id"]},
                {"$set": image_doc},
            )
            image_id = existing["_id"]
            action = "updated"
        else:
            result = await database["images"].insert_one(image_doc)
            image_id = result.inserted_id
            action = "inserted"

        return {
            "_id": str(image_id),
            "status": action,
            "image_domain": image_domain,
            "library_key": library_key,
            "original_filename": path.name,
        }

    async def import_public_references_from_directory(
        self,
        root_dir: str | Path,
        public_collection: str = "ref_database",
        limit: Optional[int] = None,
    ) -> dict:
        root = Path(root_dir)
        if not root.is_dir():
            raise ValueError(f"Public reference directory not found: {root}")

        image_paths = sorted(path for path in root.rglob("*") if path.is_file() and validate_image_file(path.name))
        if limit is not None:
            image_paths = image_paths[: max(0, int(limit))]

        inserted = 0
        updated = 0
        failed = 0
        imported = []
        errors = []

        for path in image_paths:
            try:
                result = await self.import_public_reference_image(
                    image_path=path,
                    root_dir=root,
                    public_collection=public_collection,
                )
                imported.append(result)
                if result["status"] == "inserted":
                    inserted += 1
                else:
                    updated += 1
            except Exception as exc:
                failed += 1
                errors.append({"path": str(path), "error": str(exc)})

        return {
            "root_dir": str(root.resolve()),
            "public_collection": public_collection,
            "processed": len(image_paths),
            "inserted": inserted,
            "updated": updated,
            "failed": failed,
            "images": imported,
            "errors": errors,
        }

    async def get_user_images(
        self,
        user_id: ObjectId,
        image_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list:
        query = {"user_id": user_id}
        if image_type:
            query["image_type"] = image_type
        images = await database["images"].find(query).skip(offset).limit(limit).to_list(None)
        return [self._serialize_image_doc(img) for img in images]

    async def count_user_images(self, user_id: ObjectId, image_type: Optional[str] = None) -> int:
        query = {"user_id": user_id}
        if image_type:
            query["image_type"] = image_type
        return int(await database["images"].count_documents(query))

    async def get_public_reference_images(
        self,
        image_domain: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list:
        query = {"image_type": ImageType.REFERENCE.value, "is_public": True}
        if image_domain is not None:
            if image_domain not in self.domain_to_label:
                raise ValueError("image_domain must be one of: male, female")
            query["image_domain"] = image_domain

        images = (
            await database["images"]
            .find(query)
            .sort("library_key", 1)
            .skip(offset)
            .limit(limit)
            .to_list(None)
        )
        return [self._serialize_image_doc(img) for img in images]

    async def count_public_reference_images(self, image_domain: Optional[str] = None) -> int:
        query = {"image_type": ImageType.REFERENCE.value, "is_public": True}
        if image_domain is not None:
            if image_domain not in self.domain_to_label:
                raise ValueError("image_domain must be one of: male, female")
            query["image_domain"] = image_domain
        return int(await database["images"].count_documents(query))

    async def get_image_by_id_with_ownership(self, image_id: str, user_id: ObjectId) -> dict:
        image = await database["images"].find_one(
            {
                "_id": self._coerce_object_id(image_id, "image_id"),
                "$or": [
                    {"user_id": user_id},
                    {"is_public": True},
                ],
            }
        )
        if not image:
            raise ValueError("Image not found or not authorized")
        return self._serialize_image_doc(image)

    async def get_translated_image_with_ownership(self, task_id: str, user_id: ObjectId) -> dict:
        task = await database["translation_tasks"].find_one(
            {"_id": self._coerce_object_id(task_id, "task_id"), "user_id": user_id}
        )
        if not task:
            raise ValueError("Task not found or not authorized")
        if task.get("status") != TaskStatus.COMPLETED.value:
            raise ValueError(f"Task status is {task.get('status', 'unknown')}, not completed")
        image = await database["images"].find_one(
            {"_id": self._coerce_object_id(task["translated_image_id"], "translated_image_id")}
        )
        if not image:
            raise ValueError("Image not found")
        image["_id"] = str(image["_id"])
        image["user_id"] = str(image["user_id"])
        return image

    async def create_translation_task(
        self,
        user_id: ObjectId,
        source_image_id: str,
        reference_image_id: str,
        translation_mode: str = "female_to_female",
    ) -> dict:
        source = await database["images"].find_one(
            {
                "_id": self._coerce_object_id(source_image_id, "source_image_id"),
                "user_id": user_id,
                "image_type": ImageType.SOURCE.value,
            }
        )
        reference = await self._get_accessible_reference_image(reference_image_id, user_id)
        if not source:
            raise ValueError("Source image not found")
        if not reference:
            raise ValueError("Reference image not found")

        source_face_bgr = self._normalize_stored_face_for_model(
            self._decode_stored_image(source, prefer_model_variant=True),
            target_size=256,
        )
        reference_face_bgr = self._normalize_stored_face_for_model(
            self._decode_stored_image(reference, prefer_model_variant=True),
            target_size=256,
        )
        self._assert_translation_pair_quality(source_face_bgr, reference_face_bgr)

        if translation_mode == "auto":
            src_domain, source_label = await self._ensure_image_domain_metadata(source)
            target_domain, target_label = await self._ensure_image_domain_metadata(reference)
            if target_label is None:
                raise ValueError(
                    "Could not determine reference image domain automatically. "
                    "Please re-upload the reference with image_domain=male|female or choose translation_mode explicitly."
                )
        elif translation_mode not in self.supported_translation_modes:
            raise ValueError(
                "Invalid translation_mode. Use: auto | male_to_female | female_to_male | male_to_male | female_to_female"
            )
        else:
            src_domain, target_domain = translation_mode.split("_to_")
            source_label = self.domain_to_label[src_domain]
            target_label = self.domain_to_label[target_domain]

        task_doc = {
            "user_id": user_id,
            "source_image_id": ObjectId(source_image_id),
            "reference_image_id": ObjectId(reference_image_id),
            "source_domain": src_domain,
            "target_domain": target_domain,
            "source_domain_label": source_label,
            "target_domain_label": target_label,
            "translation_mode": translation_mode,
            "translated_image_id": None,
            "status": TaskStatus.PENDING.value,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "error_message": None,
        }
        result = await database["translation_tasks"].insert_one(task_doc)
        task_id = str(result.inserted_id)
        asyncio.create_task(self._process_translation_task(task_id, source, reference, task_doc))
        return {"_id": task_id, "status": TaskStatus.PENDING.value, "message": "Translation task created."}

    async def _process_translation_task(self, task_id: str, source_doc: dict, reference_doc: dict, task_doc: dict):
        try:
            trace_context = f"task_{task_id}"
            await database["translation_tasks"].update_one(
                {"_id": ObjectId(task_id)},
                {"$set": {"status": TaskStatus.PROCESSING.value, "updated_at": datetime.utcnow()}},
            )

            source_image_bgr = self._decode_stored_image(source_doc, prefer_model_variant=True)
            reference_image_bgr = self._decode_stored_image(reference_doc, prefer_model_variant=True)
            self._trace_image(source_image_bgr, "db_load_source_in", "source", trace_context=trace_context)
            self._trace_image(reference_image_bgr, "db_load_reference_in", "reference", trace_context=trace_context)

            # Images are already cropped during upload. Avoid re-cropping to preserve quality.
            source_image_bgr = self._normalize_stored_face_for_model(source_image_bgr, target_size=256)
            reference_image_bgr = self._normalize_stored_face_for_model(reference_image_bgr, target_size=256)
            self._trace_image(source_image_bgr, "model_input_source_face", "source", trace_context=trace_context)
            self._trace_image(
                reference_image_bgr,
                "model_input_reference_face",
                "reference",
                trace_context=trace_context,
            )

            self._assert_translation_pair_quality(
                source_image_bgr,
                reference_image_bgr,
                trace_context=trace_context,
            )

            if self.generator is None or self.style_encoder is None:
                raise ValueError("Models not loaded")

            source_image = cv2.cvtColor(source_image_bgr, cv2.COLOR_BGR2RGB)
            reference_image = cv2.cvtColor(reference_image_bgr, cv2.COLOR_BGR2RGB)

            with torch.no_grad():
                device = next(self.generator.parameters()).device
                x_src = torch.from_numpy(source_image).float().permute(2, 0, 1).unsqueeze(0).to(device)
                x_ref = torch.from_numpy(reference_image).float().permute(2, 0, 1).unsqueeze(0).to(device)
                x_src = (x_src / 127.5) - 1.0
                x_ref = (x_ref / 127.5) - 1.0
                nets = SimpleNamespace(generator=self.generator, style_encoder=self.style_encoder)
                if self.face_detector is not None and getattr(self.face_detector, "model", None) is not None:
                    nets.fan = self.face_detector.model
                elif self.fan_model is not None:
                    nets.fan = self.fan_model
                infer_args = SimpleNamespace(w_hpf=float(settings.W_HPF))

                target_label = int(task_doc.get("target_domain_label", int(settings.REFERENCE_DOMAIN_LABEL)))
                if task_doc.get("translation_mode") == "auto":
                    _, inferred_target_label = await self._ensure_image_domain_metadata(
                        reference_doc,
                        image_bgr=reference_image_bgr,
                    )
                    if inferred_target_label is not None:
                        target_label = int(inferred_target_label)
                y = torch.tensor([target_label], dtype=torch.long, device=device)
                translated = translate_using_reference(nets, infer_args, x_src, x_ref, y)


            translated_bgr = self._tensor_to_bgr_image(translated)
            self._trace_image(
                translated_bgr,
                "translated_model_output",
                "translated",
                trace_context=trace_context,
            )

            final_bgr = translated_bgr
            if self.face_restoration_service is not None:
                try:
                    final_bgr = self.face_restoration_service.enhance(
                        translated_bgr,
                        outscale=float(settings.SR_OUTSCALE),
                    )
                    self._trace_image(
                        final_bgr,
                        "translated_upscaled_output",
                        "translated",
                        trace_context=trace_context,
                    )
                except Exception as exc:
                    print(f"Super-resolution skipped during translation: {exc}")

            final_bgr = self._rescue_translated_eyes(final_bgr, source_doc, trace_context=trace_context)

            ok, buffer = cv2.imencode(".png", final_bgr)
            if not ok:
                raise ValueError("Failed to encode translated output")
            translated_base64 = base64.b64encode(buffer).decode()
            self._trace_image(final_bgr, "db_save_image", "translated", trace_context=trace_context)

            translated_doc = {
                "user_id": source_doc["user_id"],
                "image_type": ImageType.TRANSLATED.value,
                "image_data": translated_base64,
                "original_filename": f"translated_{source_doc.get('original_filename', 'image.jpg')}",
                "status": ImageStatus.TRANSLATION_COMPLETED.value,
                "faces_detected": 1,
                "landmarks": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
            res = await database["images"].insert_one(translated_doc)
            await database["translation_tasks"].update_one(
                {"_id": ObjectId(task_id)},
                {
                    "$set": {
                        "status": TaskStatus.COMPLETED.value,
                        "translated_image_id": res.inserted_id,
                        "updated_at": datetime.utcnow(),
                    }
                },
            )
        except Exception as exc:
            await database["translation_tasks"].update_one(
                {"_id": ObjectId(task_id)},
                {
                    "$set": {
                        "status": TaskStatus.FAILED.value,
                        "error_message": str(exc),
                        "updated_at": datetime.utcnow(),
                    }
                },
            )

    async def get_translation_tasks(self, user_id: ObjectId, limit: int = 50, offset: int = 0) -> list:
        tasks = await database["translation_tasks"].find({"user_id": user_id}).skip(offset).limit(limit).to_list(None)
        for task in tasks:
            task["_id"] = str(task["_id"])
            task["user_id"] = str(task["user_id"])
            task["source_image_id"] = str(task["source_image_id"])
            task["reference_image_id"] = str(task["reference_image_id"])
            if task.get("translated_image_id"):
                task["translated_image_id"] = str(task["translated_image_id"])
        return tasks

    async def count_translation_tasks(self, user_id: ObjectId) -> int:
        return int(await database["translation_tasks"].count_documents({"user_id": user_id}))

    def get_tight_face_crop(self, image_bgr: np.ndarray) -> np.ndarray:
        return self._prepare_face_bgr(image_bgr, target_size=256)

    async def delete_image(self, image_id: str, user_id: ObjectId) -> bool:
        result = await database["images"].delete_one(
            {"_id": self._coerce_object_id(image_id, "image_id"), "user_id": user_id}
        )
        return result.deleted_count > 0


image_controller = ImageController()
