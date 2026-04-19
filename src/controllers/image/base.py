from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import threading
from typing import Optional

import base64
import cv2
import numpy as np
import torch
from bson import ObjectId

from ...config import settings
from ...helpers.image_helpers import convert_image_data_to_image
from ...models import database
from ...models.Enums import ImageType
from ...services.face_restoration_service import FaceRestorationService

PUBLIC_LIBRARY_USER_ID = ObjectId("000000000000000000000001")


class ImageControllerBase:
    """Shared state and storage helpers for image workflows."""

    def __init__(self):
        self.face_detector = None
        self.opencv_face_cascade = None
        self.wing_face_aligner = None
        self.fan_model = None
        self.generator = None
        self.style_encoder = None
        self.face_restoration_service = None
        self.cloudinary_service = None
        self._model_pipeline_lock = threading.RLock()
        self._public_reference_upload_executor = None
        self._executor_init_lock = threading.Lock()
        self.domain_to_label = {"female": 0, "male": 1}
        self.label_to_domain = {
            label: domain for domain, label in self.domain_to_label.items()
        }
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
                from ...wing import FaceAligner as WingFaceAligner

                self.wing_face_aligner = WingFaceAligner(
                    wing_model_path,
                    celeba_lm_path,
                    256,
                )
            except Exception as exc:
                import logging
                logging.getLogger(__name__).debug(f"WingFaceAligner init skipped: {exc}")

        if wing_model_path:
            try:
                from ...helpers.face_detection import FaceDetector

                if fan_model is not None:
                    self.face_detector = FaceDetector(fan_model=fan_model)
                else:
                    self.face_detector = FaceDetector(wing_model_path=wing_model_path)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).debug(f"FaceDetector init skipped: {exc}")

    def initialize_postprocessors(self, base_path: str, preloaded_face_restoration_service=None) -> None:
        # Use preloaded service if provided, otherwise create new one
        if preloaded_face_restoration_service is not None:
            self.face_restoration_service = preloaded_face_restoration_service
            import logging
            logging.getLogger(__name__).debug("Using preloaded face restoration service")
        elif bool(settings.SR_ENABLED):
            try:
                self.face_restoration_service = FaceRestorationService(
                    base_path=base_path,
                    model_name=settings.SR_MODEL_NAME,
                    outscale=float(settings.SR_OUTSCALE),
                    tile=int(settings.SR_TILE),
                    face_weight=float(settings.SR_FACE_WEIGHT),
                    codeformer_fidelity=float(settings.SR_CODEFORMER_FIDELITY),
                )
                import logging
                logging.getLogger(__name__).debug(
                    f"Super-resolution ready: model={settings.SR_MODEL_NAME}, outscale={settings.SR_OUTSCALE}"
                )
            except Exception as exc:
                import logging
                logging.getLogger(__name__).debug(f"Super-resolution init skipped: {exc}")
        else:
            self.face_restoration_service = None
    
    def initialize_cloudinary_service(self, cloudinary_service) -> None:
        """Initialize Cloudinary service for image storage."""
        self.cloudinary_service = cloudinary_service

    def _get_public_reference_upload_executor(self) -> ThreadPoolExecutor:
        """Create a dedicated executor so public reference uploads really use 12 threads."""
        with self._executor_init_lock:
            if self._public_reference_upload_executor is None:
                workers = max(1, int(settings.PUBLIC_REFERENCE_SYNC_WORKERS))
                self._public_reference_upload_executor = ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="public-ref-upload",
                )
        return self._public_reference_upload_executor

    def shutdown_executors(self):
        """Clean up thread executors to prevent resource leaks."""
        with self._executor_init_lock:
            if self._public_reference_upload_executor is not None:
                self._public_reference_upload_executor.shutdown(wait=False)
                self._public_reference_upload_executor = None

    @staticmethod
    def _coerce_object_id(value, field_name: str) -> ObjectId:
        if isinstance(value, ObjectId):
            return value
        try:
            return ObjectId(str(value))
        except Exception as exc:
            raise ValueError(f"Invalid {field_name}") from exc

    def _trace_image(
        self,
        image_bgr: np.ndarray,
        stage: str,
        image_type: str = "unknown",
        trace_context: Optional[str] = None,
    ) -> None:
        """Trace images - disabled to avoid disk storage."""
        # Tracing disabled to keep all data in database only
        pass

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

    def _decode_stored_image(
        self,
        image_doc: dict,
        prefer_model_variant: bool = False,
    ) -> np.ndarray:
        if prefer_model_variant and image_doc.get("model_image_data"):
            return convert_image_data_to_image(image_doc["model_image_data"])
        return convert_image_data_to_image(image_doc["image_data"])

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

    async def _get_accessible_reference_image(
        self,
        image_id: str,
        user_id: ObjectId,
    ) -> Optional[dict]:
        return await database["images"].find_one(
            {
                "_id": self._coerce_object_id(image_id, "reference_image_id"),
                "image_type": ImageType.REFERENCE.value,
                "$or": [{"user_id": user_id}, {"is_public": True}],
            }
        )
