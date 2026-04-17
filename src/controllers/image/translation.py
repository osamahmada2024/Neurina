from __future__ import annotations

import asyncio
from datetime import datetime
import logging
from types import SimpleNamespace

import cv2
import torch
from bson import ObjectId

from ...config import settings
from ...models import database
from ...models.Enums import ImageStatus, ImageType, TaskStatus
from ...services.image_translation_service import translate_using_reference

logger = logging.getLogger(__name__)


class ImageTranslationMixin:
    """Translation task orchestration and background execution."""

    async def create_translation_task(
        self,
        user_id: ObjectId,
        source_image_id: str,
        reference_image_id: str,
        translation_mode: str = "female_to_female",
        wait_for_completion: bool = True,
        timeout_seconds: int = 300,
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
                    "Please re-upload the reference with image_domain=male|female "
                    "or choose translation_mode explicitly."
                )
        elif translation_mode not in self.supported_translation_modes:
            raise ValueError(
                "Invalid translation_mode. Use: auto | male_to_female | "
                "female_to_male | male_to_male | female_to_female"
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
        
        if wait_for_completion:
            # Process synchronously and wait for result
            return await self._process_translation_task(task_id, source, reference, task_doc)
        else:
            # Launch in background (optional, not used by default)
            asyncio.create_task(
                self._process_translation_task(task_id, source, reference, task_doc)
            )
            return {
                "_id": task_id,
                "status": TaskStatus.PENDING.value,
                "message": "Translation task queued.",
            }

    async def _process_translation_task(
        self,
        task_id: str,
        source_doc: dict,
        reference_doc: dict,
        task_doc: dict,
    ) -> dict:
        """Process translation task and return result (success or failure)."""
        try:
            trace_context = f"task_{task_id}"
            await database["translation_tasks"].update_one(
                {"_id": ObjectId(task_id)},
                {
                    "$set": {
                        "status": TaskStatus.PROCESSING.value,
                        "updated_at": datetime.utcnow(),
                    }
                },
            )

            source_image_bgr = self._decode_stored_image(
                source_doc,
                prefer_model_variant=True,
            )
            reference_image_bgr = self._decode_stored_image(
                reference_doc,
                prefer_model_variant=True,
            )
            self._trace_image(
                source_image_bgr,
                "db_load_source_in",
                "source",
                trace_context=trace_context,
            )
            self._trace_image(
                reference_image_bgr,
                "db_load_reference_in",
                "reference",
                trace_context=trace_context,
            )

            source_image_bgr = self._normalize_stored_face_for_model(
                source_image_bgr,
                target_size=256,
            )
            reference_image_bgr = self._normalize_stored_face_for_model(
                reference_image_bgr,
                target_size=256,
            )
            self._trace_image(
                source_image_bgr,
                "model_input_source_face",
                "source",
                trace_context=trace_context,
            )
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

                nets = SimpleNamespace(
                    generator=self.generator,
                    style_encoder=self.style_encoder,
                )
                if self.face_detector is not None and getattr(self.face_detector, "model", None) is not None:
                    nets.fan = self.face_detector.model
                elif self.fan_model is not None:
                    nets.fan = self.fan_model

                infer_args = SimpleNamespace(w_hpf=float(settings.W_HPF))
                target_label = int(
                    task_doc.get(
                        "target_domain_label",
                        int(settings.REFERENCE_DOMAIN_LABEL),
                    )
                )
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
                    logger.debug("Translation super-resolution skipped: %s", exc)

            final_bgr = self._rescue_translated_eyes(
                final_bgr,
                source_doc,
                trace_context=trace_context,
            )
            
            # Upload to Cloudinary or fallback to base64
            translated_image_url = None
            if self.cloudinary_service is not None:
                try:
                    # Upload translated image to Cloudinary
                    upload_result = self.cloudinary_service.upload_translation_result(
                        image_bgr=final_bgr,
                        user_id=str(source_doc["user_id"]),
                        source_image_id=str(source_doc["_id"]),
                        reference_image_id=str(reference_doc["_id"])
                    )
                    translated_image_url = upload_result['secure_url']
                    
                    self._trace_image(
                        final_bgr,
                        "cloudinary_upload_success",
                        "translated",
                        trace_context=trace_context,
                    )
                    
                except Exception as exc:
                    logger.debug("Cloudinary fallback applied for translation: %s", exc)
                    # Fallback to base64 if Cloudinary fails
                    translated_image_url = self._encode_png_base64(final_bgr)
            else:
                # Fallback to base64 if Cloudinary service not available
                translated_image_url = self._encode_png_base64(final_bgr)
            
            self._trace_image(
                final_bgr,
                "db_save_image",
                "translated",
                trace_context=trace_context,
            )

            # Determine if we're using URLs or base64
            is_cloudinary = self.cloudinary_service is not None and isinstance(translated_image_url, str) and translated_image_url.startswith('http')
            
            translated_doc = {
                "user_id": source_doc["user_id"],
                "image_type": ImageType.TRANSLATED.value,
                "image_data": translated_image_url,  # URL or base64
                "original_filename": f"translated_{source_doc.get('original_filename', 'image.jpg')}",
                "status": ImageStatus.TRANSLATION_COMPLETED.value,
                "faces_detected": 1,
                "landmarks": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "storage_type": "cloudinary" if is_cloudinary else "base64",
            }
            result = await database["images"].insert_one(translated_doc)
            translated_image_id = str(result.inserted_id)
            
            await database["translation_tasks"].update_one(
                {"_id": ObjectId(task_id)},
                {
                    "$set": {
                        "status": TaskStatus.COMPLETED.value,
                        "translated_image_id": result.inserted_id,
                        "updated_at": datetime.utcnow(),
                    }
                },
            )
            
            # Return success response
            return {
                "success": True,
                "message": "Translation completed successfully",
                "task_id": task_id,
                "status": TaskStatus.COMPLETED.value,
                "translated_image_id": translated_image_id,
                "created_at": datetime.utcnow().isoformat(),
            }
        except Exception as exc:
            error_msg = str(exc)
            await database["translation_tasks"].update_one(
                {"_id": ObjectId(task_id)},
                {
                    "$set": {
                        "status": TaskStatus.FAILED.value,
                        "error_message": error_msg,
                        "updated_at": datetime.utcnow(),
                    }
                },
            )
            
            # Return failure response
            return {
                "success": False,
                "message": f"Translation failed: {error_msg}",
                "task_id": task_id,
                "status": TaskStatus.FAILED.value,
                "error_code": "translation_error",
                "details": {"error": error_msg},
                "created_at": datetime.utcnow().isoformat(),
            }

    async def get_translation_tasks(
        self,
        user_id: ObjectId,
        limit: int = 50,
        offset: int = 0,
    ) -> list:
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
