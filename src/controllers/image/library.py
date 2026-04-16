from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from bson import ObjectId

from ...config import settings
from ...helpers.image_helpers import validate_image_file
from ...models import database
from ...models.Enums import (
    ImageErrorMessage,
    ImageStatus,
    ImageType,
    TaskStatus,
    ValidationErrorMessage,
)
from ...schemes.image_schema import ImageUploadResponseSchema

from .base import PUBLIC_LIBRARY_USER_ID


class ImageLibraryMixin:
    """Image ingestion, public library sync, and query helpers."""

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
        original_image_bgr: Optional[np.ndarray] = None,  # Add original image storage
    ) -> dict:
        if image_type not in {ImageType.SOURCE.value, ImageType.REFERENCE.value}:
            raise ValueError(ValidationErrorMessage.INVALID_IMAGE_TYPE.value)

        domain_label = None
        if image_domain is not None:
            if image_domain not in self.domain_to_label:
                raise ValueError("image_domain must be one of: male, female")
            domain_label = int(self.domain_to_label[image_domain])

        self._trace_image(
            image_array,
            "upload_raw_in",
            image_type,
            trace_context=trace_context,
        )

        landmarks = None
        faces_count = 0
        
        # First count faces using cascade - faster and more accurate
        faces_count = self._count_faces_in_image(image_array)
        
        # Reject images without exactly 1 face detected
        if faces_count != 1:
            raise ValueError(
                f"Image must contain exactly 1 face. Found: {faces_count} face(s). "
                "Please upload a different image."
            )
        
        # Get landmarks for the single face
        if self.face_detector is not None:
            _, landmarks = self.face_detector.detect_landmarks(image_array)

        preprocessed_image = self._preprocess_image(
            image_array,
            landmarks,
            target_size=256,
            strict_face_detection=True,
        )
        self._trace_image(
            preprocessed_image,
            "upload_face_crop_out",
            image_type,
            trace_context=trace_context,
        )

        model_image_bgr = self._normalize_stored_face_for_model(
            preprocessed_image,
            target_size=256,
        )
        self._trace_image(
            model_image_bgr,
            "upload_model_face_final",
            image_type,
            trace_context=trace_context,
        )

        if domain_label is None:
            inferred_label = self._infer_domain_label_from_face(model_image_bgr)
            if inferred_label is not None:
                domain_label = int(inferred_label)
                image_domain = self.label_to_domain.get(domain_label)

        image_for_storage_bgr = (
            display_image_bgr.copy()
            if display_image_bgr is not None
            else model_image_bgr
        )
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
        self._trace_image(
            image_for_storage_bgr,
            "db_save_image",
            image_type,
            trace_context=trace_context,
        )

        image_doc = {
            "user_id": user_id,
            "image_type": image_type,
            "image_data": image_base64,  # Processed/cropped image
            "model_image_data": model_image_base64,
            "original_filename": filename,
            "status": ImageStatus.PREPROCESSED.value,
            "faces_detected": faces_count,
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
        
        # Store original image for uploaded images only (not for translated)
        if original_image_bgr is not None:
            original_base64 = self._encode_png_base64(original_image_bgr)
            image_doc["image_data_original"] = original_base64
            self._trace_image(
                original_image_bgr,
                "db_save_original",
                image_type,
                trace_context=trace_context,
            )
        
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

            # Store original image for uploaded images
            original_image_bgr = image_array.copy()

            image_doc = self._build_processed_image_doc(
                image_array=image_array,
                user_id=user_id,
                image_type=image_type,
                filename=filename,
                image_domain=image_domain,
                original_image_bgr=original_image_bgr,  # Pass original image
            )
            result = await database["images"].insert_one(image_doc)
            return ImageUploadResponseSchema(
                image_id=str(result.inserted_id),
                status=ImageStatus.PREPROCESSED.value,
                message=(
                    "Image uploaded and processed successfully. "
                    f"{int(image_doc['faces_detected'])} face(s) detected."
                ),
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
            raise ValueError(
                f"Unsupported image format for public reference: {path.name}"
            )

        library_key = self._relative_library_key(path, root)
        if image_domain is None:
            leading_dir = library_key.split("/", 1)[0].strip().lower()
            if leading_dir in self.domain_to_label:
                image_domain = leading_dir
        if image_domain not in self.domain_to_label:
            raise ValueError(
                f"Could not infer image_domain from {path}. Put files under male/ "
                "or female/, or pass image_domain explicitly."
            )

        # Check if image already exists - skip expensive reprocessing if unchanged
        existing = await database["images"].find_one(
            {
                "is_public": True,
                "public_collection": public_collection,
                "library_key": library_key,
            }
        )
        if existing:
            # Image already imported - skip reprocessing
            return {
                "_id": str(existing["_id"]),
                "status": "skipped",
                "image_domain": image_domain,
                "library_key": library_key,
                "original_filename": path.name,
            }

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
            original_image_bgr=image_array,  # Store original image
            extra_fields={
                "is_public": True,
                "public_collection": public_collection,
                "library_key": library_key,
                "library_source_path": str(path.resolve()),
            },
        )

        result = await database["images"].insert_one(image_doc)
        return {
            "_id": str(result.inserted_id),
            "status": "inserted",
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

        image_paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and validate_image_file(path.name)
        )
        if limit is not None:
            image_paths = image_paths[: max(0, int(limit))]

        inserted = 0
        updated = 0
        skipped = 0
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
                elif result["status"] == "skipped":
                    skipped += 1
                else:
                    updated += 1
            except Exception as exc:
                failed += 1
                errors.append({"path": str(path), "error": str(exc)})

        if inserted == 0 and updated == 0 and skipped > 0:
            # If all images were skipped, this is a successful routine startup
            print(f"✓ Public reference library up-to-date ({skipped} images already imported)")

        return {
            "root_dir": str(root.resolve()),
            "public_collection": public_collection,
            "processed": len(image_paths),
            "inserted": inserted,
            "updated": updated,
            "skipped": skipped,
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

    async def count_user_images(
        self,
        user_id: ObjectId,
        image_type: Optional[str] = None,
    ) -> int:
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

    async def count_public_reference_images(
        self,
        image_domain: Optional[str] = None,
    ) -> int:
        query = {"image_type": ImageType.REFERENCE.value, "is_public": True}
        if image_domain is not None:
            if image_domain not in self.domain_to_label:
                raise ValueError("image_domain must be one of: male, female")
            query["image_domain"] = image_domain
        return int(await database["images"].count_documents(query))

    async def get_image_by_id_with_ownership(
        self,
        image_id: str,
        user_id: ObjectId,
    ) -> dict:
        image = await database["images"].find_one(
            {
                "_id": self._coerce_object_id(image_id, "image_id"),
                "$or": [{"user_id": user_id}, {"is_public": True}],
            }
        )
        if not image:
            raise ValueError("Image not found or not authorized")
        return self._serialize_image_doc(image)

    async def get_translated_image_with_ownership(
        self,
        task_id: str,
        user_id: ObjectId,
    ) -> dict:
        task = await database["translation_tasks"].find_one(
            {"_id": self._coerce_object_id(task_id, "task_id"), "user_id": user_id}
        )
        if not task:
            raise ValueError("Task not found or not authorized")
        if task.get("status") != TaskStatus.COMPLETED.value:
            raise ValueError(f"Task status is {task.get('status', 'unknown')}, not completed")

        image = await database["images"].find_one(
            {
                "_id": self._coerce_object_id(
                    task["translated_image_id"],
                    "translated_image_id",
                )
            }
        )
        if not image:
            raise ValueError("Image not found")
        return self._serialize_image_doc(image)

    def get_tight_face_crop(self, image_bgr: np.ndarray) -> np.ndarray:
        return self._prepare_face_bgr(image_bgr, target_size=256)

    async def delete_image(self, image_id: str, user_id: ObjectId) -> bool:
        result = await database["images"].delete_one(
            {
                "_id": self._coerce_object_id(image_id, "image_id"),
                "user_id": user_id,
            }
        )
        return result.deleted_count > 0
