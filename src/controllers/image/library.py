from __future__ import annotations

import asyncio
from datetime import datetime
from functools import partial
import hashlib
import logging
from pathlib import Path
from typing import Mapping, Optional

import cv2
import numpy as np
from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

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
from ...utils.console_feedback import console_feedback, console_progress

from .base import PUBLIC_LIBRARY_USER_ID

logger = logging.getLogger(__name__)


class ImageLibraryMixin:
    """Image ingestion, public library sync, and query helpers."""

    @staticmethod
    def _build_public_reference_source_signature(path: Path) -> dict[str, int]:
        stat = path.stat()
        return {
            "library_source_size": int(stat.st_size),
            "library_source_mtime_ns": int(
                getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))
            ),
        }

    @staticmethod
    def _public_reference_source_matches(
        image_doc: Optional[dict],
        source_signature: Mapping[str, int],
    ) -> bool:
        if not image_doc:
            return False
        return bool(
            image_doc.get("library_source_size") == source_signature["library_source_size"]
            and image_doc.get("library_source_mtime_ns") == source_signature["library_source_mtime_ns"]
        )

    @staticmethod
    def _public_reference_is_invalid(image_doc: Optional[dict]) -> bool:
        return bool(image_doc and image_doc.get("sync_status") == "invalid")

    @staticmethod
    def _is_invalid_public_reference_error(exc: Exception) -> bool:
        return isinstance(exc, ValueError)

    def _prepare_processed_image_assets(
        self,
        image_array: np.ndarray,
        image_type: str,
        image_domain: Optional[str] = None,
        trace_context: Optional[str] = None,
        display_image_bgr: Optional[np.ndarray] = None,
        original_image_bgr: Optional[np.ndarray] = None,
    ) -> dict:
        if image_type not in {ImageType.SOURCE.value, ImageType.REFERENCE.value}:
            raise ValueError(ValidationErrorMessage.INVALID_IMAGE_TYPE.value)

        with self._model_pipeline_lock:
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
            faces_count = self._count_faces_in_image(image_array)
            if faces_count != 1:
                raise ValueError(
                    f"Image must contain exactly 1 face. Found: {faces_count} face(s). "
                    "Please upload a different image."
                )

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
                        # Use lower outscale for light model to reduce memory
                        upload_outscale = 1.0 if bool(settings.SR_USE_LIGHT_MODEL) else float(settings.UPLOAD_SR_OUTSCALE)
                        image_for_storage_bgr = self.face_restoration_service.enhance(
                            model_image_bgr,
                            outscale=upload_outscale,
                        )
                        self._trace_image(
                            image_for_storage_bgr,
                            "upload_face_upscaled_out",
                            image_type,
                            trace_context=trace_context,
                        )
                    except Exception as exc:
                        logger.debug("Upload super-resolution skipped: %s", exc)
            else:
                self._trace_image(
                    image_for_storage_bgr,
                    "upload_display_image_preserved",
                    image_type,
                    trace_context=trace_context,
                )

        return {
            "image_for_storage_bgr": image_for_storage_bgr,
            "original_image_bgr": original_image_bgr,
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
        }

    @staticmethod
    def _public_reference_is_ready(image_doc: Optional[dict]) -> bool:
        if not image_doc:
            return False
        return bool(
            image_doc.get("status") == ImageStatus.PREPROCESSED.value
            and image_doc.get("sync_status") == "ready"
            and image_doc.get("cloudinary_public_id_processed")
            and image_doc.get("cloudinary_public_id_original")
        )

    async def _upsert_public_reference_sync_doc(
        self,
        path: Path,
        library_key: str,
        image_domain: str,
        public_collection: str,
        source_signature: Mapping[str, int],
    ) -> dict:
        selector = {
            "is_public": True,
            "public_collection": public_collection,
            "library_key": library_key,
        }
        existing = await database["images"].find_one(selector)
        if existing:
            updated = await database["images"].find_one_and_update(
                {"_id": existing["_id"]},
                {
                    "$set": {
                        "status": ImageStatus.PROCESSING.value,
                        "sync_status": "syncing",
                        "sync_error": None,
                        "original_filename": path.name,
                        "image_domain": image_domain,
                        "library_source_path": str(path.resolve()),
                        **source_signature,
                        "updated_at": datetime.utcnow(),
                    }
                },
                return_document=ReturnDocument.AFTER,
            )
            return updated or existing

        now = datetime.utcnow()
        document = {
            "user_id": PUBLIC_LIBRARY_USER_ID,
            "image_type": ImageType.REFERENCE.value,
            "original_filename": path.name,
            "status": ImageStatus.PROCESSING.value,
            "faces_detected": 0,
            "landmarks": None,
            "image_domain": image_domain,
            "domain_label": self.domain_to_label.get(image_domain),
            "is_public": True,
            "public_collection": public_collection,
            "library_key": library_key,
            "library_source_path": str(path.resolve()),
            **source_signature,
            "sync_status": "syncing",
            "sync_error": None,
            "created_at": now,
            "updated_at": now,
        }
        try:
            result = await database["images"].insert_one(document)
            document["_id"] = result.inserted_id
            return document
        except DuplicateKeyError:
            fetched = await database["images"].find_one(selector)
            if fetched is None:
                raise
            return fetched

    async def _ensure_reference_cloudinary_asset(
        self,
        *,
        image_doc_id,
        current_url: Optional[str],
        current_public_id: Optional[str],
        desired_public_id: str,
        image_bgr: np.ndarray,
        image_type: str,
        user_id: str,
        variant: str,
        cloudinary_gate: Optional[asyncio.Semaphore] = None,
        show_feedback: bool = False,
    ) -> tuple[str, str, bool]:
        asset_public_id = current_public_id or desired_public_id

        if current_url:
            return current_url, asset_public_id, False

        async def _run_upload_io(func, /, *args, **kwargs):
            loop = asyncio.get_running_loop()
            task = partial(func, *args, **kwargs)
            return await loop.run_in_executor(
                self._get_public_reference_upload_executor(),
                task,
            )

        async def _resolve_asset() -> tuple[str, str, bool]:
            existing_info = await _run_upload_io(
                self.cloudinary_service.get_image_info,
                asset_public_id,
            )
            if existing_info and existing_info.get("secure_url"):
                await database["images"].update_one(
                    {"_id": image_doc_id},
                    {
                        "$set": {
                            "updated_at": datetime.utcnow(),
                            "storage_type": "cloudinary",
                        }
                    },
                )
                return existing_info["secure_url"], asset_public_id, False

            upload_result = await _run_upload_io(
                self.cloudinary_service.upload_processed_face_image,
                image_bgr=image_bgr,
                image_type=image_type,
                user_id=user_id,
                suffix=variant,
                full_public_id=asset_public_id,
                show_feedback=show_feedback,
            )
            return (
                upload_result["secure_url"],
                upload_result.get("public_id") or asset_public_id,
                True,
            )

        if cloudinary_gate is None:
            return await _resolve_asset()

        async with cloudinary_gate:
            return await _resolve_asset()

    async def _recover_public_reference_from_cloudinary_index(
        self,
        *,
        image_doc: dict,
        path: Path,
        library_key: str,
        image_domain: str,
        public_collection: str,
        cloudinary_assets: Optional[Mapping[str, str]],
    ) -> Optional[dict]:
        if self.cloudinary_service is None or not cloudinary_assets:
            return None

        processed_public_id = (
            image_doc.get("cloudinary_public_id_processed")
            or self.cloudinary_service.build_public_reference_public_id(
                library_key,
                "processed",
            )
        )
        original_public_id = (
            image_doc.get("cloudinary_public_id_original")
            or self.cloudinary_service.build_public_reference_public_id(
                library_key,
                "original",
            )
        )

        processed_url = cloudinary_assets.get(processed_public_id)
        original_url = cloudinary_assets.get(original_public_id)
        if not processed_url or not original_url:
            return None

        await database["images"].update_one(
            {"_id": image_doc["_id"]},
            {
                "$set": {
                    "original_filename": path.name,
                    "status": ImageStatus.PREPROCESSED.value,
                    "image_domain": image_domain,
                    "domain_label": self.domain_to_label.get(image_domain),
                    "is_public": True,
                    "public_collection": public_collection,
                    "library_key": library_key,
                    "library_source_path": str(path.resolve()),
                    "image_data": processed_url,
                    "image_data_original": original_url,
                    "cloudinary_public_id_processed": processed_public_id,
                    "cloudinary_public_id_original": original_public_id,
                    "storage_type": "cloudinary",
                    "sync_status": "ready",
                    "sync_error": None,
                    "updated_at": datetime.utcnow(),
                }
            },
        )

        return {
            "_id": str(image_doc["_id"]),
            "status": "already_synced",
            "image_domain": image_domain,
            "library_key": library_key,
            "original_filename": path.name,
        }

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
        prepared = self._prepare_processed_image_assets(
            image_array=image_array,
            image_type=image_type,
            image_domain=image_domain,
            trace_context=trace_context,
            display_image_bgr=display_image_bgr,
            original_image_bgr=original_image_bgr,
        )
        image_for_storage_bgr = prepared["image_for_storage_bgr"]
        original_image_bgr = prepared["original_image_bgr"]

        # Upload to Cloudinary or fallback to base64
        image_url = None
        original_image_url = None
        
        if self.cloudinary_service is not None:
            try:
                user_id_str = str(user_id)
                
                # Upload processed image to Cloudinary
                upload_result = self.cloudinary_service.upload_processed_face_image(
                    image_bgr=image_for_storage_bgr,
                    image_type=image_type,
                    user_id=user_id_str,
                    suffix="processed"
                )
                image_url = upload_result['secure_url']
                
                # Upload original image if available (this is what user wants)
                if original_image_bgr is not None:
                    original_upload_result = self.cloudinary_service.upload_processed_face_image(
                        image_bgr=original_image_bgr,
                        image_type=image_type,
                        user_id=user_id_str,
                        suffix="original"
                    )
                    original_image_url = original_upload_result['secure_url']
                
                self._trace_image(
                    image_for_storage_bgr,
                    "cloudinary_upload_success",
                    image_type,
                    trace_context=trace_context,
                )
                
            except Exception as exc:
                logger.debug("Cloudinary fallback applied for upload: %s", exc)
                # Fallback to base64 if Cloudinary fails
                image_url = self._encode_png_base64(image_for_storage_bgr)
                if original_image_bgr is not None:
                    original_image_url = self._encode_png_base64(original_image_bgr)
        else:
            # Fallback to base64 if Cloudinary service not available
            image_url = self._encode_png_base64(image_for_storage_bgr)
            if original_image_bgr is not None:
                original_image_url = self._encode_png_base64(original_image_bgr)
        
        self._trace_image(
            image_for_storage_bgr,
            "db_save_image",
            image_type,
            trace_context=trace_context,
        )

        # Determine if we're using URLs or base64
        is_cloudinary = self.cloudinary_service is not None and isinstance(image_url, str) and image_url.startswith('http')
        
        image_doc = {
            "user_id": user_id,
            "image_type": image_type,
            "image_data": image_url,  # URL or base64
            "original_filename": filename,
            "status": ImageStatus.PREPROCESSED.value,
            "faces_detected": prepared["faces_detected"],
            "landmarks": prepared["landmarks"],
            "image_domain": prepared["image_domain"],
            "domain_label": prepared["domain_label"],
            "display_resolution": prepared["display_resolution"],
            "model_resolution": prepared["model_resolution"],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        
        # Add storage type flag
        image_doc["storage_type"] = "cloudinary" if is_cloudinary else "base64"
        
        # Store original image for uploaded images only (not for translated)
        if original_image_bgr is not None and original_image_url is not None:
            image_doc["image_data_original"] = original_image_url
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
            
            # Calculate hash to detect duplicate images
            image_hash = hashlib.sha256(contents).hexdigest()
            
            # Check if image already exists for this user
            existing_image = await database["images"].find_one({
                "user_id": user_id,
                "image_type": image_type,
                "image_hash": image_hash
            })
            
            if existing_image:
                return ImageUploadResponseSchema(
                    image_id=str(existing_image["_id"]),
                    status=existing_image.get("status", ImageStatus.PREPROCESSED.value),
                    message="Image already exists. Returning existing image.",
                    faces_detected=int(existing_image.get("faces_detected", 0)),
                    original_image_url=existing_image.get("image_data_original"),
                    processed_image_url=existing_image.get("image_data"),
                    created_at=existing_image.get("created_at", datetime.utcnow()).isoformat() if isinstance(existing_image.get("created_at"), datetime) else existing_image.get("created_at"),
                )
            
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
            
            # Add hash to document
            image_doc["image_hash"] = image_hash
            
            result = await database["images"].insert_one(image_doc)
            
            return ImageUploadResponseSchema(
                image_id=str(result.inserted_id),
                status=ImageStatus.PREPROCESSED.value,
                message=(
                    "Image uploaded and processed successfully. "
                    f"{int(image_doc['faces_detected'])} face(s) detected."
                ),
                faces_detected=int(image_doc["faces_detected"]),
                original_image_url=image_doc.get("image_data_original"),
                processed_image_url=image_doc.get("image_data"),
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
        cloudinary_assets: Optional[Mapping[str, str]] = None,
        cloudinary_gate: Optional[asyncio.Semaphore] = None,
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
        source_signature = self._build_public_reference_source_signature(path)
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
        if self._public_reference_is_ready(existing) and (
            self._public_reference_source_matches(existing, source_signature)
            or existing.get("library_source_size") is None
        ):
            # Image already imported - skip reprocessing
            return {
                "_id": str(existing["_id"]),
                "status": "already_synced",
                "image_domain": image_domain,
                "library_key": library_key,
                "original_filename": path.name,
            }
        if self._public_reference_is_invalid(existing) and self._public_reference_source_matches(
            existing,
            source_signature,
        ):
            return {
                "_id": str(existing["_id"]),
                "status": "invalid_skipped",
                "image_domain": image_domain,
                "library_key": library_key,
                "original_filename": path.name,
            }

        image_doc = await self._upsert_public_reference_sync_doc(
            path=path,
            library_key=library_key,
            image_domain=image_domain,
            public_collection=public_collection,
            source_signature=source_signature,
        )
        if self._public_reference_is_ready(image_doc):
            return {
                "_id": str(image_doc["_id"]),
                "status": "already_synced",
                "image_domain": image_domain,
                "library_key": library_key,
                "original_filename": path.name,
            }

        recovered = await self._recover_public_reference_from_cloudinary_index(
            image_doc=image_doc,
            path=path,
            library_key=library_key,
            image_domain=image_domain,
            public_collection=public_collection,
            cloudinary_assets=cloudinary_assets,
        )
        if recovered is not None:
            return recovered

        processed_public_id = None
        original_public_id = None
        processed_uploaded = False
        original_uploaded = False

        try:
            image_array = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image_array is None:
                raise ValueError(f"Could not read image file: {path}")

            trace_context = f"public_ref_{library_key.replace('/', '_')}"
            prepared = await asyncio.to_thread(
                self._prepare_processed_image_assets,
                image_array,
                ImageType.REFERENCE.value,
                image_domain,
                trace_context,
                None,
                image_array,
            )

            image_url = image_doc.get("image_data")
            original_image_url = image_doc.get("image_data_original")
            uploaded_now = False

            if self.cloudinary_service is not None:
                processed_public_id = (
                    image_doc.get("cloudinary_public_id_processed")
                    or self.cloudinary_service.build_public_reference_public_id(
                        library_key,
                        "processed",
                    )
                )
                original_public_id = (
                    image_doc.get("cloudinary_public_id_original")
                    or self.cloudinary_service.build_public_reference_public_id(
                        library_key,
                        "original",
                    )
                )

                processed_task = self._ensure_reference_cloudinary_asset(
                    image_doc_id=image_doc["_id"],
                    current_url=image_url,
                    current_public_id=image_doc.get("cloudinary_public_id_processed"),
                    desired_public_id=processed_public_id,
                    image_bgr=prepared["image_for_storage_bgr"],
                    image_type=ImageType.REFERENCE.value,
                    user_id=str(PUBLIC_LIBRARY_USER_ID),
                    variant="processed",
                    cloudinary_gate=cloudinary_gate,
                    show_feedback=False,
                )
                original_task = self._ensure_reference_cloudinary_asset(
                    image_doc_id=image_doc["_id"],
                    current_url=original_image_url,
                    current_public_id=image_doc.get("cloudinary_public_id_original"),
                    desired_public_id=original_public_id,
                    image_bgr=prepared["original_image_bgr"],
                    image_type=ImageType.REFERENCE.value,
                    user_id=str(PUBLIC_LIBRARY_USER_ID),
                    variant="original",
                    cloudinary_gate=cloudinary_gate,
                    show_feedback=False,
                )
                (
                    (image_url, processed_public_id, processed_uploaded),
                    (original_image_url, original_public_id, original_uploaded),
                ) = await asyncio.gather(processed_task, original_task)
                uploaded_now = processed_uploaded or original_uploaded

                await database["images"].update_one(
                    {"_id": image_doc["_id"]},
                    {
                        "$set": {
                            "cloudinary_public_id_processed": processed_public_id,
                            "cloudinary_public_id_original": original_public_id,
                            "storage_type": "cloudinary",
                            "updated_at": datetime.utcnow(),
                        },
                        "$unset": {
                            "image_data": "",
                            "image_data_original": "",
                        }
                    },
                )
            else:
                if not image_url:
                    image_url = self._encode_png_base64(prepared["image_for_storage_bgr"])
                    uploaded_now = True
                if not original_image_url:
                    original_image_url = self._encode_png_base64(prepared["original_image_bgr"])
                    uploaded_now = True

            storage_type = (
                "cloudinary"
                if str(image_url).startswith("http") and str(original_image_url).startswith("http")
                else "base64"
            )

            # Build update document - don't store image_data for Cloudinary storage
            update_doc = {
                "status": ImageStatus.PREPROCESSED.value,
                "faces_detected": prepared["faces_detected"],
                "landmarks": prepared["landmarks"],
                "image_domain": prepared["image_domain"],
                "domain_label": prepared["domain_label"],
                "display_resolution": prepared["display_resolution"],
                "model_resolution": prepared["model_resolution"],
                "storage_type": storage_type,
                "sync_status": "ready",
                "sync_error": None,
                **source_signature,
                "updated_at": datetime.utcnow(),
            }
            
            # Only store image_data for base64 storage (legacy)
            if storage_type == "base64":
                update_doc["image_data"] = image_url
                update_doc["image_data_original"] = original_image_url
            
            await database["images"].update_one(
                {"_id": image_doc["_id"]},
                {"$set": update_doc},
            )

            return {
                "_id": str(image_doc["_id"]),
                "status": "uploaded" if uploaded_now else "already_synced",
                "image_domain": prepared["image_domain"],
                "library_key": library_key,
                "original_filename": path.name,
            }
        except Exception as exc:
            cleanup_tasks = []
            if self.cloudinary_service is not None:
                if processed_uploaded and processed_public_id:
                    cleanup_tasks.append(
                        asyncio.to_thread(
                            self.cloudinary_service.delete_image,
                            processed_public_id,
                        )
                    )
                if original_uploaded and original_public_id:
                    cleanup_tasks.append(
                        asyncio.to_thread(
                            self.cloudinary_service.delete_image,
                            original_public_id,
                        )
                    )
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)

            if self._is_invalid_public_reference_error(exc):
                await database["images"].update_one(
                    {"_id": image_doc["_id"]},
                    {
                        "$set": {
                            "status": ImageStatus.FAILED.value,
                            "sync_status": "invalid",
                            "sync_error": str(exc),
                            "faces_detected": 0,
                            "landmarks": None,
                            "image_domain": image_domain,
                            "domain_label": self.domain_to_label.get(image_domain),
                            "library_source_path": str(path.resolve()),
                            **source_signature,
                            "updated_at": datetime.utcnow(),
                        },
                        "$unset": {
                            "image_data": "",
                            "image_data_original": "",
                            "cloudinary_public_id_processed": "",
                            "cloudinary_public_id_original": "",
                            "display_resolution": "",
                            "model_resolution": "",
                            "storage_type": "",
                        },
                    },
                )
                return {
                    "_id": str(image_doc["_id"]),
                    "status": "invalid_skipped",
                    "image_domain": image_domain,
                    "library_key": library_key,
                    "original_filename": path.name,
                }

            await database["images"].update_one(
                {"_id": image_doc["_id"]},
                {
                    "$set": {
                        "status": ImageStatus.FAILED.value,
                        "sync_status": "failed",
                        "sync_error": str(exc),
                        "updated_at": datetime.utcnow(),
                    }
                },
            )
            raise

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

        if not image_paths:
            return {
                "root_dir": str(root.resolve()),
                "public_collection": public_collection,
                "processed": 0,
                "inserted": 0,
                "updated": 0,
                "skipped": 0,
                "already_synced": 0,
                "invalid": 0,
                "failed": 0,
                "images": [],
                "errors": [],
            }

        inserted = 0
        updated = 0
        skipped = 0
        already_synced = 0
        invalid = 0
        failed = 0
        imported = []
        errors = []
        total = len(image_paths)
        workers = max(1, min(int(settings.PUBLIC_REFERENCE_SYNC_WORKERS), total))
        cloudinary_assets: Optional[dict[str, str]] = None
        cloudinary_gate = asyncio.Semaphore(workers)

        if self.cloudinary_service is not None:
            console_feedback("Indexing existing Cloudinary public references...")
            loop = asyncio.get_running_loop()
            cloudinary_assets = await loop.run_in_executor(
                self._get_public_reference_upload_executor(),
                partial(
                    self.cloudinary_service.list_resources_by_prefix,
                    "neurina/processed_faces/public_references",
                ),
            )

        console_progress(
            "Reference sync",
            total=total,
            completed=0,
            uploaded=0,
            skipped=0,
            failed=0,
            active=workers,
        )

        semaphore = asyncio.Semaphore(workers)

        async def _run_single(image_path: Path) -> dict:
            async with semaphore:
                try:
                    result = await self.import_public_reference_image(
                        image_path=image_path,
                        root_dir=root,
                        public_collection=public_collection,
                        cloudinary_assets=cloudinary_assets,
                        cloudinary_gate=cloudinary_gate,
                    )
                    return {"path": str(image_path), "result": result}
                except Exception as exc:
                    return {"path": str(image_path), "error": str(exc)}

        tasks = [asyncio.create_task(_run_single(path)) for path in image_paths]
        completed = 0

        for task in asyncio.as_completed(tasks):
            outcome = await task
            completed += 1

            if "result" in outcome:
                result = outcome["result"]
                imported.append(result)
                if result["status"] == "uploaded":
                    inserted += 1
                elif result["status"] == "already_synced":
                    already_synced += 1
                    skipped += 1
                elif result["status"] == "invalid_skipped":
                    invalid += 1
                    skipped += 1
                else:
                    updated += 1
            else:
                failed += 1
                errors.append({"path": outcome["path"], "error": outcome["error"]})

            console_progress(
                "Reference sync",
                total=total,
                completed=completed,
                uploaded=inserted,
                skipped=skipped,
                failed=failed,
                active=workers,
            )

        if inserted == 0 and updated == 0 and skipped > 0:
            # If all images were skipped, this is a successful routine startup
            logger.debug(
                "Public reference library already up to date (%s images).",
                skipped,
            )

        return {
            "root_dir": str(root.resolve()),
            "public_collection": public_collection,
            "processed": len(image_paths),
            "inserted": inserted,
            "updated": updated,
            "skipped": skipped,
            "already_synced": already_synced,
            "invalid": invalid,
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
        query = {"user_id": user_id, "is_public": {"$ne": True}}
        if image_type:
            query["image_type"] = image_type
        images = await database["images"].find(query).skip(offset).limit(limit).to_list(None)
        return [self._serialize_image_doc(img) for img in images]

    async def count_user_images(
        self,
        user_id: ObjectId,
        image_type: Optional[str] = None,
    ) -> int:
        query = {"user_id": user_id, "is_public": {"$ne": True}}
        if image_type:
            query["image_type"] = image_type
        return int(await database["images"].count_documents(query))

    async def get_public_reference_images(
        self,
        image_domain: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list:
        query = {
            "image_type": ImageType.REFERENCE.value,
            "is_public": True,
            "status": ImageStatus.PREPROCESSED.value,
            "sync_status": "ready",
        }
        if image_domain is not None:
            if image_domain not in self.domain_to_label:
                raise ValueError("image_domain must be one of: male, female")
            query["image_domain"] = image_domain

        images = (
            await database["images"]
            .find(
                query,
                {
                    "_id": 1,
                    "image_domain": 1,
                    "image_data": 1,
                    "image_data_original": 1,
                    "storage_type": 1,
                    "cloudinary_public_id_processed": 1,
                    "library_key": 1,
                },
            )
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
        query = {
            "image_type": ImageType.REFERENCE.value,
            "is_public": True,
            "status": ImageStatus.PREPROCESSED.value,
            "sync_status": "ready",
        }
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
        object_id = self._coerce_object_id(image_id, "image_id")
        image = await database["images"].find_one(
            {
                "_id": object_id,
                "user_id": user_id,
                "is_public": {"$ne": True},
            }
        )
        if not image:
            image = await database["images"].find_one(
                {
                    "_id": object_id,
                    "is_public": True,
                    "status": ImageStatus.PREPROCESSED.value,
                    "sync_status": "ready",
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
        return result.deleted_count > 0
