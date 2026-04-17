"""
Cloudinary Service

Handles uploading processed images to Cloudinary and managing cloud storage.
Replaces local base64 storage with secure URL-based storage.
"""

import hashlib
import logging
from pathlib import Path
import re
import time
from io import BytesIO
from typing import Optional, Dict, Any

import cv2
import numpy as np
import cloudinary
import cloudinary.api
import cloudinary.uploader
from cloudinary.exceptions import Error as CloudinaryError

from ..config.cloudinary import cloudinary_settings
from ..utils.console_feedback import console_feedback

logger = logging.getLogger(__name__)


class CloudinaryService:
    """Service for uploading images to Cloudinary cloud storage."""

    @staticmethod
    def _slugify_public_id_component(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip().lower()).strip("-")
        return slug[:48] or "image"
    
    def __init__(self):
        """
        Initialize Cloudinary service with Pydantic settings.
        """
        if not cloudinary_settings.is_configured():
            raise ValueError("Cloudinary settings are not properly configured")
        
        self.cloud_name = cloudinary_settings.cloud_name
        self.api_key = cloudinary_settings.api_key
        self.api_secret = cloudinary_settings.api_secret
        self._upload_warning_emitted = False
        
        # Configure Cloudinary with Pydantic settings
        cloudinary.config(**cloudinary_settings.get_config_dict())
        
        logger.debug("Cloudinary service initialized for cloud: %s", self.cloud_name)

    @staticmethod
    def _summarize_upload_error(error: Exception) -> str:
        message = str(error).lower()
        if "failed to resolve" in message or "nameresolutionerror" in message:
            return "Cloudinary DNS lookup failed"
        if "max retries exceeded" in message or "connection broken" in message:
            return "Cloudinary connection failed"
        return "Cloudinary upload failed"
    
    def upload_bgr_image(
        self,
        image_bgr: np.ndarray,
        public_id: Optional[str] = None,
        folder: str = "neurina",
        transformation: Optional[Dict[str, Any]] = None,
        full_public_id: Optional[str] = None,
        show_feedback: bool = True,
    ) -> Dict[str, Any]:
        """
        Upload a BGR image array to Cloudinary.
        
        Args:
            image_bgr: OpenCV BGR image array
            public_id: Optional public ID for the image
            folder: Cloudinary folder to store the image
            transformation: Optional Cloudinary transformation
            
        Returns:
            Dictionary with upload results including secure_url
            
        Raises:
            CloudinaryUploadError: If upload fails
        """
        try:
            # Convert BGR to RGB for Cloudinary
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            from PIL import Image
            pil_image = Image.fromarray(image_rgb)
            
            # Convert to bytes
            img_buffer = BytesIO()
            pil_image.save(img_buffer, format='PNG', quality=95)
            img_bytes = img_buffer.getvalue()
            
            # Generate public ID if not provided
            if full_public_id is None and public_id is None:
                timestamp = int(time.time())
                public_id = f"neurina_image_{timestamp}"

            resolved_public_id = (
                str(full_public_id).strip("/")
                if full_public_id is not None
                else f"{folder}/{public_id}" if folder else str(public_id)
            )
            
            # Prepare upload parameters
            upload_params = {
                'file': img_bytes,
                'public_id': resolved_public_id,
                'resource_type': 'image',
                'format': 'png',
                'overwrite': True,
            }
            
            # Add transformation if provided
            if transformation:
                upload_params['transformation'] = transformation
            
            # Upload to Cloudinary
            if show_feedback:
                console_feedback(f"Uploading {resolved_public_id}...")
            logger.debug("Uploading image to Cloudinary: %s", resolved_public_id)
            result = cloudinary.uploader.upload(**upload_params)
            
            # Extract key information
            upload_result = {
                'secure_url': result.get('secure_url'),
                'public_id': result.get('public_id'),
                'format': result.get('format'),
                'width': result.get('width'),
                'height': result.get('height'),
                'bytes': result.get('bytes'),
                'created_at': result.get('created_at'),
                'resource_type': result.get('resource_type')
            }
            
            self._upload_warning_emitted = False
            if show_feedback:
                console_feedback(f"Uploaded {resolved_public_id}")
            logger.debug("Cloudinary upload completed: %s", upload_result["public_id"])
            return upload_result
            
        except CloudinaryError as e:
            short_error = self._summarize_upload_error(e)
            if not self._upload_warning_emitted:
                console_feedback(f"{short_error}. Using base64 fallback")
                logger.debug("%s. Using base64 fallback.", short_error)
                self._upload_warning_emitted = True
            logger.debug("Cloudinary upload error details: %s", e, exc_info=True)
            raise CloudinaryUploadError(short_error)
        except Exception as e:
            short_error = self._summarize_upload_error(e)
            if not self._upload_warning_emitted:
                console_feedback(f"{short_error}. Using base64 fallback")
                logger.debug("%s. Using base64 fallback.", short_error)
                self._upload_warning_emitted = True
            logger.debug("Unexpected Cloudinary upload error details: %s", e, exc_info=True)
            raise CloudinaryUploadError(short_error)
    
    def upload_processed_face_image(
        self,
        image_bgr: np.ndarray,
        image_type: str,
        user_id: str,
        suffix: str = "processed",
        full_public_id: Optional[str] = None,
        show_feedback: bool = True,
    ) -> Dict[str, Any]:
        """
        Upload a processed face image with standardized naming.
        
        Args:
            image_bgr: Processed face image in BGR format
            image_type: Type of image (source, reference, translated)
            user_id: User ID for organization
            suffix: Additional suffix for naming
            
        Returns:
            Dictionary with upload results
        """
        public_id = None
        if full_public_id is None:
            timestamp = int(time.time())
            public_id = f"{image_type}/{user_id}_{suffix}_{timestamp}"
        
        # Add transformation for optimization
        transformation = {
            'quality': 'auto:good',
            'fetch_format': 'auto',
        }
        
        return self.upload_bgr_image(
            image_bgr=image_bgr,
            public_id=public_id,
            folder="neurina/processed_faces",
            transformation=transformation,
            full_public_id=full_public_id,
            show_feedback=show_feedback,
        )
    
    def upload_translation_result(
        self,
        image_bgr: np.ndarray,
        user_id: str,
        source_image_id: str,
        reference_image_id: str
    ) -> Dict[str, Any]:
        """
        Upload a translation result image.
        
        Args:
            image_bgr: Translation result image
            user_id: User ID
            source_image_id: Source image ID
            reference_image_id: Reference image ID
            
        Returns:
            Dictionary with upload results
        """
        timestamp = int(time.time())
        public_id = f"translations/{user_id}_{source_image_id}_{reference_image_id}_{timestamp}"
        
        return self.upload_bgr_image(
            image_bgr=image_bgr,
            public_id=public_id,
            folder="neurina/translations"
        )

    def build_public_reference_public_id(self, library_key: str, variant: str) -> str:
        """Build a stable public ID so retries reuse the same Cloudinary asset."""
        normalized_key = str(library_key).replace("\\", "/").lower()
        digest = hashlib.sha1(normalized_key.encode("utf-8")).hexdigest()[:12]
        stem = self._slugify_public_id_component(Path(normalized_key).stem)
        return f"neurina/processed_faces/public_references/{stem}-{digest}-{variant}"
    
    def delete_image(self, public_id: str) -> bool:
        """
        Delete an image from Cloudinary.
        
        Args:
            public_id: Public ID of the image to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            result = cloudinary.uploader.destroy(public_id)
            success = result.get('result') == 'ok'
            
            if success:
                logger.debug("Deleted Cloudinary image: %s", public_id)
            else:
                logger.warning(f"Failed to delete image: {public_id}, result: {result}")
            
            return success
            
        except CloudinaryError as e:
            logger.error(f"Error deleting image {public_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting image {public_id}: {e}")
            return False
    
    def get_image_info(self, public_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an uploaded image.
        
        Args:
            public_id: Public ID of the image
            
        Returns:
            Dictionary with image information or None if not found
        """
        try:
            result = cloudinary.api.resource(public_id)
            return {
                'secure_url': result.get('secure_url'),
                'public_id': result.get('public_id'),
                'format': result.get('format'),
                'width': result.get('width'),
                'height': result.get('height'),
                'bytes': result.get('bytes'),
                'created_at': result.get('created_at')
            }
        except CloudinaryError as e:
            logger.debug("Cloudinary image info unavailable for %s: %s", public_id, e)
            return None

    def list_resources_by_prefix(
        self,
        prefix: str,
        max_results: int = 500,
    ) -> Dict[str, str]:
        """Return a public_id -> secure_url index for an uploaded asset prefix."""
        public_index: Dict[str, str] = {}
        next_cursor = None
        normalized_prefix = str(prefix).strip("/")
        page_size = max(1, min(int(max_results), 500))

        try:
            while True:
                params: Dict[str, Any] = {
                    "type": "upload",
                    "prefix": normalized_prefix,
                    "max_results": page_size,
                }
                if next_cursor:
                    params["next_cursor"] = next_cursor

                result = cloudinary.api.resources(**params)
                for resource in result.get("resources", []):
                    public_id = resource.get("public_id")
                    secure_url = resource.get("secure_url")
                    if public_id and secure_url:
                        public_index[public_id] = secure_url

                next_cursor = result.get("next_cursor")
                if not next_cursor:
                    break
        except CloudinaryError as e:
            logger.debug(
                "Cloudinary resource listing unavailable for prefix %s: %s",
                normalized_prefix,
                e,
            )
        except Exception as e:
            logger.debug(
                "Unexpected Cloudinary resource listing failure for prefix %s: %s",
                normalized_prefix,
                e,
                exc_info=True,
            )

        return public_index
    
    @staticmethod
    def create_from_env() -> 'CloudinaryService':
        """
        Create CloudinaryService from environment variables.
        
        Returns:
            Configured CloudinaryService instance
            
        Raises:
            ValueError: If required settings are missing
        """
        try:
            return CloudinaryService()
        except ValueError as e:
            raise ValueError(f"Failed to create Cloudinary service: {e}")


class CloudinaryUploadError(Exception):
    """Raised when Cloudinary upload fails."""
    pass
