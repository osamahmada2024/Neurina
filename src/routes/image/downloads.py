from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ...controllers.image_controller import image_controller
from ...services.image_download_service import ImageDownloadService
from ...config.cloudinary import cloudinary_settings
from .dependencies import build_png_response, get_current_user

router = APIRouter()


def _decode_image_bgr(image_base64: str) -> np.ndarray:
    image_data = base64.b64decode(image_base64)
    return cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)


@router.get("/download/{image_id}")
async def download_image(
    image_id: str,
    current_user: ObjectId = Depends(get_current_user),
):
    try:
        image = await image_controller.get_image_by_id_with_ownership(image_id, current_user)
        
        # Prefer Cloudinary URL if available (built from public_id)
        image_data = None
        
        # Try processed image first
        if image.get("cloudinary_public_id_processed"):
            try:
                cloud_name = cloudinary_settings.cloud_name
                public_id = image.get("cloudinary_public_id_processed")
                image_data = f"https://res.cloudinary.com/{cloud_name}/image/upload/{public_id}.png"
            except Exception:
                pass
        
        # Fallback to original image on Cloudinary
        if not image_data and image.get("cloudinary_public_id_original"):
            try:
                cloud_name = cloudinary_settings.cloud_name
                public_id = image.get("cloudinary_public_id_original")
                image_data = f"https://res.cloudinary.com/{cloud_name}/image/upload/{public_id}.png"
            except Exception:
                pass
        
        # Last resort: use stored image_data (for legacy data)
        if not image_data:
            image_data = image.get("image_data_original") or image.get("image_data")
        
        if not image_data:
            raise ValueError(f"Image data not found for {image_id}")
        
        image_bytes = ImageDownloadService.download_image(image_data, image_id)
        filename = image.get("original_filename", f"image_{image_id}.png")
        return build_png_response(image_bytes, filename)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

