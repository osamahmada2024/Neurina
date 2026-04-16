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
        
        # For uploaded images: use original, for others use processed
        image_type = image.get("image_type", "image")
        if image_type in ["source", "reference"]:
            # Prefer original if available, fallback to processed
            image_data = image.get("image_data_original") or image.get("image_data")
        else:
            # For translated/other types, use processed
            image_data = image.get("image_data")
        
        image_bytes = ImageDownloadService.download_image(image_data, image_id)
        filename = image.get("original_filename", f"{image_type}_{image_id}.png")
        return build_png_response(image_bytes, filename)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

