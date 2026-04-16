from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Query, Header
from fastapi.responses import Response, StreamingResponse
from typing import Optional
import io
import base64
import cv2
import numpy as np
from bson import ObjectId

from ..controllers.image_controller import image_controller
from ..schemes.image_schema import ImageUploadResponseSchema
from ..services import verify_access_token
from ..services.image_download_service import ImageDownloadService
from ..models.Enums import ValidationErrorMessage

router = APIRouter(
    prefix="/images",
    tags=["images"]
)


def _png_response(image_bytes: bytes, filename: str) -> Response:
    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Content-Length": str(len(image_bytes)),
            "Cache-Control": "no-store",
        },
    )


async def get_current_user(authorization: Optional[str] = Header(None)) -> ObjectId:
    """Extract user ID from authorization header"""
    if not authorization:
        raise HTTPException(status_code=401, detail=ValidationErrorMessage.MISSING_AUTHORIZATION.value)
    
    try:
        token = authorization.replace("Bearer ", "")
        if not token:
            raise HTTPException(status_code=401, detail=ValidationErrorMessage.INVALID_AUTH_HEADER.value)
        
        token_data = verify_access_token(token)
        
        if not token_data:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        
        user_id = token_data.get("user_id")
        
        if not user_id:
            raise HTTPException(status_code=401, detail=ValidationErrorMessage.INVALID_TOKEN.value)
        
        return ObjectId(user_id)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"{ValidationErrorMessage.AUTH_FAILED.value}: {str(e)}")


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    image_type: str = Query(..., description="'source' or 'reference'"),
    image_domain: Optional[str] = Query(None, description="male | female (optional, recommended)"),
    current_user: ObjectId = Depends(get_current_user)
) -> ImageUploadResponseSchema:
    """Upload image with face detection and preprocessing"""
    try:
        return await image_controller.upload_and_process_image(
            file=file,
            user_id=current_user,
            image_type=image_type,
            filename=file.filename,
            image_domain=image_domain,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/user-images")
async def get_user_images(
    image_type: str = Query(None, description="Filter by 'source', 'reference', or 'translated'"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: ObjectId = Depends(get_current_user)
) -> dict:
    """Get user's images with pagination"""
    images = await image_controller.get_user_images(
        user_id=current_user,
        image_type=image_type,
        limit=limit,
        offset=offset
    )
    total = await image_controller.count_user_images(
        user_id=current_user,
        image_type=image_type,
    )
    count = len(images)
    return {
        "count": count,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": (offset + count) < total,
        "images": images,
    }


@router.get("/public-references")
async def get_public_reference_images(
    image_domain: Optional[str] = Query(None, description="male | female"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> dict:
    """Get public reference images with pagination."""
    try:
        images = await image_controller.get_public_reference_images(
            image_domain=image_domain,
            limit=limit,
            offset=offset,
        )
        total = await image_controller.count_public_reference_images(image_domain=image_domain)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    count = len(images)
    return {
        "count": count,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": (offset + count) < total,
        "images": images,
    }


@router.get("/image/{image_id}")
async def get_image(
    image_id: str,
    current_user: ObjectId = Depends(get_current_user)
) -> dict:
    """Get specific image by ID"""
    return await image_controller.get_image_by_id_with_ownership(image_id, current_user)


@router.post("/translate")
async def create_translation_task(
    source_image_id: str = Query(...),
    reference_image_id: str = Query(...),
    translation_mode: str = Query(
        "auto",
        description="auto | male_to_female | female_to_male | male_to_male | female_to_female",
    ),
    current_user: ObjectId = Depends(get_current_user)
) -> dict:
    """Create image translation task"""
    try:
        return await image_controller.create_translation_task(
            user_id=current_user,
            source_image_id=source_image_id,
            reference_image_id=reference_image_id,
            translation_mode=translation_mode,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/translation-tasks")
async def get_translation_tasks(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: ObjectId = Depends(get_current_user)
) -> dict:
    """Get user's translation tasks"""
    tasks = await image_controller.get_translation_tasks(
        user_id=current_user,
        limit=limit,
        offset=offset
    )
    total = await image_controller.count_translation_tasks(current_user)
    count = len(tasks)
    return {
        "count": count,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": (offset + count) < total,
        "tasks": tasks,
    }


@router.get("/download/{image_id}")
async def download_image(
    image_id: str,
    current_user: ObjectId = Depends(get_current_user)
):
    """Download image as PNG"""
    try:
        image = await image_controller.get_image_by_id_with_ownership(image_id, current_user)
        image_bytes = ImageDownloadService.download_image(image["image_data"], image_id)
        return _png_response(image_bytes, f"image_{image_id}.png")
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/download-translation/{task_id}")
async def download_translated_image(
    task_id: str,
    current_user: ObjectId = Depends(get_current_user)
):
    """Download translated image"""
    try:
        image = await image_controller.get_translated_image_with_ownership(task_id, current_user)
        image_bytes = ImageDownloadService.download_image(image["image_data"], task_id)
        return _png_response(image_bytes, f"translated_{task_id}.png")
    except HTTPException:
        raise
    except ValueError as exc:
        detail = str(exc)
        status_code = 409 if "not completed" in detail else 404
        raise HTTPException(status_code=status_code, detail=detail)


@router.get("/tight-crop/{image_id}")
async def get_tight_face_crop(
    image_id: str,
    current_user: ObjectId = Depends(get_current_user)
):
    """
    Extract and download ONLY the face region (tight crop, no padding).
    Image quality and size vary based on original image resolution.
    
    Query Parameters:
        - image_id: ID of image to crop
    
    Returns:
        PNG image with just the face crop
    
    Example:
        Input: Full photo (1920×1080)
        Output: Tight face crop (~400×500)
    """
    try:
        # Get image from database
        image = await image_controller.get_image_by_id_with_ownership(image_id, current_user)
        
        # Decode base64 to OpenCV image
        image_data = base64.b64decode(image["image_data"])
        image_bgr = cv2.imdecode(
            np.frombuffer(image_data, np.uint8),
            cv2.IMREAD_COLOR
        )
        
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Get tight face crop
        cropped_bgr = image_controller.get_tight_face_crop(image_bgr)
        
        # Encode to PNG
        _, buffer = cv2.imencode('.png', cropped_bgr)
        img_bytes = io.BytesIO(buffer.tobytes())
        
        return StreamingResponse(
            img_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=face_crop_{image_id}.png"}
        )
    
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error cropping image: {str(e)}")


@router.delete("/image/{image_id}")
async def delete_image(
    image_id: str,
    current_user: ObjectId = Depends(get_current_user)
) -> dict:
    """Delete image"""
    success = await image_controller.delete_image(image_id, current_user)
    if not success:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"message": "Image deleted successfully"}
