from __future__ import annotations

from typing import Optional
from datetime import datetime

from bson import ObjectId
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from ...controllers.image_controller import image_controller
from ...schemes.image_schema import (
    UploadCompleteResponse, 
    TranslationCreatedResponse,
    SuccessResponse,
)
from .dependencies import get_current_user

router = APIRouter()


@router.post("/upload", response_model=UploadCompleteResponse, response_model_exclude_none=True)
async def upload_image(
    file: UploadFile = File(...),
    image_type: str = Query(..., description="'source' or 'reference'"),
    image_domain: Optional[str] = Query(None, description="male | female (optional, recommended)"),
    current_user: ObjectId = Depends(get_current_user),
) -> UploadCompleteResponse:
    try:
        result = await image_controller.upload_and_process_image(
            file=file,
            user_id=current_user,
            image_type=image_type,
            filename=file.filename,
            image_domain=image_domain,
        )
        
        return UploadCompleteResponse(
            success=True,
            message="Image uploaded and processed successfully",
            image_id=result.image_id,
            image_type=image_type,
            filename=file.filename,
            faces_detected=result.faces_detected,
            status=result.status,
            created_at=datetime.utcnow().isoformat(),
            original_image_url=result.original_image_url,
            processed_image_url=result.processed_image_url,
        )
    except ValueError as exc:
        return UploadCompleteResponse(
            success=False,
            message=str(exc),
            error_code="validation_error",
            details={"error": str(exc)},
        )
    except Exception as exc:
        return UploadCompleteResponse(
            success=False,
            message=f"Upload failed: {str(exc)}",
            error_code="upload_error",
            details={"error": str(exc)},
        )


@router.post("/translate", response_model=TranslationCreatedResponse, response_model_exclude_none=True)
async def create_translation_task(
    source_image_id: str = Query(...),
    reference_image_id: str = Query(...),
    translation_mode: str = Query(
        "auto",
        description="auto | male_to_female | female_to_male | male_to_male | female_to_female",
    ),
    current_user: ObjectId = Depends(get_current_user),
) -> TranslationCreatedResponse:
    try:
        result = await image_controller.create_translation_task(
            user_id=current_user,
            source_image_id=source_image_id,
            reference_image_id=reference_image_id,
            translation_mode=translation_mode,
            wait_for_completion=True,  # Wait for result
        )
        
        # Check if translation was successful
        if result.get("success"):
            return TranslationCreatedResponse(
                success=True,
                message=result.get("message", "Translation completed successfully"),
                task_id=result.get("task_id"),
                status=result.get("status"),
                translated_image_id=result.get("translated_image_id"),
                created_at=result.get("created_at"),
            )
        else:
            return TranslationCreatedResponse(
                success=False,
                message=result.get("message", "Translation failed"),
                task_id=result.get("task_id"),
                error_code=result.get("error_code"),
                details=result.get("details"),
            )
    except ValueError as exc:
        return TranslationCreatedResponse(
            success=False,
            message=str(exc),
            error_code="validation_error",
            details={"error": str(exc)},
        )
    except Exception as exc:
        return TranslationCreatedResponse(
            success=False,
            message=f"Translation failed: {str(exc)}",
            error_code="translation_error",
            details={"error": str(exc)},
        )


@router.delete("/image/{image_id}", response_model=SuccessResponse, response_model_exclude_none=True)
async def delete_image(
    image_id: str,
    current_user: ObjectId = Depends(get_current_user),
) -> SuccessResponse:
    try:
        success = await image_controller.delete_image(image_id, current_user)
        if not success:
            return SuccessResponse(
                success=False,
                message="Image not found",
                data={"image_id": image_id},
            )
        return SuccessResponse(
            success=True,
            message="Image deleted successfully",
            data={"image_id": image_id},
        )
    except Exception as exc:
        return SuccessResponse(
            success=False,
            message=f"Delete failed: {str(exc)}",
            data={"error": str(exc)},
        )
