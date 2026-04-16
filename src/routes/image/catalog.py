from __future__ import annotations

from typing import Optional
from datetime import datetime

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ...controllers.image_controller import image_controller
from ...schemes.image_schema import ImageDataResponse, ListResponse, TranslationTaskResponse
from .dependencies import get_current_user

router = APIRouter()


def _to_iso_string(value) -> str:
    """Convert datetime or any value to ISO string safely."""
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return ""
    return str(value)


class ImageListResponse(BaseModel):
    """Response for image listing endpoints."""
    count: int
    total: int
    limit: int
    offset: int
    has_more: bool
    items: list[ImageDataResponse]


class TaskListResponse(BaseModel):
    """Response for translation tasks listing."""
    count: int
    total: int
    limit: int
    offset: int
    has_more: bool
    items: list[TranslationTaskResponse]


@router.get("/user-images", response_model=ImageListResponse, response_model_exclude_none=True)
async def get_user_images(
    image_type: str = Query(None, description="Filter by 'source', 'reference', or 'translated'"),
    limit: int = Query(100, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: ObjectId = Depends(get_current_user),
) -> ImageListResponse:
    images = await image_controller.get_user_images(
        user_id=current_user,
        image_type=image_type,
        limit=limit,
        offset=offset,
    )
    total = await image_controller.count_user_images(
        user_id=current_user,
        image_type=image_type,
    )
    count = len(images)
    
    # Convert to professional response format
    formatted_images = [
        ImageDataResponse(
            _id=img.get("_id"),
            filename=img.get("original_filename", ""),
            type=img.get("image_type", ""),
            domain=img.get("image_domain"),
            status=img.get("status", ""),
            faces_detected=img.get("faces_detected", 0),
            created_at=_to_iso_string(img.get("created_at")),
        )
        for img in images
    ]
    
    return ImageListResponse(
        count=count,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + count) < total,
        items=formatted_images,
    )


@router.get("/public-references", response_model=ImageListResponse, response_model_exclude_none=True)
async def get_public_reference_images(
    image_domain: Optional[str] = Query(None, description="male | female"),
    limit: int = Query(100, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> ImageListResponse:
    try:
        images = await image_controller.get_public_reference_images(
            image_domain=image_domain,
            limit=limit,
            offset=offset,
        )
        total = await image_controller.count_public_reference_images(
            image_domain=image_domain,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    count = len(images)
    
    # Convert to professional response format
    formatted_images = [
        ImageDataResponse(
            _id=img.get("_id"),
            filename=img.get("original_filename", ""),
            type=img.get("image_type", ""),
            domain=img.get("image_domain"),
            status=img.get("status", ""),
            faces_detected=img.get("faces_detected", 0),
            created_at=_to_iso_string(img.get("created_at")),
        )
        for img in images
    ]
    
    return ImageListResponse(
        count=count,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + count) < total,
        items=formatted_images,
    )


@router.get("/image/{image_id}", response_model=ImageDataResponse, response_model_exclude_none=True)
async def get_image(
    image_id: str,
    current_user: ObjectId = Depends(get_current_user),
) -> ImageDataResponse:
    img = await image_controller.get_image_by_id_with_ownership(image_id, current_user)
    return ImageDataResponse(
        _id=img.get("_id"),
        filename=img.get("original_filename", ""),
        type=img.get("image_type", ""),
        domain=img.get("image_domain"),
        status=img.get("status", ""),
        faces_detected=img.get("faces_detected", 0),
        created_at=_to_iso_string(img.get("created_at")),
    )


@router.get("/translation-tasks", response_model=TaskListResponse, response_model_exclude_none=True)
async def get_translation_tasks(
    limit: int = Query(100, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: ObjectId = Depends(get_current_user),
) -> TaskListResponse:
    tasks = await image_controller.get_translation_tasks(
        user_id=current_user,
        limit=limit,
        offset=offset,
    )
    total = await image_controller.count_translation_tasks(current_user)
    count = len(tasks)
    
    # Convert to professional response format
    formatted_tasks = [
        TranslationTaskResponse(
            _id=task.get("_id"),
            source_image_id=str(task.get("source_image_id", "")),
            reference_image_id=str(task.get("reference_image_id", "")),
            translated_image_id=str(task.get("translated_image_id")) if task.get("translated_image_id") else None,
            status=task.get("status", "pending"),
            translation_mode=task.get("translation_mode"),
            source_domain=task.get("source_domain"),
            target_domain=task.get("target_domain"),
            created_at=_to_iso_string(task.get("created_at")),
            updated_at=_to_iso_string(task.get("updated_at")),
            error_message=task.get("error_message"),
        )
        for task in tasks
    ]
    
    return TaskListResponse(
        count=count,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + count) < total,
        items=formatted_tasks,
    )
