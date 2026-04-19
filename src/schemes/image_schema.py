from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Any, Generic, TypeVar
from bson import ObjectId
from datetime import datetime
from ..models.Enums import ImageStatus, ImageType, TaskStatus

# Custom ObjectId validator for JSON schema
class ObjectIdField(str):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema, PydanticUndefined
        
        def validate_str(value: str) -> str:
            if not ObjectId.is_valid(value):
                raise ValueError("Invalid ObjectId")
            return str(value)
        
        return core_schema.no_info_before_validator_function(
            validate_str,
            core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(str)
        )


class ImageSchema(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={
            ObjectId: lambda x: str(x),
            datetime: lambda x: x.isoformat()
        }
    )

    id: Optional[ObjectIdField] = Field(default=None, alias="_id")
    user_id: ObjectIdField
    image_type: str = Field(...)  # Using ImageType enum values
    original_filename: str
    status: str = Field(default=ImageStatus.UPLOADED.value)
    faces_detected: int = 0
    image_domain: Optional[str] = None
    domain_label: Optional[int] = None
    is_public: bool = False
    public_collection: Optional[str] = None
    library_key: Optional[str] = None
    
    # Cloudinary storage
    cloudinary_public_id_processed: Optional[str] = None
    cloudinary_public_id_original: Optional[str] = None
    storage_type: Optional[str] = None  # 'cloudinary'
    
    # Sync metadata (for public references)
    sync_status: Optional[str] = None  # 'syncing', 'ready', 'invalid', 'failed'
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TranslationTaskSchema(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={
            ObjectId: lambda x: str(x),
            datetime: lambda x: x.isoformat()
        }
    )

    id: Optional[ObjectIdField] = Field(default=None, alias="_id")
    user_id: ObjectIdField
    source_image_id: ObjectIdField
    reference_image_id: ObjectIdField
    translated_image_id: Optional[ObjectIdField] = None
    status: str = Field(default=TaskStatus.PENDING.value)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None


class ImageUploadResponseSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, exclude_none=True)

    image_id: str
    status: str
    message: str
    faces_detected: int
    original_image_url: Optional[str] = None
    processed_image_url: Optional[str] = None


# ================== Professional Response Schemas ==================

class ImageDataResponse(BaseModel):
    """Lightweight image metadata for listings."""
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True, exclude_none=True)

    id: str = Field(..., alias="_id")
    filename: str
    type: str  # source, reference, translated
    domain: Optional[str] = None
    status: str
    faces_detected: int
    created_at: str
    size_bytes: Optional[int] = None
    cloudinary_public_id: Optional[str] = None
    original_image_url: Optional[str] = None
    processed_image_url: Optional[str] = None


class PublicReferenceDataResponse(BaseModel):
    """Minimal public reference payload for the website picker."""
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True, exclude_none=True)

    id: str = Field(..., alias="_id")
    processed_url: Optional[str] = None
    original_image_url: Optional[str] = None


class TranslationTaskResponse(BaseModel):
    """Translation task details."""
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True, exclude_none=True)

    task_id: str = Field(..., alias="_id")
    source_image_id: str
    reference_image_id: str
    translated_image_id: Optional[str] = None
    status: str  # pending, processing, completed, failed
    translation_mode: Optional[str] = None
    source_domain: Optional[str] = None
    target_domain: Optional[str] = None
    created_at: str
    updated_at: str
    error_message: Optional[str] = None


class ListResponse(BaseModel):
    """Generic paginated list response."""
    model_config = ConfigDict(arbitrary_types_allowed=True, exclude_none=True)

    count: int = Field(..., description="Number of items in this page")
    total: int = Field(..., description="Total items available")
    limit: int = Field(..., description="Items per page")
    offset: int = Field(..., description="Current offset")
    has_more: bool = Field(..., description="Whether more items exist")
    items: List[Any] = Field(...)


class SuccessResponse(BaseModel):
    """Generic success response wrapper."""
    model_config = ConfigDict(arbitrary_types_allowed=True, exclude_none=True)

    success: bool = True
    message: str
    data: Optional[Any] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ErrorResponse(BaseModel):
    """Generic error response wrapper."""
    model_config = ConfigDict(arbitrary_types_allowed=True, exclude_none=True)

    success: bool = False
    error_code: str
    message: str
    details: Optional[dict] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class UploadCompleteResponse(BaseModel):
    """Response after successful image upload."""
    model_config = ConfigDict(arbitrary_types_allowed=True, exclude_none=True)

    success: bool
    message: str
    image_id: Optional[str] = None
    image_type: Optional[str] = None
    filename: Optional[str] = None
    faces_detected: Optional[int] = None
    status: Optional[str] = None
    original_image_url: Optional[str] = None
    processed_image_url: Optional[str] = None
    created_at: Optional[str] = None
    error_code: Optional[str] = None
    details: Optional[dict] = None


class TranslationCreatedResponse(BaseModel):
    """Response when translation task is created or completed."""
    model_config = ConfigDict(arbitrary_types_allowed=True, exclude_none=True)

    success: bool
    message: str
    task_id: Optional[str] = None
    status: Optional[str] = None
    translated_image_id: Optional[str] = None
    created_at: Optional[str] = None
    error_code: Optional[str] = None
    details: Optional[dict] = None
