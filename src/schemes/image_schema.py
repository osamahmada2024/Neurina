from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Any, Generic, TypeVar
from bson import ObjectId
from datetime import datetime
from ..models.Enums import ImageStatus, ImageType, TaskStatus

# Custom ObjectId validator for JSON schema
class ObjectIdField(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return str(v)
    
    @classmethod
    def __get_pydantic_json_schema__(cls, _core_schema, _handler):
        return {"type": "string", "format": "objectid"}


class ImageSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
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

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: lambda x: str(x),
            datetime: lambda x: x.isoformat()
        }


class TranslationTaskSchema(BaseModel):
    model_config = ConfigDict(
        validate_by_name=True,
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
    image_id: str
    status: str
    message: str
    faces_detected: int

    class Config:
        arbitrary_types_allowed = True
        exclude_none = True


# ================== Professional Response Schemas ==================

class ImageDataResponse(BaseModel):
    """Lightweight image metadata for listings."""
    id: str = Field(..., alias="_id")
    filename: str
    type: str  # source, reference, translated
    domain: Optional[str] = None
    status: str
    faces_detected: int
    created_at: str
    size_bytes: Optional[int] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        exclude_none = True


class PublicReferenceDataResponse(BaseModel):
    """Minimal public reference payload for the website picker."""
    id: str = Field(..., alias="_id")
    cloudinary_url: Optional[str] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        exclude_none = True


class TranslationTaskResponse(BaseModel):
    """Translation task details."""
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

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        exclude_none = True


class ListResponse(BaseModel):
    """Generic paginated list response."""
    count: int = Field(..., description="Number of items in this page")
    total: int = Field(..., description="Total items available")
    limit: int = Field(..., description="Items per page")
    offset: int = Field(..., description="Current offset")
    has_more: bool = Field(..., description="Whether more items exist")
    items: List[Any] = Field(...)

    class Config:
        arbitrary_types_allowed = True
        exclude_none = True


class SuccessResponse(BaseModel):
    """Generic success response wrapper."""
    success: bool = True
    message: str
    data: Optional[Any] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    class Config:
        arbitrary_types_allowed = True
        exclude_none = True


class ErrorResponse(BaseModel):
    """Generic error response wrapper."""
    success: bool = False
    error_code: str
    message: str
    details: Optional[dict] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    class Config:
        arbitrary_types_allowed = True
        exclude_none = True


class UploadCompleteResponse(BaseModel):
    """Response after successful image upload."""
    success: bool
    message: str
    image_id: Optional[str] = None
    image_type: Optional[str] = None
    filename: Optional[str] = None
    faces_detected: Optional[int] = None
    status: Optional[str] = None
    created_at: Optional[str] = None
    error_code: Optional[str] = None
    details: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True
        exclude_none = True


class TranslationCreatedResponse(BaseModel):
    """Response when translation task is created or completed."""
    success: bool
    message: str
    task_id: Optional[str] = None
    status: Optional[str] = None
    translated_image_id: Optional[str] = None
    created_at: Optional[str] = None
    error_code: Optional[str] = None
    details: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True
        exclude_none = True
