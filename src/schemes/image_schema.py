from pydantic import BaseModel, Field
from typing import Optional
from bson import ObjectId
from datetime import datetime
from ..models.Enums import ImageStatus, ImageType, TaskStatus


class ImageSchema(BaseModel):
    id: Optional[ObjectId] = Field(default=None, alias="_id")
    user_id: ObjectId
    image_type: str = Field(...)  # Using ImageType enum values
    image_data: str = Field(...)  # base64 or file path
    original_filename: str
    status: str = Field(default=ImageStatus.UPLOADED.value)
    faces_detected: int = 0
    landmarks: Optional[list] = None
    image_domain: Optional[str] = None
    domain_label: Optional[int] = None
    is_public: bool = False
    public_collection: Optional[str] = None
    library_key: Optional[str] = None
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
    id: Optional[ObjectId] = Field(default=None, alias="_id")
    user_id: ObjectId
    source_image_id: ObjectId
    reference_image_id: ObjectId
    translated_image_id: Optional[ObjectId] = None
    status: str = Field(default=TaskStatus.PENDING.value)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: lambda x: str(x),
            datetime: lambda x: x.isoformat()
        }


class ImageUploadResponseSchema(BaseModel):
    image_id: str
    status: str
    message: str
    faces_detected: int

    class Config:
        arbitrary_types_allowed = True
