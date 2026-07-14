from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Optional, Dict


def _strip_required_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")
    return stripped


def _strip_optional_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


class TranslateRequest(BaseModel):
    source_image_id: str
    reference_image_id: str
    translation_mode: str = "auto"


class StyleTransferRequest(BaseModel):
    """Finalize style transfer for an existing chat session."""

    session_id: str = Field(..., min_length=1, description="Same session_id used in /chat")
    source_image_id: str = Field(
        ...,
        min_length=1,
        description="Uploaded source image id from /images/upload",
    )
    reference_image_id: str = Field(
        ...,
        min_length=1,
        description="Candidate id from chat (UUID key) or uploaded reference image id",
    )
    style_description: Optional[str] = Field(
        None,
        description="Optional style label; falls back to session state if omitted",
    )

    @model_validator(mode="after")
    def _strip_ids(self) -> "StyleTransferRequest":
        self.session_id = _strip_required_string(self.session_id, "session_id")
        self.source_image_id = _strip_required_string(self.source_image_id, "source_image_id")
        self.reference_image_id = _strip_required_string(
            self.reference_image_id,
            "reference_image_id",
        )
        self.style_description = _strip_optional_string(self.style_description)
        return self


class StyleTransferChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    source_image_id: Optional[str] = Field(
        None,
        description="Set or update the user's source image id for this session",
    )

    @field_validator("session_id", "message", mode="after")
    @classmethod
    def _strip_required_fields(cls, value: str, info):
        return _strip_required_string(value, info.field_name)

    @field_validator("source_image_id", mode="after")
    @classmethod
    def _strip_optional_source(cls, value: Optional[str]) -> Optional[str]:
        return _strip_optional_string(value)


class StyleTransferResponse(BaseModel):
    success: bool
    status: str = "COMPLETED"
    message: str
    candidate_images: Optional[Dict[str, str]] = None
    reference_image_id: Optional[str] = None
    translated_image_id: Optional[str] = None
    selected_reference_url: Optional[str] = None
    quality_score: Optional[str] = None
    errors: Optional[list] = None


class AgentExecutionInput(BaseModel):
    """Validated internal input for chat execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: ObjectId
    session_id: str = Field(..., min_length=1)
    user_input: str = Field(..., min_length=1)
    auth_token: str = ""
    source_image_id: Optional[str] = None

    @field_validator("user_id", mode="before")
    @classmethod
    def _validate_user_id(cls, value) -> ObjectId:
        if isinstance(value, ObjectId):
            return value
        stripped = str(value).strip()
        if not stripped or not ObjectId.is_valid(stripped):
            raise ValueError("user_id must be a valid ObjectId")
        return ObjectId(stripped)

    @field_validator("session_id", "user_input", mode="after")
    @classmethod
    def _strip_required_fields(cls, value: str, info) -> str:
        return _strip_required_string(value, info.field_name)

    @field_validator("auth_token", mode="after")
    @classmethod
    def _strip_auth_token(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else ""

    @field_validator("source_image_id", mode="after")
    @classmethod
    def _strip_source_image_id(cls, value: Optional[str]) -> Optional[str]:
        return _strip_optional_string(value)


class AgentRequestInput(BaseModel):
    """Validated internal input for finalizing a selected style transfer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: ObjectId
    session_id: str = Field(..., min_length=1)
    source_image_id: str = Field(..., min_length=1)
    reference_image_id: str = Field(..., min_length=1)
    auth_token: str = ""
    style_description: Optional[str] = None

    @field_validator("user_id", mode="before")
    @classmethod
    def _validate_user_id(cls, value) -> ObjectId:
        if isinstance(value, ObjectId):
            return value
        stripped = str(value).strip()
        if not stripped or not ObjectId.is_valid(stripped):
            raise ValueError("user_id must be a valid ObjectId")
        return ObjectId(stripped)

    @field_validator("session_id", "source_image_id", "reference_image_id", mode="after")
    @classmethod
    def _strip_required_fields(cls, value: str, info) -> str:
        return _strip_required_string(value, info.field_name)

    @field_validator("auth_token", mode="after")
    @classmethod
    def _strip_auth_token(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else ""

    @field_validator("style_description", mode="after")
    @classmethod
    def _strip_style_description(cls, value: Optional[str]) -> Optional[str]:
        return _strip_optional_string(value)
