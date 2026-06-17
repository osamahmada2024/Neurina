from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict


class TranslateRequest(BaseModel):
    source_image_id: str
    reference_image_id: str
    translation_mode: str = "auto"


class StyleTransferRequest(BaseModel):
    """Finalize style transfer for an existing chat session."""

    session_id: str = Field(..., description="Same session_id used in /chat")
    source_image_id: str = Field(..., description="Uploaded source image id from /images/upload")
    reference_image_id: str = Field(
        ...,
        description="Candidate id from chat (UUID key) or uploaded reference image id",
    )
    style_description: Optional[str] = Field(
        None,
        description="Optional style label; falls back to session state if omitted",
    )

    @model_validator(mode="after")
    def _strip_ids(self) -> "StyleTransferRequest":
        self.session_id = self.session_id.strip()
        self.source_image_id = self.source_image_id.strip()
        self.reference_image_id = self.reference_image_id.strip()
        if self.style_description is not None:
            self.style_description = self.style_description.strip() or None
        return self


class StyleTransferChatRequest(BaseModel):
    session_id: str
    message: str
    source_image_id: Optional[str] = Field(
        None,
        description="Set or update the user's source image id for this session",
    )


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
