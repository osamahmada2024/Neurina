from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel
from typing import Optional
from bson import ObjectId

from .image.dependencies import get_current_user
from ..orchestration import StyleTransferGraph
from ..schemes.image_schema import SuccessResponse


class StyleTransferRequest(BaseModel):
    style_description: str


class StyleTransferResponse(BaseModel):
    success: bool
    message: str
    reference_image_id: str = None
    quality_score: str = None
    errors: list = None


router = APIRouter(prefix="/style-transfer", tags=["style-transfer"])


@router.post("/request", response_model=StyleTransferResponse, response_model_exclude_none=True)
async def request_style_transfer(
    request: StyleTransferRequest,
    current_user: ObjectId = Depends(get_current_user),
    authorization: Optional[str] = Header(None),
) -> StyleTransferResponse:
    try:
        if not request.style_description or len(request.style_description.strip()) < 3:
            return StyleTransferResponse(
                success=False,
                message="Style description must be at least 3 characters",
                errors=["invalid_input"],
            )

        # Extract token from Authorization header
        auth_token = authorization.replace("Bearer ", "") if authorization else ""

        # Execute workflow
        graph = StyleTransferGraph()
        result = await graph.execute(
            user_id=current_user,
            user_input=request.style_description,
            auth_token=auth_token,
        )

        return StyleTransferResponse(
            success=result.get("success", False),
            message=result.get("message", ""),
            reference_image_id=result.get("reference_image_id"),
            quality_score=result.get("quality_score"),
            errors=result.get("errors", []),
        )

    except Exception as exc:
        return StyleTransferResponse(
            success=False,
            message=f"Workflow error: {str(exc)}",
            errors=[str(exc)],
        )
