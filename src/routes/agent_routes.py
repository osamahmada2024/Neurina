from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import StreamingResponse

from .image.dependencies import get_current_user
from ..orchestration import StyleTransferGraph
from ..schemes.api_models import StyleTransferRequest, StyleTransferChatRequest, StyleTransferResponse


router = APIRouter(prefix="/style-transfer", tags=["style-transfer"])


def _sse_response(event_generator):
    return StreamingResponse(
        event_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat")
async def chat_style_transfer(
    request: StyleTransferChatRequest,
    current_user: ObjectId = Depends(get_current_user),
    authorization: Optional[str] = Header(None),
    stream: bool = Query(True, description="Stream agent steps as SSE (default: true)"),
):
    auth_token = authorization.replace("Bearer ", "") if authorization else ""
    graph = StyleTransferGraph()
    session_id = request.session_id.strip()

    if stream:
        async def event_generator():
            async for chunk in graph.execute_stream(
                user_id=current_user,
                session_id=session_id,
                user_input=request.message,
                auth_token=auth_token,
                source_image_id=request.source_image_id,
            ):
                yield chunk

        return _sse_response(event_generator())

    try:
        result = await graph.execute(
            user_id=current_user,
            session_id=session_id,
            user_input=request.message,
            auth_token=auth_token,
            source_image_id=request.source_image_id,
        )
        return StyleTransferResponse(**result)
    except PermissionError as exc:
        return StyleTransferResponse(
            success=False,
            status="FAILED",
            message=str(exc),
            errors=[str(exc)],
        )
    except Exception as exc:
        return StyleTransferResponse(
            success=False,
            status="FAILED",
            message=f"Workflow error: {str(exc)}",
            errors=[str(exc)],
        )


@router.post("/chat/sync", response_model=StyleTransferResponse, response_model_exclude_none=True)
async def chat_style_transfer_sync(
    request: StyleTransferChatRequest,
    current_user: ObjectId = Depends(get_current_user),
    authorization: Optional[str] = Header(None),
) -> StyleTransferResponse:
    """Non-streaming chat (legacy JSON response)."""
    try:
        auth_token = authorization.replace("Bearer ", "") if authorization else ""
        graph = StyleTransferGraph()
        result = await graph.execute(
            user_id=current_user,
            session_id=request.session_id.strip(),
            user_input=request.message,
            auth_token=auth_token,
            source_image_id=request.source_image_id,
        )
        return StyleTransferResponse(**result)
    except PermissionError as exc:
        return StyleTransferResponse(
            success=False,
            status="FAILED",
            message=str(exc),
            errors=[str(exc)],
        )
    except Exception as exc:
        return StyleTransferResponse(
            success=False,
            status="FAILED",
            message=f"Workflow error: {str(exc)}",
            errors=[str(exc)],
        )


@router.post("/request", response_model=StyleTransferResponse, response_model_exclude_none=True)
async def request_style_transfer(
    request: StyleTransferRequest,
    current_user: ObjectId = Depends(get_current_user),
    authorization: Optional[str] = Header(None),
) -> StyleTransferResponse:
    try:
        auth_token = authorization.replace("Bearer ", "") if authorization else ""

        graph = StyleTransferGraph()
        result = await graph.execute_request(
            user_id=current_user,
            session_id=request.session_id,
            source_image_id=request.source_image_id,
            reference_image_id=request.reference_image_id,
            auth_token=auth_token,
            style_description=request.style_description,
        )

        return StyleTransferResponse(**result)

    except PermissionError as exc:
        return StyleTransferResponse(
            success=False,
            status="FAILED",
            message=str(exc),
            errors=[str(exc)],
        )
    except Exception as exc:
        return StyleTransferResponse(
            success=False,
            status="FAILED",
            message=f"Workflow error: {str(exc)}",
            errors=[str(exc)],
        )
