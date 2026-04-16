from __future__ import annotations

from typing import Optional

from bson import ObjectId
from fastapi import Header, HTTPException, Response

from ...models.Enums import ValidationErrorMessage
from ...services import verify_access_token


def build_png_response(image_bytes: bytes, filename: str) -> Response:
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
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail=ValidationErrorMessage.MISSING_AUTHORIZATION.value,
        )

    try:
        token = authorization.replace("Bearer ", "")
        if not token:
            raise HTTPException(
                status_code=401,
                detail=ValidationErrorMessage.INVALID_AUTH_HEADER.value,
            )

        token_data = verify_access_token(token)
        if not token_data:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        user_id = token_data.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail=ValidationErrorMessage.INVALID_TOKEN.value,
            )

        return ObjectId(user_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=401,
            detail=f"{ValidationErrorMessage.AUTH_FAILED.value}: {exc}",
        )
