from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from bson import ObjectId

from ..models.Enums import ImageType
from ..utils.safe_image_download import download_public_image

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InMemoryUploadFile:
    """Small async file adapter for controller methods that expect UploadFile-like input."""

    filename: str
    content: bytes

    async def read(self) -> bytes:
        return self.content


@dataclass(frozen=True, slots=True)
class TranslationResult:
    task_id: str
    translated_image_id: str


class AgentImageService:
    """Reusable image operations for API routes and agent workflows.

    The agent workflow must not call this same FastAPI app over HTTP. This service keeps
    the business behavior centralized while allowing LangGraph nodes to call the
    controller layer directly.
    """

    def __init__(self, controller=None):
        self._controller = controller

    @property
    def controller(self):
        if self._controller is None:
            from ..controllers.image_controller import image_controller

            self._controller = image_controller
        return self._controller

    async def upload_image(
        self,
        *,
        file,
        user_id: ObjectId,
        image_type: str,
        filename: str,
        image_domain: Optional[str] = None,
    ):
        return await self.controller.upload_and_process_image(
            file=file,
            user_id=user_id,
            image_type=image_type,
            filename=filename,
            image_domain=image_domain,
        )

    async def upload_reference_from_url(
        self,
        *,
        user_id: ObjectId,
        image_url: str,
        filename: str = "reference_image.jpg",
        image_domain: Optional[str] = None,
    ) -> str:
        image_bytes = await self._download_image(image_url)
        return await self.upload_reference_from_bytes(
            user_id=user_id,
            image_bytes=image_bytes,
            filename=filename,
            image_domain=image_domain,
        )

    async def upload_reference_from_bytes(
        self,
        *,
        user_id: ObjectId,
        image_bytes: bytes,
        filename: str = "reference_image.jpg",
        image_domain: Optional[str] = None,
    ) -> str:
        upload_file = InMemoryUploadFile(filename=filename, content=image_bytes)
        result = await self.upload_image(
            file=upload_file,
            user_id=user_id,
            image_type=ImageType.REFERENCE.value,
            filename=filename,
            image_domain=image_domain,
        )
        return result.image_id

    async def upload_source_from_bytes(
        self,
        *,
        user_id: ObjectId,
        image_bytes: bytes,
        filename: str = "source_image.jpg",
        image_domain: Optional[str] = None,
    ) -> str:
        upload_file = InMemoryUploadFile(filename=filename, content=image_bytes)
        result = await self.upload_image(
            file=upload_file,
            user_id=user_id,
            image_type=ImageType.SOURCE.value,
            filename=filename,
            image_domain=image_domain,
        )
        return result.image_id

    async def translate_images(
        self,
        *,
        user_id: ObjectId,
        source_image_id: str,
        reference_image_id: str,
        translation_mode: str = "auto",
    ) -> str:
        return (
            await self.translate_images_result(
                user_id=user_id,
                source_image_id=source_image_id,
                reference_image_id=reference_image_id,
                translation_mode=translation_mode,
            )
        ).translated_image_id

    async def translate_images_result(
        self,
        *,
        user_id: ObjectId,
        source_image_id: str,
        reference_image_id: str,
        translation_mode: str = "auto",
    ) -> TranslationResult:
        result = await self.create_translation_task(
            user_id=user_id,
            source_image_id=source_image_id,
            reference_image_id=reference_image_id,
            translation_mode=translation_mode,
            wait_for_completion=True,
        )
        if not result.get("success"):
            message = result.get("message") or "Translation failed"
            raise ValueError(message)

        translated_image_id = result.get("translated_image_id")
        if not translated_image_id:
            raise ValueError("Translation completed without translated_image_id")
        task_id = result.get("task_id") or result.get("_id")
        if not task_id:
            raise ValueError("Translation completed without task_id")
        return TranslationResult(
            task_id=str(task_id),
            translated_image_id=str(translated_image_id),
        )

    async def create_translation_task(
        self,
        *,
        user_id: ObjectId,
        source_image_id: str,
        reference_image_id: str,
        translation_mode: str = "auto",
        wait_for_completion: bool = True,
    ) -> dict:
        return await self.controller.create_translation_task(
            user_id=user_id,
            source_image_id=source_image_id,
            reference_image_id=reference_image_id,
            translation_mode=translation_mode,
            wait_for_completion=wait_for_completion,
        )

    @staticmethod
    async def _download_image(image_url: str) -> bytes:
        return await download_public_image(image_url)


agent_image_service = AgentImageService()
