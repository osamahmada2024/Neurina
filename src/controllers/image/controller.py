from __future__ import annotations

from .base import ImageControllerBase
from .library import ImageLibraryMixin
from .preprocessing import ImagePreprocessingMixin
from .translation import ImageTranslationMixin


class ImageController(
    ImageTranslationMixin,
    ImageLibraryMixin,
    ImagePreprocessingMixin,
    ImageControllerBase,
):
    """Coordinates image upload, library sync, and translation workflows."""


image_controller = ImageController()
