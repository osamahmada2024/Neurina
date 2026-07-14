from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "create_access_token": ("auth_service", "create_access_token"),
    "verify_access_token": ("auth_service", "verify_access_token"),
    "verify_strong_password": ("auth_service", "verify_strong_password"),
    "verify_google_token": ("auth_service", "verify_google_token"),
    "verify_github_code": ("auth_service", "verify_github_code"),
    "create_reset_token": ("auth_service", "create_reset_token"),
    "verify_reset_token": ("auth_service", "verify_reset_token"),
    "send_reset_email_async": ("auth_service", "send_reset_email_async"),
    "send_contact_email_async": ("auth_service", "send_contact_email_async"),
    "denormalize": ("image_translation_service", "denormalize"),
    "save_image": ("image_translation_service", "save_image"),
    "translate_using_reference": (
        "image_translation_service",
        "translate_using_reference",
    ),
    "translate_batch": ("image_translation_service", "translate_batch"),
    "normalize_tensor": ("image_translation_service", "normalize_tensor"),
    "get_translation_stats": (
        "image_translation_service",
        "get_translation_stats",
    ),
    "ModelLoader": ("model_loader", "ModelLoader"),
    "ImageDownloadService": (
        "image_download_service",
        "ImageDownloadService",
    ),
    "FaceRestorationService": (
        "face_restoration_service",
        "FaceRestorationService",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'src.services' has no attribute {name!r}") from exc

    value = getattr(import_module(f".{module_name}", __name__), attr_name)
    globals()[name] = value
    return value
