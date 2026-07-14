from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "render_email_template": ("email_templates", "render_email_template"),
    "convert_image_to_base64": ("image_helpers", "convert_image_to_base64"),
    "convert_base64_to_image": ("image_helpers", "convert_base64_to_image"),
    "resize_image": ("image_helpers", "resize_image"),
    "normalize_image": ("image_helpers", "normalize_image"),
    "denormalize_image": ("image_helpers", "denormalize_image"),
    "prepare_image_for_model": ("image_helpers", "prepare_image_for_model"),
    "tensor_to_cv_image": ("image_helpers", "tensor_to_cv_image"),
    "draw_landmarks_on_image": ("image_helpers", "draw_landmarks_on_image"),
    "validate_image_file": ("image_helpers", "validate_image_file"),
    "save_upload_file": ("image_helpers", "save_upload_file"),
    "OllamaClient": ("AgentTools.ollama_client", "OllamaClient"),
    "OllamaClientError": ("AgentTools.ollama_client", "OllamaClientError"),
    "ask_ollama": ("AgentTools.ollama_client", "ask_ollama"),
    "AgentLogger": ("AgentTools.logger", "AgentLogger"),
    "score_image_quality": ("AgentTools.face_score", "score_image_quality"),
    "batch_score_images": ("AgentTools.face_score", "batch_score_images"),
    "search_images": ("AgentTools.serper_images", "search_images"),
    "ImageRoutesTool": ("AgentTools.image_routes_tool", "ImageRoutesTool"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'src.helpers' has no attribute {name!r}") from exc

    value = getattr(import_module(f".{module_name}", __name__), attr_name)
    globals()[name] = value
    return value
