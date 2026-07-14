from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentLogger": ("logger", "AgentLogger"),
    "OllamaClient": ("ollama_client", "OllamaClient"),
    "OllamaClientError": ("ollama_client", "OllamaClientError"),
    "ask_ollama": ("ollama_client", "ask_ollama"),
    "score_image_quality": ("face_score", "score_image_quality"),
    "batch_score_images": ("face_score", "batch_score_images"),
    "search_images": ("serper_images", "search_images"),
    "ImageRoutesTool": ("image_routes_tool", "ImageRoutesTool"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'src.helpers.AgentTools' has no attribute {name!r}") from exc

    value = getattr(import_module(f".{module_name}", __name__), attr_name)
    globals()[name] = value
    return value
