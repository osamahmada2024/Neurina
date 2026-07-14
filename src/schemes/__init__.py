from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "UserSchema": ("user_schema", "UserSchema"),
    "UserResponseSchema": ("user_schema", "UserResponseSchema"),
    "LoginSchema": ("user_schema", "LoginSchema"),
    "LoginProviderSchema": ("user_schema", "LoginProviderSchema"),
    "ProviderLoginRequestSchema": ("user_schema", "ProviderLoginRequestSchema"),
    "UserProfileSchema": ("user_schema", "UserProfileSchema"),
    "ForgotPasswordSchema": ("user_schema", "ForgotPasswordSchema"),
    "ResetPasswordSchema": ("user_schema", "ResetPasswordSchema"),
    "EditProfileSchema": ("user_schema", "EditProfileSchema"),
    "ContactUsSchema": ("user_schema", "ContactUsSchema"),
    "ImageSchema": ("image_schema", "ImageSchema"),
    "TranslationTaskSchema": ("image_schema", "TranslationTaskSchema"),
    "ImageUploadResponseSchema": ("image_schema", "ImageUploadResponseSchema"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'src.schemes' has no attribute {name!r}") from exc

    value = getattr(import_module(f".{module_name}", __name__), attr_name)
    globals()[name] = value
    return value
