"""
Model loading configuration.

Controls the Hugging Face toggle plus optional repository, cache, and token
settings used for remote checkpoint loading.
"""

from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class ModelLoadingSettings(BaseSettings):
    """Settings for choosing between Hugging Face and local checkpoints."""

    use_huggingface: bool = Field(
        default=False,
        description="When false, inference checkpoints must be loaded locally only.",
    )
    local_checkpoints_dir: Path = Field(
        default=Path("./checkpoints"),
        description="Directory that stores local inference checkpoints.",
    )
    hf_model_repo: str = Field(
        default="Osama12324234/face-models",
        description="Hugging Face repository that stores inference checkpoints.",
    )
    hf_cache_dir: str | None = Field(
        default=None,
        description="Optional Hugging Face cache directory override.",
    )
    hf_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN", "MODEL_LOADING_HF_TOKEN"),
        description="Optional Hugging Face token for private model repositories.",
    )

    model_config = {
        "env_prefix": "MODEL_LOADING_",
        "env_file": str(_ENV_FILE),
        "case_sensitive": False,
        "extra": "ignore",
    }

    @field_validator("hf_model_repo", "hf_cache_dir", "hf_token", mode="before")
    @classmethod
    def _normalize_optional_strings(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            return value

        cleaned = value.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
            cleaned = cleaned[1:-1].strip()
        return cleaned or None

    def get_model_source(self) -> str:
        """Return the active checkpoint source label for logs."""
        return "Hugging Face" if self.use_huggingface else "local checkpoints"

    def resolve_checkpoints_dir(self, base_path: str | Path | None = None) -> Path:
        """Resolve the configured checkpoints directory against the project root."""
        checkpoints_dir = Path(self.local_checkpoints_dir)
        if checkpoints_dir.is_absolute() or base_path is None:
            return checkpoints_dir
        return Path(base_path) / checkpoints_dir

    def resolve_checkpoint_path(
        self,
        filename: str,
        base_path: str | Path | None = None,
    ) -> Path:
        """Resolve a specific checkpoint path."""
        return self.resolve_checkpoints_dir(base_path) / filename

    def resolve_cache_dir(self) -> Path | None:
        """Resolve the configured Hugging Face cache directory, if present."""
        if not self.hf_cache_dir:
            return None
        return Path(self.hf_cache_dir).expanduser()


model_loading_settings = ModelLoadingSettings()
