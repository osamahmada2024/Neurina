"""
Model loading configuration.

Only controls the Hugging Face toggle and the local checkpoints directory used
when remote loading is disabled.
"""

from pathlib import Path

from pydantic import Field
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

    model_config = {
        "env_prefix": "MODEL_LOADING_",
        "env_file": str(_ENV_FILE),
        "case_sensitive": False,
        "extra": "ignore",
    }

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


model_loading_settings = ModelLoadingSettings()
