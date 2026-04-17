"""
Hugging Face Model Loader Utility

Centralized utility for downloading and caching ML models from Hugging Face.
Handles error cases, retry logic, and provides production-ready model loading.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

try:
    from huggingface_hub import HfHubHTTPError
except ImportError:
    # Fallback to generic HTTPError for older versions
    from requests.exceptions import HTTPError as HfHubHTTPError

logger = logging.getLogger(__name__)

# Configuration
HF_REPO_ID = "Osama12324234/face-models"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class HFModelLoadError(Exception):
    """Raised when Hugging Face model loading fails after retries."""
    pass


def download_hf_model(
    repo_id: str,
    filename: str,
    cache_dir: Optional[str] = None,
    max_retries: int = MAX_RETRIES,
    retry_delay: int = RETRY_DELAY
) -> Path:
    """
    Download a model from Hugging Face with retry logic and error handling.
    
    Args:
        repo_id: Hugging Face repository ID
        filename: Model filename to download
        cache_dir: Optional cache directory override
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Path to the downloaded model file
        
    Raises:
        HFModelLoadError: If download fails after all retries
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Downloading {filename} from {repo_id} (attempt {attempt + 1}/{max_retries + 1})")
            
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                resume_download=True,
                force_download=False
            )
            
            logger.info(f"Successfully downloaded {filename} to {model_path}")
            return Path(model_path)
            
        except RepositoryNotFoundError as e:
            error_msg = f"Repository {repo_id} not found or model {filename} doesn't exist"
            logger.error(error_msg)
            raise HFModelLoadError(error_msg) from e
            
        except RevisionNotFoundError as e:
            error_msg = f"Model {filename} not found in repository {repo_id}"
            logger.error(error_msg)
            raise HFModelLoadError(error_msg) from e
            
        except HfHubHTTPError as e:
            last_error = e
            logger.warning(f"HTTP error downloading {filename}: {e}")
            
        except Exception as e:
            last_error = e
            logger.warning(f"Unexpected error downloading {filename}: {e}")
        
        if attempt < max_retries:
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
    
    error_msg = f"Failed to download {filename} after {max_retries + 1} attempts"
    logger.error(error_msg)
    raise HFModelLoadError(error_msg) from last_error


def ensure_inference_model(filename: str) -> Path:
    """
    Ensure an inference model is downloaded and available locally.
    
    Args:
        filename: Model filename (e.g., "codeformer.pth")
        
    Returns:
        Path to the cached model file
        
    Raises:
        HFModelLoadError: If model download fails
    """
    try:
        return download_hf_model(
            repo_id=HF_REPO_ID,
            filename=filename
        )
    except HFModelLoadError as e:
        # Provide helpful error message for users
        logger.error(f"Failed to load inference model {filename}")
        logger.error(f"Please ensure the model exists in {HF_REPO_ID}")
        logger.error(f"Visit https://huggingface.co/{HF_REPO_ID} to verify")
        raise


def get_model_info() -> dict:
    """
    Get information about available models in the Hugging Face repository.
    
    Returns:
        Dictionary with model information
    """
    return {
        "repo_id": HF_REPO_ID,
        "models": {
            "stargan_nets": {
                "filename": "582000_nets.ckpt",
                "description": "StarGAN v2 main networks checkpoint"
            },
            "stargan_nets_ema": {
                "filename": "582000_nets_ema.ckpt",
                "description": "StarGAN v2 EMA networks checkpoint"
            },
            "stargan_optims": {
                "filename": "582000_optims.ckpt",
                "description": "StarGAN v2 optimizer states"
            },
            "codeformer": {
                "filename": "codeformer.pth",
                "description": "CodeFormer face restoration model"
            },
            "realesrgan": {
                "filename": "RealESRGAN_x4plus.pth", 
                "description": "Real-ESRGAN super resolution model"
            },
            "wing": {
                "filename": "wing.ckpt",
                "description": "WING face alignment model"
            },
            "celeba_landmarks": {
                "filename": "celeba_lm_mean.npz",
                "description": "CelebA mean landmarks for face alignment"
            }
        }
    }


def clear_model_cache() -> None:
    """Clear the Hugging Face model cache."""
    try:
        from huggingface_hub import snapshot_download
        cache_dir = snapshot_download(repo_id=HF_REPO_ID, allow_patterns=["*.dummy"])
        if cache_dir:
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)
            logger.info("Cleared model cache")
    except Exception as e:
        logger.warning(f"Failed to clear model cache: {e}")
