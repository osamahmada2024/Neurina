"""
Model Preloader Service

Eagerly loads all ML models at FastAPI startup to eliminate lazy loading timeouts
and ensure optimal performance during API requests.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch

from ..config import settings
from ..config.model_loading import model_loading_settings
from .face_restoration_service import FaceRestorationService
from ..utils.hf_model_loader import ensure_inference_model, HFModelLoadError
from ..services.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class ModelPreloader:
    """Service for preloading all ML models at startup."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.loaded_models: Dict[str, any] = {}
        self.loading_errors: Dict[str, str] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _resolve_configured_path(self, configured_path: str) -> Path:
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = self.base_path / path
        return path
        
    def preload_all_models(self) -> Dict[str, bool]:
        """
        Preload all required models for the application.
        
        Returns:
            Dictionary with loading status for each model type
        """
        results = {}
        
        logger.info(f"Starting model preloading on device: {self.device}")
        start_time = time.time()
        
        # 1. Load StarGAN v2 models (training checkpoints)
        results['stargan_models'] = self._preload_stargan_models()
        
        # 2. Load Face Restoration models (Hugging Face)
        results['face_restoration_models'] = self._preload_face_restoration_models()
        
        # 3. Load Wing model for face detection
        results['wing_model'] = self._preload_wing_model()
        
        total_time = time.time() - start_time
        success_count = sum(1 for success in results.values() if success)
        
        logger.info(f"Model preloading completed in {total_time:.2f}s")
        logger.info(f"Successfully loaded {success_count}/{len(results)} model groups")
        
        if not all(results.values()):
            failed_models = [name for name, success in results.items() if not success]
            logger.error(f"Failed to load models: {failed_models}")
        
        return results
    
    def _preload_stargan_models(self) -> bool:
        """Preload StarGAN v2 models from HuggingFace or local checkpoints."""
        try:
            logger.info("Loading StarGAN v2 models...")

            if model_loading_settings.use_huggingface:
                try:
                    nets_ema_path = ensure_inference_model("582000_nets_ema.ckpt")
                    nets_path = ensure_inference_model("582000_nets.ckpt")
                    logger.info("StarGAN checkpoints downloaded from Hugging Face")
                except HFModelLoadError as e:
                    logger.warning("Failed to download StarGAN checkpoints from Hugging Face: %s", e)
                    nets_ema_path = model_loading_settings.resolve_checkpoint_path("582000_nets_ema.ckpt", self.base_path)
                    nets_path = model_loading_settings.resolve_checkpoint_path("582000_nets.ckpt", self.base_path)
                    if not nets_ema_path.exists() and not nets_path.exists():
                        detail = f"Hugging Face download failed and no local fallback checkpoint exists: {e}"
                        self.loading_errors['stargan_models'] = detail
                        logger.error(detail)
                        return False
                    logger.warning("Falling back to local StarGAN checkpoints after Hugging Face failure")
            else:
                nets_ema_path = model_loading_settings.resolve_checkpoint_path("582000_nets_ema.ckpt", self.base_path)
                nets_path = model_loading_settings.resolve_checkpoint_path("582000_nets.ckpt", self.base_path)

            models = ModelLoader.load_models(str(self.base_path), checkpoint_path=nets_ema_path, fallback_path=nets_path)
            if models:
                self.loaded_models['stargan'] = models
                self.loading_errors.pop('stargan_models', None)
                logger.info("StarGAN v2 models loaded successfully")
                return True
            else:
                detail = (
                    "StarGAN checkpoints were resolved but the model state could not be constructed. "
                    f"Primary={nets_ema_path}, Fallback={nets_path}"
                )
                self.loading_errors['stargan_models'] = detail
                logger.error("Failed to load StarGAN v2 models")
                return False
                
        except Exception as e:
            self.loading_errors['stargan_models'] = str(e)
            logger.error(f"Error loading StarGAN v2 models: {e}")
            return False
    
    def _preload_face_restoration_models(self) -> bool:
        """Preload face restoration models (CodeFormer, Real-ESRGAN)."""
        try:
            logger.info(f"Loading face restoration models from {model_loading_settings.get_model_source()}...")
            
            # Create face restoration service with preloaded models
            face_service = FaceRestorationService(
                base_path=str(self.base_path),
                model_name="codeformer",
                outscale=2.0,
                tile=0,
                face_weight=0.5,
                codeformer_fidelity=0.7,
            )
            logger.info("Preloading CodeFormer model...")
            try:
                if model_loading_settings.use_huggingface:
                    codeformer_path = ensure_inference_model("codeformer.pth")
                    logger.info("CodeFormer model downloaded from Hugging Face")
                else:
                    # Use local checkpoint
                    codeformer_path = self.base_path / "checkpoints" / "codeformer.pth"
                    if not codeformer_path.exists():
                        logger.warning(f"Local CodeFormer model not found at {codeformer_path}")
                        return False
                    logger.info("CodeFormer model loaded from local checkpoint")
                
                # Trigger model initialization by accessing the model
                codeformer_net = face_service._build_codeformer()
                logger.info("CodeFormer model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CodeFormer: {e}")
                return False
            
            # Preload Real-ESRGAN model
            logger.info("Preloading Real-ESRGAN model...")
            try:
                if model_loading_settings.use_huggingface:
                    realesrgan_path = ensure_inference_model("RealESRGAN_x4plus.pth")
                    logger.info("Real-ESRGAN model downloaded from Hugging Face")
                else:
                    # Use local checkpoint
                    realesrgan_path = self.base_path / "checkpoints" / "RealESRGAN_x4plus.pth"
                    if not realesrgan_path.exists():
                        logger.warning(f"Local Real-ESRGAN model not found at {realesrgan_path}")
                        return False
                    logger.info("Real-ESRGAN model loaded from local checkpoint")
                
                # Trigger model initialization
                upsampler = face_service._build_generic_upsampler()
                logger.info("Real-ESRGAN model loaded successfully")
            except HFModelLoadError as e:
                logger.error(f"Failed to load Real-ESRGAN: {e}")
                return False
            
            # Store the initialized service
            self.loaded_models['face_restoration'] = face_service
            self.loading_errors.pop('face_restoration_models', None)
            logger.info("Face restoration models loaded successfully")
            return True
            
        except Exception as e:
            self.loading_errors['face_restoration_models'] = str(e)
            logger.error(f"Error loading face restoration models: {e}")
            return False
    
    def _preload_wing_model(self) -> bool:
        """Preload Wing model for face detection."""
        try:
            logger.info("Loading Wing model...")
            
            # Wing model path based on loading preference
            if model_loading_settings.use_huggingface:
                try:
                    wing_path = ensure_inference_model("wing.ckpt")
                    logger.info("Wing model downloaded from Hugging Face")
                except HFModelLoadError:
                    logger.warning("Wing model not available on Hugging Face, falling back to local")
                    wing_path = self._resolve_configured_path(settings.WING_MODEL_PATH)
            else:
                wing_path = self._resolve_configured_path(settings.WING_MODEL_PATH)
            
            # Verify model exists
            if wing_path.exists() and wing_path.is_file():
                self.loaded_models['wing_path'] = str(wing_path)
                self.loading_errors.pop('wing_model', None)
                logger.info(f"Wing model path verified: {wing_path}")
                return True
            else:
                self.loading_errors['wing_model'] = f"Wing model checkpoint not found at {wing_path}"
                logger.warning(f"Wing model checkpoint not found at {wing_path}")
                return False
                
        except Exception as e:
            self.loading_errors['wing_model'] = str(e)
            logger.error(f"Error loading Wing model: {e}")
            return False
    
    def get_loaded_models(self) -> Dict[str, any]:
        """Get dictionary of all loaded models."""
        return self.loaded_models.copy()
    
    def get_model(self, model_type: str) -> Optional[any]:
        """Get a specific loaded model by type."""
        return self.loaded_models.get(model_type)
    
    def is_model_loaded(self, model_type: str) -> bool:
        """Check if a specific model type is loaded."""
        return model_type in self.loaded_models
    
    def get_loading_summary(self) -> Dict[str, str]:
        """Get a summary of loaded models for logging/monitoring."""
        summary = {}
        
        if 'stargan' in self.loaded_models:
            stargan = self.loaded_models['stargan']
            summary['stargan'] = f"Generator: {stargan['generator'] is not None}, StyleEncoder: {stargan['style_encoder'] is not None}"
        
        if 'face_restoration' in self.loaded_models:
            summary['face_restoration'] = "CodeFormer and Real-ESRGAN loaded"
        
        if 'wing_path' in self.loaded_models:
            summary['wing'] = f"Path: {self.loaded_models['wing_path']}"
        
        return summary

    def get_loading_errors(self) -> Dict[str, str]:
        """Get loading errors captured during startup preloading."""
        return self.loading_errors.copy()
