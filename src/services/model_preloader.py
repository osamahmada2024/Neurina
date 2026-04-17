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

from .face_restoration_service import FaceRestorationService
from ..utils.hf_model_loader import ensure_inference_model, HFModelLoadError
from ..utils.console_feedback import console_feedback
from ..services.model_loader import ModelLoader
from ..config.model_loading import model_loading_settings

logger = logging.getLogger(__name__)


class ModelPreloader:
    """Service for preloading all ML models at startup."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.loaded_models: Dict[str, any] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_local_checkpoint_path(self, filename: str) -> Path:
        return model_loading_settings.resolve_checkpoint_path(filename, self.base_path)

    def _get_inference_checkpoint_path(self, filename: str) -> Optional[Path]:
        local_path = self._get_local_checkpoint_path(filename)

        if local_path.is_file():
            return local_path

        if not model_loading_settings.use_huggingface:
            logger.debug(
                "Local checkpoint not found with Hugging Face disabled: %s",
                local_path,
            )
            return None

        try:
            return ensure_inference_model(filename)
        except HFModelLoadError as exc:
            logger.error("Failed to load %s from Hugging Face: %s", filename, exc)
            return None
        
    def preload_all_models(self) -> Dict[str, bool]:
        """
        Preload all required models for the application.
        
        Returns:
            Dictionary with loading status for each model type
        """
        results = {}
        
        logger.debug(f"Starting model preloading on device: {self.device}")
        start_time = time.time()
        
        # 1. Load StarGAN v2 models (training checkpoints)
        results['stargan_models'] = self._preload_stargan_models()
        
        # 2. Load Face Restoration models (Hugging Face)
        results['face_restoration_models'] = self._preload_face_restoration_models()
        
        # 3. Load Wing model for face detection
        results['wing_model'] = self._preload_wing_model()
        
        total_time = time.time() - start_time
        success_count = sum(1 for success in results.values() if success)
        
        logger.debug(f"Model preloading completed in {total_time:.2f}s")
        logger.debug(f"Successfully loaded {success_count}/{len(results)} model groups")
        
        if not all(results.values()):
            failed_models = [name for name, success in results.items() if not success]
            logger.error(f"Failed to load models: {failed_models}")
        
        return results
    
    def _preload_stargan_models(self) -> bool:
        """Preload StarGAN v2 models from local checkpoints."""
        try:
            logger.debug("Loading StarGAN v2 models...")
            
            models = ModelLoader.load_models(str(self.base_path))
            if models:
                self.loaded_models['stargan'] = models
                logger.debug("StarGAN v2 models loaded successfully")
                return True
            else:
                logger.error("Failed to load StarGAN v2 models")
                return False
                
        except Exception as e:
            logger.error(f"Error loading StarGAN v2 models: {e}")
            return False
    
    def _preload_face_restoration_models(self) -> bool:
        """Preload face restoration models (CodeFormer, Real-ESRGAN)."""
        try:
            logger.debug(
                "Loading face restoration models from %s...",
                model_loading_settings.get_model_source(),
            )

            codeformer_path = self._get_inference_checkpoint_path("codeformer.pth")
            if codeformer_path is None:
                return False

            realesrgan_path = self._get_inference_checkpoint_path("RealESRGAN_x4plus.pth")
            if realesrgan_path is None:
                return False
            
            # Create face restoration service with preloaded models
            face_service = FaceRestorationService(
                base_path=str(self.base_path),
                model_name="codeformer",
                outscale=2.0,
                tile=0,
                face_weight=0.5,
                codeformer_fidelity=0.7,
            )
            logger.debug("Preloading CodeFormer model...")
            try:
                logger.debug("CodeFormer checkpoint resolved at %s", codeformer_path)
                
                # Trigger model initialization by accessing the model
                face_service._build_codeformer()
                console_feedback("CodeFormer ready")
                logger.debug("CodeFormer model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CodeFormer: {e}")
                return False
            
            # Preload Real-ESRGAN model
            logger.debug("Preloading Real-ESRGAN model...")
            try:
                logger.debug("Real-ESRGAN checkpoint resolved at %s", realesrgan_path)
                face_service._ensure_realesrgan_weights()
                console_feedback("Real-ESRGAN ready")
                
                logger.debug("Real-ESRGAN model preload verified")
            except Exception as e:
                logger.error(f"Failed to preload Real-ESRGAN: {e}")
                return False
            
            # Store the initialized service
            self.loaded_models['face_restoration'] = face_service
            console_feedback("High-resolution model ready")
            logger.debug("Face restoration models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading face restoration models: {e}")
            return False
    
    def _preload_wing_model(self) -> bool:
        """Preload Wing model for face detection."""
        try:
            logger.debug("Loading Wing model...")

            wing_path = self._get_inference_checkpoint_path("wing.ckpt")
            if wing_path is None:
                return False

            if wing_path.exists() and wing_path.is_file():
                self.loaded_models['wing_path'] = str(wing_path)
                console_feedback("Wing model ready")
                logger.debug(f"Wing model path verified: {wing_path}")
                return True
            else:
                logger.debug(f"Wing model checkpoint not found at {wing_path}")
                return False
                
        except Exception as e:
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
