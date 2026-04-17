import os
import logging
import torch
from ..models.neural_models import Generator, StyleEncoder
from ..config import settings
from ..config.model_loading import model_loading_settings
from ..wing import FAN
from ..utils.hf_model_loader import ensure_inference_model, HFModelLoadError

logger = logging.getLogger(__name__)

class ModelLoader:
    """Load neural network models on startup"""
    
    @staticmethod
    def _resolve_checkpoint_path(filename: str, base_path: str) -> str:
        """Resolve checkpoint path from HuggingFace first, fallback to local"""
        # Try HuggingFace FIRST if enabled
        if model_loading_settings.use_huggingface:
            try:
                logger.info(f"Downloading {filename} from HuggingFace ({model_loading_settings.get_model_source()})...")
                hf_path = ensure_inference_model(filename)
                logger.info(f"✓ Loaded {filename} from HuggingFace: {hf_path}")
                return str(hf_path)
            except HFModelLoadError as e:
                logger.warning(f"Failed to load {filename} from HuggingFace, trying local: {e}")
        
        # Fallback to local if HF disabled or failed
        local_path = os.path.join(base_path, "checkpoints", filename)
        if os.path.exists(local_path):
            logger.info(f"✓ Loaded {filename} from local: {local_path}")
            return local_path
        
        logger.error(f"Could not find {filename} in HuggingFace or local checkpoints")
        return None
    
    @staticmethod
    def load_models(base_path):
        """Load generator and style encoder from checkpoint"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.debug(f"Device: {device}")
            logger.info(f"Model source: {model_loading_settings.get_model_source()}")

            nets_ema_path = ModelLoader._resolve_checkpoint_path("582000_nets_ema.ckpt", base_path)
            nets_path = ModelLoader._resolve_checkpoint_path("582000_nets.ckpt", base_path)
            wing_path = ModelLoader._resolve_checkpoint_path("wing.ckpt", base_path)

            if not nets_ema_path and not nets_path:
                logger.error("Neither nets_ema nor nets checkpoint found")
                return None

            if str(settings.CHECKPOINT_VARIANT).lower() == "ema":
                checkpoint_path = nets_ema_path if nets_ema_path else nets_path
            else:
                checkpoint_path = nets_path if nets_path else nets_ema_path
            
            if not checkpoint_path:
                logger.error(f"Checkpoint not found")
                return None

            logger.info(f"Loading checkpoint: {checkpoint_path}")

            # Prefer EMA weights for inference quality, same as original StarGAN v2 sampling path.
            checkpoint = torch.load(checkpoint_path, map_location=device)
            logger.debug(f"Loaded checkpoint: {checkpoint_path}")
            logger.debug(f"Checkpoint keys: {list(checkpoint.keys())}")

            # Extract state_dicts
            generator_state = checkpoint.get('generator')
            style_encoder_state = checkpoint.get('style_encoder')

            if not generator_state or not style_encoder_state:
                print(f"✗ Missing generator or style_encoder state_dict in checkpoint")
                return None

            # Create model instances and load state_dicts
            logger.debug("Creating Generator model...")
            generator = Generator(img_size=256, style_dim=64, w_hpf=float(settings.W_HPF))
            generator.load_state_dict(generator_state)
            generator.to(device)
            generator.eval()
            logger.debug("Generator loaded successfully")
            
            logger.debug("Creating StyleEncoder model...")
            style_encoder = StyleEncoder(
                img_size=256,
                style_dim=64,
                num_domains=int(settings.NUM_DOMAINS),
            )
            style_encoder.load_state_dict(style_encoder_state)
            style_encoder.to(device)
            style_encoder.eval()
            logger.debug("StyleEncoder loaded successfully")

            fan_model = None
            fan_state = checkpoint.get("fan")
            if fan_state:
                try:
                    fan_model = FAN().to(device).eval()
                    fan_model.load_state_dict(fan_state, strict=False)
                    logger.debug("FAN loaded from checkpoint")
                except Exception as e:
                    logger.warning(f"FAN checkpoint load failed: {e}")
                    fan_model = None
            
            return {
                'generator': generator,
                'style_encoder': style_encoder,
                'fan_model': fan_model,
                'wing_path': wing_path,
                'device': device,
                'checkpoint_path': checkpoint_path,
            }
        
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}", exc_info=True)
            return None
