import os
import logging
from pathlib import Path
import torch
from ..models.neural_models import Generator, StyleEncoder
from ..config import settings
from ..wing import FAN

logger = logging.getLogger(__name__)

class ModelLoader:
    """Load neural network models on startup"""

    @staticmethod
    def _resolve_path(base_path, configured_path):
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = Path(base_path) / path
        return str(path)
    
    @staticmethod
    def load_models(base_path, checkpoint_path=None, fallback_path=None):
        """Load generator and style encoder from checkpoint
        
        Args:
            base_path: Project base path (used for wing model lookup)
            checkpoint_path: Explicit path to primary checkpoint file (e.g. from HF download)
            fallback_path: Explicit path to fallback checkpoint file
        """
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.debug(f"Device: {device}")

            wing_path = ModelLoader._resolve_path(base_path, settings.WING_MODEL_PATH)

            if checkpoint_path is not None or fallback_path is not None:
                # Use explicitly provided paths (e.g. from HuggingFace download)
                nets_ema_path = str(checkpoint_path) if checkpoint_path else None
                nets_path = str(fallback_path) if fallback_path else None
            else:
                # Legacy: look in local checkpoints directory
                nets_ema_path = os.path.join(base_path, "checkpoints", "582000_nets_ema.ckpt")
                nets_path = os.path.join(base_path, "checkpoints", "582000_nets.ckpt")

            if str(settings.CHECKPOINT_VARIANT).lower() == "ema":
                primary = nets_ema_path
                secondary = nets_path
            else:
                primary = nets_path
                secondary = nets_ema_path

            # Select the first existing path
            selected = None
            if primary and os.path.exists(primary):
                selected = primary
            elif secondary and os.path.exists(secondary):
                selected = secondary
            
            if not selected:
                checked = [p for p in [primary, secondary] if p]
                print(f"✗ Checkpoint not found at {checked}")
                return None
            checkpoint_path = selected

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
                'wing_path': wing_path if os.path.exists(wing_path) else None,
                'device': device,
                'checkpoint_path': checkpoint_path,
            }
        
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}", exc_info=True)
            return None
