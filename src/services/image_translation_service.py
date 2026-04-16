import torch
import torchvision.utils as vutils
import numpy as np
from typing import Tuple, Optional


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """
    Denormalize tensor from [-1, 1] to [0, 1]
    """
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x: torch.Tensor, ncol: int, filename: str):
    """
    Save tensor image to file
    
    Args:
        x: Image tensor
        ncol: Number of columns for grid
        filename: Output filename
    """
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


@torch.no_grad()
def translate_using_reference(
    nets: dict,
    args: object,
    x_src: torch.Tensor,
    x_ref: torch.Tensor,
    y_ref: torch.Tensor
) -> torch.Tensor:
    """
    Perform image-to-image translation using reference style
    
    Args:
        nets: Dictionary containing generator, style_encoder, and optionally fan
        args: Arguments object with configuration
        x_src: Source image tensor
        x_ref: Reference image tensor
        y_ref: Reference style label/ID
    
    Returns:
        Translated image tensor
    """
    N, C, H, W = x_src.size()
    
    # Get face heatmaps if high-pass filter is enabled
    masks = None
    if hasattr(nets, 'fan') and nets.fan is not None and hasattr(args, 'w_hpf'):
        if args.w_hpf > 0:
            try:
                masks = nets.fan.get_heatmap(x_src)
            except Exception as e:
                print(f"Warning: Could not get face heatmap: {str(e)}")
    
    # Extract reference style
    if hasattr(nets, 'style_encoder'):
        s_ref = nets.style_encoder(x_ref, y_ref)
    else:
        raise ValueError("Style encoder not available in nets")
    
    # Generate translated image
    if hasattr(nets, 'generator'):
        if masks is not None:
            x_fake = nets.generator(x_src, s_ref, masks=masks)
        else:
            x_fake = nets.generator(x_src, s_ref)
    else:
        raise ValueError("Generator not available in nets")
    
    return x_fake


@torch.no_grad()
def translate_batch(
    nets: dict,
    args: object,
    x_src: torch.Tensor,
    x_ref: torch.Tensor,
    y_ref: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Translate a batch of images
    
    Returns:
        Tuple of (source_images, translated_images)
    """
    x_fake = translate_using_reference(nets, args, x_src, x_ref, y_ref)
    return x_src, x_fake


def normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [-1, 1]"""
    return x * 2 - 1


def get_translation_stats(
    x_src: torch.Tensor,
    x_fake: torch.Tensor
) -> dict:
    """
    Calculate statistics about translation
    
    Returns:
        Dictionary with translation metrics
    """
    return {
        "source_mean": float(x_src.mean().cpu().item()),
        "source_std": float(x_src.std().cpu().item()),
        "translated_mean": float(x_fake.mean().cpu().item()),
        "translated_std": float(x_fake.std().cpu().item()),
    }
