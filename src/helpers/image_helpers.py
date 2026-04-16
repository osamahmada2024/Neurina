import cv2
import numpy as np
import torch
import base64
import io
from PIL import Image
from typing import Tuple, Optional
from ..models.Enums import ImageFormat
import os


def convert_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


def convert_base64_to_image(base64_str: str) -> np.ndarray:
    """Convert base64 string to numpy array"""
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def convert_opencv_to_pillow(cv_image: np.ndarray) -> Image.Image:
    """Convert OpenCV image (BGR) to Pillow image (RGB)"""
    return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))


def convert_pillow_to_opencv(pil_image: Image.Image) -> np.ndarray:
    """Convert Pillow image (RGB) to OpenCV image (BGR)"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def resize_image(image: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Resize image to target size"""
    return cv2.resize(image, size)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [-1, 1] range for model input"""
    image = image.astype(np.float32) / 255.0
    return image * 2 - 1


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Denormalize image from [-1, 1] to [0, 255]"""
    image = (image + 1) / 2
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)


def prepare_image_for_model(image: np.ndarray, img_size: int = 256) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Prepare image for model inference
    Returns: (tensor, original_image)
    """
    # Resize
    resized = resize_image(image, (img_size, img_size))
    
    # Normalize
    normalized = normalize_image(resized)
    
    # Convert to tensor (HWC -> CHW)
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    
    return tensor, resized


def tensor_to_cv_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert model output tensor to OpenCV image"""
    # Remove batch dimension and convert to numpy
    image = tensor.squeeze(0).cpu().detach().numpy()
    
    # CHW -> HWC
    image = image.transpose(1, 2, 0)
    
    # Denormalize
    image = denormalize_image(image)
    
    return image


def draw_landmarks_on_image(image: np.ndarray, landmarks: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw facial landmarks on image
    landmarks: numpy array of shape (98, 2) or similar
    """
    image_copy = image.copy()
    
    if landmarks is not None and len(landmarks) > 0:
        # Draw circles at each landmark
        for i, (x, y) in enumerate(landmarks.astype(int)):
            cv2.circle(image_copy, (x, y), 2, color, -1)
        
        # Draw lines connecting landmarks to form face outline
        # Assuming 98-point landmark format
        if len(landmarks) == 98:
            # Jawline
            for i in range(0, 17):
                pt1 = tuple(landmarks[i].astype(int))
                pt2 = tuple(landmarks[i+1].astype(int)) if i+1 < 17 else None
                if pt2:
                    cv2.line(image_copy, pt1, pt2, color, 1)
            
            # Left eyebrow
            for i in range(17, 22):
                pt1 = tuple(landmarks[i].astype(int))
                pt2 = tuple(landmarks[i+1].astype(int)) if i+1 < 22 else None
                if pt2:
                    cv2.line(image_copy, pt1, pt2, color, 1)
            
            # Right eyebrow
            for i in range(22, 27):
                pt1 = tuple(landmarks[i].astype(int))
                pt2 = tuple(landmarks[i+1].astype(int)) if i+1 < 27 else None
                if pt2:
                    cv2.line(image_copy, pt1, pt2, color, 1)
    
    return image_copy


def validate_image_file(filename: str, allowed_extensions: list = None) -> bool:
    
    
    if allowed_extensions is None:
        allowed_extensions = ImageFormat.get_supported_formats()
    
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in allowed_extensions


async def save_upload_file(upload_file, destination_path: str):
    """Save uploaded file to destination"""
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    
    contents = await upload_file.read()
    with open(destination_path, 'wb') as f:
        f.write(contents)
    
    return destination_path
