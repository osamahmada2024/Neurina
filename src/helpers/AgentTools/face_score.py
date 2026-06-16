import requests
import cv2
import numpy as np
from typing import Dict, Any, Optional

from ...config import settings


def download_image_from_url(image_url: str, timeout: int = 30) -> Optional[np.ndarray]:
    """Download image from URL and return as BGR numpy array."""
    try:
        response = requests.get(image_url, timeout=timeout)
        response.raise_for_status()

        image_array = np.frombuffer(response.content, np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return None
        return image_bgr
    except Exception as e:
        print(f"Error downloading image from {image_url}: {e}")
        return None


def compute_blur_score(image_bgr: np.ndarray) -> float:
    """
    Compute blur score using Laplacian variance.
    Higher = sharper, Lower = blurrier
    Normalized to 0-1 where 1 is sharp, 0 is very blurry
    """
    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Normalize: 0 = completely blurry, 1 = very sharp
        # Typical sharp images have laplacian_var > 500
        # Typical blurry images have laplacian_var < 100
        min_blur = 0.0
        max_blur = 500.0

        blur_score = min(1.0, max(min_blur, laplacian_var / max_blur))
        return blur_score
    except Exception as e:
        print(f"Error computing blur score: {e}")
        return 0.0


def compute_quality_score(image_bgr: np.ndarray) -> float:
    """
    Compute overall quality score (0-1) using multiple metrics.
    Combines: resolution, brightness, contrast, edge density
    """
    try:
        height, width = image_bgr.shape[:2]

        # Resolution score: prefer images >= 400x400
        min_dimension = min(height, width)
        resolution_score = min(1.0, max(0.0, min_dimension / 400.0))

        # Contrast score: using grayscale standard deviation
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        contrast_score = min(1.0, gray.std() / 100.0)

        # Blur score
        blur_score = compute_blur_score(image_bgr)

        # Edge density score: using Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = np.sum(edges > 0) / (height * width)
        edge_score = min(1.0, edge_ratio / 0.1)  # Expect ~10% edges in good images

        # Weighted combination (prioritize blur and contrast)
        quality = (
            resolution_score * 0.15 +
            contrast_score * 0.25 +
            blur_score * 0.35 +
            edge_score * 0.25
        )

        return float(quality)
    except Exception as e:
        print(f"Error computing quality score: {e}")
        return 0.0


def count_faces_opencv(image_bgr: np.ndarray) -> int:
    """
    Detect faces using OpenCV cascade classifiers.
    Returns number of faces detected.
    """
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            maxSize=(None, None),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return len(faces)
    except Exception as e:
        print(f"Error detecting faces with OpenCV: {e}")
        return 0


def score_image_quality(image_url: str) -> Dict[str, Any]:
    """
    Score image quality for reference image selection.

    Args:
        image_url: URL to download and score the image

    Returns:
        Dictionary with:
        - has_face: bool (at least 1 face detected)
        - face_count: int (number of faces detected)
        - quality_score: float (0-1, overall quality)
        - blur_score: float (0-1, where 1 is sharp)
        - resolution: tuple (width, height)
        - passes_gate: bool (quality_score >= 0.7 and face_count == 1)
    """

    # Download image
    image_bgr = download_image_from_url(image_url, timeout=30)

    if image_bgr is None:
        return {
            "has_face": False,
            "face_count": 0,
            "quality_score": 0.0,
            "blur_score": 0.0,
            "resolution": (0, 0),
            "passes_gate": False,
            "error": "Failed to download image",
        }

    height, width = image_bgr.shape[:2]

    # Compute metrics
    face_count = count_faces_opencv(image_bgr)
    quality_score = compute_quality_score(image_bgr)
    blur_score = compute_blur_score(image_bgr)

    # Gate criteria: exactly 1 face + high quality score
    quality_threshold = float(settings.get("QUALITY_GATE_THRESHOLD", 0.7))
    passes_gate = (face_count == 1) and (quality_score >= quality_threshold)

    return {
        "has_face": face_count > 0,
        "face_count": face_count,
        "quality_score": quality_score,
        "blur_score": blur_score,
        "resolution": (width, height),
        "passes_gate": passes_gate,
    }


def batch_score_images(image_urls: list) -> list:
    results = []

    for url in image_urls:
        score_dict = score_image_quality(url)
        results.append((url, score_dict))

    results.sort(
        key=lambda x: (x[1]["quality_score"], x[1]["blur_score"]),
        reverse=True
    )

    return results

