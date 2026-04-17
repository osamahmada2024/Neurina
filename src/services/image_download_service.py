import base64
import io

import cv2
import numpy as np
import requests
from fastapi import HTTPException


class ImageDownloadService:
    """Service for turning stored image payloads into PNG bytes."""

    PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

    @staticmethod
    def decode_base64_bytes(image_data: str) -> bytes:
        try:
            return base64.b64decode(image_data)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error decoding image bytes: {str(exc)}")

    @staticmethod
    def download_url_bytes(image_url: str) -> bytes:
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error downloading image: {str(exc)}")

    @staticmethod
    def decode_base64_image(image_data: str) -> np.ndarray:
        try:
            image_bytes = ImageDownloadService.decode_base64_bytes(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                raise ValueError("Failed to decode image")

            return img_bgr
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error decoding image: {str(exc)}")

    @staticmethod
    def encode_png_bytes(img_bgr: np.ndarray) -> bytes:
        try:
            success, buffer = cv2.imencode(".png", img_bgr)
            if not success:
                raise ValueError("Failed to encode image as PNG")
            return buffer.tobytes()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error converting to PNG: {str(exc)}")

    @staticmethod
    def download_image(image_data: str, filename: str) -> bytes:
        try:
            if image_data.startswith(("http://", "https://")):
                image_bytes = ImageDownloadService.download_url_bytes(image_data)
            else:
                image_bytes = ImageDownloadService.decode_base64_bytes(image_data)
            if image_bytes.startswith(ImageDownloadService.PNG_SIGNATURE):
                return image_bytes

            if image_data.startswith(("http://", "https://")):
                nparr = np.frombuffer(image_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise HTTPException(status_code=400, detail="Error decoding downloaded image")
            else:
                img_bgr = ImageDownloadService.decode_base64_image(image_data)
            return ImageDownloadService.encode_png_bytes(img_bgr)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(exc)}")

    @staticmethod
    def to_buffer(image_bytes: bytes) -> io.BytesIO:
        buffer = io.BytesIO(image_bytes)
        buffer.seek(0)
        return buffer
