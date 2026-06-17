import requests
import io
from typing import Optional, Union
from fastapi import UploadFile
from ...config.settings import settings

class ImageRoutesTool:
    """
    Tool for uploading and managing images via the backend API.
    Handles both URL-based and file-based uploads.
    """

    def __init__(self, token: str):
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {self.token}"
        }

    #  REFERENCE IMAGE UPLOADS

    def upload_reference_from_url(self, image_url: str) -> str:
        """Upload reference image from URL to backend."""
        img_data = requests.get(image_url, timeout=30).content

        files = {
            "file": ("reference_image.jpg", img_data, "image/jpeg")
        }

        data = {
            "type": "reference"
        }
        try:
            resp = requests.post(
                f"{settings.Backend_API_URL.rstrip('/')}/api/images/upload",
                headers=self.headers,
                files=files,
                params={"image_type": "reference"},
            )
            resp.raise_for_status()
            payload = resp.json()
            return payload.get("image_id") or (payload.get("data") or {}).get("image_id", "")
        except requests.exceptions.RequestException as e:
            print(f"API Error (upload_reference_from_url): {e}")
            raise Exception(f"Failed to upload reference image: {e}")

    def upload_reference_from_bytes(
        self,
        image_bytes: bytes,
        filename: str = "reference_image.jpg"
    ) -> str:
        """Upload reference image from bytes to backend."""
        files = {
            "file": (filename, image_bytes, "image/jpeg")
        }

        data = {
            "type": "reference"
        }
        try:
            resp = requests.post(
                f"{settings.Backend_API_URL.rstrip('/')}/api/images/upload",
                headers=self.headers,
                files=files,
                params={"image_type": "reference"},
            )
            resp.raise_for_status()
            payload = resp.json()
            return payload.get("image_id") or (payload.get("data") or {}).get("image_id", "")
        except requests.exceptions.RequestException as e:
            print(f"API Error (upload_reference_from_bytes): {e}")
            raise Exception(f"Failed to upload reference image bytes: {e}")

    #  SOURCE IMAGE UPLOADS 

    def upload_source_from_url(self, image_url: str) -> str:
        """Upload source image from URL to backend."""
        img_data = requests.get(image_url, timeout=30).content

        files = {
            "file": ("source_image.jpg", img_data, "image/jpeg")
        }

        try:
            resp = requests.post(
                f"{settings.Backend_API_URL.rstrip('/')}/api/images/upload",
                headers=self.headers,
                files=files,
                params={"image_type": "source"},
            )
            resp.raise_for_status()
            payload = resp.json()
            return payload.get("image_id") or (payload.get("data") or {}).get("image_id", "")
        except requests.exceptions.RequestException as e:
            print(f"API Error (upload_source_from_url): {e}")
            raise Exception(f"Failed to upload source image: {e}")

    def upload_source_from_bytes(
        self,
        image_bytes: bytes,
        filename: str = "source_image.jpg"
    ) -> str:
        """Upload source image from bytes to backend."""
        files = {
            "file": (filename, image_bytes, "image/jpeg")
        }

        try:
            resp = requests.post(
                f"{settings.Backend_API_URL.rstrip('/')}/api/images/upload",
                headers=self.headers,
                files=files,
                params={"image_type": "source"},
            )
            resp.raise_for_status()
            payload = resp.json()
            return payload.get("image_id") or (payload.get("data") or {}).get("image_id", "")
        except requests.exceptions.RequestException as e:
            print(f"API Error (upload_source_from_bytes): {e}")
            raise Exception(f"Failed to upload source image bytes: {e}")

    #  GENERIC UPLOAD (ANY TYPE) 

    def upload_image_from_url(
        self,
        image_url: str,
        image_type: str = "reference"
    ) -> str:
        """
        Upload image from URL to backend.

        Args:
            image_url: URL of the image
            image_type: 'source' or 'reference'

        Returns:
            image_id from backend
        """
        if image_type == "source":
            return self.upload_source_from_url(image_url)
        else:
            return self.upload_reference_from_url(image_url)

    def upload_image_from_bytes(
        self,
        image_bytes: bytes,
        image_type: str = "reference",
        filename: str = "image.jpg"
    ) -> str:
        """
        Upload image from bytes to backend.

        Args:
            image_bytes: Image data as bytes
            image_type: 'source' or 'reference'
            filename: Name of the file

        Returns:
            image_id from backend
        """
        if image_type == "source":
            return self.upload_source_from_bytes(image_bytes, filename)
        else:
            return self.upload_reference_from_bytes(image_bytes, filename)

    #  TRANSLATION 

    def translate_images(
        self,
        source_image_id: str,
        reference_image_id: str,
        translation_mode: str = "auto"
    ) -> str:
        """
        Trigger style transfer translation.

        Args:
            source_image_id: ID of source image
            reference_image_id: ID of reference image
            translation_mode: Translation mode (auto, male_to_female, etc)

        Returns:
            translated_image_id from backend
        """
        try:
            resp = requests.post(
                f"{settings.Backend_API_URL.rstrip('/')}/api/images/translate",
                headers=self.headers,
                params={
                    "source_image_id": source_image_id,
                    "reference_image_id": reference_image_id,
                    "translation_mode": translation_mode,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
            return payload.get("translated_image_id") or (payload.get("data") or {}).get(
                "translated_image_id", ""
            )
        except requests.exceptions.RequestException as e:
            print(f"API Error (translate_images): {e}")
            raise Exception(f"Translation API failed: {e}")


