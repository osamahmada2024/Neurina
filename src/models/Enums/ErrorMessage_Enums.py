from enum import Enum


class ImageErrorMessage(Enum):
    """Image processing error messages"""
    INVALID_FORMAT = "Invalid image file format. Allowed: jpg, jpeg, png, gif, bmp"
    READ_ERROR = "Could not read image file"
    NO_FACES_DETECTED = "No faces detected in image"
    FACE_DETECTION_FAILED = "Face detection failed"
    FACE_DETECTOR_NOT_INITIALIZED = "Face detector not initialized"
    PREPROCESSING_ERROR = "Image preprocessing error"
    PROCESSING_ERROR = "Error processing image"


class ValidationErrorMessage(Enum):
    """Validation error messages"""
    INVALID_IMAGE_TYPE = "image_type must be 'source' or 'reference'"
    MISSING_AUTHORIZATION = "Authorization header missing"
    INVALID_AUTH_HEADER = "Invalid authorization header"
    INVALID_TOKEN = "Invalid token"
    AUTH_FAILED = "Authentication failed"
