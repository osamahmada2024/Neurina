from enum import Enum

class ImageStatus(Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    FACE_DETECTED = "face_detected"
    PREPROCESSED = "preprocessed"
    TRANSLATION_COMPLETED = "translation_completed"
    FAILED = "failed"


class ImageType(Enum):
    SOURCE = "source"
    REFERENCE = "reference"
    TRANSLATED = "translated"
