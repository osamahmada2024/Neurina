from enum import Enum


class ImageFormat(Enum):
    """Supported image formats"""
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    BMP = "bmp"
    
    @classmethod
    def get_supported_formats(cls) -> list:
        """Get list of supported format extensions"""
        return [fmt.value for fmt in cls]
    
    @classmethod
    def is_supported(cls, extension: str) -> bool:
        """Check if format is supported"""
        return extension.lower() in cls.get_supported_formats()
