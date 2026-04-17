"""
Cloudinary Configuration

Pydantic settings for Cloudinary cloud storage integration.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional


class CloudinarySettings(BaseSettings):
    """Cloudinary configuration settings with validation."""
    
    cloud_name: str = Field(..., description="Cloudinary cloud name")
    api_key: str = Field(..., description="Cloudinary API key")
    api_secret: str = Field(..., description="Cloudinary API secret")
    secure: bool = Field(default=True, description="Use HTTPS for Cloudinary URLs")
    
    model_config = {
        "env_prefix": "CLOUDINARY_",
        "env_file": "src/.env",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra environment variables
    }
    
    @field_validator('cloud_name')
    @classmethod
    def validate_cloud_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Cloudinary cloud name cannot be empty')
        return v.strip()
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Cloudinary API key cannot be empty')
        return v.strip()
    
    @field_validator('api_secret')
    @classmethod
    def validate_api_secret(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Cloudinary API secret cannot be empty')
        return v.strip()
    
    def is_configured(self) -> bool:
        """Check if all required Cloudinary settings are configured."""
        return all([
            self.cloud_name,
            self.api_key,
            self.api_secret
        ])
    
    def get_config_dict(self) -> dict:
        """Get Cloudinary configuration as dictionary for cloudinary.config."""
        return {
            'cloud_name': self.cloud_name,
            'api_key': self.api_key,
            'api_secret': self.api_secret,
            'secure': self.secure
        }


# Global instance
cloudinary_settings = CloudinarySettings()
