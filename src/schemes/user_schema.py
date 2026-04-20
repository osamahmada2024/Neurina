from pydantic import BaseModel, EmailStr, Field, ConfigDict, field_validator
from typing import Optional, Any
from bson import ObjectId

class ObjectIdField(str):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema, PydanticUndefined
        
        def validate_str(value: str) -> str:
            if not ObjectId.is_valid(value):
                raise ValueError("Invalid ObjectId")
            return str(value)
        
        return core_schema.no_info_before_validator_function(
            validate_str,
            core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(str)
        )


class UserSchema(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={
            ObjectId: lambda x: str(x)
        }
    )
    
    id: Optional[ObjectIdField] = Field(default=None, alias="_id")
    name : str = Field(..., min_length = 2)
    email: EmailStr
    password: str = Field(..., min_length = 6)
    provider : Optional[str] = "local"
    profile_picture : Optional[str] = "https://www.radfordacademy.co.uk/content/uploads/sites/14/2024/07/Staff-placeholder-image.jpeg"


class LoginSchema(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length = 6)


class LoginProviderSchema(BaseModel):
    email: EmailStr
    name : str
    provider_id : str
    provider : str
    profile_picture : Optional[str] = "https://www.radfordacademy.co.uk/content/uploads/sites/14/2024/07/Staff-placeholder-image.jpeg"

class ProviderLoginRequestSchema(BaseModel):
    id_token: str
    provider: str


class UserResponseSchema(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={
            ObjectId: lambda x: str(x)
        }
    )
    
    id: Optional[ObjectIdField] = Field(alias="_id")
    email: EmailStr
    name: str


class UserProfileSchema(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={
            ObjectId: lambda x: str(x)
        }
    )
    
    id: Optional[ObjectIdField] = Field(alias="_id")
    email: EmailStr
    name: str
    profile_picture: Optional[str] = "https://www.radfordacademy.co.uk/content/uploads/sites/14/2024/07/Staff-placeholder-image.jpeg"


class ForgotPasswordSchema(BaseModel):
    email: EmailStr
    app_type: Optional[str] = "web"  # "web" or "mobile"


class ResetPasswordSchema(BaseModel):
    token: str
    password: str = Field(..., min_length = 6)


class EditProfileSchema(BaseModel):
    name: Optional[str] = Field(None, min_length = 2)
    profile_picture: Optional[str] = None


class ContactUsSchema(BaseModel):
    name: str = Field(..., min_length=2)
    email: EmailStr
    message: str = Field(..., min_length=10)
