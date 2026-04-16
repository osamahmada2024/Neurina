from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from bson import ObjectId

class UserSchema(BaseModel):
    id : Optional[ObjectId] = Field(default = None, alias = "_id")
    name : str = Field(..., min_length = 2)
    email: EmailStr
    password: str = Field(..., min_length = 6)
    provider : Optional[str] = "local"
    profile_picture : Optional[str] = "https://www.radfordacademy.co.uk/content/uploads/sites/14/2024/07/Staff-placeholder-image.jpeg"

    class Config:
        
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId : lambda x : str(x)
        }


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
    id : Optional[ObjectId] = Field(alias="_id")
    email : EmailStr
    name : str

    class Config:
       
       validate_by_name = True
       arbitrary_types_allowed = True
       json_encoders = {
            ObjectId : lambda x : str(x)
       }


class UserProfileSchema(BaseModel):
    id : Optional[ObjectId] = Field(alias="_id")
    email : EmailStr
    name : str
    profile_picture : Optional[str] = "https://www.radfordacademy.co.uk/content/uploads/sites/14/2024/07/Staff-placeholder-image.jpeg"

    class Config:
       
       validate_by_name = True
       arbitrary_types_allowed = True
       json_encoders = {
            ObjectId : lambda x : str(x)
       }


class ForgotPasswordSchema(BaseModel):
    email: EmailStr
    app_type: Optional[str] = "web"  # "web" or "mobile"


class ResetPasswordSchema(BaseModel):
    token: str
    password: str = Field(..., min_length = 6)


class EditProfileSchema(BaseModel):
    name: Optional[str] = Field(None, min_length = 2)
    profile_picture: Optional[str] = None
