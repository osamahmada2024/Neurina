from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from bson import ObjectId

class UserSchema(BaseModel):
    id : Optional[ObjectId] = Field(default = None,alias = "_id")
    name : str = Field(..., min_length = 2)
    email: EmailStr
    password: Optional[str] = Field(None, min_length = 6)
    provider : Optional[str] = "local"
    google_id : Optional[str] = None

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
    profile_picture : Optional[str] = None

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
       