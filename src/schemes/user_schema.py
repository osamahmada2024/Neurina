from pydantic import BaseModel, EmailStr, Field, 
from typing import Optional
from bson import ObjectId

class UserSchema(BaseModel):
    _id : Optional[ObjectId] = Field(default = None,alias = "_id")
    name : str = Field(..., min_length = 2)
    email: EmailStr
    password: Optional[str] = Field(None, min_length = 6)
    provider : Optional[str] = "local"
    google_id : Optional[str] = None

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId : lambda x : str(x)
        }




class UserResponseSchema(BaseModel):
    _id : Optional[str] = Field(alias="_id")
    email : EmailStr
    name : str

    class Config:
       
       allow_population_by_field_name = True
       arbitrary_types_allowed = True
       json_encoders = {
            ObjectId : lambda x : str(x)
       }