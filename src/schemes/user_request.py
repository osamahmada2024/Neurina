from pydantic import BaseModel

class UserStyleRequest(BaseModel):
    user_text: str
    source_image_id: str
    auth_token: str
