from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from ..config import settings
from ..models import Password_Exceeded
from ..schemes import LoginProviderSchema
import requests


def create_access_token(data: dict) -> str:

    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp" : expire})
    token = jwt.encode(to_encode, settings.SECRET_KEY, algorithm = settings.ALGORITHM)

    return token

def verify_access_token(token: str):

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms = [settings.ALGORITHM])
        return payload

    except JWTError:
        return None


def verify_strong_password(password: str) -> str:

    if len(password) < 8:
        return Password_Exceeded.EXCEED_MIN_LENGTH
    if len(password) > 128:
        return Password_Exceeded.EXCEED_MAX_LENGTH
    if not any(char.isupper() for char in password):
        return Password_Exceeded.EXCEED_REQUIRE_UPPERCASE
    if not any(char.islower() for char in password):
        return Password_Exceeded.EXCEED_REQUIRE_LOWERCASE
    if not any(char.isdigit() for char in password):
        return Password_Exceeded.EXCEED_REQUIRE_DIGIT
    if not any(char in "!@#$%^&*()-_=+[]{}|;:'\",.<>?/" for char in password):
        return Password_Exceeded.EXCEED_REQUIRE_SPECIAL_CHAR
    return Password_Exceeded.VALID


def verify_google_token(token: str) -> dict:
    try:
        google_info = id_token.verify_oauth2_token(token, requests.Request(), 
        audience = [
            settings.GOOGLE_CLIENT_ID, # web client id
            settings.GOOGLE_ANDROID_CLIENT_ID, # android client id
            settings.GOOGLE_IOS_CLIENT_ID # ios client id
        ])
        user_data = {
            "email" : google_info["email"],
            "name" : google_info.get("name", ""),
            "provider_id" : google_info["sub"],
            "provider" : "google"
        }
        return LoginProviderSchema(**user_data)
    except Exception as e:
        raise Exception("Invalid Google token")


def verify_github_token(token: str) -> dict:
    try:
        headers = {
            "Authorization": f"token {token}"
        }
        response = requests.get("https://api.github.com/user", headers = headers)
        if response.status_code != 200:
            raise Exception("Invalid GitHub token")
        github_info = response.json()
        user_dict = {
            "name": github_info.get("name", github_info.get("login", "")),
            "email": github_info.get("email", ""), 
            "provider_id": github_info["id"],
            "provider": "github"
        }
        return LoginProviderSchema(**user_dict)
    
    except Exception as e:
        raise Exception("Invalid GitHub token")
