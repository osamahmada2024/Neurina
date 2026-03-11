from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from ..config import settings
from ..models import Password_Exceeded, Providers
from ..schemes import LoginProviderSchema
from google.auth.transport.requests import Request
from google.oauth2 import id_token
import requests
from fastapi import HTTPException


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
        google_info = id_token.verify_oauth2_token(token, Request(), 
        audience = [
            settings.GOOGLE_CLIENT_ID, # web client id
            settings.GOOGLE_ANDROID_CLIENT_ID, # android client id
            settings.GOOGLE_IOS_CLIENT_ID # ios client id
        ])
        user_data = {
            "email" : google_info["email"],
            "name" : google_info.get("name", ""),
            "provider_id" : google_info["sub"],
            "provider" : Providers.GOOGLE.value,
            "profile_picture" : google_info.get("picture", None)
        }
        return LoginProviderSchema(**user_data)
    except Exception as e:
        raise Exception("Invalid Google token: " + str(e))


async def verify_github_code(code: str) -> LoginProviderSchema:

    # exchange code for access token from github api

    token_url = "https://github.com/login/oauth/access_token"

    headers = {"Accept": "application/json"}
    data = {
        "client_id": settings.GITHUB_CLIENT_ID,
        "client_secret": settings.GITHUB_CLIENT_SECRET,
        "code": code
    }

    token_response =  requests.post(token_url, headers=headers, data=data)
    token_json = token_response.json()
    print("GitHub token response:", token_json)  # Debugging line
    access_token = token_json.get("access_token")

    if not access_token:
        raise HTTPException(status_code=400, detail="Failed to get access token")

    # convert access token to user info from github api

    try:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        response =   requests.get("https://api.github.com/user", headers=headers, timeout=5)


        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Invalid GitHub token")
        
        github_info = response.json()
        email = github_info.get("email")
        if not email:
            email_response =  requests.get("https://api.github.com/user/emails", headers=headers, timeout=5)
            if email_response.status_code == 200:
                emails = email_response.json()
                primary_email = next((e for e in emails if e.get("primary")), None)
                if primary_email:
                    email = primary_email.get("email")

        user_dict = {
            "name": github_info.get("name", github_info.get("login", "")),
            "email": email,
            "provider_id": str(github_info["id"]),
            "provider": Providers.GITHUB.value,
            "profile_picture": github_info.get("avatar_url", None)
        }

        return LoginProviderSchema(**user_dict)

    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid GitHub token: " + str(e))
        