from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from ..config import settings
from ..models import Password_Exceeded

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
        return Password_Exceeded.BELOW_MIN_LENGTH
    if not any(char.isupper() for char in password):
        return Password_Exceeded.MISSING_UPPERCASE
    if not any(char.islower() for char in password):
        return Password_Exceeded.MISSING_LOWERCASE
    if not any(char.isdigit() for char in password):
        return Password_Exceeded.MISSING_DIGIT
    if not any(char in "!@#$%^&*()-_=+[]{}|;:'\",.<>?/" for char in password):
        return Password_Exceeded.MISSING_SPECIAL_CHARACTER
    return Password_Exceeded.VALID
    