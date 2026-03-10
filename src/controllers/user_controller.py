from ..models import database
from ..schemes.user_schema import (
    UserSchema,
    UserResponseSchema,
    LoginSchema,
    ProviderLoginRequestSchema
)
from ..services import (
    create_access_token,
    verify_access_token,
    verify_strong_password,
    verify_google_token,
    verify_github_token
)
from passlib.hash import bcrypt
from ..models.Enums import Password_Exceeded, Providers
from typing import Union
from google.oauth2 import id_token
from google.auth.transport import requests
from ..config import settings



async def sign_up_controller(user: UserSchema):

    # check if user already exists
    existing_user = await database["users"].find_one({
        "email" : user.email 
        })
    if existing_user: 
        raise Exception("User already exists")

    # verify password strength
    
    if verify_strong_password(user.password) != Password_Exceeded.VALID:
        raise Exception(verify_strong_password(user.password).value)

    # create new user and hash password
    user_dict = user.dict(exclude_unset = True) 
    hashed_password = bcrypt.hash(user.password)
    user_dict["password"] = hashed_password
    result = await database["users"].insert_one(user_dict)
    
    # create access token
    access_token = create_access_token({
        "user_id" : str(result.inserted_id),
        "email" : user.email
        })

    return {
        "access_token" : access_token,
        "user" : UserResponseSchema(**user_dict)
    }


async def sign_in_controller(user : LoginSchema):

    # check if user exists
    existing_user = await database["users"].find_one({
        "email" : user.email
    })
    if not existing_user:
        raise Exception("Invalid email or password")

    # verify password
    if not bcrypt.verify(user.password, existing_user["password"]):
        raise Exception("Invalid email or password")

    # create access token
    access_token = create_access_token({
        "user_id" : str(existing_user["_id"]),
        "email" : existing_user["email"]
        })

    return {
        "access_token" : access_token,
        "user" : UserResponseSchema(**existing_user)
    }


async def Provider_login_controller(user : ProviderLoginRequestSchema) : 

    if user.provider == Providers.GOOGLE.value:
        user = verify_google_token(user.token)
    elif user.provider == Providers.GITHUB.value:
        user = verify_github_token(user.token)
    else :
        raise Exception("Unsupported provider")
    
    # check if user already exists
    existing_user = await database["users"].find_one({
        "email" : user.email 
        })

    if not existing_user:
        # create new user
       
        user_dict = {
            "name" : user.name,
            "email" : user.email,
            "provider" : user.provider,
            "provider_id" : user.provider_id
        }
        
        result = await database["users"].insert_one(user_dict)
        user_dict["_id"] = result.inserted_id
        existing_user = user_dict

    access_token = create_access_token({
        "user_id" : str(existing_user["_id"]),
        "email" : existing_user["email"]
        })

    return {
        "access_token" : access_token,
        "user" : UserResponseSchema(**existing_user)
    }
