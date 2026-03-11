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
    verify_github_code
)
from passlib.hash import bcrypt
from ..models.Enums import Password_Exceeded, Providers
from typing import Union
from google.oauth2 import id_token
from google.auth.transport import requests
from ..config import settings
from fastapi import Request



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
    user_dict = user.dict(exclude={"id"}) 
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


async def Google_login_controller(user : ProviderLoginRequestSchema) : 

    user = verify_google_token(user.id_token)
    
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
            "provider_id" : user.provider_id,
            "profile_picture" : user.profile_picture
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



async def Github_login_controller(request: Request) :

    code =  request.query_params.get("code")
    if not code:
        raise Exception("Code not provided")

    user = await verify_github_code(code)

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
            "provider_id" : user.provider_id,
            "profile_picture" : user.profile_picture
        }
        result = await database["users"].insert_one(user_dict)
        user_dict["_id"] = result.inserted_id
        existing_user = user_dict


    access_token =  create_access_token({
        "user_id" : str(existing_user["_id"]),  
        "email" : existing_user["email"]
        })

    return {
        "access_token" : access_token,      
        "user" : UserResponseSchema(**existing_user)
    }
    
    
