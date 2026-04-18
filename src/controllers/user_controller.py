from ..models import database
from ..schemes.user_schema import (
    UserSchema,
    UserResponseSchema,
    LoginSchema,
    ProviderLoginRequestSchema,
    ForgotPasswordSchema,
    ResetPasswordSchema,
    EditProfileSchema,
    UserProfileSchema
)
from ..services import (
    create_access_token,
    verify_access_token,
    verify_strong_password,
    verify_google_token,
    verify_github_code,
    create_reset_token,
    verify_reset_token,
    send_reset_email_async
)
from passlib.hash import bcrypt
import hashlib
from ..models.Enums import Password_Exceeded, Providers
from typing import Union
from google.oauth2 import id_token
from google.auth.transport import requests
from ..config import settings
from fastapi import Request, HTTPException
from bson import ObjectId



async def sign_up_controller(user: UserSchema):

    # check if user already exists
    existing_user = await database["users"].find_one({
        "email" : user.email 
        })
    if existing_user: 
        raise Exception("User already exists")

    # verify password strength
    
    password_status = verify_strong_password(user.password)
    if password_status != Password_Exceeded.VALID:
        raise Exception(password_status.value)

    # create new user and hash password
    user_dict = user.model_dump(exclude={"id"})
    # Hash with SHA-256 first to handle long passwords, then bcrypt
    sha256_hash = hashlib.sha256(user.password.encode('utf-8')).hexdigest()
    hashed_password = bcrypt.hash(sha256_hash)
    user_dict["password"] = hashed_password
    result = await database["users"].insert_one(user_dict)
    user_dict["_id"] = result.inserted_id
    
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

    # verify password (support both old bcrypt-only and new SHA-256 + bcrypt)
    sha256_hash = hashlib.sha256(user.password.encode('utf-8')).hexdigest()
    # Try new method first
    if bcrypt.verify(sha256_hash, existing_user["password"]):
        pass  # New method works
    # Fall back to old method (bcrypt only)
    elif bcrypt.verify(user.password, existing_user["password"]):
        # Update to new hash format for security
        new_hash = bcrypt.hash(sha256_hash)
        await database["users"].update_one(
            {"_id": existing_user["_id"]},
            {"$set": {"password": new_hash}}
        )
    else:
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


async def forget_password_controller(request : ForgotPasswordSchema):

    existing_user = await database["users"].find_one({
        "email" : request.email
    })

    # Only send email if user exists, but always return success message for privacy
    if existing_user:
        reset_token = create_reset_token({
            "user_id" : str(existing_user["_id"]),
            "email" : existing_user["email"]
        })

        await database["users"].update_one(
            {"_id" : existing_user["_id"]},
            {"$set" : {"reset_token" : reset_token}}
        )

        await send_reset_email_async(request.email, reset_token, request.app_type)

    return {
        "message" : "If the email exists in our system, you will receive a reset link shortly"
    }


async def reset_password_controller(request : ResetPasswordSchema):

    payload = verify_reset_token(request.token)

    if not payload:
        raise Exception("Invalid or expired token")

    existing_user = await database["users"].find_one({
        "_id" : ObjectId(payload["user_id"])
    })

    if not existing_user:
        raise Exception("User not found")

    password_status = verify_strong_password(request.password)
    if password_status != Password_Exceeded.VALID:
        raise Exception(password_status.value)

    # Hash with SHA-256 first to handle long passwords, then bcrypt
    sha256_hash = hashlib.sha256(request.password.encode('utf-8')).hexdigest()
    hashed_password = bcrypt.hash(sha256_hash)

    await database["users"].update_one(
        {"_id" : existing_user["_id"]},
        {
            "$set" : {"password" : hashed_password},
            "$unset" : {"reset_token" : ""}
        }
    )

    return {
        "message" : "Password reset successfully"
    }


async def edit_profile_controller(token : str, request : EditProfileSchema):

    payload = verify_access_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    existing_user = await database["users"].find_one({
        "_id" : ObjectId(payload["user_id"])
    })

    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")

    update_data = {}

    if request.name:
        update_data["name"] = request.name

    if request.profile_picture:
        update_data["profile_picture"] = request.profile_picture

    if update_data:
        await database["users"].update_one(
            {"_id" : existing_user["_id"]},
            {"$set" : update_data}
        )

    updated_user = await database["users"].find_one({
        "_id" : existing_user["_id"]
    })

    return UserProfileSchema(**updated_user)

