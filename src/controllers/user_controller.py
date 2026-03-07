from ..models import database
from ..schemes.user_schema import UserSchema, UserResponseSchema
from ..services import create_access_token, verify_access_token
from passlib.hash import bcrypt
from ..services import verify_strong_password
from ..models.Enums import Password_Exceeded



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


async def sign_in_controller(email: str, password: str):

    # check if user exists
    user = await database["users"].find_one({
        "email" : email
    })
    if not user:
        raise Exception("Invalid email or password")

    # verify password
    if not bcrypt.verify(password, user["password"]):
        raise Exception("Invalid email or password")

    # create access token
    access_token = create_access_token({
        "user_id" : str(user["_id"]),
        "email" : user["email"]
        })

    return {
        "access_token" : access_token,
        "user" : UserResponseSchema(**user)
    }


async def google_login_controller(google_id : str, email : str, name : str) : 
    
    # check if user already exists
    existing_user = await database["users"].find_one({
        "email" : email 
        })

    if not existing_user:
        # create new user
        user_dict = {
            "name" : name,
            "email" : email,
            "provider" : "google",
            "google_id" : google_id
        }
        result = await database["users"].insert_one(user_dict)
        user_dict["_id"] = result.inserted_id
        existing_user = user_dict
    else:
        existing_user["_id"] = existing_user["_id"]
    access_token = create_access_token({
        "user_id" : str(existing_user["_id"]),
        "email" : existing_user["email"]
        })

    return {
        "access_token" : access_token,
        "user" : UserResponseSchema(**existing_user)
    }
