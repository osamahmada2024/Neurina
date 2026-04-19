from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from ..controllers import (
    sign_up_controller, 
    sign_in_controller, 
    Github_login_controller, 
    Google_login_controller,
    forget_password_controller,
    reset_password_controller,
    edit_profile_controller
)
from ..schemes import (
    UserSchema,
    LoginSchema,
    ProviderLoginRequestSchema,
    UserProfileSchema,
    ForgotPasswordSchema,
    ResetPasswordSchema,
    EditProfileSchema,
)
from ..services import verify_access_token
from fastapi.security import OAuth2PasswordBearer
from ..models import database
from bson import ObjectId

router = APIRouter(
    prefix = "/users",
    tags = ["users"]
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/users/login")

@router.post("/signup")
async def sign_up_user(user : UserSchema): 

    try : 
        return await sign_up_controller(user)
    except Exception as e:
        raise HTTPException(status_code = 400, detail = str(e))

@router.post("/login")
async def sign_in_user(login_data : LoginSchema):

    try :
        return await sign_in_controller(login_data)
    except Exception as e:
        raise HTTPException(status_code = 400, detail = str(e))


@router.post("/google-login")
async def Google_login_user(login_data : ProviderLoginRequestSchema):

    try :
        return await Google_login_controller(login_data)
    except Exception as e:
        raise HTTPException(status_code = 400, detail = str(e))


@router.get("/github-login")
async def Github_login_user(request: Request):

    try :
        return await Github_login_controller(request)
    except Exception as e:
        raise HTTPException(status_code = 400, detail = str(e))


@router.get("/profile")
async def get_profile_user(token: str = Depends(oauth2_scheme)):

    try:
        payload = verify_access_token(token)

        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")


        user = await database["users"].find_one({"_id": ObjectId(payload["user_id"])})

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return UserProfileSchema(**user)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load profile: {str(e)}")


@router.post("/forgot-password")
async def forgot_password_user(request: ForgotPasswordSchema, background_tasks: BackgroundTasks):

    try:
        return await forget_password_controller(request, background_tasks)
    except Exception as e:
        raise HTTPException(status_code = 400, detail = str(e))


@router.post("/reset-password")
async def reset_password_user(request: ResetPasswordSchema):

    try:
        return await reset_password_controller(request)
    except Exception as e:
        raise HTTPException(status_code = 400, detail = str(e))


@router.put("/edit-profile")
async def edit_profile_user(request: EditProfileSchema, token: str = Depends(oauth2_scheme)):

    try:
        return await edit_profile_controller(token, request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Failed to update profile: {str(e)}")
