from fastapi import APIRouter, HTTPException, Depends
from ..controllers import sign_up_controller, sign_in_controller, Provider_login_controller
from ..schemes import UserSchema, UserResponseSchema, LoginSchema, ProviderLoginRequestSchema
from ..services import verify_access_token
from fastapi.security import OAuth2PasswordBearer
from typing import Union
from ..models import Providers

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
        return await sign_in_controller(login_data.email, login_data.password)
    except Exception as e:
        raise HTTPException(status_code = 400, detail = str(e))

@router.post("/provider-login")
async def provider_login_user(login_data : ProviderLoginRequestSchema):

    try :
        if login_data.provider == Providers.GOOGLE.value:
            return await Provider_login_controller(login_data)
        elif login_data.provider == Providers.GITHUB.value:
            return await Provider_login_controller(login_data)

        raise HTTPException(status_code = 400, detail = "Invalid provider")

    except Exception as e:
        raise HTTPException(status_code = 400, detail = str(e))

    
@router.get("/profile")
async def get_profile_user(token : str = Depends(oauth2_scheme)):

    try : 
        payload = verify_access_token(token)
        if not payload:
            raise HTTPException(status_code = 401, detail = "Invalid token")
        return {"message": f"Hello {payload['email']}!", "user_id": payload["user_id"]}
    except Exception as e:
        raise HTTPException(status_code = 401, detail = str(e))
