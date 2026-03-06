from fastapi import APIRouter, HTTPException, Depends
from ..controllers import sign_up_controller, sign_in_controller, google_login_controller
from ..schemes import UserSchema, UserResponseSchema
from ..services import verify_access_token
from fastapi.security import OAuth2PasswordBearer

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
async def sign_in_user(email : str, password : str):

    try :
        return await sign_in_controller(email, password)
    except Exception as e:
        raise HTTPException(status_code = 400, detail = str(e))

@router.post("/google-login")
async def google_login_user(google_id : str, email : str, name : str) :

    try :
        return await google_login_controller(google_id, email, name)
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
