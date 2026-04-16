from .user_route import router as user_router
from .image_route import router as image_router
from fastapi import APIRouter

router = APIRouter()
router.include_router(user_router)
router.include_router(image_router) 
