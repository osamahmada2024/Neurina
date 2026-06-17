from fastapi import APIRouter

from .image import router as image_router
from .user_route import router as user_router
from .agent_routes import router as agent_router

router = APIRouter()
router.include_router(user_router)
router.include_router(image_router)
router.include_router(agent_router)
