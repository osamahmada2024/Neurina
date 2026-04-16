from fastapi import APIRouter

from .catalog import router as catalog_router
from .downloads import router as downloads_router
from .mutations import router as mutations_router

router = APIRouter(prefix="/images", tags=["images"])
router.include_router(catalog_router)
router.include_router(downloads_router)
router.include_router(mutations_router)
