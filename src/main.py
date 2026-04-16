from fastapi import FastAPI
from pathlib import Path
from .config import settings
from .models import init_db
from .routes import router
from .controllers import image_controller
from .services import ModelLoader
import os

app = FastAPI(
    title = settings.APP_NAME,
    version = settings.APP_VERSION,
    debug = settings.DEBUG
)

@app.get('/')
async def root():

    return {
        "App Name": settings.APP_NAME,
        "App Version": settings.APP_VERSION,
        "message" : "Welcome to the Neurina Image Translation API!"
    }

app.include_router(router, prefix="/api")


async def _sync_public_references_on_startup(base_path: Path) -> None:
    if not bool(settings.PUBLIC_REFERENCE_SYNC_ON_STARTUP):
        return

    reference_root = base_path / settings.PUBLIC_REFERENCE_DIR
    if not reference_root.is_dir():
        print(f"Public reference sync skipped: directory not found at {reference_root}")
        return

    summary = await image_controller.import_public_references_from_directory(
        root_dir=reference_root,
        public_collection=settings.PUBLIC_REFERENCE_COLLECTION,
    )
    print(
        "Public reference sync summary: "
        f"processed={summary['processed']} "
        f"inserted={summary['inserted']} "
        f"updated={summary['updated']} "
        f"failed={summary['failed']}"
    )

    if bool(settings.PUBLIC_REFERENCE_SYNC_FAIL_ON_ERROR) and summary["failed"] > 0:
        sample_error = summary["errors"][0]
        raise RuntimeError(
            "Public reference startup sync failed. "
            f"first_error={sample_error['path']}: {sample_error['error']}"
        )

@app.on_event("startup")
async def startup_event():
    """Initialize database and load models on startup"""
    await init_db()
    
    base_path = Path(os.path.dirname(os.path.dirname(__file__)))
    models = ModelLoader.load_models(base_path)

    wing_path = base_path / "checkpoints" / "wing.ckpt"
    celeba_lm_path = base_path / "checkpoints" / "celeba_lm_mean.npz"
    image_controller.initialize_models(
        generator=models["generator"] if models else None,
        style_encoder=models["style_encoder"] if models else None,
        fan_model=models.get("fan_model") if models else None,
        wing_model_path=str(wing_path),
        celeba_lm_path=str(celeba_lm_path) if celeba_lm_path.is_file() else None,
    )
    image_controller.initialize_postprocessors(str(base_path))
    await _sync_public_references_on_startup(base_path)
