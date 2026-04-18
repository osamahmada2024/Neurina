from fastapi import FastAPI
from pathlib import Path
from .config import settings
from .config.logging import configure_logging
from .models import init_db
from .routes import router
from .controllers.image import image_controller
from .services import ModelLoader
from .services.model_preloader import ModelPreloader
from .services.cloudinary_service import CloudinaryService
from .utils.console_feedback import console_feedback
import os
import logging

configure_logging()
logger = logging.getLogger(__name__)

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
        logger.debug(f"Public reference sync skipped: directory not found at {reference_root}")
        return

    console_feedback("Syncing public reference library...")
    summary = await image_controller.import_public_references_from_directory(
        root_dir=reference_root,
        public_collection=settings.PUBLIC_REFERENCE_COLLECTION,
    )

    if summary["inserted"] == 0 and summary["updated"] == 0 and summary["failed"] == 0 and summary["already_synced"] > 0:
        console_feedback(
            "Public reference library already synced "
            f"(total={summary['processed']}, skipped={summary['skipped']}, invalid={summary['invalid']})"
        )
    else:
        console_feedback(
            "Public reference library synced "
            f"(total={summary['processed']}, uploaded={summary['inserted']}, "
            f"skipped={summary['skipped']}, invalid={summary['invalid']}, failed={summary['failed']})"
        )

    if bool(settings.PUBLIC_REFERENCE_SYNC_FAIL_ON_ERROR) and summary["failed"] > 0:
        sample_error = summary["errors"][0]
        raise RuntimeError(
            "Public reference startup sync failed. "
            f"first_error={sample_error['path']}: {sample_error['error']}"
        )

@app.on_event("startup")
async def startup_event():
    """Initialize database, load models, and setup services on startup"""
    logger = logging.getLogger(__name__)
    
    console_feedback("Starting application...")
    await init_db()
    console_feedback("Database ready")
    
    base_path = Path(os.path.dirname(os.path.dirname(__file__)))
    
    # Eager model loading
    console_feedback("Loading models...")
    preloader = ModelPreloader(str(base_path))
    loading_results = preloader.preload_all_models()
    loading_errors = preloader.get_loading_errors()

    def _model_status(name: str, loaded: bool, detail: str | None = None) -> None:
        suffix = f" ({detail})" if detail else ""
        console_feedback(f"{name}: {'loaded' if loaded else 'not loaded'}{suffix}")
    
    # Check if all critical models loaded successfully
    if not loading_results.get('stargan_models', False):
        error_detail = loading_errors.get("stargan_models", "No additional detail captured")
        logger.error("Critical: StarGAN v2 models failed to load: %s", error_detail)
        console_feedback("StarGAN model loading failed")
        raise RuntimeError(f"Failed to load StarGAN v2 models - cannot start server. {error_detail}")
    
    if not loading_results.get('face_restoration_models', False):
        logger.debug("Face restoration models failed to load - image enhancement will be disabled")
        console_feedback("Face restoration disabled")
    else:
        if face_restoration_service := preloader.get_model('face_restoration'):
            console_feedback("Face restoration ready")
        else:
            console_feedback("Face restoration lazy")
    
    # Initialize image controller with loaded models
    stargan_models = preloader.get_model('stargan')
    face_restoration_service = preloader.get_model('face_restoration')
    wing_path = preloader.get_model('wing_path')
    celeba_lm_path = preloader.get_model('celeba_lm_path')
    if not celeba_lm_path:
        celeba_lm_path = Path(settings.CELEBA_LM_MEAN_PATH).expanduser()
        if not celeba_lm_path.is_absolute():
            celeba_lm_path = base_path / celeba_lm_path
        celeba_lm_path = str(celeba_lm_path)
    
    image_controller.initialize_models(
        generator=stargan_models["generator"] if stargan_models else None,
        style_encoder=stargan_models["style_encoder"] if stargan_models else None,
        fan_model=stargan_models.get("fan_model") if stargan_models else None,
        wing_model_path=wing_path,
        celeba_lm_path=celeba_lm_path if Path(celeba_lm_path).is_file() else None,
    )
    
    # Initialize postprocessors with preloaded face restoration service
    image_controller.initialize_postprocessors(str(base_path), face_restoration_service)

    _model_status("StarGAN", stargan_models is not None)
    _model_status("wing.ckpt", bool(wing_path), wing_path if wing_path else None)
    _model_status(
        "celeba_lm_mean.npz",
        bool(celeba_lm_path) and Path(celeba_lm_path).is_file(),
        celeba_lm_path if celeba_lm_path else None,
    )
    if face_restoration_service is not None:
        _model_status("Face restoration", True, "preloaded")
    else:
        preload_enabled = getattr(preloader, "face_restoration_preload_attempted", False)
        if preload_enabled:
            _model_status(
                "Face restoration",
                False,
                loading_errors.get("face_restoration_models", "preload failed"),
            )
        else:
            _model_status("Face restoration", False, "lazy")
    
    # Initialize Cloudinary service
    cloudinary_enabled = False
    try:
        console_feedback("Initializing Cloudinary...")
        cloudinary_service = CloudinaryService.create_from_env()
        image_controller.initialize_cloudinary_service(cloudinary_service)
        cloudinary_enabled = True
        console_feedback("Cloudinary ready")
    except ValueError as e:
        logger.debug("Cloudinary service initialization failed: %s", e)
        console_feedback("Cloudinary disabled")
    
    # Store preloader for potential use
    app.state.model_preloader = preloader
    
    await _sync_public_references_on_startup(base_path)
    console_feedback("Application ready")
