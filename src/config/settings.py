from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str 
    APP_VERSION: str
    MONGO_URI: str
    DB_NAME: str
    HOST: str
    PORT: int
    DEBUG: bool
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    GITHUB_CLIENT_ID: str
    GITHUB_CLIENT_SECRET: str
    GOOGLE_ANDROID_CLIENT_ID: str
    GOOGLE_IOS_CLIENT_ID: str
    SMTP_SERVER: str
    SMTP_PORT: int
    SMTP_EMAIL: str
    SMTP_PASSWORD: str
    RESET_LINK_WEB: str
    RESET_LINK_MOBILE: str
    PUBLIC_REFERENCE_DIR: str = "Ref_Database"
    PUBLIC_REFERENCE_COLLECTION: str = "ref_database"
    PUBLIC_REFERENCE_SYNC_ON_STARTUP: bool = True
    PUBLIC_REFERENCE_SYNC_FAIL_ON_ERROR: bool = True
    FACE_CROP_SERVICE_URL: str = "http://face-crop-service:8010/crop"
    FACE_CROP_TIMEOUT_SECONDS: int = 15
    W_HPF: float = 1.0
    NUM_DOMAINS: int = 2
    REFERENCE_DOMAIN_LABEL: int = 0
    CHECKPOINT_VARIANT: str = "ema"
    FACE_MARGIN_LEFT: float = 0.44
    FACE_MARGIN_RIGHT: float = 0.39
    FACE_MARGIN_TOP: float = 0.44
    FACE_MARGIN_BOTTOM: float = 0.14
    SR_ENABLED: bool = True
    SR_MODEL_NAME: str = "codeformer"
    SR_OUTSCALE: float = 2.0
    SR_TILE: int = 0
    SR_FACE_WEIGHT: float = 0.5
    SR_CODEFORMER_FIDELITY: float = 0.7
    UPLOAD_SR_ENABLED: bool = True
    UPLOAD_SR_OUTSCALE: float = 4.0
    TRANSLATION_EYE_RESCUE_ENABLED: bool = True
    TRANSLATION_EYE_RESCUE_ALPHA: float = 0.88
    TRANSLATION_QUALITY_GATE_ENABLED: bool = True
    TRANSLATION_MIN_LAPLACIAN_VAR: float = 18.0
    TRANSLATION_SOFT_MIN_LAPLACIAN_VAR: float = 35.0
    TRANSLATION_MIN_GRADIENT_P90: float = 110.0
    TRANSLATION_MIN_CONTRAST_STD: float = 55.0

    class Config:
        env_file = "src/.env"
        extra = "ignore"

settings = Settings()
