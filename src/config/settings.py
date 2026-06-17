from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str 
    APP_VERSION: str
    APP_LOG_LEVEL: str = "WARNING"
    APP_FEEDBACK_LOG_LEVEL: str = "WARNING"
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
    SMTP_SERVER: str = ""
    SMTP_PORT: int = 587
    SMTP_EMAIL: str = ""
    SMTP_PASSWORD: str = ""
    SENDGRID_API_KEY: str = ""
    RESET_LINK_WEB: str
    RESET_LINK_MOBILE: str
    MAX_UPLOAD_SIZE: int = 104857600
    ALLOWED_IMAGE_FORMATS: str = "jpg,jpeg,png,gif,bmp"
    IMG_SIZE: int = 256
    WING_MODEL_PATH: str = "./checkpoints/wing.ckpt"
    CELEBA_LM_MEAN_PATH: str = "./checkpoints/celeba_lm_mean.npz"
    USE_GPU: bool = True
    GPU_DEVICE: int = 0
    REDIS_URL: str = "redis://localhost:6379"
    PUBLIC_REFERENCE_DIR: str = "Ref_Database"
    PUBLIC_REFERENCE_COLLECTION: str = "ref_database"
    PUBLIC_REFERENCE_SYNC_ON_STARTUP: bool = True
    PUBLIC_REFERENCE_SYNC_FAIL_ON_ERROR: bool = True
    PUBLIC_REFERENCE_SYNC_WORKERS: int = 12
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

    
    # Agent config
    OLLAMA_BASE_URL: str
    Search_Secret_API_KEY: str
    SUPERVISOR_MODEL: str
    QUERY_MODEL: str 
    REASONING_MODEL: str
    MAX_REFERENCE_CANDIDATES: int 
    REQUEST_TIMEOUT: int
    VERIFY_TIMEOUT: int
    Backend_API_URL: str
    MAX_RETRIES: int = 3

    # Hugging Face Hub (RAG embeddings download / rate limits)
    HF_TOKEN: str = ""
    RAG_EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    model_config = SettingsConfigDict(env_file="src/.env", extra="ignore")



settings = Settings()
