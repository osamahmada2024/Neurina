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

    class Config:
        env_file = "src/.env"

settings = Settings()
