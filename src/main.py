from fastapi import FastAPI
from .config import settings
from .database import database
from .routes import router

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
        "message" : "Welcome to the FastAPI application!"
    }

app.include_router(router, prefix="/api")
