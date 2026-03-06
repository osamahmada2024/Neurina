from motor.motor_asyncio import AsyncIOMotorClient
from ..config import settings 

client = AsyncIOMotorClient(settings.MONGO_URI)
database = client[settings.DB_NAME]


