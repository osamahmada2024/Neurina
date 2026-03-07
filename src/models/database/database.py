from motor.motor_asyncio import AsyncIOMotorClient
from ...config import settings 

client = AsyncIOMotorClient(settings.MONGO_URI)
database = client[settings.DB_NAME]

async def init_db():
    # create indexes
    await database["users"].create_index("email", unique=True)
    await database["users"].create_index("google_id", unique=True)
    await database["users"].create_index("name", unique=True)
    