from motor.motor_asyncio import AsyncIOMotorClient
from ...config import settings 

client = AsyncIOMotorClient(settings.MONGO_URI)
database = client[settings.DB_NAME]

async def init_db():
    """Initialize database collections and indexes"""
    # Users collection
    await database["users"].create_index("email", unique=True)
    await database["users"].create_index("provider_id", unique=True, sparse=True)
    
    # Images collection
    await database["images"].create_index("user_id")
    await database["images"].create_index("image_type")
    await database["images"].create_index("status")
    await database["images"].create_index("is_public", sparse=True)
    await database["images"].create_index([("is_public", 1), ("image_type", 1), ("image_domain", 1)])
    await database["images"].create_index(
        "library_key",
        unique=True,
        partialFilterExpression={"is_public": True},
    )
    await database["images"].create_index([("user_id", 1), ("image_type", 1)])
    
    # Translation tasks collection
    await database["translation_tasks"].create_index("user_id")
    await database["translation_tasks"].create_index("status")
    await database["translation_tasks"].create_index("source_image_id")
    await database["translation_tasks"].create_index("reference_image_id")
    await database["translation_tasks"].create_index([("user_id", 1), ("status", 1)])

