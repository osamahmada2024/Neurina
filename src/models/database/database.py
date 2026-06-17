from motor.motor_asyncio import AsyncIOMotorClient
from ...config import settings 

class DatabaseProxy:
    client: AsyncIOMotorClient = None

    def __getitem__(self, name):
        if self.client is None:
            self.client = AsyncIOMotorClient(settings.MONGO_URI)
        return self.client[settings.DB_NAME][name]

database = DatabaseProxy()

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

    # Agent Sessions collection
    await database["agent_sessions"].create_index("user_id")
    await database["agent_sessions"].create_index("session_id", unique=True)
    await database["agent_sessions"].create_index("status")
