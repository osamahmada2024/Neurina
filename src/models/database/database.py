import logging

from motor.motor_asyncio import AsyncIOMotorClient
from ...config import settings 

logger = logging.getLogger(__name__)

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
    await _ensure_agent_session_indexes()


async def _ensure_agent_session_indexes() -> None:
    collection = database["agent_sessions"]
    await collection.create_index("user_id")
    await collection.create_index("status")

    # New isolation boundary: each user's session namespace is independent.
    await collection.create_index(
        [("user_id", 1), ("session_id", 1)],
        unique=True,
        name="agent_sessions_user_session_unique",
    )

    index_info = await collection.index_information()
    for index_name, metadata in index_info.items():
        if index_name == "_id_":
            continue
        keys = list(metadata.get("key", []))
        if keys == [("session_id", 1)] and metadata.get("unique"):
            try:
                await collection.drop_index(index_name)
                logger.info("Dropped legacy unique agent session index: %s", index_name)
            except Exception as exc:
                logger.warning(
                    "Could not drop legacy unique agent session index %s: %s",
                    index_name,
                    exc,
                )

    await collection.create_index(
        "session_id",
        name="agent_sessions_session_lookup",
    )
