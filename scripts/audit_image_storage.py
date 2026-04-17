"""
Database Audit: Check image storage status

Analyze which images have Cloudinary IDs vs base64 storage
"""

import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LOCAL_DB_URI = "mongodb://localhost:27017"
DB_NAME = "neurina_xai_db"


class DatabaseAudit:
    def __init__(self):
        self.client = None
        self.db = None
    
    async def connect(self) -> bool:
        try:
            logger.info("Connecting to local MongoDB...")
            self.client = AsyncIOMotorClient(LOCAL_DB_URI, serverSelectionTimeoutMS=5000)
            await self.client.admin.command('ping')
            self.db = self.client[DB_NAME]
            logger.info("✓ Connected")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    async def audit_images(self):
        """Analyze image storage status."""
        if not await self.connect():
            return
        
        col = self.db["images"]
        
        # Count by storage type
        total = await col.count_documents({})
        cloudinary_ids = await col.count_documents({"cloudinary_public_id_processed": {"$exists": True}})
        base64_only = await col.count_documents({
            "cloudinary_public_id_processed": {"$exists": False},
            "image_data": {"$exists": True}
        })
        public_images = await col.count_documents({"is_public": True})
        public_with_cloudinary = await col.count_documents({
            "is_public": True,
            "cloudinary_public_id_processed": {"$exists": True}
        })
        
        logger.info("\n" + "="*60)
        logger.info("DATABASE AUDIT - IMAGE STORAGE STATUS")
        logger.info("="*60)
        logger.info(f"Total images: {total}")
        logger.info(f"  - With Cloudinary IDs: {cloudinary_ids}")
        logger.info(f"  - Base64 only (no Cloudinary): {base64_only}")
        logger.info(f"\nPublic images: {public_images}")
        logger.info(f"  - With Cloudinary IDs: {public_with_cloudinary}")
        logger.info(f"  - Base64 only: {public_images - public_with_cloudinary}")
        logger.info("="*60 + "\n")
        
        # Sample some base64-only images
        logger.info("Sample base64-only images:")
        cursor = col.find({
            "cloudinary_public_id_processed": {"$exists": False},
            "image_data": {"$exists": True}
        }).limit(3)
        
        async for doc in cursor:
            logger.info(f"  ID: {doc['_id']}")
            logger.info(f"    Type: {doc.get('image_type')}")
            logger.info(f"    Domain: {doc.get('image_domain')}")
            logger.info(f"    Public: {doc.get('is_public')}")
            logger.info(f"    Library Key: {doc.get('library_key')}")
            logger.info(f"    Storage Type: {doc.get('storage_type')}")
        
        logger.info("\n✓ Audit complete")
        self.client.close()


async def main():
    audit = DatabaseAudit()
    await audit.audit_images()


if __name__ == "__main__":
    asyncio.run(main())
