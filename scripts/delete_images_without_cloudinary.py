"""
Delete images without Cloudinary IDs from database

Removes all images that don't have cloudinary_public_id_processed
"""

import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LOCAL_DB_URI = "mongodb://localhost:27017"
DB_NAME = "neurina_xai_db"


class ImageDeleter:
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
    
    async def delete_images_without_cloudinary_ids(self):
        """Delete all images without cloudinary_public_id_processed."""
        if not await self.connect():
            return
        
        col = self.db["images"]
        
        # Count images without Cloudinary IDs
        query = {"cloudinary_public_id_processed": {"$exists": False}}
        count = await col.count_documents(query)
        
        if count == 0:
            logger.info("✓ All images have Cloudinary IDs, nothing to delete")
            self.client.close()
            return
        
        logger.info("\n" + "="*60)
        logger.info("DELETE IMAGES WITHOUT CLOUDINARY IDs")
        logger.info("="*60)
        logger.info(f"Found {count} images without cloudinary_public_id_processed")
        
        # Show sample images
        logger.info("\nSample images to be deleted:")
        cursor = col.find(query).limit(5)
        
        sample_count = 0
        async for doc in cursor:
            logger.info(f"  ID: {doc['_id']}")
            logger.info(f"    Type: {doc.get('image_type')}")
            logger.info(f"    Filename: {doc.get('original_filename')}")
            logger.info(f"    Size: ~{len(doc.get('image_data', '')) / 1024 / 1024:.2f} MB" if doc.get('image_data') else "    Size: unknown")
            sample_count += 1
        
        logger.info("="*60 + "\n")
        
        # Confirm deletion
        response = input(f"Delete {count} images? (yes/no): ").strip().lower()
        if response != "yes":
            logger.info("Deletion cancelled")
            self.client.close()
            return
        
        # Delete images
        result = await col.delete_many(query)
        
        logger.info("\n" + "="*60)
        logger.info("DELETION COMPLETE")
        logger.info("="*60)
        logger.info(f"Deleted: {result.deleted_count} images")
        logger.info("Estimated space saved: ~{:.1f} MB".format(result.deleted_count * 0.5))
        logger.info("="*60 + "\n")
        
        self.client.close()


async def main():
    deleter = ImageDeleter()
    await deleter.delete_images_without_cloudinary_ids()


if __name__ == "__main__":
    asyncio.run(main())
