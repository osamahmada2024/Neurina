"""
Clean database: Remove all image data fields

Removes: image_data, image_data_original, model_image_data
"""

import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LOCAL_DB_URI = "mongodb://localhost:27017"
DB_NAME = "neurina_xai_db"


class CompleteImageDataCleaner:
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
    
    async def cleanup_all_image_data(self):
        """Remove ALL image data fields: image_data, image_data_original, model_image_data."""
        if not await self.connect():
            return
        
        col = self.db["images"]
        
        # Find documents with any image data fields
        query = {
            "$or": [
                {"image_data": {"$exists": True}},
                {"image_data_original": {"$exists": True}},
                {"model_image_data": {"$exists": True}}
            ]
        }
        
        count = await col.count_documents(query)
        
        if count == 0:
            logger.info("✓ No image data fields found")
            self.client.close()
            return
        
        logger.info("\n" + "="*60)
        logger.info("CLEANUP ALL IMAGE DATA FIELDS")
        logger.info("="*60)
        logger.info(f"Found {count} documents with image data fields")
        logger.info("\nWill remove:")
        logger.info("  • image_data")
        logger.info("  • image_data_original")
        logger.info("  • model_image_data")
        logger.info("="*60 + "\n")
        
        # Sample check
        sample = await col.find_one(query)
        if sample:
            logger.info("Sample document fields:")
            if sample.get("image_data"):
                logger.info(f"  • image_data: {len(sample.get('image_data', '')) / 1024 / 1024:.2f} MB")
            if sample.get("image_data_original"):
                logger.info(f"  • image_data_original: {len(sample.get('image_data_original', '')) / 1024 / 1024:.2f} MB")
            if sample.get("model_image_data"):
                logger.info(f"  • model_image_data: {len(sample.get('model_image_data', '')) / 1024 / 1024:.2f} MB")
            logger.info("")
        
        # Confirm
        response = input(f"Cleanup {count} documents? (yes/no): ").strip().lower()
        if response != "yes":
            logger.info("Cleanup cancelled")
            self.client.close()
            return
        
        # Remove all image data fields
        result = await col.update_many(
            query,
            {
                "$unset": {
                    "image_data": "",
                    "image_data_original": "",
                    "model_image_data": ""
                }
            }
        )
        
        logger.info("\n" + "="*60)
        logger.info("CLEANUP COMPLETE")
        logger.info("="*60)
        logger.info(f"Cleaned: {result.modified_count} documents")
        logger.info("Estimated space saved: ~100-200 MB")
        logger.info("="*60 + "\n")
        
        self.client.close()


async def main():
    cleaner = CompleteImageDataCleaner()
    await cleaner.cleanup_all_image_data()


if __name__ == "__main__":
    asyncio.run(main())
