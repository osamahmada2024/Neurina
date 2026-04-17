"""
Batch Upload base64 images to Cloudinary

Uploads all base64 images to Cloudinary and stores the public_id
"""

import asyncio
import logging
import base64
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LOCAL_DB_URI = "mongodb://localhost:27017"
DB_NAME = "neurina_xai_db"


class Base64ToCloudinaryUploader:
    def __init__(self):
        self.client = None
        self.db = None
        self.cloudinary_service = None
        self.stats = {
            "total": 0,
            "uploaded": 0,
            "skipped": 0,
            "errors": 0,
        }
    
    async def connect(self) -> bool:
        try:
            logger.info("Connecting to local MongoDB...")
            self.client = AsyncIOMotorClient(LOCAL_DB_URI, serverSelectionTimeoutMS=5000)
            await self.client.admin.command('ping')
            self.db = self.client[DB_NAME]
            
            # Initialize Cloudinary service
            try:
                from src.config.cloudinary import cloudinary_settings
                from src.services.cloudinary_service import CloudinaryService
                
                if cloudinary_settings.is_configured():
                    self.cloudinary_service = CloudinaryService()
                    logger.info("✓ Cloudinary service initialized")
                else:
                    logger.error("✗ Cloudinary not configured in .env")
                    return False
            except ImportError as e:
                logger.error(f"Failed to import Cloudinary modules: {e}")
                logger.info("Make sure you run this script from the project root directory")
                return False
            
            logger.info("✓ Connected to MongoDB")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    async def upload_base64_image(self, image_data: str, public_id: str) -> Optional[str]:
        """Upload base64 image to Cloudinary."""
        try:
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if image_array is None:
                logger.warning(f"Could not decode base64 for {public_id}")
                return None
            
            # Upload to Cloudinary
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.cloudinary_service.upload_bgr_image,
                image_array,
                public_id,
            )
            
            return result.get("public_id") if result else None
            
        except Exception as e:
            logger.error(f"Upload failed for {public_id}: {e}")
            return None
    
    async def migrate_base64_images(self):
        """Migrate all base64 images to Cloudinary."""
        if not await self.connect():
            return
        
        col = self.db["images"]
        
        # Find all images with base64 but no Cloudinary ID
        query = {
            "cloudinary_public_id_processed": {"$exists": False},
            "image_data": {"$exists": True, "$ne": ""}
        }
        
        total = await col.count_documents(query)
        if total == 0:
            logger.info("✓ All images already have Cloudinary IDs")
            self.client.close()
            return
        
        logger.info(f"\nFound {total} images to migrate")
        logger.info("="*60 + "\n")
        
        cursor = col.find(query)
        batch_size = 50
        batch = []
        batch_num = 0
        
        async for doc in cursor:
            batch.append(doc)
            
            if len(batch) >= batch_size:
                batch_num += 1
                logger.info(f"Processing batch {batch_num} ({len(batch)} images)...")
                await self._process_batch(batch, col)
                batch = []
        
        # Process remaining
        if batch:
            batch_num += 1
            logger.info(f"Processing final batch {batch_num} ({len(batch)} images)...")
            await self._process_batch(batch, col)
        
        logger.info("\n" + "="*60)
        logger.info("MIGRATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total: {self.stats['total']}")
        logger.info(f"Uploaded: {self.stats['uploaded']}")
        logger.info(f"Skipped: {self.stats['skipped']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info("="*60 + "\n")
        
        self.client.close()
    
    async def _process_batch(self, batch: list, col):
        """Process a batch of images."""
        for doc in batch:
            self.stats['total'] += 1
            doc_id = doc["_id"]
            image_type = doc.get("image_type", "image")
            library_key = doc.get("library_key", str(doc_id))
            
            try:
                # Build public_id
                public_id = f"neurina/processed_faces/public_references/{library_key.replace('/', '_')}-processed"
                
                # Upload processed image
                processed_id = await self.upload_base64_image(
                    doc.get("image_data", ""),
                    public_id
                )
                
                if not processed_id:
                    self.stats['errors'] += 1
                    logger.warning(f"  ✗ {doc_id}: Failed to upload processed image")
                    continue
                
                # Upload original if available
                original_id = None
                if doc.get("image_data_original"):
                    original_public_id = f"neurina/processed_faces/public_references/{library_key.replace('/', '_')}-original"
                    original_id = await self.upload_base64_image(
                        doc.get("image_data_original", ""),
                        original_public_id
                    )
                
                # Update document
                update_doc = {
                    "cloudinary_public_id_processed": processed_id,
                    "storage_type": "cloudinary",
                }
                
                if original_id:
                    update_doc["cloudinary_public_id_original"] = original_id
                
                await col.update_one(
                    {"_id": doc_id},
                    {"$set": update_doc}
                )
                
                self.stats['uploaded'] += 1
                logger.info(f"  ✓ {doc_id}: Uploaded")
                
            except Exception as e:
                self.stats['errors'] += 1
                logger.error(f"  ✗ {doc_id}: {e}")


async def main():
    uploader = Base64ToCloudinaryUploader()
    
    print("\n" + "="*60)
    print("MIGRATE BASE64 IMAGES TO CLOUDINARY")
    print("="*60)
    print("\nThis will:")
    print("  1. Find all base64 images without Cloudinary IDs")
    print("  2. Upload them to Cloudinary")
    print("  3. Store the Cloudinary public_id in DB")
    print("\nAfter this, you can safely delete image_data from DB")
    print("="*60 + "\n")
    
    response = input("Start migration? (yes/no): ").strip().lower()
    if response != "yes":
        logger.info("Migration cancelled")
        return
    
    try:
        await uploader.migrate_base64_images()
    except Exception as e:
        logger.error(f"Migration failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
