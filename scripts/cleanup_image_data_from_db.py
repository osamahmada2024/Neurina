"""
Database Cleanup Script: Remove image_data fields from DB

This script removes image_data and image_data_original from all documents 
since images are now stored on Cloudinary with only IDs needed.
"""

import asyncio
import sys
from typing import Optional
import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database URLs
LOCAL_DB_URI = "mongodb://localhost:27017"
ATLAS_DB_URI = "mongodb+srv://hmsosama773_db_user:bweqI9cPybKowOcL@cluster0.srxflho.mongodb.net/?appName=Cluster0"
DB_NAME = "neurina_xai_db"


class DatabaseCleaner:
    """Cleanup MongoDB: Remove image_data fields to save space."""
    
    def __init__(self, use_atlas: bool = False):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.db_uri = ATLAS_DB_URI if use_atlas else LOCAL_DB_URI
        self.use_atlas = use_atlas
        self.stats = {
            "collections": 0,
            "documents_cleaned": 0,
            "space_saved_estimate_mb": 0,
            "errors": 0,
        }
    
    async def connect(self) -> bool:
        """Connect to MongoDB."""
        try:
            db_type = "MongoDB Atlas" if self.use_atlas else "Local MongoDB"
            logger.info(f"Connecting to {db_type}...")
            
            self.client = AsyncIOMotorClient(self.db_uri, serverSelectionTimeoutMS=5000)
            await self.client.admin.command('ping')
            
            self.db = self.client[DB_NAME]
            logger.info(f"✓ Connected to {db_type}")
            return True
        except Exception as e:
            logger.error(f"✗ Connection failed: {e}")
            return False
    
    async def cleanup_images_collection(self) -> bool:
        """Remove image_data and image_data_original from images collection."""
        try:
            col = self.db["images"]
            
            # Find documents with image_data
            count = await col.count_documents({
                "$or": [
                    {"image_data": {"$exists": True}},
                    {"image_data_original": {"$exists": True}}
                ]
            })
            
            if count == 0:
                logger.info("[images] No documents with image_data fields")
                return True
            
            logger.info(f"[images] Found {count} documents with image_data fields")
            logger.info(f"[images] Estimated space to save: ~{count * 0.5:.1f} MB")
            
            # Update all documents - remove image_data fields
            result = await col.update_many(
                {
                    "$or": [
                        {"image_data": {"$exists": True}},
                        {"image_data_original": {"$exists": True}}
                    ]
                },
                {
                    "$unset": {
                        "image_data": "",
                        "image_data_original": ""
                    }
                }
            )
            
            logger.info(f"[images] ✓ Cleaned {result.modified_count} documents")
            self.stats["documents_cleaned"] += result.modified_count
            self.stats["space_saved_estimate_mb"] += count * 0.5
            self.stats["collections"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"[images] ✗ Cleanup failed: {e}")
            self.stats["errors"] += 1
            return False
    
    async def cleanup_translation_tasks_collection(self) -> bool:
        """Remove image_data from translation tasks if present."""
        try:
            col = self.db["translation_tasks"]
            
            count = await col.count_documents({
                "image_data": {"$exists": True}
            })
            
            if count == 0:
                logger.info("[translation_tasks] No documents with image_data fields")
                return True
            
            logger.info(f"[translation_tasks] Found {count} documents with image_data fields")
            
            result = await col.update_many(
                {"image_data": {"$exists": True}},
                {"$unset": {"image_data": ""}}
            )
            
            logger.info(f"[translation_tasks] ✓ Cleaned {result.modified_count} documents")
            self.stats["documents_cleaned"] += result.modified_count
            self.stats["collections"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"[translation_tasks] ✗ Cleanup failed: {e}")
            self.stats["errors"] += 1
            return False
    
    async def verify_cloudinary_ids(self) -> bool:
        """Verify that images still have cloudinary_public_id_processed."""
        try:
            col = self.db["images"]
            
            # Find documents without cloudinary_public_id_processed
            count = await col.count_documents({
                "is_public": True,
                "cloudinary_public_id_processed": {"$exists": False}
            })
            
            if count > 0:
                logger.warning(f"⚠ Found {count} public images without cloudinary_public_id_processed!")
                logger.warning("These images won't be retrievable after cleanup!")
                return False
            
            logger.info("✓ All public images have cloudinary_public_id_processed")
            return True
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False
    
    async def cleanup_all(self) -> bool:
        """Run all cleanup operations."""
        if not await self.connect():
            return False
        
        logger.info(f"\n{'='*60}")
        logger.info("DATABASE CLEANUP - REMOVE IMAGE_DATA FIELDS")
        logger.info(f"{'='*60}\n")
        
        # Verify before cleanup
        if not await self.verify_cloudinary_ids():
            logger.error("Cannot proceed - some images missing cloudinary_public_id")
            return False
        
        logger.info("Starting cleanup...\n")
        
        # Cleanup collections
        await self.cleanup_images_collection()
        await self.cleanup_translation_tasks_collection()
        
        logger.info(f"\n{'='*60}")
        logger.info("CLEANUP SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Collections processed: {self.stats['collections']}")
        logger.info(f"Documents cleaned: {self.stats['documents_cleaned']}")
        logger.info(f"Estimated space saved: ~{self.stats['space_saved_estimate_mb']:.1f} MB")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"{'='*60}\n")
        
        return self.stats["errors"] == 0
    
    async def close(self):
        """Close database connection."""
        if self.client:
            self.client.close()
        logger.info("Connection closed")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean MongoDB: Remove image_data fields")
    parser.add_argument(
        "--atlas",
        action="store_true",
        help="Clean MongoDB Atlas instead of local"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    if not args.confirm:
        print("\n" + "="*60)
        print("WARNING: This will remove image_data fields from DB!")
        print("="*60)
        print("\nThis operation:")
        print("  • Removes image_data and image_data_original from all images")
        print("  • Saves ~0.5 MB per image (depending on size)")
        print("  • Images will be served from Cloudinary via public_id")
        print("\nEnsure all images have cloudinary_public_id_processed!")
        print("="*60 + "\n")
        
        response = input("Type 'yes' to confirm cleanup: ").strip().lower()
        if response != "yes":
            logger.info("Cleanup cancelled")
            sys.exit(0)
    
    cleaner = DatabaseCleaner(use_atlas=args.atlas)
    
    try:
        success = await cleaner.cleanup_all()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n✗ Cleanup cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        sys.exit(1)
    finally:
        await cleaner.close()


if __name__ == "__main__":
    asyncio.run(main())
