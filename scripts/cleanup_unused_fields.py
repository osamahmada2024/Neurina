"""
Clean database: Remove unused fields from documents

Removes fields that are stored but never read in the codebase:
- library_source_path
- sync_error
- landmarks
- display_resolution
- model_resolution
"""

import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LOCAL_DB_URI = "mongodb://localhost:27017"
DB_NAME = "neurina_xai_db"

# Unused fields to remove
UNUSED_FIELDS = [
    "library_source_path",      # Stored but never read
    "sync_error",               # Set/unset but never read
    "landmarks",                # Computed & stored but never retrieved (huge!)
    "display_resolution",       # Stored but never read
    "model_resolution",         # Stored but never read
    "library_source_mtime_ns",  # For sync only - public refs don't change
    "library_source_size",      # For sync only - public refs don't change
]


class UnusedFieldsCleaner:
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
    
    async def analyze_unused_fields(self):
        """Analyze storage used by unused fields."""
        if not await self.connect():
            return
        
        col = self.db["images"]
        
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS: UNUSED FIELDS")
        logger.info("="*60)
        
        for field in UNUSED_FIELDS:
            query = {field: {"$exists": True}}
            count = await col.count_documents(query)
            
            if count == 0:
                logger.info(f"✓ {field}: Not present")
                continue
            
            # Sample one document with this field
            sample = await col.find_one(query)
            if sample:
                field_data = sample.get(field)
                if isinstance(field_data, list):
                    size_kb = len(str(field_data)) / 1024
                    logger.info(f"✗ {field}: {count} docs, ~{size_kb:.1f} KB each")
                elif isinstance(field_data, dict):
                    size_kb = len(str(field_data)) / 1024
                    logger.info(f"✗ {field}: {count} docs, ~{size_kb:.1f} KB each")
                else:
                    logger.info(f"✗ {field}: {count} docs")
        
        logger.info("="*60 + "\n")
        self.client.close()
    
    async def cleanup_unused_fields(self):
        """Remove all unused fields from database."""
        if not await self.connect():
            return
        
        col = self.db["images"]
        
        # Check which fields exist
        fields_to_remove = {}
        for field in UNUSED_FIELDS:
            count = await col.count_documents({field: {"$exists": True}})
            if count > 0:
                fields_to_remove[field] = ""
        
        if not fields_to_remove:
            logger.info("✓ No unused fields to remove")
            self.client.close()
            return
        
        logger.info("\n" + "="*60)
        logger.info("CLEANUP: REMOVE UNUSED FIELDS")
        logger.info("="*60)
        logger.info("\nFields to remove:")
        for field in fields_to_remove.keys():
            logger.info(f"  • {field}")
        logger.info("="*60 + "\n")
        
        # Confirm
        response = input(f"Remove {len(fields_to_remove)} unused fields? (yes/no): ").strip().lower()
        if response != "yes":
            logger.info("Cleanup cancelled")
            self.client.close()
            return
        
        # Remove fields
        result = await col.update_many(
            {},
            {"$unset": fields_to_remove}
        )
        
        logger.info("\n" + "="*60)
        logger.info("CLEANUP COMPLETE")
        logger.info("="*60)
        logger.info(f"Updated: {result.modified_count} documents")
        logger.info("\nRemoved fields:")
        for field in fields_to_remove.keys():
            logger.info(f"  ✓ {field}")
        logger.info("="*60 + "\n")
        
        self.client.close()


async def main():
    import sys
    
    cleaner = UnusedFieldsCleaner()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        await cleaner.analyze_unused_fields()
    else:
        await cleaner.cleanup_unused_fields()


if __name__ == "__main__":
    asyncio.run(main())
