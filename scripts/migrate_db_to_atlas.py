"""
Database Migration Script: Local MongoDB to MongoDB Atlas

This script migrates all data from local MongoDB to MongoDB Atlas cloud database.
Usage: python migrate_db_to_atlas.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import UpdateOne

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


class DatabaseMigrator:
    """Handles migration of MongoDB data from local to Atlas."""
    
    def __init__(self):
        self.local_client: Optional[AsyncIOMotorClient] = None
        self.atlas_client: Optional[AsyncIOMotorClient] = None
        self.local_db: Optional[AsyncIOMotorDatabase] = None
        self.atlas_db: Optional[AsyncIOMotorDatabase] = None
        self.stats = {
            "collections": 0,
            "documents": 0,
            "errors": 0,
        }
    
    async def connect(self) -> bool:
        """Connect to both MongoDB instances."""
        try:
            logger.info("Connecting to local MongoDB...")
            self.local_client = AsyncIOMotorClient(LOCAL_DB_URI, serverSelectionTimeoutMS=5000)
            # Test connection
            await self.local_client.admin.command('ping')
            self.local_db = self.local_client[DB_NAME]
            logger.info("✓ Connected to local MongoDB")
            
            logger.info("Connecting to MongoDB Atlas...")
            self.atlas_client = AsyncIOMotorClient(ATLAS_DB_URI, serverSelectionTimeoutMS=5000)
            # Test connection
            await self.atlas_client.admin.command('ping')
            self.atlas_db = self.atlas_client[DB_NAME]
            logger.info("✓ Connected to MongoDB Atlas")
            
            return True
        except Exception as e:
            logger.error(f"✗ Connection failed: {e}")
            return False
    
    async def get_collections(self) -> list:
        """Get list of collections from local database."""
        try:
            collections = await self.local_db.list_collection_names()
            logger.info(f"Found {len(collections)} collections: {', '.join(collections)}")
            return collections
        except Exception as e:
            logger.error(f"Failed to get collections: {e}")
            return []
    
    async def migrate_collection(self, collection_name: str) -> bool:
        """Migrate a single collection from local to Atlas."""
        try:
            logger.info(f"\n[{collection_name}] Starting migration...")
            
            local_col = self.local_db[collection_name]
            atlas_col = self.atlas_db[collection_name]
            
            # Get document count
            doc_count = await local_col.count_documents({})
            logger.info(f"[{collection_name}] Found {doc_count} documents")
            
            if doc_count == 0:
                logger.info(f"[{collection_name}] Collection is empty, skipping")
                return True
            
            # Clear existing data in Atlas (optional - comment out to append)
            # await atlas_col.delete_many({})
            # logger.info(f"[{collection_name}] Cleared existing data in Atlas")
            
            # Fetch all documents from local
            cursor = local_col.find({})
            documents = await cursor.to_list(length=None)
            
            if not documents:
                logger.info(f"[{collection_name}] No documents to migrate")
                return True
            
            # Insert into Atlas in batches
            batch_size = 1000
            inserted_count = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                try:
                    result = await atlas_col.insert_many(batch, ordered=False)
                    inserted_count += len(result.inserted_ids)
                    logger.info(f"[{collection_name}] Inserted batch {i//batch_size + 1}: {len(result.inserted_ids)} documents")
                except Exception as e:
                    # If duplicate key error, try update
                    if "duplicate" in str(e).lower():
                        logger.warning(f"[{collection_name}] Duplicate key detected, using upsert...")
                        for doc in batch:
                            try:
                                await atlas_col.replace_one(
                                    {"_id": doc["_id"]},
                                    doc,
                                    upsert=True
                                )
                                inserted_count += 1
                            except Exception as inner_e:
                                logger.error(f"[{collection_name}] Upsert failed: {inner_e}")
                                self.stats["errors"] += 1
                    else:
                        logger.error(f"[{collection_name}] Batch insert failed: {e}")
                        self.stats["errors"] += 1
            
            logger.info(f"[{collection_name}] ✓ Migration complete: {inserted_count}/{doc_count} documents")
            self.stats["documents"] += inserted_count
            self.stats["collections"] += 1
            return True
            
        except Exception as e:
            logger.error(f"[{collection_name}] ✗ Migration failed: {e}")
            self.stats["errors"] += 1
            return False
    
    async def migrate_all(self) -> bool:
        """Migrate all collections from local to Atlas."""
        if not await self.connect():
            return False
        
        collections = await self.get_collections()
        if not collections:
            logger.error("No collections found to migrate")
            return False
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting migration of {len(collections)} collections...")
        logger.info(f"{'='*60}\n")
        
        for collection in collections:
            await self.migrate_collection(collection)
        
        logger.info(f"\n{'='*60}")
        logger.info("MIGRATION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Collections migrated: {self.stats['collections']}")
        logger.info(f"Documents migrated: {self.stats['documents']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"{'='*60}\n")
        
        return self.stats["errors"] == 0
    
    async def close(self):
        """Close database connections."""
        if self.local_client:
            self.local_client.close()
        if self.atlas_client:
            self.atlas_client.close()
        logger.info("Connections closed")


async def main():
    """Main entry point."""
    migrator = DatabaseMigrator()
    
    try:
        success = await migrator.migrate_all()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n✗ Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        sys.exit(1)
    finally:
        await migrator.close()


if __name__ == "__main__":
    asyncio.run(main())
