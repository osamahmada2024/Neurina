import asyncio
import hashlib
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Set Railway MongoDB URI before importing models
import os
os.environ["MONGO_URI"] = "mongodb+srv://hmsosama773_db_user:bweqI9cPybKowOcL@cluster0.srxflho.mongodb.net/?appName=Cluster0"

from src.models import database
from src.config.cloudinary import cloudinary_settings


def build_public_reference_public_id(library_key: str, variant: str) -> str:
    """Build a stable public ID so retries reuse the same Cloudinary asset."""
    normalized_key = str(library_key).replace("\\", "/").lower()
    digest = hashlib.sha1(normalized_key.encode("utf-8")).hexdigest()[:12]
    stem = Path(normalized_key).stem
    return f"neurina/processed_faces/public_references/{stem}-{digest}-{variant}"


def slugify_public_id_component(text: str) -> str:
    """Simple slugify for public ID components."""
    import re
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    text = text.strip('-')
    return text[:100]


async def update_image_urls():
    """Update image URLs from cloudinary_public_id for all images."""
    
    try:
        # Get all images
        images = await database["images"].find({}).to_list(length=None)
        
        print(f"Found {len(images)} images")
        
        updated_count = 0
        
        for img in images:
            update_doc = {}
            
            # Build processed URL from cloudinary_public_id_processed
            if img.get("cloudinary_public_id_processed") and not img.get("image_data"):
                processed_public_id = img["cloudinary_public_id_processed"]
                processed_url = f"https://res.cloudinary.com/{cloudinary_settings.cloud_name}/image/upload/{processed_public_id}.jpg"
                update_doc["image_data"] = processed_url
                print(f"Updating processed URL from cloudinary_public_id_processed for {img['_id']}")
            
            # Build original URL from cloudinary_public_id_original
            if img.get("cloudinary_public_id_original") and not img.get("image_data_original"):
                original_public_id = img["cloudinary_public_id_original"]
                original_url = f"https://res.cloudinary.com/{cloudinary_settings.cloud_name}/image/upload/{original_public_id}.jpg"
                update_doc["image_data_original"] = original_url
                print(f"Updating original URL from cloudinary_public_id_original for {img['_id']}")
            
            # For public references without cloudinary_public_id, build from library_key
            if img.get("is_public") and img.get("library_key"):
                library_key = img["library_key"]
                processed_public_id = build_public_reference_public_id(library_key, "processed")
                original_public_id = build_public_reference_public_id(library_key, "original")
                
                # Try both .jpg and .png extensions
                processed_url_jpg = f"https://res.cloudinary.com/{cloudinary_settings.cloud_name}/image/upload/{processed_public_id}.jpg"
                original_url_jpg = f"https://res.cloudinary.com/{cloudinary_settings.cloud_name}/image/upload/{original_public_id}.jpg"
                
                # Update processed URL if missing
                if not img.get("image_data"):
                    update_doc["image_data"] = processed_url_jpg
                    print(f"Updating processed URL from library_key for {img['_id']}")
                
                # Update original URL if missing (separate check)
                if not img.get("image_data_original"):
                    update_doc["image_data_original"] = original_url_jpg
                    print(f"Updating original URL from library_key for {img['_id']}")
            
            if update_doc:
                await database["images"].update_one(
                    {"_id": img["_id"]},
                    {"$set": update_doc}
                )
                updated_count += 1
        
        print(f"Successfully updated {updated_count} images with Cloudinary URLs")
        return updated_count
        
    except Exception as exc:
        print(f"Failed to update image URLs: {str(exc)}")
        raise


if __name__ == "__main__":
    asyncio.run(update_image_urls())
