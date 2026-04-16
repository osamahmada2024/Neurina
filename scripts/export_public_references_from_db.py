from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import sys
from pathlib import Path

from motor.motor_asyncio import AsyncIOMotorClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "image"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export public reference images from MongoDB to a local folder."
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "exports" / "public_references_from_db"),
        help="Destination folder for exported images.",
    )
    parser.add_argument(
        "--include-model-variant",
        action="store_true",
        help="Also export model_image_data alongside the display image_data variant.",
    )
    return parser


async def _export_images(output_dir: Path, include_model_variant: bool) -> int:
    client = AsyncIOMotorClient(settings.MONGO_URI)
    db = client[settings.DB_NAME]
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    cursor = (
        db["images"]
        .find({"is_public": True, "image_type": "reference"})
        .sort("image_domain", 1)
        .sort("library_key", 1)
    )

    try:
        async for doc in cursor:
            image_id = str(doc["_id"])
            domain = str(doc.get("image_domain") or "unknown")
            library_key = str(doc.get("library_key") or doc.get("original_filename") or image_id)
            stem = _safe_name(Path(library_key).stem)

            domain_dir = output_dir / domain
            domain_dir.mkdir(parents=True, exist_ok=True)

            display_filename = f"{image_id}__{stem}.png"
            display_path = domain_dir / display_filename
            display_bytes = base64.b64decode(doc["image_data"])
            display_path.write_bytes(display_bytes)

            model_relpath = None
            if include_model_variant and doc.get("model_image_data"):
                model_dir = output_dir / "model_variant" / domain
                model_dir.mkdir(parents=True, exist_ok=True)
                model_filename = f"{image_id}__{stem}__model.png"
                model_path = model_dir / model_filename
                model_path.write_bytes(base64.b64decode(doc["model_image_data"]))
                model_relpath = str(model_path.relative_to(output_dir)).replace("\\", "/")

            manifest.append(
                {
                    "_id": image_id,
                    "image_domain": doc.get("image_domain"),
                    "domain_label": doc.get("domain_label"),
                    "original_filename": doc.get("original_filename"),
                    "library_key": doc.get("library_key"),
                    "public_collection": doc.get("public_collection"),
                    "display_path": str(display_path.relative_to(output_dir)).replace("\\", "/"),
                    "model_path": model_relpath,
                }
            )
    finally:
        client.close()

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    print(
        f"[OK] exported={len(manifest)} output_dir={output_dir} "
        f"manifest={manifest_path}"
    )
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(
        _export_images(
            output_dir=Path(args.output_dir),
            include_model_variant=bool(args.include_model_variant),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
