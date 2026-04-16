from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402
from src.controllers import image_controller  # noqa: E402
from src.models import init_db  # noqa: E402
from src.services import ModelLoader  # noqa: E402


async def _bootstrap_models() -> None:
    await init_db()

    base_path = str(ROOT)
    models = ModelLoader.load_models(base_path)
    if not models:
        raise RuntimeError("Failed to load models for public reference import.")

    wing_path = ROOT / "checkpoints" / "wing.ckpt"
    celeba_lm_path = ROOT / "checkpoints" / "celeba_lm_mean.npz"
    image_controller.initialize_models(
        generator=models.get("generator"),
        style_encoder=models.get("style_encoder"),
        fan_model=models.get("fan_model"),
        wing_model_path=str(wing_path),
        celeba_lm_path=str(celeba_lm_path) if celeba_lm_path.is_file() else None,
    )
    image_controller.initialize_postprocessors(base_path)
    image_controller._trace_image = lambda *args, **kwargs: None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess public reference images and store them in MongoDB for site-side selection."
    )
    parser.add_argument(
        "--source-dir",
        default=str(ROOT / settings.PUBLIC_REFERENCE_DIR),
        help="Root directory containing public references, typically with male/ and female/ subfolders.",
    )
    parser.add_argument(
        "--collection",
        default=settings.PUBLIC_REFERENCE_COLLECTION,
        help="Logical collection label stored in image documents.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for how many files to import, useful for smoke tests.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="How many imported IDs to print at the end.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    await _bootstrap_models()

    summary = await image_controller.import_public_references_from_directory(
        root_dir=args.source_dir,
        public_collection=args.collection,
        limit=args.limit,
    )

    print(
        "[SUMMARY] "
        f"processed={summary['processed']} "
        f"inserted={summary['inserted']} "
        f"updated={summary['updated']} "
        f"failed={summary['failed']} "
        f"root_dir={summary['root_dir']}"
    )

    for item in summary["images"][: max(0, int(args.show))]:
        print(
            f"[IMAGE] id={item['_id']} status={item['status']} "
            f"domain={item['image_domain']} key={item['library_key']}"
        )

    for error in summary["errors"][:10]:
        print(f"[ERROR] path={error['path']} error={error['error']}")

    return 0 if summary["processed"] > 0 else 1


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
