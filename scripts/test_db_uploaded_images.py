from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torchvision.utils as vutils
from motor.motor_asyncio import AsyncIOMotorClient
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402
from src.models.neural_models import Generator, StyleEncoder  # noqa: E402
def load_ref_tensor_from_bgr(image_bgr: np.ndarray, img_size: int, device: torch.device) -> torch.Tensor:
    t = transforms.Compose(
        [
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return t(Image.fromarray(image_rgb)).unsqueeze(0).to(device)


def save_out(t: torch.Tensor, path: Path) -> None:
    x = (t + 1) / 2
    x = x.clamp(0, 1)
    vutils.save_image(x.cpu(), str(path), nrow=1, padding=0)


def decode_b64_image(image_b64: str) -> np.ndarray | None:
    try:
        image_bytes = base64.b64decode(image_b64)
        arr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _preferred_image_b64(doc: dict) -> str | None:
    return doc.get("model_image_data") or doc.get("image_data")


async def fetch_images():
    client = AsyncIOMotorClient(settings.MONGO_URI)
    db = client[settings.DB_NAME]
    try:
        sources = await db["images"].find({"image_type": "source"}).sort("created_at", 1).to_list(None)
        refs = await db["images"].find({"image_type": "reference"}).sort("created_at", 1).to_list(None)
        return sources, refs
    finally:
        client.close()


def run_translation_for_pairs(sources: list[dict], refs: list[dict], output_dir: Path) -> int:
    ckpt_dir = ROOT / "checkpoints"
    nets_ckpt = ckpt_dir / "582000_nets.ckpt"
    if not nets_ckpt.is_file():
        print(f"[FATAL] Missing checkpoint: {nets_ckpt}")
        return 1

    if not sources:
        print("[FATAL] No source images found in DB.")
        return 1
    if not refs:
        print("[FATAL] No reference images found in DB.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    ck = torch.load(nets_ckpt, map_location=device)
    gen = Generator(img_size=256, style_dim=64, w_hpf=1).to(device).eval()
    enc = StyleEncoder(img_size=256, style_dim=64, num_domains=2).to(device).eval()
    gen.load_state_dict(ck["generator"])
    enc.load_state_dict(ck["style_encoder"])

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    translated_count = 0
    ref_doc = refs[0]
    ref_bgr = decode_b64_image(_preferred_image_b64(ref_doc))
    if ref_bgr is None:
        print("[FATAL] Could not decode selected reference image.")
        return 1
    x_ref = load_ref_tensor_from_bgr(ref_bgr, 256, device)
    y = torch.tensor([0], dtype=torch.long, device=device)

    for idx, src_doc in enumerate(sources, 1):
        src_bgr = decode_b64_image(_preferred_image_b64(src_doc))
        if src_bgr is None:
            print(f"[WARN] Skipping source {src_doc.get('_id')} decode failed.")
            continue

        rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
        x_src = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
        x_src = (x_src / 127.5) - 1.0
        x_src = x_src.to(device)

        with torch.no_grad():
            s = enc(x_ref, y)
            fake = gen(x_src, s)

        src_id = str(src_doc.get("_id"))
        ref_id = str(ref_doc.get("_id"))
        save_out(x_src, output_dir / f"{idx:03d}_source_{src_id}_{ts}.jpg")
        save_out(x_ref, output_dir / f"{idx:03d}_reference_{ref_id}_{ts}.jpg")
        save_out(fake, output_dir / f"{idx:03d}_translated_{src_id}_to_{ref_id}_{ts}.jpg")
        translated_count += 1

    print(f"[OK] translated={translated_count} output_dir={output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run style transfer tests from uploaded DB images.")
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / "Downloads" / "neurina_style_transfer"),
        help="Directory to save output images.",
    )
    args = parser.parse_args()

    sources, refs = asyncio.run(fetch_images())
    return run_translation_for_pairs(sources, refs, Path(args.output_dir))


if __name__ == "__main__":
    raise SystemExit(main())
