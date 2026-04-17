from __future__ import annotations

import importlib
import importlib.util
import os
import site
import sys
import types
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config.model_loading import model_loading_settings
from ..utils.hf_model_loader import ensure_inference_model


class FaceRestorationService:
    """
    Post-process aligned face crops using GFPGAN, CodeFormer, or Real-ESRGAN.
    
    Dependencies:
    - GFPGAN: installed via pip (see requirements.txt)
    - CodeFormer: source code in .vendor/CodeFormer/ (not available on PyPI)
    - Real-ESRGAN: installed via pip (see requirements.txt)
    
    Model weights are automatically downloaded on first use to checkpoints/face_restoration/
    """

    # Hugging Face models for inference
    HF_MODELS = {
        "codeformer": {
            "filename": "codeformer.pth",
            "scale": 2,
            "pipeline": "face_transformer",
        },
        "realesrgan_x4plus": {
            "filename": "RealESRGAN_x4plus.pth",
            "scale": 4,
            "pipeline": "generic",
        },
    }
    
    # Legacy URL models (GFPGAN remains URL-based)
    MODEL_URLS = {
        "gfpgan_v1.4": (
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.8/GFPGANv1.4.pth",
            2,
            "face",
        ),
    }

    def __init__(
        self,
        base_path: str,
        model_name: str = "gfpgan_v1.4",
        outscale: float = 2.0,
        tile: int = 0,
        device: Optional[str] = None,
        face_weight: float = 0.5,
        codeformer_fidelity: float = 0.7,
    ):
        self.base_path = Path(base_path)
        self.model_name = model_name.lower()
        self.outscale = float(outscale)
        self.tile = int(tile)
        self.device = device
        self.face_weight = float(face_weight)
        self.codeformer_fidelity = float(codeformer_fidelity)
        self._upsampler = None
        self._face_restorer = None
        self._codeformer_net = None

    @staticmethod
    def _ensure_torchvision_functional_tensor_alias() -> None:
        if "torchvision.transforms.functional_tensor" in sys.modules:
            return
        compat = types.ModuleType("torchvision.transforms.functional_tensor")
        try:
            import torchvision.transforms.functional as tvf

            compat.rgb_to_grayscale = tvf.rgb_to_grayscale
        except Exception:
            def _fallback_rgb_to_grayscale(*args, **kwargs):
                raise ImportError("torchvision is not installed")

            compat.rgb_to_grayscale = _fallback_rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = compat

    @staticmethod
    def _suppress_torchvision_pretrained_warnings() -> None:
        warnings.filterwarnings(
            "ignore",
            message=r"The parameter 'pretrained' is deprecated since 0\.13.*",
            category=UserWarning,
            module=r"torchvision\.models\._utils",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Arguments other than a weight enum or `None` for 'weights' are deprecated since 0\.13.*",
            category=UserWarning,
            module=r"torchvision\.models\._utils",
        )

    @staticmethod
    def _import_gfpganer(base_path: str):
        """Import GFPGANer from pip-installed gfpgan package."""
        FaceRestorationService._ensure_torchvision_functional_tensor_alias()
        FaceRestorationService._suppress_torchvision_pretrained_warnings()
        
        try:
            from gfpgan import GFPGANer
            return GFPGANer
        except ImportError:
            raise ImportError(
                "gfpgan is not installed. Install it with: pip install -r requirements.txt"
            )

    def _get_model_info(self) -> tuple[str, int, str]:
        # Check Hugging Face models first
        if self.model_name in self.HF_MODELS:
            model_info = self.HF_MODELS[self.model_name]
            return model_info["filename"], model_info["scale"], model_info["pipeline"]
        
        # Fallback to legacy URL models
        if self.model_name in self.MODEL_URLS:
            url, scale, pipeline = self.MODEL_URLS[self.model_name]
            return Path(url).name, scale, pipeline
            
        raise ValueError(f"Unsupported SR model: {self.model_name}")

    def _weights_dir(self, category: str) -> Path:
        path = self.base_path / "checkpoints" / category
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_local_inference_checkpoint(self, filename: str) -> Path:
        return model_loading_settings.resolve_checkpoint_path(filename, self.base_path)

    def _ensure_inference_checkpoint(self, filename: str, scale: int) -> tuple[Path, int]:
        local_path = self._resolve_local_inference_checkpoint(filename)
        if local_path.is_file():
            return local_path, scale

        if not model_loading_settings.use_huggingface:
            raise FileNotFoundError(
                f"Local checkpoint not found with Hugging Face disabled: {local_path}"
            )

        return ensure_inference_model(filename), scale

    def _ensure_realesrgan_weights(self) -> tuple[Path, int]:
        model_info = self.HF_MODELS["realesrgan_x4plus"]
        return self._ensure_inference_checkpoint(
            model_info["filename"],
            model_info["scale"],
        )

    @staticmethod
    def _find_installed_weight(package_name: str, filename: str) -> Optional[Path]:
        roots: list[str] = []
        try:
            roots.extend(site.getsitepackages())
        except Exception:
            pass
        try:
            user_site = site.getusersitepackages()
            if user_site:
                roots.append(user_site)
        except Exception:
            pass

        for root in roots:
            candidate = Path(root) / package_name / "weights" / filename
            if candidate.is_file():
                return candidate
        return None

    def _ensure_gfpgan_weights(self) -> tuple[Path, int]:
        url, scale, _ = self.MODEL_URLS["gfpgan_v1.4"]
        filename = Path(url).name
        model_path = self._weights_dir("face_restoration") / filename
        if model_path.is_file():
            return model_path, scale

        installed = self._find_installed_weight("gfpgan", filename)
        if installed is not None:
            return installed, scale

        self._ensure_torchvision_functional_tensor_alias()
        from basicsr.utils.download_util import load_file_from_url

        load_file_from_url(
            url=url,
            model_dir=str(self._weights_dir("face_restoration")),
            progress=True,
            file_name=filename,
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"Failed to download GFPGAN weights to {model_path}")
        return model_path, scale

    def _ensure_codeformer_weights(self) -> tuple[Path, int]:
        if self.model_name == "codeformer":
            model_info = self.HF_MODELS["codeformer"]
            return self._ensure_inference_checkpoint(
                model_info["filename"],
                model_info["scale"],
            )
        raise ValueError(f"Unsupported SR model: {self.model_name}")

    @staticmethod
    def _load_module_from_file(module_name: str, file_path: Path):
        if module_name in sys.modules:
            return sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module spec for {module_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _ensure_codeformer_modules(self) -> None:
        if "basicsr.archs.codeformer_arch" in sys.modules:
            return

        # Search paths for CodeFormer arch files (in priority order)
        search_paths = [
            self.base_path / ".vendor" / "CodeFormer" / "basicsr" / "archs",
            self.base_path / "src" / "codeformer_archs",
            self.base_path / "codeformer_archs",
        ]

        for codeformer_root in search_paths:
            vqgan = codeformer_root / "vqgan_arch.py"
            codeformer = codeformer_root / "codeformer_arch.py"
            if vqgan.is_file() and codeformer.is_file():
                self._load_module_from_file("basicsr.archs.vqgan_arch", vqgan)
                self._load_module_from_file("basicsr.archs.codeformer_arch", codeformer)
                logger.info(f"CodeFormer archs loaded from {codeformer_root}")
                return

        # Fallback: try importing from pip-installed basicsr / CodeFormer package
        try:
            from basicsr.archs import codeformer_arch  # noqa: F401
            from basicsr.archs import vqgan_arch  # noqa: F401
            logger.info("CodeFormer archs loaded from pip-installed basicsr")
            return
        except ImportError:
            pass

        raise FileNotFoundError(
            "CodeFormer arch files not found in any search path: "
            + ", ".join(str(p) for p in search_paths)
            + ". Ensure .vendor/CodeFormer or src/codeformer_archs/ is present."
        )

    def _build_generic_upsampler(self):
        if self._upsampler is not None:
            return self._upsampler

        self._ensure_torchvision_functional_tensor_alias()
        self._suppress_torchvision_pretrained_warnings()
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model_path, scale = self._ensure_realesrgan_weights()
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)

        half = False
        if self.device is None:
            try:
                import torch

                half = bool(torch.cuda.is_available())
            except Exception:
                half = False
        elif str(self.device).startswith("cuda"):
            half = True

        self._upsampler = RealESRGANer(
            scale=scale,
            model_path=str(model_path),
            model=model,
            tile=self.tile,
            tile_pad=10,
            pre_pad=0,
            half=half,
        )
        return self._upsampler

    def _build_face_restorer(self):
        if self._face_restorer is not None:
            return self._face_restorer

        self._suppress_torchvision_pretrained_warnings()
        GFPGANer = self._import_gfpganer(str(self.base_path))
        model_path, scale = self._ensure_gfpgan_weights()
        self._face_restorer = GFPGANer(
            model_path=str(model_path),
            upscale=scale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=self._build_generic_upsampler(),
            device=self.device,
        )
        return self._face_restorer

    def _build_codeformer(self):
        if self._codeformer_net is not None:
            return self._codeformer_net

        self._ensure_torchvision_functional_tensor_alias()
        self._suppress_torchvision_pretrained_warnings()
        self._ensure_codeformer_modules()

        import torch
        from basicsr.utils.registry import ARCH_REGISTRY

        model_path, _ = self._ensure_codeformer_weights()
        device = torch.device(
            self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(device)

        checkpoint = torch.load(str(model_path), map_location="cpu")
        params = checkpoint.get("params_ema") or checkpoint.get("params")
        if params is None:
            raise KeyError("CodeFormer checkpoint is missing params_ema/params")
        net.load_state_dict(params, strict=True)
        net.eval()
        self._codeformer_net = net
        return self._codeformer_net

    @staticmethod
    def _resize_exact(image_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        if image_bgr.shape[1] == target_w and image_bgr.shape[0] == target_h:
            return image_bgr
        return cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    def _run_realesrgan(self, image_bgr: np.ndarray, outscale: float) -> np.ndarray:
        upsampler = self._build_generic_upsampler()
        output, _ = upsampler.enhance(image_bgr, outscale=float(outscale))
        if output is None or output.size == 0:
            raise ValueError("Real-ESRGAN returned an empty image")
        return output

    def _run_gfpgan(self, image_bgr: np.ndarray, outscale: float) -> np.ndarray:
        restorer = self._build_face_restorer()
        _, restored_faces, _ = restorer.enhance(
            image_bgr,
            has_aligned=True,
            only_center_face=True,
            paste_back=False,
            weight=self.face_weight,
        )
        if not restored_faces:
            raise ValueError("GFPGAN returned no restored face")

        restored_bgr = restored_faces[0]
        target_h = max(1, int(round(image_bgr.shape[0] * float(outscale))))
        target_w = max(1, int(round(image_bgr.shape[1] * float(outscale))))

        current_h, current_w = restored_bgr.shape[:2]
        upscale_ratio = max(target_w / float(current_w), target_h / float(current_h))

        if upscale_ratio > 1.01:
            restored_bgr = self._run_realesrgan(restored_bgr, outscale=upscale_ratio)

        return self._resize_exact(restored_bgr, target_w, target_h)

    def _run_codeformer(self, image_bgr: np.ndarray, outscale: float) -> np.ndarray:
        self._ensure_torchvision_functional_tensor_alias()
        self._suppress_torchvision_pretrained_warnings()
        import torch
        from basicsr.utils import img2tensor, tensor2img
        from torchvision.transforms.functional import normalize

        net = self._build_codeformer()
        device = next(net.parameters()).device

        aligned_bgr = cv2.resize(image_bgr, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        face_tensor = img2tensor(aligned_bgr / 255.0, bgr2rgb=True, float32=True)
        normalize(face_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        face_tensor = face_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = net(face_tensor, w=self.codeformer_fidelity, adain=True)[0]
            restored_bgr = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

        restored_bgr = restored_bgr.astype("uint8")
        target_h = max(1, int(round(image_bgr.shape[0] * float(outscale))))
        target_w = max(1, int(round(image_bgr.shape[1] * float(outscale))))

        current_h, current_w = restored_bgr.shape[:2]
        upscale_ratio = max(target_w / float(current_w), target_h / float(current_h))
        if upscale_ratio > 1.01:
            restored_bgr = self._run_realesrgan(restored_bgr, outscale=upscale_ratio)

        return self._resize_exact(restored_bgr, target_w, target_h)

    def enhance(self, image_bgr: np.ndarray, outscale: Optional[float] = None) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Invalid image for super resolution")

        actual_outscale = float(outscale or self.outscale)
        if actual_outscale <= 0:
            raise ValueError("outscale must be greater than zero")

        _, _, pipeline = self._get_model_info()
        if pipeline == "face_transformer":
            return self._run_codeformer(image_bgr, actual_outscale)
        if pipeline == "face":
            return self._run_gfpgan(image_bgr, actual_outscale)
        return self._run_realesrgan(image_bgr, actual_outscale)
