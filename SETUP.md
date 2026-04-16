# Setup Guide - Neurina Project

## Prerequisites

- Python 3.8+
- CUDA 11.8+ (recommended for GPU acceleration, CPU fallback available)

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **GFPGAN** (1.3.8) - Face restoration
- **Real-ESRGAN** (0.3.0) - Super-resolution
- **PyTorch** (2.2.0) with CUDA support
- **FastAPI** - Web framework
- **MongoDB driver** - Database
- And other dependencies

### 2. Set Up Environment Variables

```bash
cp src/.env.example src/.env
```

Edit `src/.env` with your configuration (see comments in the file for details).

### 3. CodeFormer Support (Optional but Recommended)

CodeFormer source code is included in `.vendor/CodeFormer/` (not available on PyPI).

**No additional setup required** - it's already set up in the repository.

If you need to manually reinstall CodeFormer source files:

```bash
git clone https://github.com/sczhou/CodeFormer.git .vendor/CodeFormer
```

### 4. Download Model Weights

Model weights are **automatically downloaded on first use** to `checkpoints/face_restoration/`:

- GFPGAN: `GFPGANv1.4.pth`
- CodeFormer: `codeformer.pth`
- Real-ESRGAN: `RealESRGAN_x4plus.pth`

Alternatively, pre-download manually:

```bash
python -c "from src.services.face_restoration_service import FaceRestorationService; FaceRestorationService('.')._ensure_gfpgan_weights(); FaceRestorationService('.')._ensure_codeformer_weights()"
```

### 5. Start MongoDB (Docker)

```bash
docker compose -f docker/docker-compose.yml up -d
```

Verify: `docker compose -f docker/docker-compose.yml ps`

### 6. Run the Application

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Access API: http://localhost:8000/docs

## Directory Structure

```
Neurina/
├── .vendor/
│   └── CodeFormer/          # CodeFormer source (required)
│       ├── basicsr/         # Architecture modules
│       └── ...
├── checkpoints/             # Model weights (auto-downloaded)
│   ├── face_restoration/
│   ├── super_resolution/
│   └── ...
├── src/
│   ├── main.py             # FastAPI entry point
│   ├── services/           # Business logic
│   └── ...
├── requirements.txt        # Python dependencies
└── SETUP.md               # This file
```

## Troubleshooting

### GFPGAN Import Errors
```
ImportError: gfpgan is not installed
```
**Solution**: Run `pip install -r requirements.txt`

### CodeFormer Module Not Found
```
FileNotFoundError: CodeFormer vendor archs not found
```
**Solution**: Ensure `.vendor/CodeFormer/basicsr/archs/` exists. Reinstall if missing:
```bash
git clone https://github.com/sczhou/CodeFormer.git .vendor/CodeFormer
```

### CUDA Out of Memory
Set GPU to CPU mode in `.env`:
```
DEVICE=cpu
```

Or use tiling in `.env`:
```
SR_TILE=256
```

### Models Fail to Download
Check internet connectivity and GitHub releases availability:
- GFPGAN: https://github.com/TencentARC/GFPGAN/releases/tag/v1.3.8
- CodeFormer: https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0

## Performance Tips

1. **GPU Acceleration**: Ensure CUDA is available (`nvidia-smi`)
2. **Batch Processing**: Use batch endpoints for better throughput
3. **Caching**: Models are cached in memory after first load
4. **Tiling**: Reduce `SR_TILE` in `.env` for large images

## Development

See README.md for basic usage and API documentation.
