---
title: NeurinaXAI
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# NeurinaXAI

FastAPI service for face preprocessing, public reference sync, and StarGAN v2 image translation.

## Hugging Face Space

This repository is configured to run as a Hugging Face Docker Space.

Required Space secrets or variables:

- `APP_NAME`
- `APP_VERSION`
- `MONGO_URI`
- `DB_NAME`
- `HOST`
- `PORT`
- `DEBUG`
- `SECRET_KEY`
- `ALGORITHM`
- `ACCESS_TOKEN_EXPIRE_MINUTES`
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_ANDROID_CLIENT_ID`
- `GOOGLE_IOS_CLIENT_ID`
- `GITHUB_CLIENT_ID`
- `GITHUB_CLIENT_SECRET`
- `SMTP_SERVER`
- `SMTP_PORT`
- `SMTP_EMAIL`
- `SMTP_PASSWORD`
- `RESET_LINK_WEB`
- `RESET_LINK_MOBILE`
- `WING_MODEL_PATH`
- `CELEBA_LM_MEAN_PATH`
- `FACE_CROP_SERVICE_URL`
- `FACE_CROP_TIMEOUT_SECONDS`
- `W_HPF`
- `NUM_DOMAINS`
- `REFERENCE_DOMAIN_LABEL`
- `CHECKPOINT_VARIANT`
- `FACE_MARGIN_LEFT`
- `FACE_MARGIN_RIGHT`
- `FACE_MARGIN_TOP`
- `FACE_MARGIN_BOTTOM`
- `CLOUDINARY_CLOUD_NAME`
- `CLOUDINARY_API_KEY`
- `CLOUDINARY_API_SECRET`
- `MODEL_LOADING_USE_HUGGINGFACE`
- `MODEL_LOADING_HF_MODEL_REPO`
- `MODEL_LOADING_HF_CACHE_DIR`
- `MODEL_LOADING_LOCAL_CHECKPOINTS_DIR`
- `MODEL_LOADING_PRELOAD_FACE_RESTORATION_ON_STARTUP`

Recommended for Hugging Face Spaces:

- `HOST=0.0.0.0`
- `PORT=8000`
- `MODEL_LOADING_USE_HUGGINGFACE=true`
- `MODEL_LOADING_PRELOAD_FACE_RESTORATION_ON_STARTUP=false`

## Quick Start

```bash
pip install -r requirements.txt
docker compose -f docker/docker-compose.yml up -d
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Access the API at: http://localhost:8000/docs

## Setup & Configuration

For detailed setup instructions, see [SETUP.md](SETUP.md).

**Quick Config:**
- Copy `src/.env.example` to `src/.env` and customize settings
- Model weights download automatically on first use
- `Ref_Database` syncs into MongoDB on startup

## Dependencies

- **GFPGAN** - installed via pip (face restoration)
- **CodeFormer** - source in `.vendor/CodeFormer/` (fine-grained face restoration)  
- **Real-ESRGAN** - installed via pip (super-resolution)
- **PyTorch** - GPU-accelerated deep learning
- **FastAPI** - async web framework
- **MongoDB** - database (via Docker)
