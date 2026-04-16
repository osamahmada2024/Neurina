# Neurina

FastAPI service for face preprocessing, public reference sync, and StarGAN v2 image translation.

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
