# Neurina

FastAPI service for face preprocessing, public reference sync, and StarGAN v2 image translation.

## Run

```bash
pip install -r requirements.txt
docker compose -f docker/docker-compose.yml up -d
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Config

- Copy `C:\Users\osama\Projects\Neurina\src\.env.example` to `C:\Users\osama\Projects\Neurina\src\.env`
- Set production secrets before deploy
- `Ref_Database` is synced into MongoDB on startup
