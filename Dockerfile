FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV, Pillow, and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install additional dependencies for face restoration
# Pin versions compatible with numpy 1.26.4 (numpy 2.x causes 'expected np.ndarray' errors)
RUN pip install --no-cache-dir \
    'numpy<2' \
    gfpgan==1.3.8 \
    realesrgan==0.3.0 \
    basicsr==1.4.2

# Copy the vendor directory (CodeFormer needs this)
COPY .vendor /app/.vendor

# Copy the source code
COPY src /app/src

# Create checkpoints directory and pre-download models from HuggingFace during build
# This avoids long startup downloads that cause crash loops
RUN mkdir -p /app/checkpoints
RUN python << 'EOF'
from huggingface_hub import hf_hub_download
import os
repo = 'Osama12324234234/face-models'
dst = '/app/checkpoints'
files = ['582000_nets_ema.ckpt', '582000_nets.ckpt', 'codeformer.pth', 'RealESRGAN_x4plus.pth', 'wing.ckpt']
for f in files:
    out = os.path.join(dst, f)
    if not os.path.exists(out):
        print(f'Downloading {f}...')
        hf_hub_download(repo_id=repo, filename=f, local_dir=dst)
        print(f'Downloaded {f}')
    else:
        print(f'Already cached: {f}')
print('All models ready')
EOF

# Set environment variables - use local checkpoints since we pre-downloaded them
ENV PYTHONPATH=/app
ENV MODEL_LOADING_USE_HUGGINGFACE=false

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
