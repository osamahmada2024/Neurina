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
RUN pip install --no-cache-dir \
    gfpgan \
    realesrgan \
    basicsr

# Copy the vendor directory (CodeFormer needs this)
COPY .vendor /app/.vendor

# Copy the source code
COPY src /app/src

# Create checkpoints directory
RUN mkdir -p /app/checkpoints

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_LOADING_USE_HUGGINGFACE=false

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
