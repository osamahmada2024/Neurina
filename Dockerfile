FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel --break-system-packages

# Copy requirements first to leverage Docker cache


COPY requirements.txt .
RUN python -m pip install \
    --break-system-packages \
    -v \
    -r requirements.txt



# Copy project
COPY .vendor /app/.vendor
COPY src /app/src

# Create checkpoints directory
RUN mkdir -p /app/checkpoints

ENV PYTHONPATH=/app
ENV MODEL_LOADING_USE_HUGGINGFACE=true

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
