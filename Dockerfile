# REAL WAN Image-to-Video Container for RunPod Serverless
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py /app/handler.py

# Pre-download models to reduce cold start time
RUN python -c "from diffusers import MotionAdapter; MotionAdapter.from_pretrained('guoyww/animatediff-motion-adapter-v1-5-2', torch_dtype=torch.float16)" || echo "Model download failed, will download on first run"

# Set entrypoint
CMD ["python", "/app/handler.py"]