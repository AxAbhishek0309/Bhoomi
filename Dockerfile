# AgriBotX Pro Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libpq-dev \
    libgdal-dev \
    gdal-bin \
    libspatialindex-dev \
    libgeos-dev \
    libproj-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/mock_soil_reports data/crop_images data/ndvi_samples models logs

# Set permissions
RUN chmod +x main.py cli/agribotx_cli.py ui/dashboard.py

# Expose ports
EXPOSE 7860 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860')" || exit 1

# Default command
CMD ["python", "main.py", "--mode", "web", "--host", "0.0.0.0", "--port", "7860"]