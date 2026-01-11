# =========================================
# AMC Project - Dockerfile
# Automatic Modulation Classification
# =========================================

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY app/ ./app/
COPY config.yaml .
COPY README.md .

# Create necessary directories
RUN mkdir -p models results logs

# Expose Streamlit port
EXPOSE 8501

# Default command (can be overridden)
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
