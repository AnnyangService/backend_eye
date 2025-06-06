# Stage 1: Builder
FROM python:3.12 AS builder

# Set working directory
WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install PyTorch CPU version first (smaller size)
RUN pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim AS base

# Set working directory
WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Set common environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=run.py

# development stage
FROM base AS development
ENV FLASK_ENV=development
ENV FLASK_CONFIG=development
# Run the application using Python directly
CMD ["python", "run.py"]

# production stage
FROM base AS production
ENV FLASK_ENV=production
ENV FLASK_CONFIG=production
# Run the application using Gunicorn
CMD ["python", "run.py"]
