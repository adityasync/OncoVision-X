# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary project files (avoiding huge data directories)
COPY app.py .
COPY demo.py .
COPY predict.py .
COPY requirements.txt .
COPY configs/ ./configs/
COPY frontend/ ./frontend/
COPY src/ ./src/
COPY pretrained/ ./pretrained/
COPY experiments/full_model/checkpoints/best.pth ./experiments/full_model/checkpoints/best.pth

# Ensure other expected directories exist
RUN mkdir -p logs reports results

# Set permissions for Hugging Face Spaces
RUN chmod -R 777 /app

# Expose the port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
