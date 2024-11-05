# Use an official Python runtime as a base image
FROM python:3.10.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /

# Copy requirements file and install dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 5000 for the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "main/app.py"]
# CMD ["uvicorn", "main.app:app", "--host", "0.0.0.0", "--port", "5000"]
