# Use an official Python runtime as a base image
FROM python:3.10.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create a directory for the app
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port your app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "main\app.py"]
