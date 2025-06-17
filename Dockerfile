# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including OpenCV's libGL dependency and git for cloning)
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    gcc \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Clone YOLOv5 locally and install its dependencies
RUN git clone --depth 1 https://github.com/ultralytics/yolov5.git /app/yolov5 \
 && pip install --no-cache-dir -r /app/yolov5/requirements.txt

# Copy application code and model
COPY app.py best.pt requirements.txt /app/

# Install remaining Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for Flask to use the correct port
ENV PORT=8080

# Expose the port Flask will run on
EXPOSE 8080

# Start the Flask app
CMD ["python", "app.py"]
