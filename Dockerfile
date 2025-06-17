# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including OpenCV's libGL dependency)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    gcc \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy necessary files
COPY app.py /app/app.py
COPY best.pt /app/best.pt
COPY requirements.txt /app/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for Flask to use the correct port
ENV PORT=8080

# Expose the port Flask will run on
EXPOSE 8080

# Start the Flask app
CMD ["python", "app.py"]
