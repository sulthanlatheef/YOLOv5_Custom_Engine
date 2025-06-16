# Use official Python base image
# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev gcc

# Copy necessary files
COPY app.py /app/app.py
COPY best.pt /app/best.pt
COPY requirements.txt /app/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Cloud Run expects
ENV PORT=8080

# Start the Flask app
CMD ["python", "app.py"]
