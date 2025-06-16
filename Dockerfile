# Use official Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app code
COPY . .

# Expose the port Flask will use (Cloud Run expects 8080)
EXPOSE 8080

# Start the Flask app
CMD ["python", "app.py"]
