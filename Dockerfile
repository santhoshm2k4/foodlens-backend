# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies, including Tesseract OCR
# Update package list, install tesseract, then clean up apt cache to keep image small
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on (Render usually maps this automatically)
EXPOSE 8000

# Command to run the application using Uvicorn
# Render's $PORT variable will be automatically available inside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]