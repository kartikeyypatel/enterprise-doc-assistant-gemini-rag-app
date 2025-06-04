# Start with an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if any are needed, e.g., for certain Python packages)
# For now, we assume most are handled by pip. If you encounter missing system libs later, add them here.
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# This includes main_api.py and the 'docs' folder if you want to include
# initial documents for population if the Pinecone index is empty.
# For a production setup, the 'docs' for initial population might be handled
# by a separate ingestion job, but for testing this container, it's okay.
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# We use 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]