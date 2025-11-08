## Use the official TensorFlow image as a base
FROM python:3.11-slim-bullseye

# Set the working directory
WORKDIR /app

# Create a non-root user and grant ownership of the /app directory
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app

# Install system-level build dependencies
RUN apt-get update && apt-get install -y build-essential libffi-dev && rm -rf /var/lib/apt/lists/*

# Switch to the non-root user
USER appuser
ENV PATH="/home/appuser/.local/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy only the requirements file first to leverage Docker caching
COPY --chown=appuser:appuser requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=appuser:appuser . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application with Uvicorn
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]