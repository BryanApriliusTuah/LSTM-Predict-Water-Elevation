FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
	build-essential \
	libglib2.0-0 \
	libsm6 \
	libxext6 \
	libxrender-dev \
	&& rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
