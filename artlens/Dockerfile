FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# Install system dependencies for OpenCV and scikit-image
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_spaces.txt .
RUN pip install --no-cache-dir -r requirements_spaces.txt

COPY app.py .

# HuggingFace Spaces runs on port 7860
EXPOSE 7860

CMD ["python", "app.py"]