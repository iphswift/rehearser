# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    build-essential \
    espeak \
    libespeak-dev \
    wget \
    libsndfile1 \
    libmagic1 \
    unzip \
    libsndfile1-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir numpy==1.21.2
RUN pip install --no-cache-dir -r requirements.txt

# Set NLTK data environment and download resources
ENV NLTK_DATA /app/nltk_data
RUN python -m nltk.downloader -d /app/nltk_data averaged_perceptron_tagger
RUN python -m nltk.downloader -d /app/nltk_data cmudict

# Download the pre-trained TTS model from Zenodo (for example, FastSpeech2)
# You can replace this URL with the actual URL of the model you need
RUN python -c "\
from espnet_model_zoo.downloader import ModelDownloader; \
d = ModelDownloader(); \
d.download('kan-bayashi/ljspeech_fastspeech2')"

# Copy the entire app directory and app.py into the container
COPY . /app

# Expose port 5000 for the Flask app
EXPOSE 5000

# Command to run on container start
CMD ["flask", "run", "--host=0.0.0.0"]
