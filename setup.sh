#!/bin/bash

echo "Setting up audio processing environment..."

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1

# Clean up cache
echo "Cleaning up cache..."
rm -rf ~/.cache/huggingface/hub/models--openai--whisper-large-v2

# Create directories for models
mkdir -p ~/.cache/huggingface

# Install Python dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Pre-download the Whisper medium model (smaller than large-v2)
echo "Pre-downloading Whisper medium model..."
python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; processor = WhisperProcessor.from_pretrained('openai/whisper-medium'); model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')"

echo "Setup complete! You can now run: python audio_processor.py --audio your_audio_file.wav"
