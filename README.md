# Audio Processing with Whisper large-v2

This repository contains a speech-to-text and emotion detection system using the Whisper large-v2 model via Hugging Face Transformers. It can transcribe audio, detect the language (English or Hindi), translate Hindi to English, and classify emotions.

## Features

- Transcription using Whisper large-v2 model
- Language detection (English/Hindi)
- Translation of Hindi speech to English text
- Emotion detection using pre-trained MLP classifiers
- Persistent model caching to avoid repeated downloads
- Optimized for GitHub Codespaces

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- Trained emotion classifier models:
  - `mlp_classifier_hindi.model`
  - `mlp_classifier_english.model`

## Setup Instructions for GitHub Codespaces

### 1. Create a New Codespace

1. Open your GitHub repository
2. Click the "Code" button
3. Select the "Codespaces" tab
4. Click "Create codespace on main"

### 2. Run Setup Script

Once your Codespace is ready, run the following commands in the terminal:

```bash
chmod +x setup.sh
sudo ./setup.sh
```

This will:
- Install system dependencies (including FFmpeg)
- Install required Python packages
- Pre-download and cache the Whisper large-v2 model

### 3. Upload Emotion Classifier Models

Upload your trained emotion classifier models to the root directory of your Codespace:
- `mlp_classifier_hindi.model`
- `mlp_classifier_english.model`

You can do this by dragging and dropping files into the Explorer panel in VS Code, or using `curl` to download them from a source.

### 4. Run the Audio Processor

```bash
python audio_processor.py --audio path/to/your/audio.wav
```

If you don't provide an audio file path, the script will prompt you to enter one.

## Command Line Arguments

- `--audio`: Path to the audio file you want to process
- `--reload`: Force reload all models (use if you encounter caching issues)

Example:
```bash
python audio_processor.py --audio samples/recording.wav --reload
```

## System Requirements

- **Memory**: At least 8GB RAM recommended for Whisper large-v2
- **Disk Space**: At least 2GB free space for model storage
- **Processing**: CPU is sufficient but GPU will significantly improve performance

## Project Structure

```
├── audio_processor.py     # Main script
├── requirements.txt       # Python dependencies
├── setup.sh               # Setup script for Codespaces
├── .devcontainer/         # Codespaces configuration
│   └── devcontainer.json  # Devcontainer configuration
├── mlp_classifier_hindi.model    # Pre-trained model (you must provide)
└── mlp_classifier_english.model  # Pre-trained model (you must provide)
```

## Technical Details

### Models Used

- **Speech Recognition**: Whisper large-v2 (Transformers implementation)
- **Translation**: Helsinki-NLP/opus-mt-hi-en
- **Emotion Detection**: Custom MLP classifiers (Hindi and English)

### Audio Feature Extraction

The system extracts the following features for emotion detection:
- MFCC (Mel-frequency cepstral coefficients)
- Chroma
- Mel Spectrogram

## Troubleshooting

### Model Download Issues

If you encounter issues downloading the models, you can manually trigger downloads:

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
```

### Out of Memory Errors

If you encounter out of memory errors, consider:
1. Switching to a smaller Whisper model (medium or small)
2. Upgrading your Codespace to a larger machine type
3. Processing shorter audio clips

### Audio Format Issues

If your audio isn't being processed correctly:
1. Ensure FFmpeg is installed correctly
2. Convert your audio to WAV format with: `ffmpeg -i input.mp3 output.wav`
3. Try resampling to 16kHz: `ffmpeg -i input.wav -ar 16000 output_16k.wav`

## Files Included

### audio_processor.py

The main Python script that:
- Loads models only once (cached)
- Transcribes audio using Whisper large-v2
- Detects language and emotion
- Provides management responses

### requirements.txt

```
torch>=2.0.0
transformers>=4.30.0
datasets
librosa
soundfile
numpy
joblib
sentencepiece
protobuf
accelerate
```

### setup.sh

```bash
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
```

### .devcontainer/devcontainer.json

```json
{
    "name": "Audio Processing Environment",
    "image": "mcr.microsoft.com/devcontainers/python:3.10",
    "features": {},
    "postCreateCommand": "sudo apt-get update && sudo apt-get install -y ffmpeg && pip install -r requirements.txt",
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-python.vscode-pylance"
            ]
        }
    },
    "mounts": [
        "source=whisper-cache,target=/home/vscode/.cache/huggingface,type=volume"
    ]
}
```
