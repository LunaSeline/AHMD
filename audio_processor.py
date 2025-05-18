import torch
import numpy as np
import os
import json
import random
import joblib
import argparse
import librosa
import soundfile
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

# Global variables for models
WHISPER_PROCESSOR = None
WHISPER_MODEL = None
TRANSLATOR = None
HINDI_EMOTION_MODEL = None
ENGLISH_EMOTION_MODEL = None

def load_models(force_reload=False, model_size="medium"):
    """Load all required models only once."""
    global WHISPER_PROCESSOR, WHISPER_MODEL, TRANSLATOR, HINDI_EMOTION_MODEL, ENGLISH_EMOTION_MODEL
    
    # Check if models are already loaded
    if WHISPER_PROCESSOR is None or WHISPER_MODEL is None or force_reload:
        model_name = f"openai/whisper-{model_size}"
        print(f"Loading Whisper {model_size} model...")
        WHISPER_PROCESSOR = WhisperProcessor.from_pretrained(model_name)
        WHISPER_MODEL = WhisperForConditionalGeneration.from_pretrained(model_name)
        WHISPER_MODEL.config.forced_decoder_ids = None
    
    if TRANSLATOR is None or force_reload:
        print("Loading translator model...")
        TRANSLATOR = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")
    
    try:
        if HINDI_EMOTION_MODEL is None or force_reload:
            print("Loading Hindi emotion model...")
            HINDI_EMOTION_MODEL = joblib.load("mlp_classifier_hindi.model")
        
        if ENGLISH_EMOTION_MODEL is None or force_reload:
            print("Loading English emotion model...")
            ENGLISH_EMOTION_MODEL = joblib.load("mlp_classifier_english.model")
    except FileNotFoundError:
        print("Warning: Emotion classifier models not found. Emotion detection will be disabled.")
        # Create dummy emotion models that return "unknown" for testing
        class DummyModel:
            def predict(self, X):
                return ["unknown"]
        
        HINDI_EMOTION_MODEL = DummyModel()
        ENGLISH_EMOTION_MODEL = DummyModel()

def transcribe(wav_file):
    """Transcribes audio using Whisper and detects language."""
    # Ensure models are loaded
    if WHISPER_MODEL is None or WHISPER_PROCESSOR is None:
        load_models()
    
    try:
        # Load audio file using librosa
        print(f"Loading audio file: {wav_file}")
        audio, sample_rate = librosa.load(wav_file, sr=16000)
        
        # Process audio for Whisper model
        print("Processing audio features...")
        input_features = WHISPER_PROCESSOR(
            audio, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).input_features
        
        # Generate transcription
        print("Generating transcription...")
        predicted_ids = WHISPER_MODEL.generate(input_features)
        transcription = WHISPER_PROCESSOR.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        print(f"Raw transcription: {transcription}")
        
        # Detect language based on Unicode character ranges
        detected_language = "Hindi" if any('\u0900' <= char <= '\u097F' for char in transcription) else "English"

        if detected_language == "Hindi":
            print("Translating Hindi to English...")
            english_translation = TRANSLATOR(transcription)[0]['translation_text']
            return detected_language, transcription, english_translation
        return detected_language, transcription, None
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        # Return placeholder values for testing
        return "Unknown", "Error in transcription", None

def extract_features(file_name):
    """
    Extracts MFCC, Chroma, and Mel Spectrogram features (matches training process).
    """
    try:
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate

            # Compute Short-Term Fourier Transform (STFT) if required
            stft = np.abs(librosa.stft(X))

            # Initialize feature vector
            result = np.array([])

            # Extract MFCC
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

            # Extract Chroma features
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))

            # Extract Mel Spectrogram features
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))

        return np.expand_dims(result, axis=0)  # Match input shape for model
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        # Return placeholder features for testing
        return np.zeros((1, 128))  # Dummy features

def detect_emotion(audio_file, language):
    """
    Predicts emotion using the trained MLP model based on the detected language.
    """
    try:
        # Ensure models are loaded
        if HINDI_EMOTION_MODEL is None or ENGLISH_EMOTION_MODEL is None:
            load_models()
            
        features = extract_features(audio_file)
        model = HINDI_EMOTION_MODEL if language == "Hindi" else ENGLISH_EMOTION_MODEL
        emotion_label = model.predict(features)[0]  # Extract single prediction

        return emotion_label
    except Exception as e:
        print(f"Error in emotion detection: {str(e)}")
        return "unknown"

def process_audio(audio_file):
    """Runs transcription, emotion detection, and emotion management pipeline."""
    print("\nProcessing audio file...")

    # Step 1: Transcription and Translation
    language, transcription, translation = transcribe(audio_file)
    if language == "Unknown":
        print("Error: Could not process audio.")
        return None

    # Step 2: Emotion Detection
    emotion = detect_emotion(audio_file, language)

    # Print Results
    print("\n-----RESULTS-----")
    print(f"Detected Language: {language}")
    print(f"Transcription: {transcription}")
    if translation:
        print(f"Translation: {translation}")
    print(f"Emotion: {emotion}")

    return language, transcription, translation, emotion

def main():
    parser = argparse.ArgumentParser(description='Audio processing with emotion detection')
    parser.add_argument('--audio', type=str, default=None, help='Path to audio file')
    parser.add_argument('--reload', action='store_true', help='Force reload all models')
    parser.add_argument('--model-size', type=str, default="medium", 
                        choices=["tiny", "base", "small", "medium"], 
                        help='Whisper model size (default: medium)')
    args = parser.parse_args()
    
    # Load models once at startup
    load_models(force_reload=args.reload, model_size=args.model_size)
    
    if args.audio:
        audio_path = args.audio
    else:
        audio_path = input("Enter the path to the audio file: ")
    
    process_audio(audio_path)

if __name__ == "__main__":
    main()
