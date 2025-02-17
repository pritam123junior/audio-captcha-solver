import torch
import os
import librosa
import cv2
import pytesseract
import re 
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ForCTC
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from Levenshtein import ratio  # For accuracy calculation
import numpy as np
import webrtcvad

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained(
    "models/facebook/wav2vec2-large-960h"
)
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)


#  Define CRNN with CTCLoss
class CRNN(nn.Module):
    def __init__(self, input_size=100, hidden_size=256, num_classes=63):
        super(CRNN, self).__init__()
        self.cnn = nn.Conv1d(1, 100, kernel_size=3, padding=1)
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.cnn(x))
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# Load trained CRNN model
crnn_model = CRNN().to(device)
crnn_model.load_state_dict(torch.load('model_checkpoint_epoch_10.pth'))
crnn_model.eval()

#  Data Augmentation (Pitch Shift & Time Stretch)
def augment_audio(audio, sr=16000):
    audio = librosa.effects.time_stretch(audio, rate=0.9)  # Slow down
    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)  # Shift pitch
    return audio

# Preprocess audio with noise reduction
def preprocess_audio(audio_file):
    audio, sr = librosa.load(audio_file, sr=16000)
    audio = augment_audio(audio, sr)

    vad = webrtcvad.Vad()
    frame_duration = 10  # ms
    frames = []
    for i in range(0, len(audio), int(sr * frame_duration / 1000)):
        frame = audio[i:i + int(sr * frame_duration / 1000)]
        if len(frame) < int(sr * frame_duration / 1000):
            frame = np.pad(frame, (0, int(sr * frame_duration / 1000) - len(frame)), 'constant')
        frames.append(frame)

    processed_audio = []
    for frame in frames:
        frame_bytes = (frame * 32768).astype(np.int16).tobytes()
        is_speech = vad.is_speech(frame_bytes, sr)
        if is_speech:
            processed_audio.extend(frame)

    processed_audio = np.array(processed_audio)
    if len(processed_audio) == 0:
        return np.zeros(16000)

    processed_audio = librosa.util.normalize(processed_audio)
    return processed_audio

# Transcribe audio (Remove extra letters, Normalize case)
def transcribe_audio(file_path):
    audio = preprocess_audio(file_path)
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    transcription = re.sub(r'\b(?:SMALL|CAPITAL)\b', '', transcription, flags=re.IGNORECASE)  # Remove extra words
    transcription = re.sub(r'[^a-zA-Z0-9]', '', transcription)  # Keep only alphanumeric

    # Remove extra repeating characters
    transcription = re.sub(r'(.)\1+', r'\1', transcription)

    return transcription.strip().upper()  #  Normalize case

# Extract text from image using OCR
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 --psm 7'
    text = pytesseract.image_to_string(binary, config=custom_config)

    return re.sub(r'[^a-zA-Z0-9]', '', text.strip()) #  Normalize case

# Predict function
def predict(audio_file, image_file):
    transcribed_text = transcribe_audio(audio_file)
    ground_truth_text = extract_text_from_image(image_file)

    print(f" Predicted Text: {transcribed_text}")
    print(f" Ground Truth: {ground_truth_text}")

    accuracy = ratio(transcribed_text, ground_truth_text) * 100
    print(f"Character-Level Accuracy: {accuracy:.2f}%")

# Test prediction
audio_path = "data/audio/captcha_0031.wav"
image_path = "data/images/captcha_0031.png"

predict(audio_path, image_path)
