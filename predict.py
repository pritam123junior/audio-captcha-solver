import torch
import os
import librosa
import cv2
import pytesseract
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from Levenshtein import ratio  # For better accuracy calculation
from transformers import Wav2Vec2ProcessorWithLM
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained("models/facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("models/facebook/wav2vec2-large-960h").to(device)

# Load trained CRNN model
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
crnn_model.load_state_dict(torch.load('model_checkpoint_epoch_4.pth'))
crnn_model.eval()

# Load and preprocess audio with noise reduction
def load_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr)
    audio = librosa.effects.preemphasis(audio)  # Apply noise reduction
    audio = librosa.util.normalize(audio)
    return audio

# Transcribe audio
import re

def transcribe_audio(file_path):
    audio = load_audio(file_path)
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    # Only keep alphanumeric characters (remove words like 'SMALL', 'CAPITAL')
   
    return transcription.strip()

# Extract text from image using OCR
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    text = pytesseract.image_to_string(binary, config="--psm 7")

    # Normalize text: remove spaces and convert to uppercase
    return text.strip()

# Predict function
def predict(audio_file, image_file):
    transcribed_text = transcribe_audio(audio_file)
    ground_truth_text = extract_text_from_image(image_file)
    
    print(f"Predicted Text: {transcribed_text}")
    print(f"Ground Truth: {ground_truth_text}")
    
    # Use Levenshtein Ratio for better accuracy calculation
    accuracy = ratio(transcribed_text, ground_truth_text) * 100
    
    print(f"Character-Level Accuracy: {accuracy:.2f}%")

# Test prediction
audio_path = "data/audio/captcha_0003.wav"
image_path = "data/images/captcha_0003.png"


predict(audio_path, image_path)
