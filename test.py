import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import speech_recognition as sr
from PIL import Image
import pytesseract
image_path = "data/images/captcha_0001.png"
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    print(f"Raw Extracted Text: {repr(text)}")  # Show the raw output with whitespace
    return text.strip()
import cv2
import numpy as np

def extract_text_from_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)  # Apply thresholding
    image = Image.fromarray(image)
    
    text = pytesseract.image_to_string(image, config="--psm 6")  # OCR processing
    print(f"Raw Extracted Text: {repr(text)}")
    
    return text.strip()
from PIL import Image

 # Open image to check visibility
print(extract_text_from_image(image_path))