import os
import torch
import librosa
import cv2
import pytesseract
from PIL import Image
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm  

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set Tesseract OCR path (change as needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Load Wav2Vec2 model
processor = Wav2Vec2Processor.from_pretrained("models/facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("models/facebook/wav2vec2-large-960h").to(device)

# Load and preprocess audio
def load_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr)
    audio = librosa.util.normalize(audio)  # Normalize
    return audio

# Convert audio to text using Wav2Vec2
def transcribe_audio(file_path):
    audio = load_audio(file_path)
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.strip()

# Extract text from image using OCR
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    text = pytesseract.image_to_string(binary, config="--psm 7")
    return text.strip()

# Define vocabulary mapping for text (Add more characters based on your needs)
vocab = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")}
vocab["<unk>"] = len(vocab)  # "<unk>" to the vocab for unknown characters

# Padding function for sequences
def pad_sequence(sequence, max_length):
    return F.pad(sequence, (0, max_length - len(sequence)), value=0)  # Pad with 0s (or another value)

# Dataset class
class CaptchaDataset(Dataset):
    def __init__(self, audio_dir, image_dir, vocab, max_audio_len=100, max_image_len=50):
        self.audio_files = sorted(os.listdir(audio_dir))
        self.image_files = sorted(os.listdir(image_dir))
        self.audio_dir = audio_dir
        self.image_dir = image_dir
        self.vocab = vocab
        self.max_audio_len = max_audio_len
        self.max_image_len = max_image_len

    def __len__(self):
        return len(self.audio_files)

    def text_to_sequence(self, text):
        return [self.vocab.get(char, self.vocab["<unk>"]) for char in text]

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        image_path = os.path.join(self.image_dir, self.image_files[idx])

        audio_text = transcribe_audio(audio_path)
        image_text = extract_text_from_image(image_path)

        audio_sequence = self.text_to_sequence(audio_text)
        image_sequence = self.text_to_sequence(image_text)

        # Pad sequences to the max length
        audio_sequence = pad_sequence(torch.tensor(audio_sequence, dtype=torch.float32), self.max_audio_len)
        image_sequence = pad_sequence(torch.tensor(image_sequence, dtype=torch.float32), self.max_image_len)

        # Ensure input shape matches the expected format: [batch_size, channels, length]
      
        audio_sequence = audio_sequence.unsqueeze(0)  # channel dimension
        image_sequence = image_sequence.unsqueeze(0)  #  channel dimension

        return audio_sequence, image_sequence

# Define CRNN model (CNN + RNN for captcha decoding)
class CRNN(nn.Module):
    def __init__(self, input_size=100, hidden_size=256, num_classes=len(vocab)):
        super(CRNN, self).__init__()
        self.cnn = nn.Conv1d(1, 100, kernel_size=3, padding=1)  # Match LSTM input size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.cnn(x))  # CNN output: (batch, channels=100, seq_len)
        x = x.permute(0, 2, 1)  # Change to (batch, seq_len, channels) for LSTM
        x, _ = self.rnn(x)  # LSTM processing
        x = self.fc(x)  # Change here: output shape should be (batch, seq_len, num_classes)
        return x

def main():
    # Initialize model
    model = CRNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load dataset
    dataset = CaptchaDataset("data/audio", "data/images", vocab)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0
        start_time = time.time()

        #  tqdm for progress bar
        for audio_text, image_text in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            optimizer.zero_grad()

            # Move tensors to device
            audio_text = audio_text.to(device)
            image_text = image_text.to(device).long()  # Ensure target tensor is of type long

            # Forward pass: Get model outputs
            outputs = model(audio_text)  # Shape: [batch_size, seq_length, num_classes]

            # Reshape the target tensor to match the output shape
            batch_size = outputs.shape[0]
            seq_length = outputs.shape[1]
            num_classes = outputs.shape[2]

            image_text = pad_sequence(image_text.view(-1), seq_length * batch_size).view(batch_size, seq_length)

            # Ensure the shapes match
            assert outputs.shape[0] == image_text.shape[0], f"Mismatch: {outputs.shape[0]} != {image_text.shape[0]}"
            assert outputs.shape[1] == image_text.shape[1], f"Mismatch: {outputs.shape[1]} != {image_text.shape[1]}"

            # Compute the loss using cross-entropy
            outputs = outputs.view(-1, num_classes)  # Flatten: [batch_size * seq_length, num_classes]
            image_text = image_text.view(-1)  # Flatten target to [batch_size * seq_length]

            # Compute the loss using cross-entropy
            loss = criterion(outputs, image_text)  # Now both are 1D tensors: [batch_size * seq_length]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy calculation
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == image_text).sum().item()
            total += image_text.size(0)

        train_accuracy = (correct / total) * 100
        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Time: {epoch_time:.2f}s")

        # Save model checkpoint
        torch.save(model.state_dict(), f'model_checkpoint_epoch_{epoch+1}.pth')

    # Final accuracy evaluation
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for audio_text, image_text in dataloader:
            audio_text = audio_text.to(device)
            image_text = image_text.to(device).squeeze(1).long()

            outputs = model(audio_text)
            predicted = torch.argmax(outputs, dim=2)  # Adjust here to match 3D output

            correct += (predicted == image_text).sum().item()
            total += image_text.size(0) * image_text.size(1)  # Adjust to account for sequence length

    accuracy = (correct / total) * 100
    print(f"Final Accuracy: {accuracy:.2f}%")

    # Save the final model
    torch.save(model.state_dict(), 'final_model.pth')

if __name__ == "__main__":
    main()
