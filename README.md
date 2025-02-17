# Audio Captcha Solver

Project Title

Audio Captcha Recognizer using Wav2Vec2 and CRNN

Description

This project focuses on recognizing and decoding audio captchas using a deep learning approach. It leverages Wav2Vec2 for speech-to-text transcription and CRNN (CNN + RNN) for text-based captcha recognition. Additionally, Tesseract OCR is used for text extraction from image-based captchas.

Features

*Audio Captcha Processing: Uses Wav2Vec2 for transcription.
*Image Captcha Decoding: Utilizes Tesseract OCR for text extraction.
*Preprocessing: Noise reduction, normalization, and augmentation techniques for robust feature extraction.
*Deep Learning Model: A CRNN-based model for accurate captcha decoding.
*Accuracy Calculation: Uses Levenshtein distance to compare predictions with ground truth.

## Setup

1.  **Clone the repository (or extract the zip file):**
    ```bash
    git clone [https://github.com/yourusername/audio-captcha-solver.git](https://github.com/yourusername/audio-captcha-solver.git)  # Or unzip the archive
    cd audio-captcha-solver
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv .venv  # Create a virtual environment
    source .venv/bin/activate  # Activate the environment (Linux/macOS)
    .venv\Scripts\activate  # Activate the environment (Windows)
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Data Preparation:**
    *   Place your audio captcha files (.wav) in the `data/audio` directory.
    *   Place the corresponding image captcha files (.png) in the `data/image` directory.
    *   **Crucially:** Ensure that the filenames of the audio and image files match (e.g., `0001.wav` and `0001.png`). The image filename *must* contain the correct captcha text.

## Usage

### Training the model

```bash
python train.py

### for check the model prediction
python prdict.py