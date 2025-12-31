# Automatic Speech Recognition (ASR) System — From Scratch

An end-to-end **Automatic Speech Recognition (ASR)** system built **from scratch using PyTorch**, covering the complete pipeline from raw audio preprocessing to model training, evaluation, and a live **Gradio web demo**.\
The project also includes **transfer learning with OpenAI Whisper** for comparison and advanced performance.

---

##   Project Highlights

- End-to-end ASR pipeline (audio → text)
- Log-Mel spectrogram based acoustic modeling
- CNN + BiGRU + CTC architecture
- Character Error Rate (CER) & Word Error Rate (WER) evaluation
- Model checkpointing & validation
- Greedy + Beam Search CTC decoding
- Interactive **Gradio web application**
- Whisper-based zero-shot transcription for comparison

---

##  Model Architecture

```
Audio (.wav/.flac)
   ↓
Log-Mel Spectrogram
   ↓
2D Convolutional Layers (feature extraction)
   ↓
Bidirectional GRU (sequence modeling)
   ↓
Fully Connected Layer
   ↓
CTC Loss
   ↓
Text Transcription
```

---

##  Project Structure

```
ASR/
├── src/
│   ├── audio.py          # Audio loading & preprocessing
│   ├── dataset.py        # PyTorch Dataset & Collate Function
│   ├── model.py          # CNN + BiGRU ASR model
│   ├── train.py          # Training & validation loop
│   ├── infer.py          # Inference utilities
│   ├── decoder.py        # Greedy CTC decoder
│   ├── beam_decoder.py   # Beam search decoder
│
├── notebooks/
│   ├── 01_data_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│   ├── 04_evaluation.ipynb
│
├── app.py                # Gradio demo app
├── requirements.txt
├── README.md
```

---

##  Dataset

- **LibriSpeech (dev-clean subset)**
- Sampling rate: **16 kHz**
- Feature type: **80-dim Log-Mel Spectrogram**
- Vocabulary:
  ```
  A–Z, space, apostrophe + CTC blank token
  ```

---

##  Performance (Custom ASR)

| Metric           | Value                    |
| ---------------- | ------------------------ |
| CER (Validation) | \~0.54                   |
| WER (Validation) | \~0.90                   |
| Model            | CNN + BiGRU (512 hidden) |

> Note: Performance improves with longer training and language model decoding.

---

## Installation & Setup

### Clone Repository

```bash
git clone https://github.com/<your-username>/asr-from-scratch.git
cd asr-from-scratch
```

---

### Create Virtual Environment

```bash
python -m venv asr_env
```

Activate:

**Windows**

```bash
asr_env\Scripts\activate
```

**Linux / macOS**

```bash
source asr_env/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

> If using GPU, make sure CUDA-compatible PyTorch is installed.

---

### Download Dataset (LibriSpeech)

Download and extract `dev-clean` from:

```
http://www.openslr.org/12
```

Place it under:

```
data/librispeech/
```

---

## Training the ASR Model

```bash
python -m src.train
```

What happens:

- Loads dataset
- Trains model with CTC loss
- Runs validation after each epoch
- Saves best model automatically (`best_model_512.pt`)

---

## Inference (Command-line)

```python
from src.infer import transcribe_audio

text = transcribe_audio("sample.wav")
print(text)
```

---

## Run Gradio Demo App

```bash
python app.py
```

Open browser at:

```
http://127.0.0.1:7860
```

Features:

- Upload audio file
- Live transcription
- End-to-end inference

---

## Whisper (Transfer Learning)

The project also demonstrates **zero-shot transcription** using **OpenAI Whisper** via Hugging Face Transformers.

Advantages:

- Much lower CER/WER
- Strong language modeling
- Production-ready baseline

Used for:

- Performance comparison
- Advanced decoding reference

---

## Future Improvements

- Language Model (KenLM) integration
- Beam search with LM scoring
- Whisper fine-tuning
- Streaming ASR
- Transformer-based acoustic model

---

## Author

**Kshithij Singhania**\
Electrical Engineering @ IIT Indore\
Interests: Machine Learning, Speech Processing, Deep Learning, Systems

---

## Acknowledgements

- LibriSpeech Dataset
- PyTorch
- Hugging Face Transformers
- OpenAI Whisper
- Gradio

---

## License

This project is for educational and research purposes.
