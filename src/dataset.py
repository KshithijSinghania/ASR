import pandas as pd
import torch
from torch.utils.data import Dataset

from src.audio import extract_logmel_normalized


# --------------------
# Vocabulary (CTC)
# --------------------
VOCAB = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ '")
BLANK_TOKEN = "<blank>"

char2idx = {c: i + 1 for i, c in enumerate(VOCAB)}
char2idx[BLANK_TOKEN] = 0

idx2char = {i: c for c, i in char2idx.items()}


# --------------------
# Text preprocessing
# --------------------
def normalize_text(text: str) -> str:
    return text.upper()


def filter_text(text: str) -> str:
    return "".join([c for c in text if c in char2idx])


def text_to_int(text: str):
    text = normalize_text(text)
    text = filter_text(text)
    return [char2idx[c] for c in text]


def int_to_text(indices):
    return "".join([idx2char[i] for i in indices if i != 0])


# --------------------
# Dataset
# --------------------
class ASRDataset(Dataset):
    def __init__(self, manifest_path):
        self.df = pd.read_csv(manifest_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Audio → features
        features = extract_logmel_normalized(row["audio_path"])
        features = torch.tensor(features, dtype=torch.float32).transpose(0, 1)
        # (T, 80)

        # Text → integers
        transcript = torch.tensor(
            text_to_int(row["transcript"]),
            dtype=torch.long
        )

        return features, transcript


# --------------------
# Collate function
# --------------------
def collate_fn(batch):
    features, transcripts = zip(*batch)

    feature_lengths = torch.tensor([f.shape[0] for f in features])
    transcript_lengths = torch.tensor([t.shape[0] for t in transcripts])

    features_padded = torch.nn.utils.rnn.pad_sequence(
        features, batch_first=True
    )
    transcripts_padded = torch.nn.utils.rnn.pad_sequence(
        transcripts, batch_first=True
    )

    return (
        features_padded,
        transcripts_padded,
        feature_lengths,
        transcript_lengths,
    )
