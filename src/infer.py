import torch
import numpy as np

from src.model import ASRModel
from src.audio import extract_logmel_normalized
from src.dataset import idx2char
from src.beam_decoder import ctc_beam_search
from src.decoder import decode_logits


def load_model(model_path, device):
    model = ASRModel(len(idx2char)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def transcribe_audio(audio_path, model, device, beam=True):
    features = extract_logmel_normalized(audio_path)
    features = torch.tensor(features, dtype=torch.float32).transpose(0, 1)
    features = features.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(features)
        log_probs = logits.log_softmax(dim=-1)

    if beam:
        text = ctc_beam_search(
            log_probs[0].cpu(),
            idx2char,
            beam_width=5
        )
    else:
        text = decode_logits(log_probs[0], idx2char)

    return text
