import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from jiwer import cer
import os

from src.dataset import ASRDataset, collate_fn, idx2char
from src.transformer_model import TransformerASR
from src.decoder import decode_logits
from src.decoder import beam_search_decode

from src.augment import SpecAugmentGPU


# --------------------
# Config
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 2          # Transformers are memory heavy
EPOCHS = 12
LR = 1e-4               # VERY important for Transformers
VAL_RATIO = 0.2

MANIFEST = "data/librispeech/dev_clean_manifest.csv"
BEST_MODEL_PATH = "best_model_transformer.pt"


# --------------------
# Dataset split
# --------------------
full_dataset = ASRDataset(MANIFEST)

val_size = int(len(full_dataset) * VAL_RATIO)
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)


# --------------------
# Model
# --------------------
num_classes = len(idx2char)

model = TransformerASR(vocab_size=num_classes).to(DEVICE)

if os.path.exists(BEST_MODEL_PATH):
    print("🔁 Loading existing Transformer checkpoint...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))


criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_cer = float("inf")



spec_augment = SpecAugmentGPU()


# --------------------
# Training loop
# --------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for features, transcripts, feat_lens, txt_lens in train_loader:
        features = features.to(DEVICE)
        features = spec_augment(features)
        transcripts = transcripts.to(DEVICE)

        # CTC expects lengths on CPU
        feat_lens = feat_lens.cpu()
        txt_lens = txt_lens.cpu()

        optimizer.zero_grad()

        logits = model(features, feat_lens)
        log_probs = logits.log_softmax(dim=-1)
        log_probs = log_probs.permute(1, 0, 2)  # (T, B, C)

        loss = criterion(log_probs, transcripts, feat_lens, txt_lens)
        loss.backward()

        # Critical for Transformer stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch} | Train Loss: {avg_loss:.4f}")

    # --------------------
    # Validation
    # --------------------
    model.eval()
    cer_scores = []

    with torch.no_grad():
        for features, transcripts, feat_lens, _ in val_loader:
            features = features.to(DEVICE)

            logits = model(features, feat_lens)
            log_probs = logits.log_softmax(dim=-1)

            pred_text = beam_search_decode(
                log_probs[0],
                idx2char,
                beam_width=10
            )

            true_text = "".join(
                [idx2char[i.item()] for i in transcripts[0] if i.item() != 0]
            )

            cer_scores.append(cer(true_text, pred_text))


    val_cer = sum(cer_scores) / len(cer_scores)
    print(f"Epoch {epoch} | Val CER: {val_cer:.4f}")

    # --------------------
    # Checkpoint
    # --------------------
    if val_cer < best_cer:
        best_cer = val_cer
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"✅ Saved new best Transformer model (CER={best_cer:.4f})")

    torch.cuda.empty_cache()

print("\nTransformer training complete.")
print(f"Best validation CER: {best_cer:.4f}")
