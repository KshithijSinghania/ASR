import gradio as gr
import torch
import tempfile

from src.infer import load_model, transcribe_audio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model_gru256.pt"

model = load_model(MODEL_PATH, DEVICE)


def asr_gradio(audio):
    if audio is None:
        return ""

    sr, waveform = audio

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        import soundfile as sf
        sf.write(f.name, waveform, sr)
        text = transcribe_audio(f.name, model, DEVICE, beam=True)

    return text


demo = gr.Interface(
    fn=asr_gradio,
    inputs=gr.Audio(type="numpy", label="Upload or record audio"),
    outputs=gr.Textbox(label="Transcription"),
    title="Custom ASR System (CTC + GRU)",
    description="End-to-end Automatic Speech Recognition demo"
)

demo.launch()
