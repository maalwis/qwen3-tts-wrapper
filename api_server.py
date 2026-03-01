import io
import sys
import os
import glob
import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

# ── point at your local package ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# Use HF_HOME from environment if provided (Docker sets this), otherwise default.
HF_HOME = os.environ.get("HF_HOME", os.path.join(os.path.dirname(__file__), "hf-model-cache"))
os.environ["HF_HOME"] = HF_HOME

# Keep offline flags (or let Docker override)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# ── device & precision config ─────────────────────────────────────────────────
# Automatically use CUDA if available, fall back to CPU gracefully.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use bfloat16 on CUDA (stable on Ampere+ GPUs like A5000) or float32 on CPU.
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

print(f"[device] Using: {DEVICE}  |  dtype: {DTYPE}", flush=True)

# Allow overriding the model repo dir (inside HF cache) via env var
MODEL_REPO_DIR = os.environ.get(
    "QWEN_TTS_REPO_DIR",
    os.path.join(HF_HOME, "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"),
)

# If a snapshot is specified, use it; otherwise pick the newest snapshot folder.
snapshot_override = os.environ.get("QWEN_TTS_SNAPSHOT")
if snapshot_override:
    MODEL_PATH = os.path.join(MODEL_REPO_DIR, "snapshots", snapshot_override)
else:
    snapshots = sorted(glob.glob(os.path.join(MODEL_REPO_DIR, "snapshots", "*")))
    if not snapshots:
        raise RuntimeError(
            f"No snapshots found under {MODEL_REPO_DIR}. "
            f"Is hf-model-cache mounted correctly to HF_HOME={HF_HOME}?"
        )
    MODEL_PATH = snapshots[-1]

# ── app ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Qwen3-TTS API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

tts: Optional[Qwen3TTSModel] = None


@app.on_event("startup")
def load_model():
    global tts
    print(f"Loading Qwen3-TTS model on {DEVICE} ({DTYPE}) …", flush=True)

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        attn_implementation="eager",
        torch_dtype=DTYPE,          # load weights directly in target precision
    )

    # Move every sub-module to the target device
    tts.model.to(device=DEVICE, dtype=DTYPE)

    # Also move the speech tokenizer codec if it exposes a .to() interface
    if hasattr(tts, "speech_tokenizer") and hasattr(tts.speech_tokenizer, "to"):
        tts.speech_tokenizer.to(device=DEVICE, dtype=DTYPE)

    tts.model.eval()

    if DEVICE.type == "cuda":
        allocated = torch.cuda.memory_allocated(DEVICE) / 1024 ** 3
        reserved  = torch.cuda.memory_reserved(DEVICE)  / 1024 ** 3
        print(
            f"Model loaded. VRAM: {allocated:.2f} GB allocated / "
            f"{reserved:.2f} GB reserved.",
            flush=True,
        )
    else:
        print("Model loaded on CPU.", flush=True)


# ── helpers ──────────────────────────────────────────────────────────────────
def wav_to_bytes(audio: np.ndarray, sample_rate: int) -> io.BytesIO:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    info = {
        "status": "ok",
        "model_loaded": tts is not None,
        "device": str(DEVICE),
        "dtype": str(DTYPE),
    }
    if DEVICE.type == "cuda":
        info["vram_allocated_gb"] = round(torch.cuda.memory_allocated(DEVICE) / 1024 ** 3, 2)
        info["vram_reserved_gb"]  = round(torch.cuda.memory_reserved(DEVICE)  / 1024 ** 3, 2)
        info["cuda_device_name"]  = torch.cuda.get_device_name(DEVICE)
    return info


@app.post("/tts/clone")
async def tts_clone(
    text: str = Form(...),
    ref_text: str = Form(""),
    language: str = Form("Auto"),
    x_vector_only: bool = Form(False),
    temperature: float = Form(0.9),
    top_p: float = Form(1.0),
    max_new_tokens: int = Form(2048),
    ref_audio: UploadFile = File(...),
):
    """
    Voice-clone endpoint — requires a reference WAV file.

    Form fields:
      text          : text to synthesise
      ref_text      : transcript of ref_audio  (required when x_vector_only=false)
      language      : e.g. "Auto", "English", "Chinese"
      x_vector_only : set true to clone voice without ref_text transcript
      temperature   : sampling temperature
      top_p         : nucleus sampling
      max_new_tokens: max codec tokens
      ref_audio     : WAV file (multipart/form-data)
    """
    if tts is None:
        raise HTTPException(503, "Model not loaded yet")
    if not text.strip():
        raise HTTPException(400, "text cannot be empty")
    if not x_vector_only and not ref_text.strip():
        raise HTTPException(400, "ref_text is required when x_vector_only=false")

    audio_bytes = await ref_audio.read()

    import tempfile, pathlib
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # autocast keeps generation in bfloat16 on GPU for speed & VRAM savings;
        # on CPU it is a no-op (enabled=False).
        autocast_ctx = torch.amp.autocast(
            device_type=DEVICE.type,
            dtype=DTYPE,
            enabled=(DEVICE.type == "cuda"),
        )
        with autocast_ctx:
            wavs, fs = tts.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=tmp_path,
                ref_text=ref_text if not x_vector_only else None,
                x_vector_only_mode=x_vector_only,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)

    buf = wav_to_bytes(wavs[0], fs)
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="output.wav"'},
    )


@app.get("/info")
def info():
    """Returns supported languages / speakers if exposed by the model."""
    if tts is None:
        raise HTTPException(503, "Model not loaded yet")
    return {
        "model_type": getattr(tts.model, "tts_model_type", "unknown"),
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "supported_languages": tts.get_supported_languages(),
        "supported_speakers": tts.get_supported_speakers(),
    }