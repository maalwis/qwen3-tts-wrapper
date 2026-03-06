# api_server.py
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

HF_HOME = os.environ.get("HF_HOME", os.path.join(os.path.dirname(__file__), "hf-model-cache"))
os.environ["HF_HOME"] = HF_HOME
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# ── model paths ───────────────────────────────────────────────────────────────

def _resolve_model_path(repo_dir: str) -> str:
    """Pick the latest snapshot from a HF cache repo directory."""
    snapshots = sorted(glob.glob(os.path.join(repo_dir, "snapshots", "*")))
    if not snapshots:
        raise RuntimeError(
            f"No snapshots found under {repo_dir}. "
            f"Is hf-model-cache mounted correctly to HF_HOME={HF_HOME}?"
        )
    return snapshots[-1]


# Base model (voice clone)
CLONE_REPO_DIR = os.environ.get(
    "QWEN_TTS_REPO_DIR",
    os.path.join(HF_HOME, "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base"),
)
CLONE_MODEL_PATH = (
    os.path.join(CLONE_REPO_DIR, "snapshots", os.environ["QWEN_TTS_SNAPSHOT"])
    if os.environ.get("QWEN_TTS_SNAPSHOT")
    else _resolve_model_path(CLONE_REPO_DIR)
)

# CustomVoice model
CUSTOM_REPO_DIR = os.environ.get(
    "QWEN_TTS_CUSTOM_REPO_DIR",
    os.path.join(HF_HOME, "models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice"),
)
CUSTOM_MODEL_PATH = (
    os.path.join(CUSTOM_REPO_DIR, "snapshots", os.environ["QWEN_TTS_CUSTOM_SNAPSHOT"])
    if os.environ.get("QWEN_TTS_CUSTOM_SNAPSHOT")
    else _resolve_model_path(CUSTOM_REPO_DIR)
)

# # VoiceDesign model
# DESIGN_REPO_DIR = os.environ.get(
#     "QWEN_TTS_DESIGN_REPO_DIR",
#     os.path.join(HF_HOME, "models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
# )
# DESIGN_MODEL_PATH = (
#     os.path.join(DESIGN_REPO_DIR, "snapshots", os.environ["QWEN_TTS_DESIGN_SNAPSHOT"])
#     if os.environ.get("QWEN_TTS_DESIGN_SNAPSHOT")
#     else _resolve_model_path(DESIGN_REPO_DIR)
# )

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Qwen3-TTS API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tts_clone: Optional[Qwen3TTSModel] = None
tts_custom: Optional[Qwen3TTSModel] = None
tts_design: Optional[Qwen3TTSModel] = None

# Set MODEL_MODE before starting — only one model loads at a time:
#   MODEL_MODE=clone   uvicorn api_server:app
#   MODEL_MODE=custom  uvicorn api_server:app
#   MODEL_MODE=design  uvicorn api_server:app
MODEL_MODE = os.environ.get("MODEL_MODE")
VALID_MODES = {"clone", "custom", "design"}


@app.on_event("startup")
def load_models():
    global tts_clone, tts_custom, tts_design

    if MODEL_MODE not in VALID_MODES:
        raise RuntimeError(
            f"MODEL_MODE='{MODEL_MODE}' is invalid. "
            f"Set MODEL_MODE to exactly one of: {', '.join(sorted(VALID_MODES))}."
        )

    from transformers import AutoConfig, AutoModel, AutoProcessor
    from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    if MODEL_MODE == "clone":
        print("Loading Qwen3-TTS clone model …", flush=True)
        tts_clone = Qwen3TTSModel.from_pretrained(CLONE_MODEL_PATH, attn_implementation="eager")
        tts_clone.model.eval()
        print("Clone model ready.", flush=True)

    elif MODEL_MODE == "custom":
        print("Loading Qwen3-TTS custom-voice model …", flush=True)
        tts_custom = Qwen3TTSModel.from_pretrained(CUSTOM_MODEL_PATH, attn_implementation="eager")
        tts_custom.model.eval()
        print("Custom-voice model ready.", flush=True)

    # elif MODEL_MODE == "design":
    #     print("Loading Qwen3-TTS voice-design model …", flush=True)
    #     tts_design = Qwen3TTSModel.from_pretrained(DESIGN_MODEL_PATH, attn_implementation="eager")
    #     tts_design.model.eval()
    #     print("Voice-design model ready.", flush=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def wav_to_bytes(audio: np.ndarray, sample_rate: int) -> io.BytesIO:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "clone_model_loaded": tts_clone is not None,
        "custom_model_loaded": tts_custom is not None,
        # "design_model_loaded": tts_design is not None,
    }


@app.post("/tts/clone")
async def tts_clone_endpoint(
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
      ref_text      : transcript of ref_audio (required when x_vector_only=false)
      language      : e.g. "Auto", "English", "Chinese"
      x_vector_only : clone tone/timbre without a transcript
      temperature   : sampling temperature
      top_p         : nucleus sampling
      max_new_tokens: max codec tokens
      ref_audio     : WAV file (multipart/form-data)
    """
    if tts_clone is None:
        raise HTTPException(503, "Clone model not loaded. Restart with MODEL_MODE=clone.")
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
        wavs, fs = tts_clone.generate_voice_clone(
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


@app.post("/tts/custom")
async def tts_custom_endpoint(
    text: str = Form(...),
    speaker: str = Form(...),
    language: str = Form("Auto"),
    instruct: str = Form(""),
    temperature: float = Form(0.9),
    top_p: float = Form(1.0),
    max_new_tokens: int = Form(2048),
):
    """
    Custom-voice endpoint — uses a predefined speaker from the CustomVoice model.

    Form fields:
      text          : text to synthesise
      speaker       : speaker name (see GET /info/custom for available speakers)
      language      : e.g. "Auto", "English", "Chinese"
      instruct      : optional style instruction (ignored by 0.6B model)
      temperature   : sampling temperature
      top_p         : nucleus sampling
      max_new_tokens: max codec tokens
    """
    if tts_custom is None:
        raise HTTPException(503, "Custom model not loaded. Restart with MODEL_MODE=custom.")
    if not text.strip():
        raise HTTPException(400, "text cannot be empty")
    if not speaker.strip():
        raise HTTPException(400, "speaker cannot be empty")

    try:
        wavs, fs = tts_custom.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct or None,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    buf = wav_to_bytes(wavs[0], fs)
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="output.wav"'},
    )


# @app.post("/tts/design")
# async def tts_design_endpoint(
#     text: str = Form(...),
#     instruct: str = Form(...),
#     language: str = Form("Auto"),
#     temperature: float = Form(0.9),
#     top_p: float = Form(1.0),
#     max_new_tokens: int = Form(2048),
# ):
#     """
#     Voice-design endpoint — describe the desired voice in natural language.

#     Form fields:
#       text          : text to synthesise
#       instruct      : natural-language voice description
#                       e.g. "A calm, deep male voice with a slight British accent"
#       language      : e.g. "Auto", "English", "Chinese"
#       temperature   : sampling temperature
#       top_p         : nucleus sampling
#       max_new_tokens: max codec tokens
#     """
#     if tts_design is None:
#         raise HTTPException(503, "Voice-design model not loaded. Restart with MODEL_MODE=design.")
#     if not text.strip():
#         raise HTTPException(400, "text cannot be empty")
#     if not instruct.strip():
#         raise HTTPException(400, "instruct cannot be empty — describe the voice you want")

#     try:
#         wavs, fs = tts_design.generate_voice_design(
#             text=text,
#             instruct=instruct,
#             language=language,
#             temperature=temperature,
#             top_p=top_p,
#             max_new_tokens=max_new_tokens,
#         )
#     except ValueError as e:
#         raise HTTPException(400, str(e))

#     buf = wav_to_bytes(wavs[0], fs)
#     return StreamingResponse(
#         buf,
#         media_type="audio/wav",
#         headers={"Content-Disposition": 'attachment; filename="output.wav"'},
#     )


@app.get("/info")
def info():
    """Returns info for the clone model."""
    if tts_clone is None:
        raise HTTPException(503, "Clone model not loaded yet")
    return {
        "model_type": getattr(tts_clone.model, "tts_model_type", "unknown"),
        "supported_languages": tts_clone.get_supported_languages(),
        "supported_speakers": tts_clone.get_supported_speakers(),
    }


@app.get("/info/custom")
def info_custom():
    """Returns info for the custom-voice model, including available speakers."""
    if tts_custom is None:
        raise HTTPException(503, "Custom-voice model not loaded yet")
    return {
        "model_type": getattr(tts_custom.model, "tts_model_type", "unknown"),
        "supported_languages": tts_custom.get_supported_languages(),
        "supported_speakers": tts_custom.get_supported_speakers(),
    }


# @app.get("/info/design")
# def info_design():
#     """Returns info for the voice-design model."""
#     if tts_design is None:
#         raise HTTPException(503, "Voice-design model not loaded yet")
#     return {
#         "model_type": getattr(tts_design.model, "tts_model_type", "unknown"),
#         "supported_languages": tts_design.get_supported_languages(),
#         "supported_speakers": tts_design.get_supported_speakers(),
#     }