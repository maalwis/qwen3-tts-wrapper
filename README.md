# qwen3-tts-base

A FastAPI server wrapping **Qwen3-TTS** (12Hz, 0.6B and 1.7B variants) for voice cloning and text-to-speech synthesis. Designed to run locally or on cloud GPU providers such as RunPod.

---

## Features

- 🎙️ **Voice cloning** — synthesise speech in any reference voice from a short WAV sample
- 🌐 **Multi-language** — Auto-detect or specify English, Chinese, and more
- 📦 **Fully offline** — model weights loaded from local HF cache, no internet required at runtime
- 🐳 **Docker-ready** — includes Dockerfile for containerised deployment

---

## Project Structure

```
qwen3-tts-base/
├── api_server.py          # FastAPI application (main entry point)
├── Dockerfile             # Container definition
├── frontend.html          # Simple browser UI
├── frontend-docker.html   # Browser UI configured for Docker
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Package metadata
├── hf-model-cache/        # Local HuggingFace model weights (not committed)
└── qwen_tts/              # Core model inference package
    ├── inference/         # High-level TTS model wrapper
    └── core/              # Model architecture, tokeniser, codec
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Download model weights

Place the HuggingFace snapshot under `hf-model-cache/` or set `HF_HOME` to point at an existing cache:

```
hf-model-cache/
└── models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/
    └── snapshots/<hash>/
```

### 3. Run the server

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check — returns model load status |
| GET | `/info` | Model metadata, supported languages & speakers |
| POST | `/tts/clone` | Voice-clone synthesis (multipart form) |

### POST `/tts/clone`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | ✅ | Text to synthesise |
| `ref_audio` | file (WAV) | ✅ | Reference voice sample |
| `ref_text` | string | ✅* | Transcript of `ref_audio` (* unless `x_vector_only=true`) |
| `language` | string | ❌ | `"Auto"` (default), `"English"`, `"Chinese"`, … |
| `x_vector_only` | bool | ❌ | Clone voice style without needing a transcript |
| `temperature` | float | ❌ | Sampling temperature (default `0.9`) |
| `top_p` | float | ❌ | Nucleus sampling (default `1.0`) |
| `max_new_tokens` | int | ❌ | Max codec tokens (default `2048`) |

**Response:** `audio/wav` binary stream.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HOME` | `./hf-model-cache` | Path to HuggingFace model cache |
| `QWEN_TTS_REPO_DIR` | `$HF_HOME/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base` | Specific model repo dir |
| `QWEN_TTS_SNAPSHOT` | *(latest)* | Pin to a specific snapshot hash |
| `HF_HUB_OFFLINE` | `1` | Disable HF Hub network calls |
| `TRANSFORMERS_OFFLINE` | `1` | Disable Transformers network calls |

---

## Branches

| Branch | Description |
|--------|-------------|
| `main` | Stable CPU-based inference |
| `feature/gpu-support` | CUDA GPU acceleration with float16 & autocast |

---

## RunPod Deployment

See the **[RunPod Deployment Guide](docs/runpod-deployment.md)** for step-by-step instructions to run this server on a GPU pod.

---

## License

Model weights are subject to the [Qwen License](https://huggingface.co/Qwen). This wrapper code is MIT licensed.