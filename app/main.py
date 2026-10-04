"""FastAPI бекенд + Gradio UI на одному порту (8002)."""
import base64
import io
import os
import tempfile

import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import tts_pipeline as pipe
from app import ui
from app import verbalizer

import gradio as gr

app = FastAPI(title="HolosTTS hybrid", version="1.0")


class SynthRequest(BaseModel):
    text: str
    voice: str | None = None          # ім'я пресету
    audio_base64: str | None = None   # аудіо у base64 для клонування голосу
    auto_verbalize: bool = True       # автоматично вербалізувати цифри/дати/час


class VerbalizeRequest(BaseModel):
    text: str


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "engine": "hybrid-onnx-int8",
        "provider": ",".join(pipe.session.get_providers()),
        "input_binding": pipe.INPUT_BINDING,
        "threads": pipe.THREADS,
    }


@app.get("/api/voices")
def voices():
    return {"voices": pipe.list_voices()}


@app.post("/api/verbalize")
def verbalize_text(req: VerbalizeRequest):
    try:
        return {"text": verbalizer.verbalize(req.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/synthesize")
def synthesize(req: SynthRequest):
    try:
        voice = req.voice or pipe.list_voices()[0]
        if req.audio_base64:
            raw = base64.b64decode(req.audio_base64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(raw)
                tmp_path = f.name
            try:
                voice = pipe.encode_voice(tmp_path)
            finally:
                os.unlink(tmp_path)
        text = verbalizer.auto_verbalize(req.text) if req.auto_verbalize else req.text
        sr, data = pipe.synthesize(text, voice)
        buf = io.BytesIO()
        sf.write(buf, data, sr, format="WAV")
        return {
            "sample_rate": sr,
            "audio_base64": base64.b64encode(buf.getvalue()).decode(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


demo = ui.create_demo()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
