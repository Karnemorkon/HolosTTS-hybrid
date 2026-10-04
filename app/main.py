"""FastAPI бекенд + Gradio UI на одному порту (8002)."""
import base64
import io
import os
import struct
import tempfile

import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
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


_OPENAI_VOICES = ("alloy", "echo", "fable", "onyx", "nova", "shimmer")


class OpenAISpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str | None = None
    response_format: str = "mp3"
    speed: float = 1.0
    stream: bool = False


def _pick_voice(name):
    """Пресет сервісу або мапінг OpenAI-імені на пресет; None = невідомий."""
    presets = pipe.list_voices()
    if not name:
        return presets[0]
    if name in presets:
        return name
    if name in _OPENAI_VOICES:
        return presets[_OPENAI_VOICES.index(name) % len(presets)]
    return None


def _wav_bytes(sr, data):
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


def _mp3_bytes(sr, data):
    from pydub import AudioSegment
    seg = AudioSegment(data.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    buf = io.BytesIO()
    seg.export(buf, format="mp3")
    return buf.getvalue()


def _wav_header():
    """RIFF-заголовок із невідомою довжиною (0xFFFFFFFF) для стрімінгу."""
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF, b"WAVE",
        b"fmt ", 16, 1, 1,
        pipe.SAMPLE_RATE, pipe.SAMPLE_RATE * 2, 2, 16,
        b"data", 0xFFFFFFFF,
    )


@app.post("/v1/audio/speech")
def openai_speech(req: OpenAISpeechRequest):
    """OpenAI-сумісний Audio Speech API (автентифікація не потрібна)."""
    if req.response_format not in ("wav", "pcm", "mp3"):
        raise HTTPException(status_code=400, detail="response_format: підтримуються wav, pcm, mp3")
    if not (req.input or "").strip():
        raise HTTPException(status_code=400, detail="input обов'язковий і не може бути порожнім")
    if not 0.25 <= req.speed <= 4.0:
        raise HTTPException(status_code=400, detail="speed мусить бути в діапазоні 0.25-4.0")
    voice = _pick_voice(req.voice)
    if voice is None:
        raise HTTPException(status_code=400, detail="Невідомий voice: %s" % req.voice)
    try:
        text = verbalizer.auto_verbalize(req.input)
        if len(text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Текст закороткий (мінімум 10 символів)")
        if len(text) > 50000:
            raise HTTPException(status_code=400, detail="Текст мусить бути менше 50k символів")

        if not req.stream:
            sr, data = pipe.synthesize(text, voice, speed=req.speed)
            if req.response_format == "wav":
                return Response(content=_wav_bytes(sr, data), media_type="audio/wav")
            if req.response_format == "pcm":
                return Response(content=data.tobytes(), media_type="application/octet-stream")
            return Response(content=_mp3_bytes(sr, data), media_type="audio/mpeg")

        def _chunks():
            if req.response_format == "wav":
                yield _wav_header()
            for fchunk in pipe.synthesize_stream(text, voice, speed=req.speed):
                pcm = pipe.float_to_pcm16(fchunk)
                if req.response_format == "pcm":
                    yield pcm.tobytes()
                else:
                    from pydub import AudioSegment
                    buf = io.BytesIO()
                    AudioSegment(
                        pcm.tobytes(), frame_rate=pipe.SAMPLE_RATE,
                        sample_width=2, channels=1,
                    ).export(buf, format="mp3")
                    yield buf.getvalue()

        media = {"wav": "audio/wav", "pcm": "application/octet-stream", "mp3": "audio/mpeg"}
        return StreamingResponse(_chunks(), media_type=media[req.response_format])
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


demo = ui.create_demo()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
