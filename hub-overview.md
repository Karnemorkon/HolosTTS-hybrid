`# HolosTTS-hybrid`

Незалежна Docker-обгортка для **українського синтезу мови** / Independent Docker wrapper for **Ukrainian text-to-speech**.

## Українською

Виробнича Docker-обгортка: REST API + Gradio Web UI на порту **8002**. Multi-arch: **amd64 + arm64** (Raspberry Pi та інші SBC — тег `embedded`), з opt-in прискоренням `cuda` (NVIDIA GPU) та `openvino` (сучасні x86-64).

Навколо [patriotyk/HolosTTS](https://github.com/patriotyk/HolosTTS): PyTorch-енкодер голосу + **INT8 ONNX**-генератор через ONNX Runtime, автоматичне керування потоками CPU, кеш і prefetch речень, **вербалізація чисел, дат і часу** (`250 грн` → «двісті пʼятдесят гривень»).

### Можливості

- Синтез 24 кГц українською, **27 пресетів голосів** + клонування з аудіо-промпта із **збереженням власного голосу** у Web UI (далі — звичайний `voice` для API)
- **OpenAI-сумісний** `POST /v1/audio/speech` — `wav` / `pcm` / `mp3`, `speed`, `stream`
- `POST /api/synthesize` — WAV/MP3/OGG, WebSocket-стрім `/api/ws/synthesize`, `GET /api/voices`, `POST /api/verbalize`, `GET /healthz`
- Наголошування: швидке словникове (`rule`, типово) або нейромережеве ByT5 (`HOLOS_STRESS=byt5`)
- Автопотоки CPU (80% хоста, до cgroup-квоти), healthcheck, ліміти ресурсів, ротація логів
- Перший запуск довантажує моделі (~3.5 ГБ) у named volume — один раз

### Швидкий старт

```bash
docker pull morkon06/holostts-hybrid:latest

docker run -d --name holos-tts   -p 8002:8002   -v holos_models_cache:/app/models   --restart unless-stopped   morkon06/holostts-hybrid:latest

# Web UI: http://localhost:8002/
curl http://localhost:8002/healthz
```

Запит у форматі OpenAI:

```bash
curl http://localhost:8002/v1/audio/speech   -H "Content-Type: application/json"   -d "{"input":"Привіт! Це український TTS.","voice":"nova","response_format":"wav"}"   --output speech.wav
```

### Теги

| Тег | Архітектури | Що це |
|---|---|---|
| `latest`, `cpu`, `embedded` | **amd64 + arm64** | **дефолт**, CPUExecutionProvider; будь-які x86-64 (включно зі старими) та arm64-плати (RPi 4/5). `embedded` — алиас для SBC |
| `cuda` | amd64 | GPU NVIDIA (CUDA 12.6; потрібен NVIDIA Container Toolkit) |
| `openvino` | amd64 | **лише сучасні x86-64 з AVX512/VNNI** (~2024+); на старих CPU не використовувати |
| `1.0.2` | amd64 + arm64 | пін релізу (версійні теги — з git-тегів `v*`) |

## English

**Production-ready Dockerized Ukrainian text-to-speech** — REST API + Gradio Web UI on port **8002**. Multi-arch: **amd64 + arm64** (Raspberry Pi and other SBCs via the `embedded` tag), with opt-in `cuda` (NVIDIA GPU) and `openvino` (modern x86-64) acceleration.

An independent wrapper around [patriotyk/HolosTTS](https://github.com/patriotyk/HolosTTS): PyTorch voice encoder + **INT8 ONNX** generator via ONNX Runtime, automatic CPU threads, sentence caching/prefetch, and **verbalization of numbers, dates and time**.

### Highlights

- 24 kHz synthesis, **27 voice presets** + cloning from an audio prompt with **saving your own voice** in the Web UI (then a regular `voice` for the API)
- **OpenAI-compatible** `POST /v1/audio/speech` — `wav` / `pcm` / `mp3`, `speed`, `stream`
- Native `POST /api/synthesize` (WAV/MP3/OGG), WebSocket streaming `/api/ws/synthesize`, `GET /api/voices`, `POST /api/verbalize`, `GET /healthz`
- Stress modes: fast dictionary (`rule`, default) or neural ByT5 (`HOLOS_STRESS=byt5`)
- Auto CPU threads (80% of host, clamped to cgroup quota), healthcheck, resource limits, log rotation
- First start downloads models (~3.5 GB) into a named volume — once

### Quick start

```bash
docker pull morkon06/holostts-hybrid:latest

docker run -d --name holos-tts   -p 8002:8002   -v holos_models_cache:/app/models   --restart unless-stopped   morkon06/holostts-hybrid:latest

# Web UI: http://localhost:8002/
curl http://localhost:8002/healthz
```

OpenAI-compatible request:

```bash
curl http://localhost:8002/v1/audio/speech   -H "Content-Type: application/json"   -d "{"input":"Привіт! Це український TTS.","voice":"nova","response_format":"wav"}"   --output speech.wav
```

### Tags

| Tag | Architectures | What it is |
|---|---|---|
| `latest`, `cpu`, `embedded` | **amd64 + arm64** | **default**, CPUExecutionProvider; any x86-64 (incl. old CPUs) and arm64 boards (RPi 4/5). `embedded` is an alias for SBC users |
| `cuda` | amd64 | NVIDIA GPU (CUDA 12.6; needs NVIDIA Container Toolkit) |
| `openvino` | amd64 | **modern x86-64 with AVX512/VNNI only** (~2024+); do not use on old CPUs |
| `1.0.2` | amd64 + arm64 | release pin (version tags come from `v*` git tags) |

## Source & license / Джерело та ліцензія

- Source: https://github.com/Karnemorkon/HolosTTS-hybrid — MIT
- Оригінальні моделі та ваги: MIT від [patriotyk](https://huggingface.co/patriotyk)

> **Відмова від відповідальності / Disclaimer**
>
> * **UA:** Це незалежна Docker-обгортка і **не є офіційним продуктом** автора оригінальної нейромережі. Вона жодним чином не афілійована, не спонсорується і не підтримується [patriotyk](https://github.com/patriotyk). Усі права на оригінальні моделі та ваги належать їхньому законному автору. Проєкт надається "як є" (as is).
> * **EN:** An independent Docker wrapper, **not an official product** of the original neural network creator. Not affiliated with, endorsed, or sponsored by [patriotyk](https://github.com/patriotyk). Provided "as is", without warranty of any kind.
