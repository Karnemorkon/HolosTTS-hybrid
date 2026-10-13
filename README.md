# HolosTTS-hybrid

[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/banner-direct-single.svg)](https://stand-with-ukraine.pp.ua)
[![Made in Ukraine](https://img.shields.io/badge/made_in-Ukraine-ffd700.svg?labelColor=0057b7)](https://stand-with-ukraine.pp.ua)
[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/badges/StandWithUkraine.svg)](https://stand-with-ukraine.pp.ua)
[![Russian Warship Go Fuck Yourself](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/badges/RussianWarship.svg)](https://stand-with-ukraine.pp.ua)

---

> ⚠️ **Відмова від відповідальності / Disclaimer**
>
> * **UA:** Цей проєкт є незалежною аматорською розробкою (Docker-обгорткою) і **не є офіційним продуктом** автора оригінальної нейромережі. Він жодним чином не афілійований, не спонсорується і не підтримується розробником [patriotyk](https://github.com/patriotyk). Усі права на оригінальні моделі, ваги та препроцес належать їхньому законному автору. Проєкт надається "як є" (as is), автор цієї обгортки не несе відповідальності за можливі збої чи використання сервісу.
> * **EN:** This project is an independent, community-driven Docker wrapper and **is not an official product** of the original neural network creator. It is not affiliated with, endorsed, or sponsored by [patriotyk](https://github.com/patriotyk). All rights to the original models and weights belong to their respective owner. The software is provided "as is", without warranty of any kind.

> **English summary:** production-ready Dockerized **Ukrainian text-to-speech** service
> (REST API + Web UI, port 8002). It runs the [HolosTTS](https://github.com/patriotyk/HolosTTS)
> neural network with a **PyTorch voice encoder + INT8 ONNX generator** on CPU via
> ONNX Runtime, adds automatic CPU thread management, sentence-level caching/prefetching,
> and **verbalization of numbers, dates and time** (M2M100 + CTranslate2) before synthesis.
> Model weights are MIT-licensed by [patriotyk](https://huggingface.co/patriotyk).

Український синтез мови у Docker-контейнері: FastAPI (+ Gradio UI), виключно CPU,
жодних GPU-залежностей. Перший запуск сам довантажує моделі (~3.5 ГБ) у named volume.

## Можливості

- **Синтез українською** 24 кГц з 27 пресетами голосів + клонування голосу з аудіо-промпта
  (у Web UI клон можна зберегти під власним імʼям і використовувати через API)
- **Формати** WAV/MP3/OGG у `/api/synthesize` та wav/pcm/mp3 у `/v1/audio/speech`
- **Потоковий синтез**: REST-стрім (`stream=true`) і WebSocket `/api/ws/synthesize` (PCM16-чунки)
- **Наголошування**: швидкий словниковий режим (`rule`, типово) або нейромережевий
  ByT5-CTranslate2 (`HOLOS_STRESS=byt5` — якісніший на рідкісних словах, завантажує ~1.2 ГБ один раз)
- **INT8 ONNX-генератор** (`holos_cpu_int8.onnx`) на ONNX Runtime — швидше за fp32 на CPU
- **Автовербалізація**: `250 грн`, `22.08.2025 о 15:30` → «двісті пʼятдесят гривень…»
  (M2M100-CTranslate2, ліниве завантаження ~1.9 ГБ один раз на volume)
- **Автопотоки CPU**: 80% потоків хоста, обрізаних до cgroup-квоти (або `TTS_THREADS`)
- **Нарізка на речення + LRU-кеш препроцесу + prefetch** наступної частини
- **Продакшен-compose**: healthcheck, ліміти CPU/RAM, ротація логів, автоперезапуск

## Швидкий старт

```bash
git clone https://github.com/Karnemorkon/HolosTTS-hybrid.git
cd HolosTTS-hybrid
docker compose up -d --build
# перший запуск: довантаження моделей у volume (кілька хвилин), потім:
curl http://localhost:8002/healthz
# Web UI: http://localhost:8002/
# логи першого запуску: docker compose logs -f holos-hybrid
```

Ознака готовності: `{"status":"ok","engine":"hybrid-onnx-int8",...,"threads":12}`.

## Розгортання через Docker Hub

Образ: [`morkon06/holostts-hybrid`](https://hub.docker.com/r/morkon06/holostts-hybrid).

```bash
docker pull morkon06/holostts-hybrid:latest

docker run -d --name holos-tts \
  -p 8002:8002 \
  -v holos_models_cache:/app/models \
  --restart unless-stopped \
  morkon06/holostts-hybrid:latest
```

Теги (деталі — таблиця «Варіанти збірки» вище): `latest`/`cpu`/`embedded` — дефолт, multi-arch amd64+arm64; `cuda`, `openvino` — прискорення; версійні (`1.0.2`) — пін релізу.

Compose без збірки (замість `build: .`):

```yaml
services:
  holos-hybrid:
    image: morkon06/holostts-hybrid:latest
    container_name: holos-hybrid
    ports:
      - "8002:8002"
    environment:
      - HF_HOME=/app/models
    volumes:
      - holos_models_cache:/app/models
    restart: unless-stopped

volumes:
  holos_models_cache:
```

Ліміти CPU/RAM, healthcheck і ротацію логів — копіюйте із `docker-compose.yml` репозиторію.


## API

### `POST /api/synthesize`

| Поле | Тип | Опис |
|---|---|---|
| `text` | str (обовʼязкове) | Текст, 10…50000 символів |
| `voice` | str | Імʼя пресету з `/api/voices` (типово — перший) |
| `audio_base64` | str | WAV у base64 — клонування голосу замість пресету |
| `auto_verbalize` | bool | Автовербалізація цифр/дат/часу (типово `true`) |
| `format` | str | `wav` (типово), `mp3`, `ogg` |
| `speed` | float | Швидкість синтезу 0.25–4.0 (типово 1.0) |
| `raw_stress` | bool | `true` → без автонаголосу, власні `+`/U+0301 |

Відповідь: `{"format","sample_rate","duration_sec","voice","text","verbalized_text","audio_base64"}`.

```bash
curl -s http://localhost:8002/api/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Привіт! Це тест українського синтезу мови.","voice":"Speaker_84"}' \
  | python3 -c 'import sys,json,base64; open("out.wav","wb").write(base64.b64decode(json.load(sys.stdin)["audio_base64"]))'
```

### `POST /api/verbalize`

```bash
curl -s http://localhost:8002/api/verbalize \
  -H 'Content-Type: application/json' \
  -d '{"text":"22.08.2025 о 15:30, 250 грн"}'
# {"text":"двадцять другого серпня дві тисячі двадцять пʼятого року о пʼятнадцятій тридцять, ціна двісті пʼятдесят гривень..."}
```

### `GET /api/voices`, `GET /healthz`

```bash
curl -s http://localhost:8002/api/voices      # `{"voices":[...],"custom":[...]}`
curl -s http://localhost:8002/healthz         # статус, провайдер, потоки, режим наголосу
```

**Збережені голоси.** Клонування — у Web UI: завантажте аудіо → прослухайте синтез →
введіть імʼя і натисніть «Зберегти голос». Ембеддінг зберігається у volume
`holos_models_cache` (`speakers/<імʼя>.npy`) і стає звичайним `voice` для REST/OpenAI API.
### `POST /v1/audio/speech` (OpenAI-сумісний)

Формат [OpenAI Audio Speech API](https://platform.openai.com/docs/api-reference/audio/createSpeech) — працює з будь-яким клієнтом, що вміє підмінювати `base_url` (SillyTavern, OpenWebUI, LibreChat, HA-плагіни тощо):

| Поле | Опис |
|---|---|
| `input` | текст (автовербалізація цифр) |
| `voice` | пресет (`Speaker_84`) або імена OpenAI (`alloy`, `nova`, …) |
| `response_format` | `wav`, `pcm`, `mp3` (типово `mp3`) |
| `speed` | 0.25–4.0 (типово 1.0) |
| `stream` | `true` → потокова відповідь (частини по мірі генерації) |

```bash
curl http://localhost:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Привіт! Це український TTS.","voice":"Speaker_84","response_format":"wav"}' \
  --output speech.wav
```

Автентифікація не потрібна; поле `model` приймається і ігнорується.


### `WS /api/ws/synthesize` (потоковий)

Двосторонній WebSocket: перший кадр — JSON-конфіг від клієнта, далі — бінарні
PCM16-чунки (24 кГц, mono), фінальний текстовий кадр — JSON-метадані
(`{"done": true, "sample_rate": 24000, "duration_sec": ...}`).

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect('ws://localhost:8002/api/ws/synthesize') as ws:
        cfg = {'text': 'Привіт! Це потоковий синтез мови.', 'voice': 'Speaker_84'}
        await ws.send(json.dumps(cfg))
        with open('stream.pcm', 'wb') as f:
            async for msg in ws:
                if isinstance(msg, bytes):
                    f.write(msg)
                else:
                    print(json.loads(msg))

asyncio.run(main())
```

Конфіг: `text` (обовʼязково), `voice`, `speed`, `auto_verbalize`, `raw_stress`.
Помилка сервера → текстовий кадр `{"error": "..."}`.

## Конфігурація

| Змінна | Типово | Що робить |
|---|---|---|
| `TTS_THREADS` | `min(80% потоків хоста, cgroup-quota)` | PyTorch/препроцес |
| `ORT_INTRA_THREADS` | `min(TTS_THREADS, 9)` | ONNX Runtime intra-op |
| `HF_HOME` | `/app/models` | кеш моделей (volume `holos_models_cache`) |
| `HOLOS_STRESS` | `rule` | Наголошування: `rule` (словниковий) або `byt5` (ByT5-CT2, +1.2 ГБ) |
| `ORT_PROVIDER` | `auto` | EP ONNX Runtime: `cpu`, `cuda`, `openvino`, `auto` |

### Варіанти збірки (теги образу)

| Тег / `ORT_VARIANT` | Архітектури | EP | Для чого |
|---|---|---|---|
| `latest` / `cpu` / `embedded` (**дефолт**) | **linux/amd64 + linux/arm64** | CPUExecutionProvider | будь-які x86-64 (включно зі старими — перевірено на Xeon E5-2650, AVX без AVX2) та arm64-плати (Raspberry Pi 4/5, Ampere тощо). `embedded` — алиас `cpu` для зручності SBC-користувачів |
| `openvino` | linux/amd64 | OpenVINOExecutionProvider | **лише сучасні x86-64 CPU з AVX512/VNNI** (~2024+); на старих процесорах один синтез займає >500 с — не використовувати |
| `cuda` | linux/amd64 | CUDAExecutionProvider | GPU NVIDIA (потрібен NVIDIA Container Toolkit; база `nvidia/cuda:12.6.3`) |

`cpu`/`openvino`/`cuda` підміняє пакет onnxruntime:
`docker build --build-arg ORT_VARIANT=openvino .`

Примітка: `onnxruntime-openvino==1.24.1` — це **найновіша** версія на PyPI
(EP-білди ONNX Runtime for OpenVINO виходять рідше за основний onnxruntime).
Повільність на старих CPU зумовлена відсутністю інструкцій AVX512/VNNI, а не версією пакета.

Приклади — у [`.env.example`](.env.example). Ліміти контейнера (`cpus: "12.0"`,
`memory: 8g`) задані у `docker-compose.yml` — під ваш хост правте їх разом із
`TTS_THREADS` (типово quota = авто-значення `TTS_THREADS`).

## Архітектура

```text
текст ─ автовербалізація (M2M100-CTranslate2, ліниво) ─► текст із розгорнутими числами
     ─ наголос (ukrainian-word-stress / ByT5-CT2) ─► ipa-uk ─► CharTokenizer ─► tokens ┐
                                                                               ├─ ONNX Runtime
аудіо ─ VoiceEncoder (PyTorch, voice_encoder.pt) ─► ембеддінг [256] ───────────┘
        holos_cpu_int8.onnx ─► float32 wav ─► нормалізація ─► WAV 24 кГц
```

## Бенчмарки (Xeon E5-2650, 12 потоків, теплий стан)

| Сценарій | Час |
|---|---|
| Одне речення (теплий замір) | **1.3–1.9 с** |
| 3 довгі речення (3 частини) | **~8.4 с** |
| Перший запуск (завантаження моделей) | до ~10 хв (volume cache) |

Для порівняння: у парних A/B-замірах чистий PyTorch-бекенд (official HolosTTS)
на тих самих текстах давав 5.3 с / 15.0 с — гібридний INT8-ONNX-шлях швидший на 27–35%.

## Home Assistant

Сервіс — звичайний HTTP-сервер, тому його можна підʼєднати до Home Assistant кількома шляхами:

| Шлях | Що дає |
|---|---|
| **Wyoming-адаптер** (рекомендовано) | міст Wyoming-протокол → `/api/synthesize`; HA (з 2023.5) має вбудовану TTS-платформу Wyoming з Assist-pipeline |
| Кастомна TTS-інтеграція (`custom_components`) | прямий виклик REST API, повний контроль над entity |
| `rest_command` в automations | прості оголошення без TTS-entity |

Деякі HA-плагіни, що працюють з OpenAI TTS API, підключаться безпосередньо до `/v1/audio/speech`.

Приклад `rest_command` у `configuration.yaml`:

```yaml
rest_command:
  holos_tts:
    url: "http://localhost:8002/api/synthesize"
    method: POST
    content_type: "application/json"
    payload: '{"text": "{{ message }}", "auto_verbalize": true}'
```


## Ліцензія та подяки

- **Цей репозиторій** (обгортка, сервіс, UI, оптимізації): [MIT](LICENSE).
- **Нейромережа HolosTTS** (ваги `holos_cpu_int8.onnx`, `voice_encoder.pt`, `voices.pt`)
  належить автору **[patriotyk](https://github.com/patriotyk)** і розповсюджується за MIT:
  [github.com/patriotyk/HolosTTS](https://github.com/patriotyk/HolosTTS) ·
  [huggingface.co/patriotyk/HolosTTS](https://huggingface.co/patriotyk/HolosTTS).
  Велика подяка за модель та препроцес-пайплайн!
- **Вербалізація**: код адаптовано з MIT-space
  [patriotyk/styletts2-ukrainian](https://huggingface.co/spaces/patriotyk/styletts2-ukrainian)
  (модель [skypro1111/m2m100-ukr-verbalization-ct2](https://huggingface.co/skypro1111/m2m100-ukr-verbalization-ct2)).
- **Залежності**: [ukrainian-word-stress](https://github.com/patriotyk/ukrainian-word-stress)
  (MIT, lang-uk) · [ipa-uk](https://github.com/patriotyk/ipa-uk) (MIT, lang-uk) ·
  [ukrainian-accentor](https://github.com/egorsmkv/ukrainian-accentor) · ONNX Runtime,
  PyTorch, FastAPI, Gradio тощо — див. `requirements.txt`.

Проєкт не афілійований з автором моделі; права на нейромережу належать її автору.
