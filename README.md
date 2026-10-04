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
```

Ознака готовності: `{"status":"ok","engine":"hybrid-onnx-int8",...,"threads":12}`.

## Розгортання через Docker Hub

Образ: [`morkon06/holostts-hybrid`](https://hub.docker.com/r/morkon06/holostts-hybrid).

```bash
docker pull morkon06/holostts-hybrid:main

docker run -d --name holos-tts \
  -p 8002:8002 \
  -v holos_models_cache:/app/models \
  --restart unless-stopped \
  morkon06/holostts-hybrid:main
```

Теги: `main` — останній збір із гілки `main`; `latest` та версійні (`1.0.0`) — із git-тегів (`v1.0.0`).

Compose без збірки (замість `build: .`):

```yaml
services:
  holos-hybrid:
    image: morkon06/holostts-hybrid:main
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

Відповідь: `{"sample_rate": 24000, "audio_base64": "..."}` (WAV int16).

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
curl -s http://localhost:8002/api/voices      # {"voices":["Speaker_84",...]}
curl -s http://localhost:8002/healthz         # статус, провайдер, потоки
```
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


## Конфігурація

| Змінна | Типово | Що робить |
|---|---|---|
| `TTS_THREADS` | `min(80% потоків хоста, cgroup-quota)` | PyTorch/препроцес |
| `ORT_INTRA_THREADS` | `min(TTS_THREADS, 9)` | ONNX Runtime intra-op |
| `HF_HOME` | `/app/models` | кеш моделей (volume `holos_models_cache`) |

Приклади — у [`.env.example`](.env.example). Ліміти контейнера (`cpus: "12.0"`,
`memory: 8g`) задані у `docker-compose.yml` — під ваш хост правте їх разом із
`TTS_THREADS` (типово quota = авто-значення `TTS_THREADS`).

## Архітектура

```text
текст ─ автовербалізація (M2M100-CTranslate2, ліниво) ─► текст із розгорнутими числами
     ─ Stressifier (ukrainian-word-stress) ─► ipa-uk ─► CharTokenizer ─► tokens ┐
                                                                               ├─ ONNX Runtime
аудіо ─ VoiceEncoder (PyTorch, voice_encoder.pt) ─► ембеддінг [256] ───────────┘
        holos_cpu_int8.onnx ─► float32 wav ─► нормалізація ─► WAV 24 кГц
```

Оптимізації, зроблені в цьому репозиторії (виміряно на Xeon E5-2650, 12 потоків):

- `intra_op_num_threads = min(TTS_THREADS, 9)` — виміряно як найшвидше
- `session.run(["audio"], ...)` — відкидання зайвого виходу `audio_lengths`
- `lru_cache` на препроцесі частин + **prefetch** підготовки частини `i+1` під час генерації частини `i`
- послідовна обробка частин (запити обслуговуються по черзі)

## Бенчмарки (Xeon E5-2650, 12 потоків, теплий стан)

| Сценарій | Час |
|---|---|
| Одне речення (теплий замір) | **1.3–1.9 с** |
| 3 довгі речення (3 частини) | **~8.4 с** |
| Перший запуск (завантаження моделей) | до ~10 хв (volume cache) |

Для порівняння: у парних A/B-замірах чистий PyTorch-бекенд (official HolosTTS)
на тих самих текстах давав 5.3 с / 15.0 с — гібридний INT8-ONNX-шлях швидший на 27–35%.

## Розробка і тестування

```bash
python -m py_compile app/*.py          # синтаксис (те саме робить CI)
docker compose up -d --build           # збірка + запуск
docker compose logs -f holos-hybrid     # логи (перший запуск качає моделі)
```
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
