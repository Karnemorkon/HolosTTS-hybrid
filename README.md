# HolosTTS-hybrid

> **English summary:** production-ready Dockerized **Ukrainian text-to-speech** service
> (REST API + Web UI, port 8002). It runs the [HolosTTS](https://github.com/patriotyk/HolosTTS)
> neural network with a **PyTorch voice encoder + INT8 ONNX generator** on CPU via
> ONNX Runtime, adds automatic CPU thread management, sentence-level caching/prefetching,
> and **verbalization of numbers, dates and time** (M2M100 + CTranslate2) before synthesis.
> Model weights are MIT-licensed by [patriotyk](https://huggingface.co/patriotyk).

Український синтез мови у Docker-контейнері: FastAPI (+ Gradio UI), виключно CPU,
жодних GPU-залежностей. Перший запуск сам довантажує моделі (~3.5 ГБ) у named volume.

## Можливості

- **Синтез українською** 24 кГц з 28 пресетами голосів + клонування голосу з аудіо-промпта
- **INT8 ONNX-генератор** (`holos_cpu_int8.onnx`) на ONNX Runtime — швидше за fp32 на CPU
- **Автовербалізація**: `250 грн`, `22.08.2025 о 15:30` → «двісті пʼятдесят гривень…»
  (M2M100-CTranslate2, ліниве завантаження ~1.9 ГБ один раз на volume)
- **Автопотоки CPU**: 80% потоків хоста, обрізаних до cgroup-квоти (або `TTS_THREADS`)
- **Нарізка на речення + LRU-кеш препроцесу + prefetch** наступної частини
- **Продакшен-compose**: healthcheck, ліміти CPU/RAM, ротація логів, автоперезапуск

## Швидкий старт

```bash
git clone <this-repo-url>
cd HolosTTS-hybrid
docker compose up -d --build
# перший запуск: довантаження моделей у volume (кілька хвилин), потім:
curl http://localhost:8002/healthz
# Web UI: http://localhost:8002/
```

Ознака готовності: `{status:ok,engine:hybrid-onnx-int8,...,threads:12}`.

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

## Конфігурація

| Змінна | Типово | Що робить |
|---|---|---|
| `TTS_THREADS` | `min(80% потоків хоста, cgroup-quota)` | PyTorch/препроцес |
| `ORT_INTRA_THREADS` | `min(TTS_THREADS, 9)` | ONNX Runtime intra-op |
| `HF_HOME` | `/app/models` | кеш моделей (volume `holos_models_cache`) |

Приклади — у [`.env.example`](.env.example). Ліміти контейнера (`cpus: 12.0`,
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
- `session.run([audio], ...)` — відкидання зайвого виходу `audio_lengths`
- `lru_cache` на препроцесі частин + **prefetch** підготовки частини `i+1` під час генерації частини `i`
- послідовна обробка частин (запити обслуговуються по черзі)

## Бенчмарки (той самий хост, теплий A/B почерзі)

| Сценарій | Час |
|---|---|
| Одне речення (середнє з 3) | **~3.9 с** |
| 3 довгі речення (3 частини) | **~9.7 с** |
| Перший запуск (завантаження моделей) | до ~10 хв (volume cache) |

Для порівняння: чистий PyTorch-бекенд (official HolosTTS) на тому ж хості давав
5.3 с / 15.0 с на тих самих текстах — гібридний INT8-ONNX-шлях швидший на 27–35%.

## Розробка і тестування

```bash
python -m py_compile app/*.py          # синтаксис (те саме робить CI)
docker compose up -d --build           # збірка + запуск
docker compose logs -f holos-hybrid     # логи (перший запуск качає моделі)
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
