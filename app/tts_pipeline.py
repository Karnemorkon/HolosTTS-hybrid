"""HolosTTS HYBRID pipeline: PyTorch-енкодер голосу + ONNX-генератор (INT8, CPU).

СХЕМА ЗШИВАННЯ (див. також _bind_inputs):

    текст ──Stressifier (ukrainian-word-stress)──> текст із наголосами
         ──ipa (ipa-uk)──> фонемна стрічка
         ──CharTokenizer(vocab.json)──> int64 tokens [1, N]  ────────┐
                                                                     ├─> ONNX Runtime
    аудіо ──VoiceEncoder (офіційний PyTorch-клас)──> emb [1, 256]  ──┘   holos_cpu_int8.onnx
                                                                          ──> float32 audio

Фактична схема графа (перевірено інтроспекцією сесії):
  inputs : tokens int64 [1, text_length] | voice float32 [1, 256] | speed float32 [1]
  outputs: audio float32 [1, audio_samples] | audio_lengths int64 [1]

Офіційні пакети автора (holos, ukrainian-word-stress, ipa-uk) у цьому варіанті
використовуються ЛИШЕ для:
  1) коректного завантаження об'єкта класу VoiceEncoder з voice_encoder.pt
     (клас + state_dict визначені саме в пакеті holos);
  2) препроцесингу тексту (наголоси + IPA-фонеми) перед токенізацією.
Синтез звуку виконує ВИКЛЮЧНО onnxruntime (CPUExecutionProvider).
"""
import functools
import json
import os
import re
import threading
from unicodedata import normalize

from app.threads_config import apply_threads

THREADS = apply_threads()  # до import torch: OMP/MKL env + torch.set_num_threads

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download

from ipa_uk import ipa
from ukrainian_word_stress import Stressifier, StressSymbol
from holos.models import VoiceEncoder

MODEL_REPO = "patriotyk/HolosTTS"
ONNX_FILE = "holos_cpu_int8.onnx"
SAMPLE_RATE = 24000
VOICE_DIM = 256  # 128 acoustic + 128 prosodic (див. HolosTTS.encode)

MODELS_DIR = os.environ.get("HF_HOME", "/app/models")

_INFER_LOCK = threading.Lock()

_stressify = Stressifier()

# Режим автоматичного наголошування: 'rule' (словниковий, швидший, ~0.5 с
# на речення) або 'byt5' (нейромережевий patriotyk/stressifier-byt5-g2p-
# ctranslate2: якісніший на рідкісних/власних назвах, але ~1.7 с на CPU).
_STRESS_MODE = os.environ.get('HOLOS_STRESS', 'rule').strip().lower()
if _STRESS_MODE not in ('rule', 'byt5'):
    print('[hybrid] HOLOS_STRESS невідомий, використовую rule', flush=True)
    _STRESS_MODE = 'rule'

_byt5_instance = None
_byt5_lock = threading.Lock()


def _get_byt5():
    # Ленива ініціалізація: модель ~1.2 ГБ качається у volume HF_HOME один раз.
    global _byt5_instance
    if _byt5_instance is None:
        with _byt5_lock:
            if _byt5_instance is None:
                import ctranslate2
                from ukrainian_stressifier_byt5_ct2 import UkrainianStressifier
                _orig = ctranslate2.Translator

                def _patched(*args, **kwargs):
                    # Бібліотека не приймає параметри потоків - підміняємо конструктор.
                    kwargs.setdefault('intra_threads', min(THREADS, 8))
                    kwargs.setdefault('inter_threads', 1)
                    return _orig(*args, **kwargs)

                ctranslate2.Translator = _patched
                try:
                    _byt5_instance = UkrainianStressifier()
                finally:
                    ctranslate2.Translator = _orig
                print('[hybrid] ByT5-стресифікатор готово', flush=True)
    return _byt5_instance


def _auto_stress(text):
    if _STRESS_MODE == 'byt5':
        return _get_byt5().apply_stress_marks(text)
    return _stressify(text)


# ----------------------------------------------------------------------------
# 1. ЗАВАНТАЖЕННЯ АРТЕФАКТІВ (у спільний volume holos_models_cache:/app/models)
# ----------------------------------------------------------------------------
def _hf(filename):
    return hf_hub_download(repo_id=MODEL_REPO, filename=filename)


def _ensure_vocab():
    """
    Словник токенізації живе всередині holos.pt (ключ 'vocab') і є звичайним
    списком символів. Один раз витягуємо його у vocab.json, щоб далі не
    вантажити 542-мб holos.pt взагалі: holos.pt потрібен лише для видобутку
    словника - качається рівно один раз у спільний volume.
    """
    vocab_path = os.path.join(MODELS_DIR, "vocab.json")
    if os.path.isfile(vocab_path):
        with open(vocab_path, encoding="utf-8") as f:
            return json.load(f)
    params = torch.load(_hf("holos.pt"), map_location="cpu", weights_only=True)
    raw = params["vocab"]
    vocab = raw if isinstance(raw, list) else list(raw)
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    return vocab


class CharTokenizer:
    """Посимвольний токенізатор за словником із holos.pt.

    Поведінка відповідає Tokenizer з пакета holos (символ -> id, невідомі
    символи пропускаються); реалізація власна, мінімально достатня для
    укладання токенів у ONNX-вхід.
    """

    def __init__(self, vocab):
        self.index = {ch: i for i, ch in enumerate(vocab)}

    def __call__(self, text):
        # невідомі символи пропускаються (поведінка як у пакеті holos)
        return [self.index[ch] for ch in text if ch in self.index]


tokenizer = CharTokenizer(_ensure_vocab())

# Офіційний енкодер голосу: клас VoiceEncoder + voice_encoder.pt з пакета holos.
voice_encoder = VoiceEncoder(MODEL_REPO, device="cpu")

# Пресети голосів - словник тензорів [256] з voices.pt (34 kB).
voices = torch.load(_hf("voices.pt"), map_location="cpu", weights_only=True)

# ONNX-генератор: квантована INT8 модель на CPUExecutionProvider.
_sess_options = ort.SessionOptions()
_ort_env = os.environ.get("ORT_INTRA_THREADS", "").strip()
_sess_options.intra_op_num_threads = int(_ort_env) if _ort_env else min(THREADS, 9)
_sess_options.inter_op_num_threads = 1
_sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# Вибір EP: ORT_PROVIDER=cpu|cuda|openvino|auto (auto пробує прискорення,
# фільтрує за наявністю у цьому білді onnxruntime).
_PROVIDER = os.environ.get("ORT_PROVIDER", "auto").strip().lower()
_PROVIDERS_MAP = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "openvino": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    "auto": ["CUDAExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"],
}
_wanted = _PROVIDERS_MAP.get(_PROVIDER, _PROVIDERS_MAP["auto"])
_available = ort.get_available_providers()
_providers = [p for p in _wanted if p in _available] or ["CPUExecutionProvider"]
print("[hybrid] ORT_PROVIDER=%s -> %s (наявні: %s)" % (_PROVIDER, _providers, _available), flush=True)
session = ort.InferenceSession(
    _hf(ONNX_FILE),
    sess_options=_sess_options,
    providers=_providers,
)


# ----------------------------------------------------------------------------
# 2. БІНДІНГ ВХІДНИХ НОД ONNX-СЕСІЇ
# ----------------------------------------------------------------------------
# Автор не публікує схему графа у документації, тому імена/форми нод
# визначаємо детерміновано за сигнатурою КОЖНОЇ ноди:
#   string                          -> одразу фонемна стрічка (без токенізації)
#   float32 з останнім виміром 256  -> voice  (ембеддінг з PyTorch-енкодера)
#   int64  ранг 1 ([1])             -> lengths (довжина послідовності)
#   int64  ранг 2 ([1, N])          -> tokens (id фонем з CharTokenizer)
#   float32 ранг 1 ([1])            -> контроли (speed і т.п.), за замовч. 1.0
# Біндінг кешується і друкується у лог при старті.
def _bind_inputs(sess):
    binding = {"controls": {}}
    unknown = []
    for spec in sess.get_inputs():
        name, typ, shape = spec.name, spec.type, spec.shape
        rank = len(shape) if isinstance(shape, (list, tuple)) else 0
        last_dim = shape[-1] if shape else None
        low = name.lower()
        if typ == "tensor(string)":
            binding["phonemes"] = name
        elif typ in ("tensor(float)", "tensor(float32)") and (last_dim == VOICE_DIM or "voice" in low or "style" in low):
            binding["voice"] = name
        elif typ == "tensor(int64)" and rank <= 1:
            binding["lengths"] = name
        elif typ == "tensor(int64)":
            binding["tokens"] = name
        elif typ in ("tensor(float)", "tensor(float32)") and rank <= 1:
            # скалярні керуючі входи (speed тощо)
            binding["controls"][name] = 1.0
        else:
            unknown.append(name)
    if unknown:
        raise RuntimeError("Невідомі вхідні ноди ONNX: %s" % unknown)
    if "phonemes" not in binding and not ("tokens" in binding and "voice" in binding):
        raise RuntimeError("Не вдалося зв'язати входи ONNX: %s" % binding)
    return binding


INPUT_BINDING = _bind_inputs(session)
_OUTPUT_NAMES = [o.name for o in session.get_outputs()]
print("[hybrid] ONNX inputs:", {s.name: (s.type, s.shape) for s in session.get_inputs()}, flush=True)
print("[hybrid] ONNX outputs:", {o.name: (o.type, o.shape) for o in session.get_outputs()}, flush=True)
print("[hybrid] input binding:", INPUT_BINDING, flush=True)


# ----------------------------------------------------------------------------
# 3. ЕНКОДЕР ГОЛОСУ (PyTorch) -> numpy для ONNX
# ----------------------------------------------------------------------------
SPEAKERS_DIR = os.path.join(MODELS_DIR, 'speakers')
_VOICE_ID_RE = re.compile(r'^[\w \-]{1,40}$', re.UNICODE)
_saved_voices = {}


def _load_saved_voices():
    # Збережені клони з UI (volume speakers/*.npy) -> кеш _saved_voices.
    if not os.path.isdir(SPEAKERS_DIR):
        return
    for fn in sorted(os.listdir(SPEAKERS_DIR)):
        if fn.endswith('.npy'):
            name = fn[:-4]
            try:
                arr = np.load(os.path.join(SPEAKERS_DIR, fn))
                _saved_voices[name] = arr.astype(np.float32).reshape(-1)
            except Exception as e:
                print('[hybrid] пропущено пошкоджений голос %s: %s' % (name, e), flush=True)


_load_saved_voices()


def save_voice(name, emb):
    # Зберігає ембеддінг [256] під ім'ям (викликає UI після прослуховування).
    name = (name or '').strip()
    if not _VOICE_ID_RE.match(name):
        raise ValueError('Некоректна назва голосу: 1-40 символів (літери/цифри/пробіл/або _-)')
    if name in voices:
        raise ValueError('Назва збігається з пресетом: %s' % name)
    arr = np.asarray(emb, dtype=np.float32).reshape(-1)
    if arr.size != VOICE_DIM:
        raise ValueError('Ембеддінг має бути розміром %d' % VOICE_DIM)
    os.makedirs(SPEAKERS_DIR, exist_ok=True)
    path = os.path.join(SPEAKERS_DIR, name + '.npy')
    np.save(path, arr)
    _saved_voices[name] = arr
    print('[hybrid] голос збережено: %s' % path, flush=True)
    return name


def custom_voices():
    # Рескан каталогу: .npy, покладені у volume, зʼявляються без рестарту.
    if os.path.isdir(SPEAKERS_DIR):
        for fn in sorted(os.listdir(SPEAKERS_DIR)):
            if fn.endswith(".npy") and fn[:-4] not in _saved_voices:
                try:
                    arr = np.load(os.path.join(SPEAKERS_DIR, fn))
                    _saved_voices[fn[:-4]] = arr.astype(np.float32).reshape(-1)
                except Exception:
                    pass
    return sorted(_saved_voices.keys())


def list_voices():
    return list(voices.keys()) + custom_voices()


def encode_voice(audio_path):
    """Аудіо -> ембеддінг [256] float32 numpy (вхідна нода ONNX 'voice')."""
    with _INFER_LOCK:
        with torch.inference_mode():
            emb = voice_encoder.encode(audio_path)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)


def _resolve_voice(voice):
    if isinstance(voice, np.ndarray):
        return voice.astype(np.float32).reshape(1, VOICE_DIM)
    if isinstance(voice, torch.Tensor):
        return voice.cpu().numpy().astype(np.float32).reshape(1, VOICE_DIM)
    if isinstance(voice, str):
        if voice in _saved_voices:
            return _saved_voices[voice].reshape(1, VOICE_DIM)
        if voice not in voices:
            path = os.path.join(SPEAKERS_DIR, voice + ".npy")
            if os.path.isfile(path):
                arr = np.load(path).astype(np.float32).reshape(-1)
                _saved_voices[voice] = arr
                return arr.reshape(1, VOICE_DIM)
            raise ValueError("Невідомий голос: %s" % voice)
        return voices[voice].cpu().numpy().astype(np.float32).reshape(1, VOICE_DIM)
    raise ValueError("Невідомий тип голосу: %s" % type(voice))


# ----------------------------------------------------------------------------
# 4. ТЕКСТ -> ФОНЕМИ (офіційні пакети автора) -> СИНТЕЗ ЧЕРЕЗ ONNX
# ----------------------------------------------------------------------------
# Межі нарізки тексту та нормалізація пунктуації (див. split_to_parts).
_BREAK_CHARS = frozenset('.?!:')
_GROUP_LIMIT = 100
_TERMINAL_CHARS = frozenset('.?!:-')
_DASH_CLASS = re.compile('[\u1806\u2010\u2011\u2012\u2013\u2014\u2015\u207b\u208b\u2212\u2e3a\u2e3b]')


def _collapse_newlines(text):
    """Переноси рядка -> '. ' або пробіли (щоб не різати посеред слова)."""
    text = re.sub(r'(\w+[^.,!:?\-])\n', r'\1. ', text)
    return text.replace('\n', ' ')


def split_to_parts(text, group=True):
    """Нарізка тексту на частини по реченнях.

    Поведінка відповідає препроцесу HolosTTS (аналогічно автор ріже текст
    у своєму gradio-демо: https://github.com/patriotyk/HolosTTS); реалізація
    тут власна: речення групуються у частини завдовжки до 100 символів
    (group=True), розрив можливий лише після '.?!:' з наступним пробілом.
    """
    text = _collapse_newlines(text)
    parts = []
    tail = []
    last_index = len(text) - 1
    for pos, char in enumerate(text):
        tail.append(char)
        if char in _BREAK_CHARS and pos < last_index and text[pos + 1] == ' ':
            if group and len(tail) <= _GROUP_LIMIT:
                continue
            parts.append(''.join(tail))
            tail = []
    parts.append(''.join(tail))
    return parts


def _strip_wrappers(text):
    """Прибрати крайові пробіли та ASCII-лапки."""
    return text.strip().replace('"', '')


def _apply_stress_markup(text):
    """'+' як маркер наголосу -> комбінований наголос + NFKC-нормалізація."""
    text = text.replace('+', StressSymbol.CombiningAcuteAccent)
    return normalize('NFKC', text)


def _standardize_dashes(text):
    """Екзотичні тире/мінуси -> звичайний '-'."""
    return _DASH_CLASS.sub('-', text)


def _ensure_terminal_punct(text):
    """Речення мусить завершуватися пунктуацією."""
    if text[-1] not in _TERMINAL_CHARS:
        text += '.'
    return text


@functools.lru_cache(maxsize=2048)
def _prepare_part(part, raw_stress=False):
    # Кеш lru_cache; raw_stress=True пропускає автонаголос (власні +/U+0301).
    part = _strip_wrappers(part)
    if not part:
        return None
    part = _apply_stress_markup(part)
    part = _standardize_dashes(part)
    part = _ensure_terminal_punct(part)
    part = part.replace(' - ', ': ')
    if raw_stress:
        return ipa(part)
    return ipa(_auto_stress(part))


def _generate_part(phonemes, voice_arr, speed=1.0):
    """Одна частина: масиви numpy -> onnxruntime -> float32 wav."""
    # --- МІСЦЕ ЗШИВАННЯ -------------------------------------------------
    # Вектор ембеддінгу, отриманий офіційним ПайТорч-енкодером (VoiceEncoder),
    # прокидається у вхідну ноду ONNX-сесії як numpy float32 [1, 256].
    # Текстова частина: фонеми -> токені -> int64 нода 'tokens'.
    # --------------------------------------------------------------------
    feeds = {
        name: np.asarray(
            [speed if name.lower() == "speed" else value], dtype=np.float32
        )
        for name, value in INPUT_BINDING.get("controls", {}).items()
    }
    if "phonemes" in INPUT_BINDING:
        # граф сам приймає фонемну стрічку - токенізація не потрібна
        feeds[INPUT_BINDING["phonemes"]] = np.asarray([[phonemes]], dtype=object)
    else:
        tokens = [0] + tokenizer(phonemes) + [0]  # BOS/EOS = 0, як у HolosTTS.generate
        feeds[INPUT_BINDING["tokens"]] = np.asarray([tokens], dtype=np.int64)
        if "lengths" in INPUT_BINDING:
            feeds[INPUT_BINDING["lengths"]] = np.asarray([len(tokens)], dtype=np.int64)
    feeds[INPUT_BINDING["voice"]] = voice_arr

    outputs = session.run(["audio"], feeds)  # audio_lengths не потрібен
    return np.asarray(outputs[0], dtype=np.float32).reshape(-1)


def _validate_text(text):
    if len(text.strip()) < 10:
        raise ValueError("Текст закороткий (мінімум 10 символів)")
    if len(text) > 50000:
        raise ValueError("Текст мусить бути менше 50k символів")


def synthesize_stream(text, voice, speed=1.0, raw_stress=False):
    """Генератор синтезу: yield частини хвилі (float32) по мірі генерації.

    Використовується для потокового /v1/audio/speech. Блокування утримується
    на весь час ітерації - запити обслуговуються по черзі.
      voice: str (пресет), np.ndarray або torch.Tensor (клонований ембеддінг).
    """
    _validate_text(text)
    voice_arr = _resolve_voice(voice)
    parts = list(split_to_parts(text))
    with _INFER_LOCK:
        if parts:
            # препроцес частини i+1 йде паралельно з генерацією частини i
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(_prepare_part, parts[0], raw_stress)
                for i, part in enumerate(parts):
                    phonemes = pending.result()
                    if i + 1 < len(parts):
                        pending = pool.submit(_prepare_part, parts[i + 1], raw_stress)
                    if phonemes:
                        yield _generate_part(phonemes, voice_arr, speed)


def float_to_pcm16(chunk):
    """float32 частини -> int16 (для потокової відповіді).

    Глобальну пікову нормалізацію у стрімінгу застосувати неможливо,
    тому частина лише відтинається, якщо вийшла за межі [-1, 1].
    """
    peak = np.abs(chunk).max()
    if peak > 1.0:
        chunk = chunk / peak
    return np.clip(chunk * 32767, -32768, 32767).astype(np.int16)


def synthesize(text, voice, speed=1.0, raw_stress=False):
    """
    Синтез мови цілком.
      voice: str (пресет), np.ndarray або torch.Tensor (клонований ембеддінг).
      Повертає (SAMPLE_RATE, int16 numpy).
    """
    chunks = list(synthesize_stream(text, voice, speed, raw_stress))
    if not chunks:
        raise ValueError("Не вдалося отримати фонеми з тексту")

    data = np.concatenate(chunks)
    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak
    data = np.clip(data * 32767, -32768, 32767).astype(np.int16)
    return SAMPLE_RATE, data
