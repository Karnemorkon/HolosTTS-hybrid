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
session = ort.InferenceSession(
    _hf(ONNX_FILE),
    sess_options=_sess_options,
    providers=["CPUExecutionProvider"],
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
def list_voices():
    return list(voices.keys())


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
        if voice not in voices:
            raise ValueError("Невідомий пресет голосу: %s" % voice)
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
def _prepare_part(part):
    """Текст -> IPA-фонемна стрічка (результат кешується lru_cache)."""
    part = _strip_wrappers(part)
    if not part:
        return None
    part = _apply_stress_markup(part)
    part = _standardize_dashes(part)
    part = _ensure_terminal_punct(part)
    part = part.replace(' - ', ': ')
    return ipa(_stressify(part))


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


def synthesize_stream(text, voice, speed=1.0):
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
                pending = pool.submit(_prepare_part, parts[0])
                for i, part in enumerate(parts):
                    phonemes = pending.result()
                    if i + 1 < len(parts):
                        pending = pool.submit(_prepare_part, parts[i + 1])
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


def synthesize(text, voice, speed=1.0):
    """
    Синтез мови цілком.
      voice: str (пресет), np.ndarray або torch.Tensor (клонований ембеддінг).
      Повертає (SAMPLE_RATE, int16 numpy).
    """
    chunks = list(synthesize_stream(text, voice, speed))
    if not chunks:
        raise ValueError("Не вдалося отримати фонеми з тексту")

    data = np.concatenate(chunks)
    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak
    data = np.clip(data * 32767, -32768, 32767).astype(np.int16)
    return SAMPLE_RATE, data
