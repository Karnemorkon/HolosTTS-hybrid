"""Вербалізація цифр, дат і часу в текст перед синтезом (M2M100 + CTranslate2).

Модель: Morkon/holos-native-models, subfolder verbalizer/ct2-int8 (CTranslate2,
int8); токенізатор: той самий repo, subfolder verbalizer/onnx-int8.
Код адаптовано з авторського
https://huggingface.co/spaces/patriotyk/styletts2-ukrainian (verbalizer.py).

Відмінність від автора: усі речення обробляються ОДНИМ translate_batch
(автор викликав по одному реченню - повільніше).

Модель (~470 МБ stored-int8) завантажується ліниво при першому виклику у
спільний volume HF_HOME (holos_models_cache) - качається рівно один раз.
"""
import re
import threading

from app.threads_config import resolve_threads

MODEL_ID = "Morkon/holos-native-models"
TOKENIZER_ID = "Morkon/holos-native-models"

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")

_instance = None
_lock = threading.Lock()


def needs_verbalize(text):
    """True, якщо текст містить цифри (числа, дати, час)."""
    return any(ch.isdigit() for ch in text or "")


class Verbalizer:
    def __init__(self):
        import ctranslate2
        from huggingface_hub import snapshot_download
        from transformers import M2M100Tokenizer

        print("[verbalizer] завантаження моделі " + MODEL_ID + " ...", flush=True)
        import os
        root = snapshot_download(
            repo_id=MODEL_ID,
            allow_patterns=["verbalizer/ct2-int8/*"],
        )
        model_path = os.path.join(root, "verbalizer", "ct2-int8")
        self.translator = ctranslate2.Translator(
            model_path,
            device="cpu",
            compute_type="int8",
            intra_threads=resolve_threads(),
        )
        self.tokenizer = M2M100Tokenizer.from_pretrained(
            TOKENIZER_ID, subfolder="verbalizer/onnx-int8")
        self.tokenizer.src_lang = "uk"
        print("[verbalizer] готово", flush=True)

    def verbalize(self, text):
        """Вербалізує ВСІ речення тексту одним батчем translate_batch."""
        text = (text or "").strip()
        if not text:
            return text
        parts = [p for p in _SENT_SPLIT.split(text) if p]
        sources = [
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(p))
            for p in parts
        ]
        lang_token = self.tokenizer.lang_code_to_token["uk"]
        results = self.translator.translate_batch(
            sources,
            target_prefix=[[lang_token] for _ in parts],
            beam_size=1,
            num_hypotheses=1,
            use_vmap=True,
        )
        out = []
        for res in results:
            target = res.hypotheses[0][1:]  # без мовного токена
            out.append(
                self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target))
            )
        return " ".join(s.strip() for s in out)


def _get():
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = Verbalizer()
    return _instance


def verbalize(text):
    """Публічний виклик: вербалізує текст (лінива ініціалізація)."""
    return _get().verbalize(text)


def auto_verbalize(text):
    """Вербалізує лише якщо у тексті є цифри, інакше повертає як є."""
    if needs_verbalize(text):
        return verbalize(text)
    return text
