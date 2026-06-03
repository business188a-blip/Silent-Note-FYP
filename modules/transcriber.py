"""
modules/transcriber.py
──────────────────────────────────────────────────────────────────────────────
Wraps OpenAI Whisper for fully offline, CPU-based speech-to-text.
──────────────────────────────────────────────────────────────────────────────
"""

import threading

import whisper

from modules.urdu_learning import clean_and_learn


# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_NAME    = "small"
DEVICE        = "cpu"
CHUNK_SECONDS = 30

URDU_PROMPT = "یہ اردو میں ایک گفتگو ہے۔"

SCRIPT_PROMPTS = {
    "ur": URDU_PROMPT,
}

FAST_OPTIONS = dict(
    fp16                        = False,
    beam_size                   = 1,
    best_of                     = 1,
    temperature                 = 0,
    condition_on_previous_text  = False,
    no_speech_threshold         = 0.6,
    compression_ratio_threshold = 2.4,
)

QUALITY_OPTIONS = dict(
    fp16                        = False,
    beam_size                   = 5,
    best_of                     = 5,
    temperature                 = 0,
    condition_on_previous_text  = False,
    no_speech_threshold         = 0.45,
    compression_ratio_threshold = 2.4,
)

QUALITY_LANGUAGES = {"en", "ur"}

URDU_REPLACEMENTS = {
    "ویبسائٹ": "ویب سائٹ",
    "ویب سائٹ": "ویب سائٹ",
    "دزائن": "ڈیزائن",
    "دیزائن": "ڈیزائن",
    "دیت لائن": "ڈیڈ لائن",
    "ڈیٹ لائن": "ڈیڈ لائن",
    "کنفم": "کنفرم",
    "نیکسٹ میٹنگ": "اگلی میٹنگ",
    "مندے": "پیر",
}


class Transcriber:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model      = None
            cls._instance._loaded     = False
            cls._instance._load_lock  = threading.Lock()
            cls._instance._infer_lock = threading.Lock()
        return cls._instance

    def load_model(self, on_progress=None):
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            if on_progress:
                on_progress("Loading Whisper small model...")
            self._model = whisper.load_model(
                MODEL_NAME,
                device    = DEVICE,
                in_memory = True,
            )
            self._loaded = True
            if on_progress:
                on_progress("Whisper model ready.")

    def is_loaded(self):
        return self._loaded

    def transcribe_file(self, wav_path, language=None, task="transcribe"):
        self._ensure_loaded()
        if language is None or language == "auto":
            language = self.detect_language(wav_path)
        options = dict(QUALITY_OPTIONS if language in QUALITY_LANGUAGES else FAST_OPTIONS)
        options["task"]     = task
        options["language"] = language
        if language in SCRIPT_PROMPTS:
            options["initial_prompt"] = SCRIPT_PROMPTS[language]
        with self._infer_lock:
            raw = self._model.transcribe(wav_path, **options)
        segments = []
        for seg in raw.get("segments", []):
            segments.append({
                "id":      seg["id"],
                "start":   round(seg["start"], 2),
                "end":     round(seg["end"],   2),
                "text":    self._clean_text(seg["text"].strip(), language),
                "speaker": "Speaker 1",
            })
        return {
            "text":     self._clean_text(raw["text"].strip(), language),
            "segments": segments,
            "language": raw.get("language", language),
        }

    def transcribe_chunks(self, wav_path, language=None):
        self._ensure_loaded()
        if language is None or language == "auto":
            language = self.detect_language(wav_path)
        audio         = whisper.load_audio(wav_path)
        total_samples = len(audio)
        chunk_samples = CHUNK_SECONDS * whisper.audio.SAMPLE_RATE
        decode_params = QUALITY_OPTIONS if language in QUALITY_LANGUAGES else FAST_OPTIONS
        decode_opts = whisper.DecodingOptions(
            task        = "transcribe",
            fp16        = decode_params["fp16"],
            language    = language,
            beam_size   = decode_params["beam_size"],
            best_of     = decode_params["best_of"],
            temperature = decode_params["temperature"],
            prompt      = SCRIPT_PROMPTS.get(language, ""),
        )
        offset = 0
        while offset < total_samples:
            chunk = audio[offset: offset + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = whisper.pad_or_trim(chunk)
            mel = whisper.log_mel_spectrogram(chunk).to(DEVICE)
            with self._infer_lock:
                result = whisper.decode(self._model, mel, decode_opts)
            text = result.text.strip()
            if text:
                yield self._clean_text(text, language)
            offset += chunk_samples

    def detect_language(self, wav_path):
        self._ensure_loaded()
        audio = whisper.load_audio(wav_path)
        audio = whisper.pad_or_trim(audio)
        mel   = whisper.log_mel_spectrogram(audio).to(DEVICE)
        with self._infer_lock:
            _, probs = self._model.detect_language(mel)

        # Urdu aur Hindi acoustically similar hain — small model often
        # misidentifies Urdu as Hindi. Agar top language Hindi hai aur
        # Urdu probability bhi 10% se zyada hai toh Urdu prefer karo.
        top_lang = max(probs, key=probs.get)
        if top_lang == "hi" and probs.get("ur", 0) > 0.10:
            return "ur"

        return top_lang

    def _ensure_loaded(self):
        if not self._loaded:
            self.load_model()

    @staticmethod
    def _clean_text(text, language):
        if language != "ur":
            return text
        cleaned = text
        for wrong, right in URDU_REPLACEMENTS.items():
            cleaned = cleaned.replace(wrong, right)
        return clean_and_learn(cleaned)
