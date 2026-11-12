import sys
import os
import datetime
import json
import re
import subprocess
import tempfile
import threading
import time
import wave
from enum import Enum
from abc import abstractmethod
import openai
from sdk import audio_format
from sdk.transcription_result import TranscriptSegment, TranscriptionHypothesis
from tsutils import utilities
# torch, whisper, faster_whisper and deepgram are imported where they are used: importing
# torch alone takes seconds, and only the selected speech to text engine needs any of them.


class STTEnum(Enum):
    """Supported Speech To Text Models
    """
    WHISPER_LOCAL = 1
    WHISPER_API = 2
    WHISPER_CPP = 3
    DEEPGRAM_API = 4
    SENSEVOICE_LOCAL = 5


MODELS_DIR = f"{utilities.get_data_path(app_name='Transcribe')}/models/"
FASTER_WHISPER = "faster-whisper"
OPENAI_WHISPER = "openai-whisper"


def _installed(module: str) -> bool:
    import importlib.util  # pylint: disable=import-outside-toplevel
    if module in sys.modules:
        return sys.modules[module] is not None
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def pick_local_whisper_engine(setting: str = "auto") -> str:
    """Choose the local Whisper engine.

    faster-whisper is the standard engine (installed by requirements.txt). openai-whisper needs
    PyTorch (gigabytes) and is only used when it is chosen, or installed and faster-whisper is not.
    """
    setting = str(setting or "auto").strip().lower()
    if setting in (FASTER_WHISPER, OPENAI_WHISPER):
        return setting
    if _installed("faster_whisper") or not _installed("whisper"):
        return FASTER_WHISPER
    return OPENAI_WHISPER


# Models openai-whisper's own download code (below) knows; others are loaded by name.
OPENAI_WHISPER_URLS = ("tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3")


def gpu_check() -> int:
    """0 = faster-whisper can use an NVIDIA GPU, 2 = no NVIDIA GPU, 1 = a GPU but its libraries do not load."""
    add_nvidia_dll_directories()
    try:
        import ctranslate2  # pylint: disable=import-outside-toplevel
        count = ctranslate2.get_cuda_device_count()
    except Exception as exception:  # pylint: disable=broad-except
        print(f"CTranslate2 could not be loaded: {exception}")
        return 1
    if count <= 0:
        print("No NVIDIA GPU was found.")
        return 2
    if sys.platform == "win32":
        import ctypes  # pylint: disable=import-outside-toplevel
        for library in ("cublas64_12.dll", "cudnn64_9.dll"):
            try:
                ctypes.WinDLL(library)
            except OSError as exception:
                print(f"{library} could not be loaded: {exception}")
                return 1
    print(f"NVIDIA GPUs found: {count}; CTranslate2 {ctranslate2.__version__}")
    return 0


def add_nvidia_dll_directories():
    """Windows: let CTranslate2 find the CUDA libraries that setup-gpu.bat installs with pip
    (nvidia-cublas-cu12, nvidia-cudnn-cu12). They live in site-packages and are not on PATH."""
    if sys.platform != "win32":
        return []
    import importlib.util  # pylint: disable=import-outside-toplevel
    added = []
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        spec = None
    for root in (list(spec.submodule_search_locations or []) if spec else []):
        for package in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
            folder = os.path.join(root, package, "bin")
            if os.path.isdir(folder) and folder not in added:
                try:
                    os.add_dll_directory(folder)  # pylint: disable=no-member
                except (AttributeError, OSError):
                    pass
                os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
                added.append(folder)
    return added


class STTModelFactory:
    """Factory class to get the appropriate STT Model
    """
    def get_stt_model_instance(self, stt_model: STTEnum, stt_model_config: dict):
        """Get the appropriate STT model class instance
        Args:
          stt_model: Speech to Text Model
          config: dict: Used to pass all configuration parameters
          model_file: str: OpenAI Transcription model for local transcription
        """
        if not isinstance(stt_model, STTEnum):
            raise TypeError('STTModelFactory: stt_model should be an instance of STTEnum')

        if stt_model == STTEnum.WHISPER_LOCAL:
            # How do we get a different model for whisper, tiny vs base vs medium
            # Model value is derived from command line args
            return WhisperSTTModel(stt_model_config=stt_model_config,
                                   background=bool(stt_model_config.get("background")))
        elif stt_model == STTEnum.WHISPER_API:
            return APIWhisperSTTModel(stt_model_config=stt_model_config)
        elif stt_model == STTEnum.WHISPER_CPP:
            return WhisperCPPSTTModel(stt_model_config=stt_model_config)
        elif stt_model == STTEnum.DEEPGRAM_API:
            return DeepgramSTTModel(stt_model_config=stt_model_config)
        elif stt_model == STTEnum.SENSEVOICE_LOCAL:
            return SenseVoiceSTTModel(stt_model_config=stt_model_config)
        raise ValueError("Unknown Speech to Text Model Type")


class STTModelInterface:
    """Interface all Speech To Text Models must adhere to
    """
    # True when the microphone and speaker workers may call transcribe_pcm at the same time.
    thread_safe = False

    def transcribe_pcm(self, pcm: bytes, sample_rate: int = audio_format.TARGET_SAMPLE_RATE):
        """Transcribe 16-bit mono PCM held in memory (the live path).

        Engines that can take audio from memory override this. The default writes one
        temporary WAV file for engines that only read files.
        """
        file_descriptor, path = tempfile.mkstemp(suffix=".wav")
        os.close(file_descriptor)
        try:
            audio_format.write_pcm16_wav(path, pcm, sample_rate)
            return self.get_transcription(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    @abstractmethod
    def get_transcription(self, wav_file_path: str):
        """Get transcription from the provided audio file
        """
        pass  # pylint: disable=W0107

    @abstractmethod
    def get_sentences(self, wav_file_path: str):
        """Get transcription from the provided audio file
           as individual sentences
        """
        pass  # pylint: disable=W0107

    @abstractmethod
    def process_response(self, response) -> str:
        """Extract transcription from the response of the specific STT Model
        """
        pass  # pylint: disable=W0107

    @abstractmethod
    def normalize_response(
        self,
        response,
        audio_start_seconds: float = 0.0,
        audio_end_seconds: float = 0.0,
    ) -> TranscriptionHypothesis:
        """Normalize a provider response for live transcription reconciliation."""
        pass  # pylint: disable=W0107


def _segment_from_mapping(segment_id: int, segment: dict, offset_seconds: float = 0.0) -> TranscriptSegment:
    return TranscriptSegment(
        id=int(segment.get("id", segment_id)),
        start_seconds=offset_seconds + float(segment.get("start", 0.0)),
        end_seconds=offset_seconds + float(segment.get("end", 0.0)),
        text=str(segment.get("text", "")).strip(),
        confidence=segment.get("confidence"),
    )


def _single_segment(
    text: str,
    audio_start_seconds: float,
    audio_end_seconds: float,
) -> list[TranscriptSegment]:
    text = str(text or "").strip()
    if not text:
        return []
    return [
        TranscriptSegment(
            id=0,
            start_seconds=audio_start_seconds,
            end_seconds=audio_end_seconds,
            text=text,
        )
    ]


class SenseVoiceSTTModel(STTModelInterface):
    """Optional Windows-only SenseVoiceSmall transcription backend."""

    SUPPORTED_LANGUAGES = {"zh", "en", "yue", "ja", "ko", "auto"}
    LANGUAGE_NAMES = {
        "chinese": "zh",
        "english": "en",
        "cantonese": "yue",
        "japanese": "ja",
        "korean": "ko",
        "auto": "auto",
    }

    def __init__(self, stt_model_config: dict):
        if sys.platform != "win32":
            raise RuntimeError("The experimental SenseVoice backend currently supports Windows only.")

        try:
            from funasr import AutoModel  # pylint: disable=import-outside-toplevel
            from funasr.utils.postprocess_utils import (  # pylint: disable=import-outside-toplevel
                rich_transcription_postprocess,
            )
        except ImportError as exception:
            raise RuntimeError(
                "SenseVoice dependencies are not installed. Run "
                "'pip install -r requirements-sensevoice.txt' from app\\transcribe."
            ) from exception

        self.lang = self._normalize_language(stt_model_config.get("audio_lang", "auto"))
        self.use_itn = bool(stt_model_config.get("use_itn", True))
        self.postprocess = rich_transcription_postprocess
        model_name = stt_model_config.get("model", "FunAudioLLM/SenseVoiceSmall")
        device = stt_model_config.get("device", "auto")
        if device == "auto":
            import torch  # pylint: disable=import-outside-toplevel
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print("[INFO] Using SenseVoiceSmall for transcription.")
        self.audio_model = AutoModel(
            model=model_name,
            vad_model="fsmn-vad",
            hub="hf",
            disable_pbar=True,
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
        )

    @classmethod
    def _normalize_language(cls, lang: str) -> str:
        normalized = str(lang).strip().lower()
        normalized = cls.LANGUAGE_NAMES.get(normalized, normalized)
        return normalized if normalized in cls.SUPPORTED_LANGUAGES else "auto"

    def set_lang(self, lang: str):
        """Set the recognition language, falling back to automatic detection."""
        self.lang = self._normalize_language(lang)

    @staticmethod
    def _audio_duration(wav_file_path: str) -> float:
        with wave.open(wav_file_path, "rb") as wav_file:
            return wav_file.getnframes() / float(wav_file.getframerate())

    @staticmethod
    def _sentence_segments(text: str, duration_seconds: float) -> list[dict]:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s*", text) if part.strip()]
        if not sentences:
            return []

        total_chars = sum(len(sentence) for sentence in sentences)
        start = 0.0
        segments = []
        for segment_id, sentence in enumerate(sentences):
            fraction = len(sentence) / total_chars if total_chars else 1 / len(sentences)
            end = duration_seconds if segment_id == len(sentences) - 1 else start + duration_seconds * fraction
            segments.append({"id": segment_id, "start": start, "end": end, "text": sentence})
            start = end
        return segments

    def get_transcription(self, wav_file_path: str) -> dict:
        """Transcribe a WAV file and normalize the result for the application."""
        raw_response = self.audio_model.generate(
            input=wav_file_path,
            cache={},
            language=self.lang,
            use_itn=self.use_itn,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        text = " ".join(
            self.postprocess(item.get("text", "")).strip()
            for item in raw_response
            if item.get("text")
        ).strip()
        return {
            "text": text,
            "segments": self._sentence_segments(text, self._audio_duration(wav_file_path)),
            "raw_response": raw_response,
        }

    def process_response(self, response) -> str:
        """Extract normalized transcription text."""
        return response.get("text", "").strip() if response else ""

    def normalize_response(
        self,
        response,
        audio_start_seconds: float = 0.0,
        audio_end_seconds: float = 0.0,
    ) -> TranscriptionHypothesis:
        """Normalize SenseVoice's application response shape."""
        text = self.process_response(response)
        segments = [
            _segment_from_mapping(index, segment, offset_seconds=audio_start_seconds)
            for index, segment in enumerate(response.get("segments", []))
        ] if response else []
        return TranscriptionHypothesis(
            provider="sensevoice",
            text=text,
            segments=segments or _single_segment(text, audio_start_seconds, audio_end_seconds),
            audio_start_seconds=audio_start_seconds,
            audio_end_seconds=audio_end_seconds,
        )

    def get_sentences(self, wav_file_path: str) -> list[str]:
        """Return timestamped sentences for batch file transcription."""
        response = self.get_transcription(wav_file_path)
        return [
            f"{datetime.timedelta(seconds=int(segment['start']))} - "
            f"{datetime.timedelta(seconds=int(segment['end']))}: {segment['text']}"
            for segment in response["segments"]
        ]

LOAD_RETRY_SECONDS = 30
# Used when Hugging Face itself cannot be reached (it is blocked or slow in some countries).
MODEL_DOWNLOAD_MIRROR = "https://hf-mirror.com"
_MODEL_FILES = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]


def bundled_model_dir(model: str) -> str | None:
    """A model shipped next to Transcribe.exe (models/faster-whisper-<name>), if there is one."""
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.getcwd()
    folder = os.path.join(base, "models", f"faster-whisper-{model}")
    return folder if os.path.isfile(os.path.join(folder, "model.bin")) else None


class WhisperSTTModel(STTModelInterface):
    """Speech to Text with a local Whisper model.

    Uses faster-whisper (CTranslate2, int8 on CPU): about four times faster than openai-whisper
    on a CPU, and it does not need PyTorch. openai-whisper is optional and used only when
    ``General.local_engine`` chooses it (or when it is the only engine installed).
    """
    use_vad = True

    def __init__(self, stt_model_config: dict, background: bool = False):
        self.model = stt_model_config['local_transcripton_model_file']
        self.lang = stt_model_config['audio_lang']
        model_filename = MODELS_DIR + self.model + ".pt"
        self.model_name = self.model + ".pt"
        self.model_filename = os.path.join(MODELS_DIR, model_filename)
        self.engine = pick_local_whisper_engine(stt_model_config.get('local_engine', 'auto'))
        self.use_gpu = False
        self.audio_model = None
        self._switch_lock = threading.Lock()
        # faster-whisper runs two transcriptions at once (num_workers=2); openai-whisper cannot.
        self.thread_safe = self.engine == FASTER_WHISPER
        # "loading", "ready" or the reason it failed; the window shows it in its title.
        self.load_state = "loading"
        self._loaded = threading.Event()
        if background:
            # The window opens at once; the model (seconds, or a download the first time) loads meanwhile.
            threading.Thread(target=self._load_in_background, name="LoadSpeechModel", daemon=True).start()
        else:
            self._load()

    def _load_in_background(self):
        """Load the model; if that fails (typically: the first download without internet), say
        why and keep trying, so the program recovers by itself once the connection is back."""
        while True:
            try:
                self._load()
                return
            except Exception as exception:  # pylint: disable=broad-except
                self.load_state = str(exception)
                self._loaded.set()      # requests fail with this reason instead of waiting
                print(f'[ERROR] {exception}')
            time.sleep(LOAD_RETRY_SECONDS)

    def _load(self):
        started = time.monotonic()
        if self.engine == FASTER_WHISPER:
            try:
                self.audio_model = self._load_faster_whisper()
            except Exception as exception:  # pylint: disable=broad-except
                if not _installed("whisper"):
                    raise RuntimeError(
                        f'The local speech to text model "{self.model}" could not be loaded ({exception}). '
                        'It is downloaded from Hugging Face the first time it is used: check the internet '
                        'connection (or set a proxy in Settings > General).') from exception
                # openai-whisper is installed too (optional): use it and its .pt file instead
                print(f'[WARN] faster-whisper could not load model {self.model} ({exception}); '
                      'using openai-whisper instead.')
                self.engine = OPENAI_WHISPER
        if self.engine == OPENAI_WHISPER:
            try:
                import torch  # pylint: disable=import-outside-toplevel
                import whisper  # pylint: disable=import-outside-toplevel
            except ImportError as exception:
                raise RuntimeError(
                    'local_engine is set to openai-whisper, but it is not installed. It is optional '
                    '(it needs PyTorch, several GB): run setup-openai-whisper.bat, or set '
                    'local_engine: auto to use faster-whisper.') from exception
            self.use_gpu = bool(torch.cuda.is_available())
            if self.model in OPENAI_WHISPER_URLS:
                self.download_model()
                self.audio_model = whisper.load_model(self.model_filename)
            else:       # e.g. turbo: openai-whisper downloads it by name
                utilities.ensure_directory_exists(MODELS_DIR)
                self.audio_model = whisper.load_model(self.model, download_root=MODELS_DIR)
        self.thread_safe = self.engine == FASTER_WHISPER
        self.load_state = "ready"
        self._loaded.set()
        print(f'[INFO] Speech To Text - local Whisper ({self.engine}, model {self.model}), '
              f'using GPU: {self.use_gpu}, loaded in {time.monotonic() - started:.1f} s')

    def _load_faster_whisper(self, allow_gpu: bool = True):
        # Hugging Face (where the model is downloaded from) otherwise prints notes about tokens
        # and optional packages that mean nothing to the user.
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        if allow_gpu:
            add_nvidia_dll_directories()
        from faster_whisper import WhisperModel  # pylint: disable=import-outside-toplevel
        download_root = os.path.join(MODELS_DIR, "faster-whisper")
        utilities.ensure_directory_exists(download_root)

        def build(device, compute_type):
            # The model shipped with the program needs no download at all.
            bundled = bundled_model_dir(self.model)
            if bundled is not None:
                return WhisperModel(bundled, device=device, compute_type=compute_type, num_workers=2)
            # A model that is already on disk is opened without asking Hugging Face first:
            # that check went over the internet at every start and could take minutes where
            # the site is slow or blocked. Only a missing model is downloaded.
            try:
                return WhisperModel(self.model, device=device, compute_type=compute_type,
                                    download_root=download_root, num_workers=2, local_files_only=True)
            except Exception:  # pylint: disable=broad-except
                pass
            print(f'[INFO] Downloading the speech to text model "{self.model}" (first use only)...')
            self.load_state = "downloading"
            try:
                return WhisperModel(self.model, device=device, compute_type=compute_type,
                                    download_root=download_root, num_workers=2)
            except Exception as exception:  # pylint: disable=broad-except
                print(f'[WARN] Download from Hugging Face failed ({exception}); trying {MODEL_DOWNLOAD_MIRROR}')
                try:
                    path = self._download_from_mirror(download_root)
                except Exception as mirror_exception:  # pylint: disable=broad-except
                    print(f'[WARN] Download from the mirror failed too ({mirror_exception}).')
                    raise exception from mirror_exception
                return WhisperModel(path, device=device, compute_type=compute_type, num_workers=2)

        if allow_gpu:
            try:
                import ctranslate2  # pylint: disable=import-outside-toplevel
                if ctranslate2.get_cuda_device_count() > 0:
                    model = build("cuda", "float16")
                    self.use_gpu = True
                    self._gpu_model = model
                    return model
            except Exception as exception:  # pylint: disable=broad-except
                print(f'[WARN] faster-whisper could not use the GPU ({exception}); using the CPU.')
        model = build("cpu", "int8")
        self.use_gpu = False
        return model

    def _download_from_mirror(self, download_root: str) -> str:
        """Download the model files from the mirror into the same cache; returns their folder."""
        import huggingface_hub  # pylint: disable=import-outside-toplevel
        from faster_whisper import utils as fw_utils  # pylint: disable=import-outside-toplevel
        repo_id = getattr(fw_utils, "_MODELS", {}).get(self.model, self.model)
        return huggingface_hub.snapshot_download(
            repo_id, cache_dir=download_root, endpoint=MODEL_DOWNLOAD_MIRROR, allow_patterns=_MODEL_FILES)

    def download_model(self):
        """Download the appropriate OpenAI model if needed"""

        if os.path.exists(self.model_filename):
            return
        print(f'Could not find the transcription model file: {self.model_filename}')
        utilities.ensure_directory_exists(MODELS_DIR)
        if self.model == 'tiny':
            file_url = 'https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/' + self.model_name
            utilities.download_using_bits(file_url=file_url, file_path=self.model_filename)
        elif self.model == 'base':
            file_url = 'https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/' + self.model_name
            utilities.download_using_bits(file_url=file_url, file_path=self.model_filename)
        elif self.model == 'small':
            file_url = 'https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/' + self.model_name
            utilities.download_using_bits(file_url=file_url, file_path=self.model_filename)
        elif self.model == 'medium':
            file_url = 'https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/' + self.model_name
            utilities.download_using_bits(file_url=file_url, file_path=self.model_filename)
        elif self.model == 'large':
            file_url = 'https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/' + self.model_name
            utilities.download_using_bits(file_url=file_url, file_path=self.model_filename)
        elif self.model == 'large-v1':
            file_url = 'https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/' + self.model_name
            utilities.download_using_bits(file_url=file_url, file_path=self.model_filename)
        elif self.model == 'large-v2':
            file_url = 'https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/' + self.model_name
            utilities.download_using_bits(file_url=file_url, file_path=self.model_filename)
        elif self.model == 'large-v3':
            file_url = 'https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/' + self.model_name
            utilities.download_using_bits(file_url=file_url, file_path=self.model_filename)
        else:
            print(f'The model "{self.model}" is not available for openai-whisper '
                  '(use faster-whisper: local_engine: auto).')
            sys.exit()

    def _transcribe(self, audio) -> dict:
        """Transcribe a file path or a 16 kHz float32 array into openai-whisper's result shape."""
        self._loaded.wait()
        if self.load_state in ("loading", "downloading"):
            raise RuntimeError("The speech to text model is still being downloaded.")
        if self.load_state != "ready":
            raise RuntimeError(self.load_state)      # (the reason, worded for the user)
        language = self.lang or None
        if self.engine == FASTER_WHISPER:
            model = self.audio_model
            try:
                return self._faster_whisper_transcribe(model, audio, language)
            except Exception as exception:  # pylint: disable=broad-except
                if WhisperSTTModel.use_vad and any(word in str(exception).lower() for word in ("onnx", "silero", "vad")):
                    # The voice activity filter cannot run here: go on without it.
                    print(f'[WARN] Voice activity filter is off ({exception}).')
                    WhisperSTTModel.use_vad = False
                    return self._transcribe(audio)
                if model is not getattr(self, "_gpu_model", None):
                    raise
                # The GPU was found but its CUDA libraries (cuBLAS / cuDNN) are missing: this
                # shows up only on the first transcription. Continue on the CPU for good. The
                # other source's worker may fail on the GPU at the same moment: one load only.
                with self._switch_lock:
                    if self.audio_model is model:
                        print('[WARN] faster-whisper failed on the GPU; switching to the CPU.')
                        self.audio_model = self._load_faster_whisper(allow_gpu=False)
                return self._faster_whisper_transcribe(self.audio_model, audio, language)
        # condition_on_previous_text=False: short live windows do not benefit from it and it
        # can make Whisper loop on repeated text.
        return self.audio_model.transcribe(audio, fp16=self.use_gpu, language=language,
                                           temperature=0, condition_on_previous_text=False)

    def get_sentences(self, wav_file_path) -> dict:
        """Get transcription from the provided audio file as individual sentences
        """
        result = self._transcribe(wav_file_path)
        sentences = []
        for segment in result['segments']:
            start = str(datetime.timedelta(seconds=int(segment['start'])))
            end = str(datetime.timedelta(seconds=int(segment['end'])))
            sentences.append(f"{start} - {end}: {segment['text']}")
        return sentences

    def get_transcription(self, wav_file_path) -> dict:
        """Get transcription from the provided audio file
        """
        try:
            return self._transcribe(wav_file_path)
        except Exception as exception:
            print('WhisperSTTModel:get_transcription - Encountered error')
            print(exception)
            return ''

    @staticmethod
    def _faster_whisper_transcribe(model, audio, language) -> dict:
        # vad_filter: parts without speech (noise, music, a pause) are not given to the model,
        # which otherwise tends to invent text for them. Timestamps stay those of the audio.
        segments, info = model.transcribe(
            audio, language=language, beam_size=1, temperature=0,
            condition_on_previous_text=False, vad_filter=WhisperSTTModel.use_vad)
        items = [{"id": index, "start": float(segment.start), "end": float(segment.end),
                  "text": segment.text} for index, segment in enumerate(segments)]
        return {"text": "".join(item["text"] for item in items).strip(),
                "segments": items, "language": getattr(info, "language", language)}

    def transcribe_pcm(self, pcm: bytes, sample_rate: int = audio_format.TARGET_SAMPLE_RATE):
        """Both engines take a 16 kHz float32 array directly: no file and no ffmpeg."""
        if sample_rate != audio_format.TARGET_SAMPLE_RATE:
            return super().transcribe_pcm(pcm, sample_rate)
        # Errors go to the caller: the live transcriber shows them to the user.
        return self._transcribe(audio_format.pcm16_to_float32(pcm))

    def set_lang(self, lang: str):
        """Set Language for STT (the same model serves every language)."""
        self.lang = lang

    def process_response(self, response) -> str:
        """
        Returns transcription from the response of transcription.
        """
        # response['text'] = transcription text
        # response['language'] = language of transcription
        # response['segments'] = list of segments.
        # response['segments']['text'] = text of the segment.
        # response['segments']['start'] = start time of segment.
        # response['segments']['end'] = end time of segment.
        # Each segment is a dict
        #
        # for segment in response['segments']:
        #     start = str(datetime.timedelta(seconds=int(segment['start'])))
        #     end = str(datetime.timedelta(seconds=int(segment['end'])))
        #     print(f"{start} - {end}: {segment['text']}")
        # pprint.pprint(response)
        return response['text'].strip()

    def normalize_response(
        self,
        response,
        audio_start_seconds: float = 0.0,
        audio_end_seconds: float = 0.0,
    ) -> TranscriptionHypothesis:
        """Normalize local Whisper responses."""
        text = self.process_response(response) if response else ""
        segments = [
            _segment_from_mapping(index, segment, offset_seconds=audio_start_seconds)
            for index, segment in enumerate(response.get("segments", []))
        ] if response else []
        return TranscriptionHypothesis(
            provider="whisper",
            text=text,
            segments=segments or _single_segment(text, audio_start_seconds, audio_end_seconds),
            audio_start_seconds=audio_start_seconds,
            audio_end_seconds=audio_end_seconds,
        )


class APIWhisperSTTModel(STTModelInterface):
    """Speech to Text using the Whisper API
    """
    thread_safe = True
    def __init__(self, stt_model_config: dict):
        # Check for api_key
        if stt_model_config["api_key"] is None:
            raise Exception("Attempt to create Open AI Whisper STT Model without an api key.")  # pylint: disable=W0719
        print('[INFO] Using Open AI Whisper API for transcription.')
        # One retry at most: a live request that is retried for long only holds back newer audio.
        # The same Base URL as the replies: an OpenAI-compatible provider can serve both.
        self.stt_client = openai.OpenAI(api_key=stt_model_config['api_key'],
                                        base_url=stt_model_config.get('base_url') or None, max_retries=1)
        self.timeout = stt_model_config['timeout']
        self.model = stt_model_config.get('model') or 'whisper-1'
        # lang parameter is not required for API invocation. This exists solely
        # to support --api option from command line.
        # A better solution is to create a base class for APIWhisperSTTModel,
        # WhisperSTTModel and create set_lang method there and remove it from
        # this class
        self.lang = stt_model_config['audio_lang']

    def set_lang(self, lang: str):
        """Set STT Language"""
        self.lang = lang

    def _request(self, file) -> dict:
        kwargs = {"model": self.model, "file": file, "timeout": self.timeout}
        if isinstance(self.lang, str) and len(self.lang) == 2:
            kwargs["language"] = self.lang      # the API accepts only ISO-639-1 codes ("en")
        if self.model.startswith("whisper"):
            # verbose_json adds segment timestamps, which let the live transcript commit
            # finished text and keep every request short.
            kwargs["response_format"] = "verbose_json"
        return self.stt_client.audio.transcriptions.create(**kwargs)

    def get_transcription(self, wav_file_path) -> dict:
        """Get transcription from the provided audio file
        """
        try:
            with open(wav_file_path, "rb") as audio_file:
                return self._request(audio_file)
        except Exception as exception:
            print('Exception in transcribing audio using whisper API.')
            print(exception)
            return ''

    def transcribe_pcm(self, pcm: bytes, sample_rate: int = audio_format.TARGET_SAMPLE_RATE):
        """Upload 16 kHz mono WAV straight from memory."""
        # Errors go to the caller: the live transcriber shows them to the user.
        return self._request(("audio.wav", audio_format.pcm16_to_wav_bytes(pcm, sample_rate), "audio/wav"))

    def process_response(self, response) -> str:
        """
        Returns transcription from the response of transcription.
        """
        # response['text'] = transcription text
        # response['language'] = language of transcription
        # response['segments'] = list of segments.
        # response['segments']['text'] = text of the segment.
        # response['segments']['start'] = start time of segment.
        # response['segments']['end'] = end time of segment.
        # Each segment is a dict
        #
        # for segment in response['segments']:
        #     start = str(datetime.timedelta(seconds=int(segment['start'])))
        #     end = str(datetime.timedelta(seconds=int(segment['end'])))
        #     print(f"{start} - {end}: {segment['text']}")
        # pprint.pprint(response)
        return response.text.strip()

    def get_sentences(self, wav_file_path) -> dict:
        """Get transcription from the provided audio file as individual sentences
        """
        try:
            with open(wav_file_path, "rb") as audio_file:
                result = self.stt_client.audio.transcriptions.create(model='whisper-1', file=audio_file,
                                                                     language=self.lang)
        except Exception as exception:
            print('Exception in transcribing audio using whisper API.')
            print(exception)
            return ''

        return [result.text]

    def normalize_response(
        self,
        response,
        audio_start_seconds: float = 0.0,
        audio_end_seconds: float = 0.0,
    ) -> TranscriptionHypothesis:
        """Normalize Whisper API responses (segments come with verbose_json)."""
        text = self.process_response(response) if response else ""
        segments = []
        for index, segment in enumerate(getattr(response, "segments", None) or []):
            item = segment if isinstance(segment, dict) else {
                "id": getattr(segment, "id", index), "start": getattr(segment, "start", 0.0),
                "end": getattr(segment, "end", 0.0), "text": getattr(segment, "text", "")}
            segments.append(_segment_from_mapping(index, item, offset_seconds=audio_start_seconds))
        return TranscriptionHypothesis(
            provider="whisper-api",
            text=text,
            segments=segments or _single_segment(text, audio_start_seconds, audio_end_seconds),
            audio_start_seconds=audio_start_seconds,
            audio_end_seconds=audio_end_seconds,
        )


class WhisperCPPSTTModel(STTModelInterface):
    """Speech to Text using the local whisper cpp exes.
    It primarily deals with interacting with the whisper CPP API model.
    This model works best when used with GPU
    """
    def __init__(self, stt_model_config: dict):
        self.lang = stt_model_config['audio_lang']
        model = stt_model_config['local_transcripton_model_file']
        # whisper.cpp names the turbo model by its full name
        model = {"ggml-turbo": "ggml-large-v3-turbo"}.get(model, model)
        self.model_filename = MODELS_DIR + model + ".bin"
        self.model = model

        if not os.path.isfile(self.model_filename):
            print(f'Could not find the transcription model file: {self.model_filename}')
            utilities.ensure_directory_exists(MODELS_DIR)
            file_url = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/' + model + '.bin'
            utilities.download_using_bits(file_url=file_url, file_path=self.model_filename)

        print('[INFO] Using Whisper CPP for transcription.')
        self.model = 'base'

    def set_lang(self, lang: str):
        """Set STT Language"""
        self.lang = lang

    # whisper.cpp renamed main.exe to whisper-cli.exe; both are accepted.
    EXECUTABLES = ("../../bin/whisper-cli.exe", "../../bin/main.exe",
                   "./bin/whisper-cli.exe", "./bin/main.exe")

    @classmethod
    def _find_executable(cls):
        for candidate in cls.EXECUTABLES:
            if os.path.isfile(candidate):
                return candidate
        return None

    def get_transcription(self, wav_file_path: str):
        """Get text using STT (the input must already be 16 kHz mono 16-bit WAV)."""
        executable = self._find_executable()
        if executable is None:
            print('ERROR: whisper.cpp was not found. Put whisper-cli.exe (or main.exe) in the bin folder.')
            return None
        json_file_path = wav_file_path + ".json"
        log_file = f"{utilities.get_data_path(app_name='Transcribe')}/logs/whisper.cpp.txt"
        try:
            with open(file=log_file, mode='a', encoding='utf-8') as log_handle:
                # A hung process must not silence the source for good (it is killed after the timeout).
                subprocess.call([executable, wav_file_path, '-oj', '-m', self.model_filename,
                                 '-l', self.lang],
                                stdout=log_handle, stderr=subprocess.STDOUT, timeout=120)
            with open(json_file_path, mode="r", encoding='utf-8') as text_file:
                return json.loads(text_file.read())
        except Exception as exception:
            print(f'ERROR: converting wav file {wav_file_path} to text using whisper.cpp.')
            print(exception)
            return None
        finally:
            if os.path.exists(json_file_path):
                os.unlink(json_file_path)

    def process_response(self, response) -> str:
        # response is of type PrerecordedTranscriptionResponse
        # convert result to the appropriate dict format
        text = ''
        for segment in response["transcription"]:
            if segment["text"].strip() == '[BLANK_AUDIO]':
                continue
            text += segment["text"]
        # print(f'Transcript: {text}')
        return text

    def get_sentences(self, wav_file_path: str):
        """Not Implemented
        """
        transcript = ''
        response = self.get_transcription(wav_file_path=wav_file_path)
        for segment in response["transcription"]:
            if segment["text"].strip() == '[BLANK_AUDIO]':
                continue
            transcript += segment["text"]
        return transcript

        # raise Exception('Method not implemnted')  # pylint: disable=W0719

    def normalize_response(
        self,
        response,
        audio_start_seconds: float = 0.0,
        audio_end_seconds: float = 0.0,
    ) -> TranscriptionHypothesis:
        """Normalize whisper.cpp JSON responses."""
        text = self.process_response(response) if response else ""
        segments = []
        if response:
            for index, segment in enumerate(response.get("transcription", [])):
                segment_text = str(segment.get("text", "")).strip()
                if segment_text == "[BLANK_AUDIO]":
                    continue
                offsets = segment.get("offsets", {})
                segments.append(
                    TranscriptSegment(
                        id=index,
                        start_seconds=audio_start_seconds + float(offsets.get("from", 0)) / 1000,
                        end_seconds=audio_start_seconds + float(offsets.get("to", 0)) / 1000,
                        text=segment_text,
                    )
                )
        return TranscriptionHypothesis(
            provider="whisper.cpp",
            text=text.strip(),
            segments=segments or _single_segment(text, audio_start_seconds, audio_end_seconds),
            audio_start_seconds=audio_start_seconds,
            audio_end_seconds=audio_end_seconds,
        )


class DeepgramSTTModel(STTModelInterface):
    """Speech to Text using the Deepgram API.
    It primarily deals with interacting with the Deepgram API.
    """
    thread_safe = True
    def __init__(self, stt_model_config: dict):
        # Check for api_key
        if stt_model_config["api_key"] is None:
            raise Exception("Attempt to create Deepgram STT Model without an api key.")  # pylint: disable=W0719

        # This parameter exists primarily to adhere to the interface.
        # Deepgram does auto language detection.
        # self.lang = 'en-US'
        self.lang = stt_model_config['audio_lang']
        self.model = stt_model_config.get("model", "nova-3")

        print(f'[INFO] Using Deepgram API for transcription. Model: {self.model}')
        from deepgram import DeepgramClient  # pylint: disable=import-outside-toplevel
        self.audio_model = DeepgramClient(stt_model_config["api_key"])

    def set_lang(self, lang: str):
        """Set STT Language"""
        self.lang = lang

    def _transcribe_buffer(self, buffer_data: bytes):
        from deepgram import PrerecordedOptions  # pylint: disable=import-outside-toplevel
        # The chosen language is sent; detection per short request could change language mid-talk.
        language_options = {"language": self.lang} if self.lang else {"detect_language": True}
        options = PrerecordedOptions(
            model=self.model,
            smart_format=True,
            utterances=True,
            punctuate=True,
            paragraphs=True,
            **language_options)
        return self.audio_model.listen.prerecorded.v("1").transcribe_file({"buffer": buffer_data}, options)

    def get_transcription(self, wav_file_path: str):
        """Get text using STT
        """
        try:
            with open(wav_file_path, "rb") as audio_file:
                return self._transcribe_buffer(audio_file.read())
        except Exception as exception:
            print(exception)
        return None

    def transcribe_pcm(self, pcm: bytes, sample_rate: int = audio_format.TARGET_SAMPLE_RATE):
        """Send the WAV from memory (no file, and no debug copy of every response on disk)."""
        # Errors go to the caller: the live transcriber shows them to the user.
        return self._transcribe_buffer(audio_format.pcm16_to_wav_bytes(pcm, sample_rate))

    def get_sentences(self, wav_file_path: str):
        """Get transcription from the provided audio file as individual sentences
        """
        try:
            with open(wav_file_path, "rb") as audio_file:
                buffer_data = audio_file.read()

            from deepgram import PrerecordedOptions  # pylint: disable=import-outside-toplevel
            payload = {
                "buffer": buffer_data
                }
            if self.lang.startswith('en'):
                options = PrerecordedOptions(
                    model=self.model,
                    smart_format=True,
                    utterances=True,
                    punctuate=True,
                    paragraphs=True,
                    detect_language=True,
                    language=self.lang)
            else:
                options = PrerecordedOptions(
                    model=self.model,
                    smart_format=True,
                    utterances=True,
                    punctuate=True,
                    paragraphs=True,
                    detect_language=True,
                    language=self.lang)

            response = self.audio_model.listen.prerecorded.v("1").transcribe_file(payload, options)
            # This is not necessary and just a debugging aid
            log_file = f"{utilities.get_data_path(app_name='Transcribe')}/logs/deep.json"
            with open(log_file, mode='a', encoding='utf-8') as deep_log:
                deep_log.write(response.to_json(indent=4))
            results = []
            for utterance in response.results.utterances:
                start = str(datetime.timedelta(seconds=int(utterance['start'])))
                end = str(datetime.timedelta(seconds=int(utterance['end'])))
                results.append(f"{start} - {end} : {utterance['transcript']}")

            return results
        except Exception as exception:
            print(exception)

        return None

    def process_response(self, response) -> str:
        # response is of type PrerecordedTranscriptionResponse
        # convert result to the appropriate dict format
        text = response.results.channels[0].alternatives[0].transcript
        # print(f'Transcript: {text}')
        return text

    def normalize_response(
        self,
        response,
        audio_start_seconds: float = 0.0,
        audio_end_seconds: float = 0.0,
    ) -> TranscriptionHypothesis:
        """Normalize Deepgram prerecorded responses."""
        text = self.process_response(response) if response else ""
        segments = []
        try:
            utterances = response.results.utterances
        except AttributeError:
            utterances = []

        for index, utterance in enumerate(utterances or []):
            start = utterance.get("start") if isinstance(utterance, dict) else getattr(utterance, "start", 0.0)
            end = utterance.get("end") if isinstance(utterance, dict) else getattr(utterance, "end", 0.0)
            transcript = (
                utterance.get("transcript")
                if isinstance(utterance, dict)
                else getattr(utterance, "transcript", "")
            )
            segments.append(
                TranscriptSegment(
                    id=index,
                    start_seconds=audio_start_seconds + float(start),
                    end_seconds=audio_start_seconds + float(end),
                    text=str(transcript).strip(),
                )
            )

        return TranscriptionHypothesis(
            provider="deepgram",
            text=text.strip(),
            segments=segments or _single_segment(text, audio_start_seconds, audio_end_seconds),
            audio_start_seconds=audio_start_seconds,
            audio_end_seconds=audio_end_seconds,
        )
