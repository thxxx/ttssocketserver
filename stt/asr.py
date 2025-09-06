from __future__ import annotations
from typing import Optional, Protocol, Literal
import os
import re
import numpy as np
import librosa
import torch
import tqdm

# 선택적 의존성 (사용 시에만 import)
try:
    import nemo.collections.asr as nemo_asr  # type: ignore
except Exception:
    nemo_asr = None
try:
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq  # type: ignore
except Exception:
    AutoProcessor = AutoModelForSpeechSeq2Seq = pipeline = None
try:
    from transformers import pipeline
except Exception as e:
    raise RuntimeError("transformers pipeline is not installed (import failed).") from e

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

tqdm.tqdm = lambda *a, **k: a[0] if a else None

WHISPER_SR = 16000

class ASRBackend(Protocol):
    def transcribe_pcm(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        channels: int,
        language: Optional[str] = None,
    ) -> str: ...
    

def _pcm_int16_to_f32_mono(pcm_bytes: bytes, channels: int) -> np.ndarray:
    """int16 PCM → float32 mono [-1, 1]"""
    if not pcm_bytes:
        return np.zeros(0, dtype=np.float32)
    x = np.frombuffer(pcm_bytes, dtype=np.int16)
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return (x.astype(np.float32) / 32768.0).copy()


def _ensure_sr(audio: np.ndarray, orig_sr: int, target_sr: int = WHISPER_SR) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


# ---------------------------
# HF Whisper(Turbo) Backend
# ---------------------------
class HFWhisperBackend:
    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if pipeline is None:
            raise RuntimeError("transformers pipeline is not installed.")
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)

        processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=self.device,
            torch_dtype=self.dtype,
        )

    def transcribe_pcm(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        channels: int,
        language: Optional[str] = "korean",
    ) -> str:
        audio = _pcm_int16_to_f32_mono(pcm_bytes, channels)
        if audio.size == 0:
            return ""
        audio = _ensure_sr(audio, sample_rate, WHISPER_SR)

        # transformers ASR pipeline: numpy 1D float32
        out = self.pipe(audio, generate_kwargs={"language": language} if language else None)
        text = (out[0].get("text") if isinstance(out, list) else out.get("text", "")) or ""
        return text.strip()


# ---------------------------
# NeMo Canary Backend
# ---------------------------
class NemoASRBackend:
    def __init__(self, pretrained_name: str = "nvidia/canary-1b-v2", device: Optional[str] = None):
        if nemo_asr is None:
            raise RuntimeError("NVIDIA NeMo is not installed.")
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = nemo_asr.models.ASRModel.from_pretrained(pretrained_name)
        self.model = self.model.to(self.device)

    def transcribe_pcm(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        channels: int,
        language: Optional[str] = None,
    ) -> str:
        audio = _pcm_int16_to_f32_mono(pcm_bytes, channels)
        if audio.size == 0:
            return ""
        # NeMo 입력 SR 요구 사항이 모델마다 다를 수 있지만, 대부분 16k OK
        audio = _ensure_sr(audio, sample_rate, WHISPER_SR)

        result = self.model.transcribe(audio, source_lang='en', target_lang='en')
        
        if isinstance(result, (list, tuple)) and len(result) > 0:
            # canary 계열은 특수 토큰이 섞일 수 있음
            text = getattr(result[0], "text", "") or str(result[0])
        else:
            text = str(result)

        text = re.sub(r"<\|.*?\|>", "", text)  # 특수 토큰 제거
        return text.strip()


# ---------------------------
# Factory
# ---------------------------
BackendKind = Literal["korean", "nemo"]

def load_asr_backend(
    kind: Optional[BackendKind] = None,
    **kwargs,
) -> ASRBackend:
    """
    kind:
      - "korean"   : Hugging Face transformers Whisper-like
      - "nemo" : NVIDIA NeMo Canary 등
    kwargs:
      - korean: model_id, device, dtype
      - nemo: pretrained_name, device
    """
    kind = (kind or os.getenv("ASR_BACKEND", "korean")).lower()

    if kind == "korean":
        model_id = kwargs.get("model_id") or os.getenv(
            "HF_ASR_MODEL",
            "/home/khj6051/whisper/whisper-turbo-ko-sixth/checkpoint-6000"
            # "o0dimplz0o/Whisper-Large-v3-turbo-STT-Zeroth-KO-v2",
        )
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        return HFWhisperBackend(model_id=model_id, device=device, dtype=dtype)
    
    pretrained_name = kwargs.get("pretrained_name") or os.getenv(
        "NEMO_PRETRAINED", "nvidia/canary-1b-v2"
    )
    device = kwargs.get("device")
    return NemoASRBackend(pretrained_name=pretrained_name, device=device)
