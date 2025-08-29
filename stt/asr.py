from __future__ import annotations
from typing import Optional, Protocol, Literal
import os
import re
import numpy as np
import librosa
import torch
# 선택적 의존성 (사용 시에만 import)
try:
    import nemo.collections.asr as nemo_asr  # type: ignore
except Exception:
    nemo_asr = None

try:
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline  # type: ignore
except Exception:
    AutoProcessor = AutoModelForSpeechSeq2Seq = pipeline = None


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

        processor = AutoProcessor.from_pretrained(model_id)
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

        # NeMo는 파일/경로 입력이 기본이지만 numpy 1D도 허용하는 구현이 있습니다.
        # 모델 버전에 따라 다르면 torchaudio.save(tmp) 방식으로 우회하세요.
        result = self.model.transcribe(audio)
        # 반환 형태가 버전에 따라 다를 수 있어 방어적으로 처리
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
BackendKind = Literal["hf", "nemo"]

def load_asr_backend(
    kind: Optional[BackendKind] = None,
    **kwargs,
) -> ASRBackend:
    """
    kind:
      - "hf"   : Hugging Face transformers Whisper-like
      - "nemo" : NVIDIA NeMo Canary 등
    kwargs:
      - hf: model_id, device, dtype
      - nemo: pretrained_name, device
    """
    kind = (kind or os.getenv("ASR_BACKEND", "korean")).lower()

    if kind == "korean":
        model_id = kwargs.get("model_id") or os.getenv(
            "HF_ASR_MODEL",
            "o0dimplz0o/Whisper-Large-v3-turbo-STT-Zeroth-KO-v2",
        )
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        return HFWhisperBackend(model_id=model_id, device=device, dtype=dtype)
    
    pretrained_name = kwargs.get("pretrained_name") or os.getenv(
        "NEMO_PRETRAINED", "nvidia/canary-1b-v2"
    )
    device = kwargs.get("device")
    return NemoASRBackend(pretrained_name=pretrained_name, device=device)
