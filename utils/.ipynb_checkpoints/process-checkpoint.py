
import base64
import torch
import torchaudio
import re
import time
import numpy as np

def process_data_to_audio(aud, input_sample_rate: int, whisper_sr: int) -> np.ndarray:
    """
    process input audio data to audio
    """
    if isinstance(aud, dict) and "data" in aud:
        pcm_bytes = bytes(aud["data"])  # int 배열
    elif isinstance(aud, str):
        pcm_bytes = base64.b64decode(aud)
    else:
        pcm_bytes = None
    
    if pcm_bytes:
        x = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio = x.astype(np.float32) / 32768.0
        
        if input_sample_rate != whisper_sr:
            audio_t = torch.from_numpy(audio).unsqueeze(0)  # (1, n)
            audio_resampled = torchaudio.functional.resample(audio_t, orig_freq=input_sample_rate, new_freq=whisper_sr)
            audio = audio_resampled.squeeze(0).contiguous().numpy()
        return audio
    else:
        return None

def feed_pcm16le_bytes(x):
    t = torch.from_numpy(x)[:512] # 16kHz 기준, 길이가 512의 배수여야 한다.
    
    label = vad_iter(t, return_seconds=False)  # {'end': 212448} or {'start': 212448}

    if label is not None:
        if "start" in label:
            return 'start'
        if "end" in label:
            return 'end'
    return None

# === Whisper 전사 함수 구현 ===
async def transcribe_pcm_with_whisper(audios, sample_rate: int, channels: int) -> str:
    stts = time.time()
    result = asr_model.transcribe(audios)
    text = re.sub(r"<\|.*?\|>", "", result[0].text).strip()
    
    return text