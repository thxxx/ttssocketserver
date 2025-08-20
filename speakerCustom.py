import asyncio
import json
from datetime import datetime
from typing import Dict, Optional
from llm.openai import translate_speaker, translate_speaker_make_end
from stt.openai import open_openai_ws
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed
import orjson as oj  # loads/dumps 호환 아님
import os
import websockets
import base64
import time
import numpy as np
import librosa
import re
import torch
import torchaudio
import numpy as np
import nemo.collections.asr as nemo_asr

print("=== speakerServer.py loaded ===")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

INPUT_SAMPLE_RATE = 24000
INPUT_FRAME_MS = 32

WHISPER_SR = 16000
SAMPLE_RATE_FOR_VAD = 16000
SAMPLES_PER_FRAME = SAMPLE_RATE_FOR_VAD * INPUT_FRAME_MS // 1000

# Silero VAD 로드 (CPU OK)
silero, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
(get_speech_timestamps, _, read_audio, VADIterator, collect_chunks) = utils
vad_iter = VADIterator(silero, threshold=0.35, sampling_rate=SAMPLE_RATE_FOR_VAD, min_silence_duration_ms=150)

# ASR model 로드
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-1b-v2").to(device)
print("Loaded")

def jdumps(o): return oj.dumps(o).decode()  # bytes -> str

app = FastAPI()

DEBUG = True
def dprint(*a, **k): 
    if DEBUG: print(*a, **k)

LOGG = True
def lprint(*a, **k): 
    if LOGG: print(*a, **k)

# 클라이언트 1명당 OpenAI Realtime WS를 하나씩 유지하기 위한 세션 상태
class Session:
    def __init__(self):
        self.oai_ws = None  # OpenAI WS 연결
        self.oai_task: Optional[asyncio.Task] = None  # OAI -> Client listener task
        self.running = True

        # 오디오 누적 버퍼 (commit 시점에만 소비)
        self.audio_buf = bytearray()
        self.audios = np.empty(0, dtype=np.float32)
        
        self.input_sample_rate = INPUT_SAMPLE_RATE
        self.input_channels = 1

        self.current_transcript: str = ''
        self.transcripts: list[str] = []
        self.current_translated: str = ''
        self.translateds: list[str] = []

        # 송신을 Queue로 관리
        self.out_q: asyncio.Queue[str] = asyncio.Queue()
        self.sender_task: Optional[asyncio.Task] = None

        # --- TTS용 필드, text buffer 단위 speech 생성 ---
        self.tts_ws = None
        self.tts_task: Optional[asyncio.Task] = None
        self.tts_in_q: asyncio.Queue[str] = asyncio.Queue(maxsize=256)

        # onToken에서 단어/프레이즈 coalescing용
        self.tts_buf: list[str] = []
        self.tts_debounce_task: Optional[asyncio.Task] = None
        self.buf_count = 0

        # variables for logging
        self.start_scripting_time = 0
        self.end_scripting_time = 0
        self.end_translation_time = 0
        self.first_translated_token_output_time = 0
        self.end_tts_time = 0
        self.end_audio_input_time = 0

        # time logging
        self.connection_start_time = 0
        self.llm_cached_token_count = 0
        self.llm_input_token_count = 0
        self.llm_output_token_count = 0
        self.stt_output_token_count = 0

        self.is_network_logging = False

        self.new_speech_start = 0

sessions: Dict[int, Session] = {}  # id(ws)로 매핑

@app.websocket("/speakerws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    sess = Session()
    sessions[id(ws)] = sess

    sess.sender_task = asyncio.create_task(outbound_sender(sess, ws))

    try:
        while True:
            msg = await ws.receive()
            if msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await ws.send_text(jdumps({"type": "error", "message": "Invalid JSON"}))
                    continue

                t = data.get("type")

                # 1) 세션 시작: OpenAI Realtime WS 연결
                if t == "scriptsession.start":
                    if sess.oai_ws is None:
                        # sess.oai_ws = await open_openai_ws()
                        sess.connection_start_time = time.time()

                        # OpenAI 이벤트를 client로 릴레이하는 백그라운드 태스크
                        await ws.send_text(jdumps({"type": "scriptsession.started"}))
                    else:
                        await ws.send_text(jdumps({"type": "warn", "message": "already started"}))

                # 2) 오디오 append → Open AI로 그대로 전달
                elif t == "input_audio_buffer.append":
                    # 한 5개 쌓일때마다 한번씩?
                    # sess.buf_count += 1
                    # if sess.buf_count % 15 == 14:
                    #     # 지금까지 누적된 PCM을 Whisper에 태워서 script 추출
                    #     buf = bytes(sess.audio_buf)

                    #     # 비동기 실행 (블로킹 피함)
                    #     transcript = await transcribe_pcm(sess.audios)
                    #     await ws.send_text(jdumps({
                    #         "type": "delta",
                    #         "text": transcript,
                    #         "final": False
                    #     }))
                    try:
                        aud = data.get("audio")
                        if aud:
                            audio = process_data_to_audio(aud)
                            if audio is not None:
                                sess.audios = np.concatenate([sess.audios, audio])
                                
                                # === 여기서 VAD 검사 ===
                                vads = time.time()
                                vad_event = feed_pcm16le_bytes(audio)
                                
                                if vad_event == "end" and sess.audios.size>WHISPER_SR*1.5:
                                    target_audios = sess.audios
                                    sess.audios = np.empty(0, dtype=np.float32)
                                    
                                    t00 = time.time()
                                    transcript = await transcribe_pcm(audios=target_audios)
                                    dprint(f"[{time.time() - t00:.2f}s] scripting : ", transcript, f"  {target_audios.size}", "\n")
                                    if len(transcript)<3 or set(transcript) == " ":
                                        continue
    
                                    # 클라이언트로 최종 전사 전송
                                    await sess.out_q.put(jdumps({
                                        "type": "transcript", "text": transcript, "final": True
                                    }))

                                    sess.end_scripting_time = time.time()
                                    await translate_next(sess, transcript)
                            
                            else:
                                dprint("audio None")
                    except Exception as e:
                     dprint("Error : ", e)
                # 3) 커밋 신호 전달 (chunk 경계)
                elif t == "input_audio_buffer.commit":
                    lprint("input_audio_buffer.commit")
                    sess.end_audio_input_time = time.time()
                    sess.end_tts_time = 0
                    
                    # 지금까지 누적된 PCM을 Whisper에 태워서 script 추출
                    # buf = bytes(sess.audios)

                    # 비동기 실행 (블로킹 피함)
                    transcript = await transcribe_pcm(audios=sess.audios)
                    dprint("Transcript : ", transcript)

                    # 클라이언트로 최종 전사 전송
                    await ws.send_text(jdumps({
                        "type": "transcript",
                        "text": transcript,
                        "final": True
                    }))
                    continue

                # 4) 세션 종료
                elif t == "session.close":
                    await ws.send_text(jdumps({
                        "type": "session.close",
                        "payload": {"status": "closed successfully"},
                        "connected_time": time.time() - sess.connection_start_time,
                        "llm_cached_token_count": sess.llm_cached_token_count,
                        "llm_input_token_count": sess.llm_input_token_count,
                        "llm_output_token_count": sess.llm_output_token_count,
                        "stt_output_token_count": sess.stt_output_token_count,
                    }))
                    break

                else:
                    # 필요시 기타 타입 처리
                    pass

    except WebSocketDisconnect:
        pass
    finally:
        await teardown_session(sess)
        sessions.pop(id(ws), None)

def process_data_to_audio(aud):
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
        
        if INPUT_SAMPLE_RATE != WHISPER_SR:
            audio_t = torch.from_numpy(audio).unsqueeze(0)  # (1, n)
            audio_resampled = torchaudio.functional.resample(audio_t, orig_freq=INPUT_SAMPLE_RATE, new_freq=WHISPER_SR)
            audio = audio_resampled.squeeze(0).contiguous().numpy()
        return audio
    else:
        return None

def feed_pcm16le_bytes(x):
    t = torch.from_numpy(x)[-512:] # 16kHz 기준, 길이가 512의 배수여야 한다.
    
    label = vad_iter(t, return_seconds=False)  # {'end': 212448} or {'start': 212448}

    if label is not None:
        if "start" in label:
            return 'start'
        if "end" in label:
            return 'end'
    return None

# === Whisper 전사 함수 구현 ===
async def transcribe_pcm(audios) -> str:
    stts = time.time()
    result = asr_model.transcribe(audios)
    text = re.sub(r"<\|.*?\|>", "", result[0].text).strip()
    
    return text

async def translate_next(sess: Session, final_text: str):
    # 3-2) 연결별 누적 전사 업데이트
    if sess.current_transcript:
        sess.current_transcript += " " + final_text
    else:
        sess.current_transcript = final_text
    
    # 너무 짧은 문장이나 단어는 굳이 바로 번역하지 않기.
    if len(sess.current_transcript) < 6:
        return
    
    dprint("\nprevScripts", sess.transcripts[-5:])
    dprint("current_scripted_sentence", sess.current_transcript)
    dprint("current_translated", sess.current_translated)

    # --- 여기부터 교체 ---
    translated_text, is_cached = await run_translate_async(sess)
    sess.end_translation_time = time.time()
    lprint(f"[{is_cached}, {sess.end_translation_time - sess.end_scripting_time:.2f}] translation : ", translated_text)

    translated_text = translated_text.replace("<SKIP>", "")
    translated_text = translated_text.replace("...", "")
    if translated_text == "" or translated_text is None:
        return

    # 3-5) 누적 번역 업데이트
    if sess.current_translated and "<CORRECTED>" not in translated_text:
        sess.current_translated += " " + translated_text
    else:
        sess.current_translated = translated_text

    # 3-6) 문장 종료(<END>) 처리. client 단과 액션을 맞춰야함.
    if "<END>" in translated_text:
        translated_text = translated_text.replace("<END>", "").strip()

        sess.transcripts.append(sess.current_transcript)
        sess.translateds.append(translated_text)

        # 다음 문장 누적용 버퍼 비우기
        sess.current_transcript = ""
        sess.current_translated = ""


async def teardown_session(sess: Session):
    sess.running = False

    tasks = [sess.oai_task, sess.sender_task]
    # 1) 모두 취소
    for t in tasks:
        if t and not t.done():
            t.cancel()
    for t in tasks:
        if t:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
    if sess.oai_ws:
        with contextlib.suppress(Exception):
            await sess.oai_ws.close()

async def outbound_sender(sess: Session, client_ws: WebSocket):
    try:
        while sess.running:
            msg = await sess.out_q.get()
            await client_ws.send_text(msg)
    except Exception:
        pass

# --- 추가: 동기 translate를 스레드에서 돌리고, 콜백은 루프-세이프로 브리지 ---
async def run_translate_async(sess: Session) -> str:
    """
    sess.current_transcript, sess.current_translated, sess.transcripts[-5:]
    를 사용해서 동기 translate를 스레드에서 실행.
    onToken 콜백은 루프-세이프로 sess.out_q에 push.
    최종 완성 문자열을 반환.
    """
    loop = asyncio.get_running_loop()

    def run_blocking():
        context = ' '.join(sess.transcripts[-8:])
        
        # 여기서 만약 sess.current_transcript가 너무 길면, 최대한 전체를 다 번역하도록 하는게 좋지 않을까?
        if len(sess.current_transcript) >= 150 and sess.current_transcript[-1] in [".", "?", "!"]:
            dprint("MAKE END를 호출")
            return translate_speaker_make_end(
                prevScripts=context,
                current_scripted_sentence=sess.current_transcript,
                current_translated=sess.current_translated,
            )
        else:
            return translate_speaker(
                prevScripts=context,
                current_scripted_sentence=sess.current_transcript,
                current_translated=sess.current_translated,
            )

    # 동기 작업을 thread로
    loop = asyncio.get_running_loop()
    output = await loop.run_in_executor(None, run_blocking)
    # 최종 결과부터 바로 알림
    await sess.out_q.put(jdumps({"type": "translated", "text": output["text"]}))
    
    sess.llm_cached_token_count += output["prompt_tokens_cached"]
    sess.llm_input_token_count += output["prompt_tokens"]
    sess.llm_output_token_count += output["completion_tokens"]
    
    return output["text"], True if output["prompt_tokens_cached"]>0 else False

# 필요 import
import contextlib
