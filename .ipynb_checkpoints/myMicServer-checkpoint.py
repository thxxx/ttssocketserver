import asyncio
import json
from typing import Dict, Optional
from llm.openai import translate, translate_simple
from stt.openai import open_openai_ws
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed
import orjson as json
import os
import websockets
import base64
import re
import time
from websockets.asyncio.client import connect as ws_connect
import numpy as np
import librosa
import torch
import contextlib

from stt.asr import load_asr_backend
from stt.vad import check_audio_state
from utils.process import process_data_to_audio
from tts.zipvoice_infer import load_infer_context, generate_sentence

# 모델 로드 (기본값 사용)
ctx = load_infer_context()
ZIP_PROMPT_WAV_PATH = os.environ.get(
    "ZIPVOICE_PROMPT_WAV",
    "/workspace/ttssocketserver/tts/voice.wav",
)
ZIP_PROMPT_TEXT = os.environ.get(
    "ZIPVOICE_PROMPT_TEXT",
    "This is a test.",
)

DEBUG = True
def dprint(*a, **k): 
    if DEBUG: print(*a, **k)

LOGG = True
def lprint(*a, **k): 
    if LOGG: print(*a, **k)

def jdumps(o): return json.dumps(o).decode()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

app = FastAPI()

ASR = load_asr_backend(kind="korean")

INPUT_SAMPLE_RATE = 24000
WHISPER_SR = 16000

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
        self.current_audio_sate = "none"

        self.new_speech_start = 0

sessions: Dict[int, Session] = {}  # id(ws)로 매핑

async def transcribe_pcm_generic(audios, sample_rate: int, channels: int) -> str:
    if not audios:
        return ""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: ASR.transcribe_pcm(audios, sample_rate, channels, language="korean")
    )

@app.websocket("/ws")
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

                # (A) 핑-퐁: 시계 오프셋 추정용
                if t == "latency.ping":
                    t1 = int(time.time() * 1000)   # server recv
                    t2 = int(time.time() * 1000)   # server send (즉시)
                    await ws.send_text(jdumps({
                        "type": "latency.pong",
                        "t0": data["t0"], "t1": t1, "t2": t2
                    }))
                    if not sess.is_network_logging:
                        sess.is_network_logging = True
                        lprint("network latency logging started")
                    continue

                # 1) 세션 시작: OpenAI Realtime WS 연결
                if t == "scriptsession.start":
                    if sess.oai_ws is None:
                        sess.oai_ws = await open_openai_ws()
                        sess.connection_start_time = time.time()
                        
                        try:
                            sess.tts_task = asyncio.create_task(
                                zipvoice_streamer(
                                    sess,
                                    prompt_text=ZIP_PROMPT_TEXT,
                                    prompt_wav_path=ZIP_PROMPT_WAV_PATH,
                                )
                            )
    
                            # OpenAI 이벤트를 client로 릴레이하는 백그라운드 태스크
                            await ws.send_text(jdumps({"type": "scriptsession.started"}))
                        except Exception as e:
                            dprint("TTS connection error ", e)
                    else:
                        await ws.send_text(jdumps({"type": "warn", "message": "already started"}))

                # 2) 오디오 append → Open AI로 그대로 전달
                elif t == "input_audio_buffer.append":
                    print("--")
                    try:
                        aud = data.get("audio")
                        if aud:
                            audio = process_data_to_audio(aud, input_sample_rate=INPUT_SAMPLE_RATE, whisper_sr=WHISPER_SR)
                            if audio is None:
                                dprint("[NO AUDIO]")
                                continue
                            
                            sess.audios = np.concatenate([sess.audios, audio])
                            sess.buf_count += 1

                            # === 그냥 1초 단위로 해보자 ===
                            if sess.buf_count%16==15 and sess.current_audio_sate == "start":
                                st = time.time()
                                pcm_bytes = (np.clip(sess.audios, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                                transcript = await transcribe_pcm_generic(
                                    audios=pcm_bytes,
                                    sample_rate=16000,
                                    channels=sess.input_channels
                                )
                                # dprint(f"[WHILE {time.time() - st:.2f}s] - {transcript}\n")
                                await sess.out_q.put(jdumps({
                                    "type": "delta", "text": transcript, "final": False
                                }))
                                sess.buf_count = 0
                            
                            # === 여기서 VAD 검사 ===
                            vads = time.time()
                            vad_event = check_audio_state(audio)

                            if vad_event == "start":
                                sess.current_audio_sate = "start"
                                continue
                            if vad_event == "end" and transcript != "":
                                # 이때까지 자동으로 script 따던게 있을테니 그걸 리턴한다.
                                print("END - ", transcript)
                                await sess.out_q.put(jdumps({
                                    "type": "transcript", "text": transcript, "final": True
                                }))
                                sess.current_audio_sate = "none"
                                sess.audios = np.empty(0, dtype=np.float32)
                                
                                await translate_next(sess, transcript)
                                transcript = ""
                                continue
                            
                            if vad_event == "end" and sess.audios.size>WHISPER_SR*1.5:
                                target_audios = sess.audios
                                sess.audios = np.empty(0, dtype=np.float32)
                                
                                t00 = time.time()
                                pcm_bytes = (np.clip(target_audios, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                                transcript = await transcribe_pcm_generic(
                                    audios=pcm_bytes,
                                    sample_rate=16000,
                                    channels=sess.input_channels
                                )
                                dprint(f"[SCRIPT {time.time() - t00:.2f}s] - {transcript}\n")

                                # 클라이언트로 최종 전사 전송
                                await sess.out_q.put(jdumps({
                                    "type": "transcript", "text": transcript, "final": True
                                }))

                                sess.end_scripting_time = time.time()
                                # await translate_next(sess, transcript)
                                
                    except Exception as e:
                     dprint("Error : ", e)
                
                # 3) 커밋 신호 전달 (chunk 경계) = 현재 세팅에서는 VAD를 여기서 검사하기 때문에, 들어올일이 없다.
                elif t == "input_audio_buffer.commit":
                    lprint("input_audio_buffer.commit")
                    continue

                elif t == "test":
                    ct = data.get("current_time")
                    if ct:
                        lprint("network latency : ", time.time()*1000 - ct)
                elif t == "session.close":
                    await ws.send_text(jdumps({
                        "type": "session.close",
                        "payload": {"status": "closed successfully"},
                        "connected_time": time.time() - sess.connection_start_time,
                        "llm_cached_token_count": sess.llm_cached_token_count,
                        "llm_input_token_count": sess.llm_input_token_count,
                        "llm_output_token_count": sess.llm_output_token_count,
                    }))
                    break

                else:
                    # 필요시 기타 타입 처리
                    pass

            elif msg.get("bytes") is not None:
                # 바이너리로도 보낼 수 있다면 여기서 OAI로 전달하는 변형 가능
                buf: bytes = msg["bytes"]
                await ws.send_text(jdumps({
                    "type": "binary_ack",
                    "payload": {"received_bytes": len(buf)}
                }))

    except WebSocketDisconnect:
        pass
    finally:
        await teardown_session(sess)
        sessions.pop(id(ws), None)

async def translate_next(sess: Session, final_text: str):
    if sess.current_transcript:
        sess.current_transcript += " " + final_text
    else:
        sess.current_transcript = final_text
    
    # 너무 짧은 문장이나 단어는 굳이 바로 번역하지 않기.
    if len(sess.current_transcript)<6 or set(final_text) == " ":
        return
    
    # dprint("\nprevScripts", sess.transcripts[-5:])
    # dprint("current_scripted_sentence", sess.current_transcript)
    # dprint("current_translated", sess.current_translated)

    # run_translate_async안에서 바로 결과를 client에게 보낸다.
    st = time.time()
    translated_text = await run_translate_async(sess)
    
    print(f"[Translate {time.time() - st:.2f}s] - {translated_text}")
    sess.current_transcript = ""
    return

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

async def relay_openai_to_client(sess: Session, client_ws: WebSocket):
    try:
        async for raw in sess.oai_ws:
            try:
                evt = json.loads(raw)
            except Exception:
                await sess.out_q.put(raw)
                continue

            etype = evt.get("type", "")

            if etype.endswith(".delta"):
                text = evt.get("delta") or evt.get("text") or evt.get("content") or ""
                await sess.out_q.put(jdumps({"type": "delta", "text": text})) # 거의 걸리지 않음.
                # pass
            elif etype.endswith(".completed"):
                # lprint("latency to transcribe end : ", time.time() - sess.end_audio_input_time)

                sess.first_translated_token_output_time = 0
                # 3-1) 최종 전사 수신
                final_text = (evt.get("transcript") or evt.get("content") or "").strip()
                sess.end_scripting_time = time.time()
                
                await sess.out_q.put(jdumps({
                    "type": "transcript", "text": final_text, "final": True
                }))

                # 3-2) 연결별 누적 전사 업데이트
                if sess.current_transcript:
                    sess.current_transcript += " " + final_text
                else:
                    sess.current_transcript = final_text
                
                if len(sess.current_transcript) < 3:
                    continue
                
                dprint("prevScripts", sess.transcripts[-5:])
                dprint("current_scripted_sentence", sess.current_transcript)
                dprint("current_translated", sess.current_translated)

                # --- 여기부터 교체 ---
                # 동기 translate를 스레드에서 비동기처럼 실행(스트리밍 콜백 유지)
                translated_text = await run_translate_async(sess)
                # --- 교체 끝 ---
                sess.end_translation_time = time.time()
                lprint("[Latency logging] script end to translate end time : ", sess.end_translation_time - sess.end_scripting_time)
                lprint("[Latency logging] first to last translate token time : ", sess.end_translation_time - sess.first_translated_token_output_time)

                translated_text = translated_text.replace("<SKIP>", "")
                if translated_text == "" or translated_text is None:
                    continue

                # 3-5) 누적 번역 업데이트 (공백 관리)
                if sess.current_translated:
                    sess.current_translated += " " + translated_text
                else:
                    sess.current_translated = translated_text

                # 3-6) 문장 종료(<END>) 처리
                if "<END>" in translated_text:
                    # 저장 시에는 <END> 제거해서 넣는 걸 권장
                    translated_text = translated_text.replace("<END>", "").strip()

                    sess.transcripts.append(sess.current_transcript)
                    sess.translateds.append(translated_text)

                    # 다음 문장 누적용 버퍼 비우기
                    sess.current_transcript = ""
                    sess.current_translated = ""

            elif etype == "error":
                await sess.out_q.put(jdumps({
                    "type": "error", "message": evt.get("error", evt)
                }))
            else:
                await sess.out_q.put(jdumps({
                    "type": "oai_event", "event": evt
                }))

    except ConnectionClosed:
        pass
    except Exception as e:
        await sess.out_q.put(jdumps({
            "type": "error", "message": f"OAI relay error: {e}"
        }))


async def teardown_session(sess: Session):
    sess.running = False

    tasks = [sess.tts_task, sess.oai_task, sess.sender_task]
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
    if sess.tts_ws:
        with contextlib.suppress(Exception):
            if sess.tts_ws.open:
                await sess.tts_ws.send(jdumps({"text": ""}))  # EOS
            await sess.tts_ws.wait_closed()
        sess.tts_ws = None

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

    def flush_tts_chunk():
        if not sess.tts_buf or len(sess.tts_buf) == 0:
            return
        
        chunk = "".join(sess.tts_buf).strip()
        chunk = re.sub(r"<[^>]*>", "", chunk)

        # dprint("[flush_tts_chunk] ", repr(chunk))
        sess.tts_buf.clear()
        try:
            # sess.tts_in_q.put_nowait(chunk) # 마지막 청크
            # chunks = chunk.split(" ")
            # for i in range(0, len(chunks), 3):
            #     if i >= len(chunks) - 4:
            #         sess.tts_in_q.put_nowait(" ".join(chunks[i:])) # 마지막 청크
            #         break
            #     else:
            #         sess.tts_in_q.put_nowait(" ".join(chunks[i:i+3]) + " ") # 기본적으로는 단어를 3개씩 끊어서 보내기
            sess.tts_in_q.put_nowait(chunk) # 기본적으로는 단어를 3개씩 끊어서 보내기
        except asyncio.QueueFull:
            dprint("[flush_tts_chunk] WARN: tts_in_q full, dropping chunk")

    async def debounce_flush(delay_ms: int = 110):
        try:
            await asyncio.sleep(delay_ms / 1000)
            flush_tts_chunk()
        except Exception as e:
            dprint("[debounce_flush] WARN: debounce_flush cancelled", e)

    def on_token(tok: str):
        # 다른 스레드에서 불릴 수 있으므로 루프 스레드로 래핑
        if sess.first_translated_token_output_time == 0:
            sess.first_translated_token_output_time = time.time()

        def _append_and_schedule():
            sess.tts_buf.append(tok)
            if sess.tts_debounce_task and not sess.tts_debounce_task.done():
                sess.tts_debounce_task.cancel()
            sess.tts_debounce_task = asyncio.create_task(debounce_flush(110))
        loop.call_soon_threadsafe(_append_and_schedule)

    def run_blocking():
        # 동기 translate 호출
        return translate_simple(
            prevScripts=sess.transcripts[-5:],
            current_scripted_sentence=sess.current_transcript,
            current_translated=sess.current_translated,
            onToken=on_token,
        )

    # 동기 작업을 thread로
    loop = asyncio.get_running_loop()
    output = await loop.run_in_executor(None, run_blocking)
    
    final_text = output.get("text", "")
    if final_text == "":
        dprint("No translated text")
        return ''
    
    sess.llm_cached_token_count += output["prompt_tokens_cached"]
    sess.llm_input_token_count += output["prompt_tokens"]
    sess.llm_output_token_count += output["completion_tokens"]

    if sess.tts_debounce_task and not sess.tts_debounce_task.done():
        sess.tts_debounce_task.cancel()

    def _final_flush():
        if sess.tts_debounce_task and not sess.tts_debounce_task.done():
            sess.tts_debounce_task.cancel()
        flush_tts_chunk()

    loop.call_soon_threadsafe(_final_flush)

    # 최종 결과 알림
    await sess.out_q.put(jdumps({"type": "translated", "text": final_text}))
    return final_text

@torch.inference_mode()
async def zipvoice_streamer(sess: Session, prompt_text: str, prompt_wav_path: str):
    """
    - sess.tts_in_q 에 들어오는 text chunk 를 받아 로컬 ZipVoice로 음성 생성
    - 생성된 오디오를 base64 PCM16LE로 보내서 클라이언트가 즉시 재생
    - 순서 보장을 위해 worker 1개로 처리 (번역/ASR과는 병렬)
    """
    dprint("[zipvoice_streamer] START")
    try:
        loop = asyncio.get_running_loop()

        async def consume_loop():
            while sess.running:
                text_chunk = await sess.tts_in_q.get()
                if not text_chunk or not text_chunk.strip():
                    continue

                t0 = time.time()

                # 블로킹 TTS를 thread executor에서 실행해 이벤트 루프 블로킹 방지
                def _run_blocking():
                    wav, info = generate_sentence(
                        prompt_text=prompt_text,
                        prompt_wav_path=prompt_wav_path,
                        text=text_chunk,
                        ctx=ctx,
                    )
                    return wav, info

                try:
                    wav, info = await loop.run_in_executor(None, _run_blocking)
                except Exception as e:
                    dprint("[zipvoice_streamer] TTS error:", e)
                    continue

                # wav: torch.Tensor, shape (N,) 또는 (1, N)
                if hasattr(wav, "numpy"):
                    arr = wav.numpy()
                else:
                    arr = np.asarray(wav)
                arr = np.squeeze(arr)
                if arr.ndim != 1:
                    arr = arr.reshape(-1)

                # float32(-1..1) -> int16 PCM
                pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                b64 = base64.b64encode(pcm).decode()

                # 클라이언트로 전송 (형식 메타 포함)
                await sess.out_q.put(jdumps({
                    "type": "tts_audio",
                    "format": "pcm16le",                  # 클라에서 디코딩할 포맷
                    "sample_rate": ctx["sampling_rate"],  # 24000
                    "channels": 1,
                    "audio": b64,
                    "isFinal": False,
                }))

                lprint(f"[zipvoice_streamer] chunk TTS {len(arr)/ctx['sampling_rate']:.2f}s "
                      f"→ {time.time()-t0:.2f}s")
        await consume_loop()
    except asyncio.CancelledError:
        dprint("[zipvoice_streamer] CANCELLED")
        raise
    except Exception as e:
        dprint("[zipvoice_streamer] ERROR:", e)
        await sess.out_q.put(jdumps({"type": "tts_error", "message": str(e)}))
    finally:
        dprint("[zipvoice_streamer] END")