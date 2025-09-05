# --- MUST be at the very top (before any vllm/torch import) ---
import app.boot
import os, socket, multiprocessing as mp
import asyncio
import json
from typing import Dict, Optional
from llm.openai import translate, translate_simple
from llm.openai_translate_simple import translate_simple
from llm.custom import load_llm
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
import os, multiprocessing as mp
import math

from stt.asr import load_asr_backend
from stt.vad import check_audio_state
from utils.process import process_data_to_audio
from tts.zipvoice_infer import load_infer_context, generate_sentence
from app.session import Session
from app.session_control import teardown_session, outbound_sender
from utils.text_process import text_pr

from chatterbox_infer.tts import ChatterboxTTS

tts_model = ChatterboxTTS.from_pretrained(device="cuda")

# 모델 로드 (기본값 사용)
ctx = load_infer_context()
ZIP_PROMPT_WAV_PATH = os.environ.get(
    "ZIPVOICE_PROMPT_WAV",
    "/workspace/ttssocketserver/tts/khj.wav",
)
ZIP_PROMPT_TEXT = os.environ.get(
    "ZIPVOICE_PROMPT_TEXT",
    "Please limit your phone booth usage to 1 hour. and do not leave your belongings unattended for your neighborhoods.",
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
LLM = None

INPUT_SAMPLE_RATE = 24000
WHISPER_SR = 16000

@app.on_event("startup")
def init_models():
    # 멀티프로세싱/환경 변수는 가장 먼저
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    # 파이썬 멀티프로세싱 start method도 spawn으로
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    global LLM
    # if LLM is None:
    #     LLM = load_llm()

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
    sess = Session(input_sr=INPUT_SAMPLE_RATE, input_channels=1)
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
                    try:
                        aud = data.get("audio")
                        if aud:
                            audio = process_data_to_audio(aud, input_sample_rate=INPUT_SAMPLE_RATE, whisper_sr=WHISPER_SR)
                            if audio is None:
                                dprint("[NO AUDIO]")
                                continue

                            # === 여기서 VAD 검사 ===
                            vads = time.time()
                            vad_event = check_audio_state(audio)

                            if sess.current_audio_sate != "start":
                                sess.pre_roll.append(audio)

                                if vad_event == "start":
                                    print("[Voice Start]")
                                    sess.current_audio_sate = "start"
                                    # pre-roll(최대 5개) + 이번 조각으로 audios 시드 생성
                                    if len(sess.pre_roll) > 0:
                                        sess.audios = np.concatenate(list(sess.pre_roll) + [audio]).astype(np.float32, copy=False)
                                    else:
                                        sess.audios = audio.astype(np.float32, copy=False)
                                    sess.pre_roll.clear()
                                    self.buf_count=0
                                # 아직 start가 아니면(=무음 지속) 계속 프리롤만 업데이트하고 다음 루프
                                continue
                            
                            sess.audios = np.concatenate([sess.audios, audio])
                            sess.buf_count += 1

                            if sess.buf_count%16==15 and sess.current_audio_sate == "start":
                                st = time.time()
                                sess.audios = sess.audios[-16000*20:]
                                pcm_bytes = (np.clip(sess.audios, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                                newScript = await transcribe_pcm_generic(
                                    audios=pcm_bytes,
                                    sample_rate=16000,
                                    channels=sess.input_channels
                                )
                                sess.transcript = text_pr(sess.transcript, newScript)
                                print(f"script - {sess.transcript}")
                                await sess.out_q.put(jdumps({"type": "delta", "text": sess.transcript, "is_final": False}))
                                sess.buf_count = 0

                            # if vad_event == "end":
                            #     print(f"[Voice end] - {sess.transcript}")
                            if vad_event == "end" and sess.transcript != "":
                                # 이때까지 자동으로 script 따던게 있을테니 그걸 리턴한다.
                                print("[Voice End] - ", sess.transcript)
                                await sess.out_q.put(jdumps({"type": "transcript", "text": sess.transcript, "is_final": True}))
                                sess.current_audio_sate = "none"
                                sess.audios = np.empty(0, dtype=np.float32)
                                sess.end_scripting_time = time.time()
                                
                                await translate_next(sess, sess.transcript)
                                sess.transcript = ""
                                continue
                                
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

async def translate_next(sess: Session, transcript: str):
    sess.current_transcript = sess.current_transcript + " " + transcript if sess.current_transcript else transcript
    
    # 너무 짧은 문장이나 단어는 굳이 바로 번역하지 않기.
    if len(sess.current_transcript)<6 or transcript.strip() == "":
        return

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

# # --- 추가: 동기 translate를 스레드에서 돌리고, 콜백은 루프-세이프로 브리지 ---
# async def run_translate_async(sess: Session) -> str:
#     """
#     sess.current_transcript, sess.current_translated, sess.transcripts[-5:]
#     를 사용해서 동기 translate를 스레드에서 실행.
#     onToken 콜백은 루프-세이프로 sess.out_q에 push.
#     최종 완성 문자열을 반환.
#     """
#     loop = asyncio.get_running_loop()

#     def flush_tts_chunk():
#         if not sess.tts_buf or len(sess.tts_buf) == 0:
#             return
        
#         chunk = "".join(sess.tts_buf).strip()
#         chunk = re.sub(r"<[^>]*>", "", chunk)

#         dprint("[flush_tts_chunk] ", repr(chunk))
#         sess.tts_buf.clear()
#         try:
#             sess.tts_in_q.put_nowait(chunk) # 기본적으로는 단어를 3개씩 끊어서 보내기
#         except asyncio.QueueFull:
#             dprint("[flush_tts_chunk] WARN: tts_in_q full, dropping chunk")

#     async def debounce_flush(delay_ms: int = 110):
#         try:
#             await asyncio.sleep(delay_ms / 1000)
#             flush_tts_chunk()
#         except Exception as e:
#             dprint("[debounce_flush] WARN: debounce_flush cancelled", e)

#     current_translated = ""
#     def on_token(tok: str):
#         current_translated += tok
#         if tok[-1] == '.':
#             sess.tts_in_q.put(current_translated)
#             sess.out_q.put(jdumps({"type": "translated", "text": current_translated, "is_final": False}))
#         # def _append_and_schedule():
#         #     sess.tts_buf.append(tok)
#         #     if sess.tts_debounce_task and not sess.tts_debounce_task.done():
#         #         sess.tts_debounce_task.cancel()
#         #     sess.tts_debounce_task = asyncio.create_task(debounce_flush(110))
#         # loop.call_soon_threadsafe(_append_and_schedule)

#     def run_blocking():
#         # 동기 translate 호출
#         return translate(
#             prevScripts=sess.transcripts[-5:],
#             current_scripted_sentence=sess.current_transcript,
#             current_translated=sess.current_translated,
#             onToken=on_token,
#         )

#     # 동기 작업을 thread로
#     loop = asyncio.get_running_loop()
#     output = await loop.run_in_executor(None, run_blocking)
    
#     translated_text = output.get("text", "")
#     if translated_text == "":
#         dprint("No translated text")
#         return ''
    
#     # sess.llm_cached_token_count += output["prompt_tokens_cached"]
#     # sess.llm_input_token_count += output["prompt_tokens"]
#     # sess.llm_output_token_count += output["completion_tokens"]

#     if sess.tts_debounce_task and not sess.tts_debounce_task.done():
#         sess.tts_debounce_task.cancel()

#     def _final_flush():
#         if sess.tts_debounce_task and not sess.tts_debounce_task.done():
#             sess.tts_debounce_task.cancel()
#         flush_tts_chunk()

#     loop.call_soon_threadsafe(_final_flush)
#     await sess.tts_in_q.put(translated_text)

#     # 최종 결과 알림
#     await sess.out_q.put(jdumps({"type": "translated", "text": translated_text, "is_final": True}))
#     return translated_text

async def run_translate_async(sess: Session) -> str:
    loop = asyncio.get_running_loop()
    current_translated = ""
    sent_chars = 0

    def safe_push_tts(text: str):
        # 스레드→루프 안전 (콜백에서만 사용)
        def _f():
            try:
                sess.tts_in_q.put_nowait(text)
            except asyncio.QueueFull:
                pass
        loop.call_soon_threadsafe(_f)

    def safe_push_out(msg: dict):
        def _f():
            try:
                sess.out_q.put_nowait(jdumps(msg))
            except asyncio.QueueFull:
                pass
        loop.call_soon_threadsafe(_f)

    def clean_text(text: str) -> str:
        return re.sub(r"<[^>]*>", "", text)


    def on_token(tok: str):
        nonlocal current_translated, sent_chars
        if not tok:
            return
        current_translated += tok

        # 문장 경계에서만 스트림 전송
        if tok.endswith(('.', '!', '?', '。', '！', '？')) and len(current_translated) > 6:
            # 아직 안 보낸 구간만 전송
            segment = clean_text(current_translated[sent_chars:]).strip()
            if segment:
                safe_push_tts(segment)
                safe_push_out({"type": "translated", "text": current_translated, "is_final": False})
                sent_chars = len(current_translated)

    def run_blocking():
        return translate(
            prevScripts=sess.transcripts[-5:],
            current_scripted_sentence=sess.current_transcript,
            current_translated=sess.current_translated,
            onToken=on_token,
        )

    output = await loop.run_in_executor(None, run_blocking)
    translated_text = output.get("text", "") or ""

    if translated_text == "":
        return ""
    # 끝났을 때, 아직 안 보낸 꼬리만 (있으면) 전송
    tail = clean_text(translated_text[sent_chars:]).strip()
    if len(tail) > 2:
        await sess.tts_in_q.put(tail)
        sent_chars = len(translated_text)

    # 최종 알림은 중복 없이 한 번
    await sess.out_q.put(jdumps({"type": "translated", "text": translated_text, "is_final": True}))
    return translated_text

@torch.inference_mode()
async def chatter_streamer(sess: Session, prompt_text: str, prompt_wav_path: str):
    dprint("[chatter_streamer] START")
    try:
        loop = asyncio.get_running_loop()
        sr = 24000
        FRAME_SEC = 0.5
        FRAME_SAMPLES = int(sr * FRAME_SEC)
        OVERLAP = int(0.05 * sr)              # 50ms 교차페이드

        async def consume_loop():
            while sess.running:
                text_chunk = await sess.tts_in_q.get()
                if not text_chunk or not text_chunk.strip():
                    continue

                t0 = time.time()
                dprint(f"[zipvoice_streamer] ▶ start streaming for chunk: {text_chunk!r}")

                # --- 스트리밍 상태 변수 (청크별로 리셋) ---
                last_length = 0               # 모델이 만든 '전체' wav 길이 (누적)
                last_tail: Optional[torch.Tensor] = None  # 직전 출력 청크의 꼬리 (OVERLAP 샘플)
                frame_buf: Optional[torch.Tensor] = None  # 0.5초 프레임 배출용 누적 버퍼 (mono)

                async def emit_frames_from(out_chunk: torch.Tensor, is_final: bool = False):
                    nonlocal frame_buf

                    if out_chunk.ndim == 2 and out_chunk.shape[0] == 1:
                        out_chunk = out_chunk[0]
                    elif out_chunk.ndim == 2 and out_chunk.shape[0] > 1:
                        # 다채널이면 첫 채널만 사용(필요시 여기서 downmix로 변경 가능)
                        out_chunk = out_chunk[0]
                    elif out_chunk.ndim == 1:
                        pass
                    else:
                        out_chunk = out_chunk.reshape(-1)

                    out_chunk = out_chunk.detach()

                    if frame_buf is None:
                        frame_buf = out_chunk
                    else:
                        frame_buf = torch.cat([frame_buf, out_chunk], dim=-1)

                    # 정해진 프레임 길이만큼 계속 배출
                    while frame_buf.numel() >= FRAME_SAMPLES:
                        part = frame_buf[:FRAME_SAMPLES]
                        frame_buf = frame_buf[FRAME_SAMPLES:]

                        # float32/float(-1..1) → int16
                        pcm = (torch.clamp(part, -1.0, 1.0) * 32767.0).to(torch.int16).cpu().numpy().tobytes()
                        b64 = base64.b64encode(pcm).decode()

                        await sess.out_q.put(jdumps({
                            "type": "tts_audio",
                            "format": "pcm16le",
                            "sample_rate": sr,
                            "channels": 1,
                            "audio": b64,
                            "isFinal": False,
                        }))

                    # 마지막이면 프레임에 못 미치는 잔여분까지 모두 내보내고 마지막 패킷에 isFinal=True
                    if is_final:
                        if frame_buf is not None and frame_buf.numel() > 0:
                            part = frame_buf
                            frame_buf = None
                            pcm = (torch.clamp(part, -1.0, 1.0) * 32767.0).to(torch.int16).cpu().numpy().tobytes()
                            b64 = base64.b64encode(pcm).decode()
                            await sess.out_q.put(jdumps({
                                "type": "tts_audio",
                                "format": "pcm16le",
                                "sample_rate": sr,
                                "channels": 1,
                                "audio": b64,
                                "isFinal": True,
                            }))
                        else:
                            pass

                # --- 모델 스트리밍 루프 ---
                try:
                    async for evt in tts_model.generate_stream(
                        text_chunk,
                        audio_prompt_path=prompt_wav_path,
                        # 필요시 추가 파라미터들: temperature, top_p, cfg_weight, exaggeration 등
                    ):
                        if evt.get("type") == "chunk":
                            wav: torch.Tensor = evt["audio"]  # shape: (ch, T_total_so_far)

                            # 누적 길이에서 '새로 추가된' 만큼만 절단
                            new_total = wav.shape[-1]
                            delta = new_total - last_length
                            if delta <= 0:
                                continue

                            new_part = wav[:, last_length:new_total]  # (ch, delta)

                            # 교차페이드(OVERLAP)
                            if last_tail is None:
                                out_chunk = new_part
                            else:
                                L = min(OVERLAP, new_part.shape[-1], last_tail.shape[-1])
                                if L > 0:
                                    device = wav.device
                                    dtype = wav.dtype
                                    fade_in  = torch.linspace(0, 1, L, device=device, dtype=dtype)
                                    fade_out = 1.0 - fade_in
                                    mixed = last_tail[:, -L:] * fade_out + new_part[:, :L] * fade_in
                                    tail = new_part[:, L:]
                                    out_chunk = torch.cat([mixed, tail], dim=-1)
                                else:
                                    out_chunk = new_part

                            # 다음 교차페이드를 위해 최신 꼬리 보관
                            new_tail_start = max(0, new_total - OVERLAP)
                            last_tail = wav[:, new_tail_start:new_total].detach()

                            # 0.5초 프레임 단위로 전송
                            await emit_frames_from(out_chunk, is_final=False)

                            last_length = new_total

                        elif evt.get("type") == "eos":
                            # 스트림 종료: 잔여 프레임 flush + 마지막 패킷 isFinal=True
                            await emit_frames_from(torch.empty(0, device=last_tail.device if last_tail is not None else "cpu"), is_final=True)
                            dprint("✅ [zipvoice_streamer] streaming EOS for chunk")
                            break

                except Exception as e:
                    dprint("[chatter_streamer] TTS stream error:", e)
                    # 에러 발생 시, 남은 버퍼가 있다면 마무리 패킷 전송 시도
                    try:
                        await emit_frames_from(torch.empty(0), is_final=True)
                    except Exception:
                        pass
                    continue

                lprint(f"[zipvoice_streamer] streamed {last_length/sr:.2f}s "
                       f"→ {time.time()-t0:.2f}s")

        await consume_loop()

    except asyncio.CancelledError:
        dprint("[chatter_streamer] CANCELLED")
        raise
    except Exception as e:
        dprint("[chatter_streamer] ERROR:", e)
        await sess.out_q.put(jdumps({"type": "tts_error", "message": str(e)}))
    finally:
        dprint("[chatter_streamer] END")


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
                print("Speech after Talking end. - ", time.time() - sess.end_scripting_time, "s")

                if hasattr(wav, "numpy"):
                    arr = wav.numpy()
                else:
                    arr = np.asarray(wav)
                arr = np.squeeze(arr)
                if arr.ndim != 1:
                    arr = arr.reshape(-1)

                # float32(-1..1) -> int16 PCM
                sr = ctx["sampling_rate"]   # 보통 24000
                chunk_size = int(sr * 0.5)  # 0.5초 단위 샘플 수
                
                total_samples = len(arr)
                num_chunks = math.ceil(total_samples / chunk_size)
                
                for i in range(num_chunks):
                    start = i * chunk_size
                    end = min((i + 1) * chunk_size, total_samples)
                
                    chunk_arr = arr[start:end]
                
                    # float32(-1..1) → int16 PCM
                    pcm = (np.clip(chunk_arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                    b64 = base64.b64encode(pcm).decode()
                
                    await sess.out_q.put(jdumps({
                        "type": "tts_audio",
                        "format": "pcm16le",
                        "sample_rate": sr,
                        "channels": 1,
                        "audio": b64,
                        # 마지막 청크일 때만 True로 표시
                        "isFinal": (i == num_chunks - 1),
                    }))
                # pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                # b64 = base64.b64encode(pcm).decode()

                # # 클라이언트로 전송 (형식 메타 포함)
                # await sess.out_q.put(jdumps({
                #     "type": "tts_audio",
                #     "format": "pcm16le",                  # 클라에서 디코딩할 포맷
                #     "sample_rate": ctx["sampling_rate"],  # 24000
                #     "channels": 1,
                #     "audio": b64,
                #     "isFinal": False,
                # }))

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