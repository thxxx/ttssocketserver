# --- MUST be at the very top (before any vllm/torch import) ---
import os, socket, multiprocessing as mp
import asyncio
from typing import Dict, Optional
from llm.openai_test import translate_simple, translate
from llm.custom import load_llm
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed
import orjson as json
import os
import re
import time
from websockets.asyncio.client import connect as ws_connect
import numpy as np
import librosa
import torch
import os, multiprocessing as mp
import random
from librosa.util import normalize

from stt.asr import load_asr_backend
from stt.vad import check_audio_state
from utils.process import process_data_to_audio
from app.session import Session, arm_end_timer, cancel_end_timer
from app.session_control import teardown_session, outbound_sender
from utils.text_process import text_pr

from chatterbox_infer.mtl_tts import ChatterboxMultilingualTTS

# 파일 상단
from concurrent.futures import ThreadPoolExecutor
ENC_EXEC = ThreadPoolExecutor(max_workers=2)

def likely_voice(audio: np.ndarray, rms_dbfs_thresh: float = -45.0, min_peak: float = 0.01):
    rms = float(np.sqrt(np.mean(audio**2)) + 1e-12)
    dbfs = 20.0 * np.log10(rms)
    peak = float(np.max(np.abs(audio)))
    return (dbfs, peak)
    # return (dbfs > rms_dbfs_thresh) and (peak > min_peak)

def pcm16_b64(part: torch.Tensor) -> str:
    arr = (torch.clamp(part, -1.0, 1.0).to(torch.float32) * 32767.0) \
            .to(torch.int16).cpu().numpy().tobytes()
    import base64
    return base64.b64encode(arr).decode()

filler_audios_path = ["./utils/hmhm.wav", "./utils/uhuh.wav", "./utils/ohoh.wav", "./utils/uhmuhm.wav"]
filler_audios = []
for p in filler_audios_path:
    audiod, sr = librosa.load(p, sr=24000, mono=True)
    audiod = torch.tensor(audiod)
    filler_audios.append(pcm16_b64(audiod))

ref_audio_sets = []
# audiod, sr = librosa.load('./haha.mp3', sr=24000, mono=True)
# audiod = torch.tensor(audiod)
ref_audio_sets.append('./haha.mp3')

prompt_wav_path = "./elv.mp3"
# audiod, sr = librosa.load(prompt_wav_path, sr=24000, mono=True)
# audiod = torch.tensor(audiod)
ref_audio_sets.append(prompt_wav_path)


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

global ASR, tts_model

ASR = None
tts_model = None
LLM = None
ASR = load_asr_backend(kind="en")
tts_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

INPUT_SAMPLE_RATE = 24000
WHISPER_SR = 16000
END_WORDS = [".", "!", "?", "。", "！", "？"]

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
                    t1 = int(time.time() * 1000)
                    t2 = int(time.time() * 1000)
                    await ws.send_text(jdumps({
                        "type": "latency.pong",
                        "t0": data["t0"], "t1": t1, "t2": t2
                    }))
                    if not sess.is_network_logging:
                        sess.is_network_logging = True
                        lprint("network latency logging started")
                    continue

                if t == 'scriptsession.setvoice':
                    aud = data.get("audio")
                    lprint("Got ref voice!")
                    if aud:
                        audio = process_data_to_audio(aud, input_sample_rate=24000, whisper_sr=WHISPER_SR)
                        if audio is None:
                            dprint("[NO AUDIO]")
                            continue
                        sess.ref_audios.put(audio)
                        continue

                # 1) 세션 시작: OpenAI Realtime WS 연결
                if t == "scriptsession.start":
                    global ASR, tts_model
                    lprint("Start ", data);

                    # if sess.in_language != data.get("in_language", "ko") or ASR is None:
                    #     sess.in_language = data.get("in_language", "ko")
                    #     ASR = load_asr_backend(kind=sess.in_language)

                    # if sess.out_language != data.get("out_language", "en") or tts_model is None:
                    sess.out_language = data.get("out_language", "en")
                    if tts_model is None:
                        tts_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
                    
                    if sess.tts_task is None:
                        try:
                            sess.tts_task = asyncio.create_task(chatter_streamer(sess))
                            sess.translator_task = asyncio.create_task(translator_worker(sess))
                            sess.is_use_filler = data.get("use_filler", False)
    
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
                            vad_event = check_audio_state(audio)

                            if sess.current_audio_state != "start":
                                sess.pre_roll.append(audio)

                                if vad_event == "start":
                                    cancel_end_timer(sess)
                                    print("Come", likely_voice(np.concatenate(list(sess.pre_roll) + [audio])))
                                    if not likely_voice(np.concatenate(list(sess.pre_roll) + [audio]).astype(np.float32, copy=False))[1] > 0.02:
                                        continue
                                    sess.current_audio_state = "start"
                                    if len(sess.pre_roll) > 0:
                                        sess.audios = np.concatenate(list(sess.pre_roll) + [audio]).astype(np.float32, copy=False)
                                    else:
                                        sess.audios = audio.astype(np.float32, copy=False)
                                    print("[Voice Start] ", likely_voice(sess.audios))
                                    sess.pre_roll.clear()
                                    sess.buf_count = 0
                                # 아직 start가 아니면(=무음 지속) 계속 프리롤만 업데이트하고 다음 루프
                                continue
                            
                            sess.audios = np.concatenate([sess.audios, audio])
                            sess.buf_count += 1

                            # if vad_event == "end" and sess.transcript != "":
                            #     # 이때까지 자동으로 script 따던게 있을테니 그걸 리턴한다.
                            #     print("[Voice End] - ", sess.transcript)
                            #     await sess.out_q.put(jdumps({"type": "transcript", "text": sess.transcript.strip(), "is_final": True}))
                            #     sess.current_audio_state = "none"
                            #     if "하하" in sess.transcript:
                            #         lprint("\nHaha generation!\n")
                            #         sess.ref_audios.put(ref_audio_sets[0])
                            #     else:
                            #         sess.ref_audios.put(ref_audio_sets[1])
                            #     sess.audios = np.empty(0, dtype=np.float32)
                            #     sess.end_scripting_time = time.time()
                                
                            #     try:
                            #         sess.translate_q.put_nowait(sess.transcript)
                            #     except:
                            #         print("\n\n\nMax translation queue!\n\n")
                            #     # arm_end_timer(sess, delay=3)
                            #     sess.transcript = ""
                            #     continue
                            
                            if sess.buf_count%11==10 and sess.current_audio_state == "start":
                                st = time.time()
                                sess.audios = sess.audios[-16000*20:]
                                
                                pcm_bytes = (np.clip(sess.audios, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                                newScript = await transcribe_pcm_generic(
                                    audios=pcm_bytes,
                                    sample_rate=16000,
                                    channels=sess.input_channels
                                )
                                if (len(newScript.split(" "))>6 and len(set(newScript.split(" ")))<2) or newScript in ["감사합니다.", "시청해주셔서 감사합니다."]:
                                    sess.audios = np.empty(0, dtype=np.float32)
                                    sess.buf_count = 0
                                    continue
                                sess.transcript = text_pr(sess.transcript, newScript)
                                print(f"[{time.time() - st:.4f}s] script - {sess.transcript}")
                                await sess.out_q.put(jdumps({"type": "delta", "text": sess.transcript, "is_final": False}))
                                sess.buf_count = 0
                                
                    except Exception as e:
                     dprint("Error : ", e)
                
                # 3) 커밋 신호 전달 (chunk 경계) = 현재 세팅에서는 VAD를 여기서 검사하기 때문에, 들어올일이 없다.
                elif t == "input_audio_buffer.commit":
                    lprint("input_audio_buffer.commit - ", sess.transcript)
                    if sess.transcript is not None and sess.transcript != "":
                        await sess.out_q.put(jdumps({"type": "transcript", "text": sess.transcript, "is_final": True}))
                    
                    if sess.transcript is not None and sess.transcript != "":
                        try:
                            sess.translate_q.put_nowait(sess.transcript)
                        except:
                            print("\n\n\nMax translation queue!\n\n")
                        sess.transcript = ""
                        # arm_end_timer(sess, delay=3)
                    
                    sess.current_audio_state = "none"
                    # sess.ref_audios.put(sess.audios)
                    sess.audios = np.empty(0, dtype=np.float32)

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

def reset_translation(sess: Session):
    lprint("reset_translation - ", sess.current_transcript, sess.current_translated)
    sess.transcripts.append(sess.current_transcript)
    sess.translateds.append(sess.current_translated)
    sess.current_transcript = ""
    sess.current_translated = ""

async def translator_worker(sess: Session):
    while sess.running:
        text = await sess.translate_q.get()
        await translate_one(sess, text)

async def translate_one(sess: Session, transcript: str):
    if transcript == "<END>":
        reset_translation(sess)
        return
    
    st = time.time()
    sess.current_transcript += " " + transcript
    translated_text = await run_translate_async(sess)  # 내부에서 run_in_executor 사용
    dprint(f"[Translate {time.time() - st:.2f}s] - {translated_text!r}")

    translated_text = (translated_text or "").replace("<SKIP>", "").replace("...", "").strip()
    sess.current_translated += " " + translated_text

    if len(translated_text) > 5 and translated_text[-1] in END_WORDS and translated_text[-2:] != "..":
        reset_translation(sess)
        return

async def run_translate_async(sess: Session) -> str:
    loop = asyncio.get_running_loop()
    current_translated = ""
    sent_chars = 0
    
    def safe_push_tts(text: str):
        def _f():
            sess.tts_in_q.put_nowait(text)  # 절대 await 안 함
        loop.call_soon_threadsafe(_f)

    def safe_push_out(msg: dict):
        def _f():
            sess.out_q.put_nowait(jdumps(msg))
        loop.call_soon_threadsafe(_f)

    def clean_text(text: str) -> str:
        return re.sub(r"<[^>]*>", "", text)

    def on_token(tok: str):
        safe_push_out({"type": "translated", "text": tok, "is_final": False})
        return
        nonlocal current_translated, sent_chars
        if not tok:
            return
        current_translated += tok
        # 문장 경계에서만 보냄 (너무 자잘한 조각 방지)
        if tok.endswith(('.', '!', '?', '。', '！', '？')) and len(current_translated[sent_chars:]) > 6:
            segment = clean_text(current_translated[sent_chars:]).strip()
            if segment:
                safe_push_tts(segment)
                safe_push_out({"type": "translated", "text": current_translated, "is_final": False})
                sent_chars = len(current_translated)

    def run_blocking():
        lprint("translate", sess.transcripts[-5:], sess.current_transcript, sess.current_translated, sess.in_language, sess.out_language)
        return translate(
            prevScripts=sess.transcripts[-5:],
            current_scripted_sentence=sess.current_transcript,
            current_translated=sess.current_translated,
            onToken=on_token,
            input_language=sess.in_language,
            output_language=sess.out_language,
        )

    output = await loop.run_in_executor(None, run_blocking)
    translated_text = output.get("text", "") or ""

    # 남은 꼬리 한 번만 푸시 (여기도 await 쓰지 않는 게 포인트)
    tail = clean_text(translated_text[sent_chars:]).strip()
    
    if len(tail) > 2:
        def _f():
            sess.tts_in_q.put_nowait(tail)
        loop.call_soon_threadsafe(_f)

    cts = sess.current_transcript
    if len(translated_text) > 5 and translated_text[-1] in END_WORDS and translated_text[-3:] != "...":
        translated_text += "<END>"
        reset_translation(sess)
    
    # 최종 알림(중복 방지)
    def _g():
        sess.out_q.put_nowait(jdumps({"type": "translated", "script": cts, "text": translated_text, "is_final": True}))
    loop.call_soon_threadsafe(_g)

    try:
        if sess.is_use_filler:
            sess.out_q.put_nowait(jdumps({
                "type": "tts_audio",
                "format": "pcm16le",
                "sample_rate": sr,
                "channels": 1,
                "audio": random.sample(filler_audios, k=1)[0],
                "isFinal": False,
            }))
    except Exception as e:
        print("error ", e)
    return translated_text

@torch.inference_mode()
async def chatter_streamer(sess: Session):
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
                # --- 스트리밍 상태 변수 (청크별로 리셋) ---
                last_length = 0               # 모델이 만든 '전체' wav 길이 (누적)
                last_tail: Optional[torch.Tensor] = None  # 직전 출력 청크의 꼬리 (OVERLAP 샘플)
                frame_buf: Optional[torch.Tensor] = None  # 0.5초 프레임 배출용 누적 버퍼 (mono)

                async def emit_frames_from(out_chunk: torch.Tensor):
                    b64 = await loop.run_in_executor(ENC_EXEC, pcm16_b64, out_chunk)

                    print("Chunk send ", time.time() - t0)
                    sess.out_q.put_nowait(jdumps({
                        "type": "tts_audio",
                        "format": "pcm16le",
                        "sample_rate": sr,
                        "channels": 1,
                        "audio": b64,
                        "isFinal": False,
                    }))
                    await asyncio.sleep(0)

                # --- 모델 스트리밍 루프 ---
                try:
                    if not sess.ref_audios.empty():
                        ref_audio = sess.ref_audios.get()
                        # sess.ref_audios.put(ref_audio)
                        # ref_audio = normalize(ref_audio[-int(16000*9):])*0.9
                    else:
                        ref_audio = prompt_wav_path
                    lprint("TTS generation - ", text_chunk)
                    
                    async for evt in tts_model.generate_stream(
                        text_chunk,
                        audio_prompt_path=ref_audio,
                        language_id=sess.out_language,
                        chunk_size=25,
                        temperature=1.1
                    ):
                        if evt.get("type") == "chunk":
                            wav: torch.Tensor = evt["audio"]

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
                                    dtype = wav.dtype
                                    fade_in  = torch.linspace(0, 1, L, device=device, dtype=dtype)
                                    fade_out = 1.0 - fade_in
                                    mixed = last_tail[:, -L:] * fade_out + new_part[:, :L] * fade_in
                                    tail = new_part[:, L:]
                                    out_chunk = torch.cat([mixed, tail], dim=-1)
                                else:
                                    out_chunk = new_part

                            new_tail_start = max(0, new_total - OVERLAP)
                            last_tail = wav[:, new_tail_start:new_total].detach()

                            await emit_frames_from(out_chunk)
                            last_length = new_total

                        elif evt.get("type") == "eos":
                            # 스트림 종료: 잔여 프레임 flush + 마지막 패킷 isFinal=True
                            await emit_frames_from(torch.empty(0, device=last_tail.device if last_tail is not None else "cpu"))
                            dprint("✅ [chatter_streamer] streaming EOS for chunk")
                            break

                except Exception as e:
                    dprint("[chatter_streamer] TTS stream error:", e)
                    # 에러 발생 시, 남은 버퍼가 있다면 마무리 패킷 전송 시도
                    try:
                        await emit_frames_from(torch.empty(0))
                    except Exception:
                        pass
                    continue

                lprint(f"[chatter_streamer] for {last_length/sr:.2f}s "
                       f"→ {time.time()-t0:.2f}s taken")

        await consume_loop()

    except asyncio.CancelledError:
        dprint("[chatter_streamer] CANCELLED")
        raise
    except Exception as e:
        dprint("[chatter_streamer] ERROR:", e)
        await sess.out_q.put(jdumps({"type": "tts_error", "message": str(e)}))
    finally:
        dprint("[chatter_streamer] END")
