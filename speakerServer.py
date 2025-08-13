import asyncio
import json
from datetime import datetime
from typing import Dict, Optional
from llm.openai import translate
from stt.openai import open_openai_ws
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed
import orjson as json  # loads/dumps 호환 아님
import os
import websockets
import base64
import re
import time

def jdumps(o): return json.dumps(o).decode()  # bytes -> str

VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "wj5ree7FcgKDPFphpPWQ")

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

        # variables for logging
        self.start_scripting_time = 0
        self.end_scripting_time = 0
        self.end_translation_time = 0
        self.end_tts_time = 0

        # time logging
        self.connection_start_time = 0
        self.llm_cached_token_count = 0
        self.llm_input_token_count = 0
        self.llm_output_token_count = 0
        self.stt_output_token_count = 0

sessions: Dict[int, Session] = {}  # id(ws)로 매핑

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

                # 1) 세션 시작: OpenAI Realtime WS 연결
                if t == "scriptsession.start":
                    if sess.oai_ws is None:
                        sess.oai_ws = await open_openai_ws()
                        sess.connection_start_time = time.time()
                        model_name = data.get("model") or 'gpt-4o-mini-transcribe'
                        
                        initMsg = {
                            "type": 'transcription_session.update',
                            "session": {
                                "input_audio_format": 'pcm16',
                                "input_audio_transcription": {
                                    "model": model_name,
                                    "prompt": '',
                                    "language": (data.get("language") or "en")[:2],
                                },
                                "turn_detection": {
                                    "type": 'server_vad',
                                    "threshold": 0.4,
                                    "prefix_padding_ms": 200,
                                    "silence_duration_ms": 150,
                                },
                                "input_audio_noise_reduction": { "type": 'far_field' },
                            },
                        };
                        # 연결 직후 OpenAI 세션에 초기 메시지 전송 for setting up session
                        await sess.oai_ws.send(jdumps(initMsg))

                        # OpenAI 이벤트를 client로 릴레이하는 백그라운드 태스크
                        sess.oai_task = asyncio.create_task(relay_openai_to_client(sess, ws))
                        await ws.send_text(jdumps({"type": "scriptsession.started"}))
                    else:
                        await ws.send_text(jdumps({"type": "warn", "message": "already started"}))

                # 2) 오디오 append → Open AI로 그대로 전달
                elif t == "input_audio_buffer.append":
                    if not sess.oai_ws:
                        await ws.send_text(jdumps({"type": "error", "message": "session not started"}))
                        continue
                    
                    # 현재는 base64가 아닌 PCM이 온다.
                    if data.get("audio") and 'data' in data.get("audio"):
                        b64 = base64.b64encode(bytes(data.get("audio")['data'])).decode('ascii')
                        await sess.oai_ws.send(jdumps({
                            "type": "input_audio_buffer.append",
                            "audio": b64
                        }))

                # 3) 커밋 신호 전달 (chunk 경계)
                elif t == "input_audio_buffer.commit":
                    if not sess.oai_ws:
                        await ws.send_text(jdumps({"type": "error", "message": "session not started"}))
                        continue
                    await sess.oai_ws.send(jdumps({"type": "input_audio_buffer.commit"}))

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

# 3) relay_openai_to_client 수정본
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
                lprint("translation time : ", sess.end_translation_time - sess.end_scripting_time)

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

                    sess.transcripts.append(translated_text)
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

        dprint("[flush_tts_chunk] ", repr(chunk))
        sess.tts_buf.clear()
        try:
            # sess.tts_in_q.put_nowait(chunk) # 마지막 청크
            chunks = chunk.split(" ")
            for i in range(0, len(chunks), 3):
                if i >= len(chunks) - 4:
                    sess.tts_in_q.put_nowait(" ".join(chunks[i:])) # 마지막 청크
                    break
                else:
                    sess.tts_in_q.put_nowait(" ".join(chunks[i:i+3])) # 기본적으로는 단어를 3개씩 끊어서 보내기
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
        def _append_and_schedule():
            sess.tts_buf.append(tok)
            if sess.tts_debounce_task and not sess.tts_debounce_task.done():
                sess.tts_debounce_task.cancel()
            sess.tts_debounce_task = asyncio.create_task(debounce_flush(110))
        loop.call_soon_threadsafe(_append_and_schedule)

    def run_blocking():
        # 동기 translate 호출
        return translate(
            prevScripts=sess.transcripts[-5:],
            current_scripted_sentence=sess.current_transcript,
            current_translated=sess.current_translated,
            onToken=on_token,
        )

    # 동기 작업을 thread로
    loop = asyncio.get_running_loop()
    output = await loop.run_in_executor(None, run_blocking)
    dprint("llm output : ", output)
    final_text = output["text"]
    dprint("final_text : ", final_text)
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

# 필요 import
import contextlib
