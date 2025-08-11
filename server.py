import asyncio
import json
from datetime import datetime
from typing import Dict, Optional
from llm.openai import translate
from stt.openai import open_openai_ws
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed
import orjson as json  # loads/dumps 호환 아님

def jdumps(o): return json.dumps(o).decode()  # bytes -> str

app = FastAPI()

DEBUG = False
def dprint(*a, **k): 
    if DEBUG: print(*a, **k)

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
                        initMsg = {
                        "type": 'transcription_session.update',
                        "session": {
                            "input_audio_format": 'pcm16',
                            "input_audio_transcription": {
                                "model": 'gpt-4o-mini-transcribe',
                                "prompt": '',
                                "language": (data.get("language") or "en")[:2],
                            },
                            "turn_detection": {
                                "type": 'server_vad',
                                "threshold": 0.4,
                                "prefix_padding_ms": 200,
                                "silence_duration_ms": 100,
                            },
                            "input_audio_noise_reduction": { "type": 'far_field' },
                        },
                        };

                        # 연결 직후 OpenAI 세션에 초기 메시지 전송 for setting up session
                        await sess.oai_ws.send(jdumps(initMsg))

                        # OpenAI 이벤트를 클라로 릴레이하는 백그라운드 태스크
                        sess.oai_task = asyncio.create_task(relay_openai_to_client(sess, ws))
                        await ws.send_text(jdumps({"type": "scriptsession.started"}))
                    else:
                        await ws.send_text(jdumps({"type": "warn", "message": "already started"}))

                # 2) 오디오 append → OAI로 그대로 전달
                elif t == "input_audio_buffer.append":
                    if not sess.oai_ws:
                        await ws.send_text(jdumps({"type": "error", "message": "session not started"}))
                        continue
                    # data = { type, audio (base64) }
                    await sess.oai_ws.send(jdumps({
                        "type": "input_audio_buffer.append",
                        "audio": data.get("audio")
                    }))

                # 3) 커밋 신호 전달 (chunk 경계)
                elif t == "input_audio_buffer.commit":
                    if not sess.oai_ws:
                        await ws.send_text(jdumps({"type": "error", "message": "session not started"}))
                        continue
                    await sess.oai_ws.send(jdumps({"type": "input_audio_buffer.commit"}))

                elif t == "text":
                    txt = data["payload"]["text"]
                    await ws.send_text(jdumps({
                        "type": "echo",
                        "payload": {"text": f"server echo: {txt}"}
                    }))

                elif t == "session.close":
                    await ws.send_text(jdumps({
                        "type": "session.close",
                        "payload": {"status": "closed"}
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
                await sess.out_q.put(jdumps({"type": "delta", "text": text}))
            elif etype.endswith(".completed"):
                # 3-1) 최종 전사 수신
                final_text = (evt.get("transcript") or evt.get("content") or "").strip()
                await sess.out_q.put(jdumps({
                    "type": "transcript", "text": final_text, "final": True
                }))

                # 3-2) 연결별 누적 전사 업데이트
                # 공백 관리: 앞에 뭔가 있으면 한 칸 띄우고 붙이기
                if sess.current_transcript:
                    sess.current_transcript += " " + final_text
                else:
                    sess.current_transcript = final_text
                
                if len(sess.current_transcript) < 5:
                    continue
                
                print("prevScripts", sess.transcripts[-5:])
                print("current_scripted_sentence", sess.current_transcript)
                print("current_translated", sess.current_translated)

                # --- 여기부터 교체 ---
                # 동기 translate를 스레드에서 비동기처럼 실행(스트리밍 콜백 유지)
                translated_text = await run_translate_async(sess)
                # --- 교체 끝 ---

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
                    def strip_end(s: str) -> str:
                        return s.replace("<END>", "").strip()

                    sess.transcripts.append(sess.current_transcript.strip())
                    sess.translateds.append(strip_end(sess.current_translated))

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
    if sess.oai_task:
        sess.oai_task.cancel()
        with contextlib.suppress(Exception):
            await sess.oai_task
    if sess.oai_ws:
        try:
            await sess.oai_ws.close()
        except Exception:
            pass

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

    # 콜백: 스레드에서 실행됨. 이벤트 루프에 안전하게 put 예약
    def on_token(tok: str):
        # (A) 아주 가벼운 브리지: put_nowait를 스레드세이프로 예약
        loop.call_soon_threadsafe(
            sess.out_q.put_nowait,
            jdumps({"type": "translated_delta", "text": tok})
        )

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
    final_text = await loop.run_in_executor(None, run_blocking)
    # final_text = await asyncio.to_thread(run_blocking)

    # 최종 결과 알림
    await sess.out_q.put(jdumps({"type": "translated", "text": final_text}))
    return final_text


# 필요 import
import contextlib
