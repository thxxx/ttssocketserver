import os
import asyncio
import json
from datetime import datetime
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.asyncio.client import connect as ws_connect  # pip install websockets
from websockets.exceptions import ConnectionClosed

OPENAI_KEY = os.environ.get("OPENAI_KEY")

app = FastAPI()

# 클라이언트 1명당 OpenAI Realtime WS를 하나씩 유지하기 위한 세션 상태
class Session:
    def __init__(self):
        self.oai_ws = None  # OpenAI WS 연결
        self.oai_task: Optional[asyncio.Task] = None  # OAI -> Client listener task
        self.running = True

sessions: Dict[int, Session] = {}  # id(ws)로 매핑

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    sess = Session()
    sessions[id(ws)] = sess

    try:
        while True:
            msg = await ws.receive()
            if msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await ws.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                    continue

                t = data.get("type")

                # 1) 세션 시작: OpenAI Realtime WS 연결
                if t == "scriptsession.start":
                    if sess.oai_ws is None:
                        sess.oai_ws = await open_openai_ws()
                        # OpenAI 이벤트를 클라로 릴레이하는 백그라운드 태스크
                        sess.oai_task = asyncio.create_task(relay_openai_to_client(sess, ws))
                        await ws.send_text(json.dumps({"type": "scriptsession.started"}))
                    else:
                        await ws.send_text(json.dumps({"type": "warn", "message": "already started"}))

                # 2) 오디오 append → OAI로 그대로 전달
                elif t == "input_audio_buffer.append":
                    if not sess.oai_ws:
                        await ws.send_text(json.dumps({"type": "error", "message": "session not started"}))
                        continue
                    # data = { type, audio (base64) }
                    await sess.oai_ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": data.get("audio")
                    }))

                # 3) 커밋 신호 전달 (chunk 경계)
                elif t == "input_audio_buffer.commit":
                    if not sess.oai_ws:
                        await ws.send_text(json.dumps({"type": "error", "message": "session not started"}))
                        continue
                    await sess.oai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

                elif t == "text":
                    txt = data["payload"]["text"]
                    await ws.send_text(json.dumps({
                        "type": "echo",
                        "payload": {"text": f"server echo: {txt}"}
                    }))

                elif t == "session.close":
                    await ws.send_text(json.dumps({
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
                await ws.send_text(json.dumps({
                    "type": "binary_ack",
                    "payload": {"received_bytes": len(buf)}
                }))

    except WebSocketDisconnect:
        pass
    finally:
        await teardown_session(sess)
        sessions.pop(id(ws), None)

async def open_openai_ws():
    """
    OpenAI Realtime(Transcribe) WS 연결 생성.
    문서 기준:
    - URL: wss://api.openai.com/v1/realtime?model=gpt-4o-mini-transcribe
      (또는 transcription intent를 쓰는 변형)
    - Headers:
        Authorization: Bearer <OPENAI_KEY>
        OpenAI-Beta: realtime=v1
    """
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_KEY not set")

    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-transcribe"
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }
    # websockets.connect 에서는 'additional_headers' 인자 사용
    return await ws_connect(url, additional_headers=headers)


async def relay_openai_to_client(sess: Session, client_ws: WebSocket):
    """
    OpenAI → 서버 → 클라이언트 릴레이.
    OpenAI가 보내는 이벤트 타입 예:
      - transcript.delta / transcript.completed (모델/모드에 따라)
      - response.delta / response.completed (응답 스트림 계열)
      - error
    들어오는 raw JSON을 그대로 넘기거나, 필요한 최소만 매핑해서 넘긴다.
    """
    try:
        async for raw in sess.oai_ws:
            # OpenAI 쪽은 문자열 프레임(JSON)로 온다
            try:
                evt = json.loads(raw)
            except Exception:
                # 파싱 실패 시 원문 그대로 전달
                await client_ws.send_text(raw)
                continue

            # 최소 매핑: delta 텍스트만 빠르게 중계
            etype = evt.get("type", "")
            # 전사 스트림 계열 가정
            if etype.endswith(".delta"):
                # 가능한 필드 이름 케이스를 호환
                # 예: { type: "transcript.delta", delta: "..." } or { ... "text": "..." }
                text = evt.get("delta") or evt.get("text") or evt.get("content") or ""
                await client_ws.send_text(json.dumps({"type": "delta", "text": text}))
            elif etype.endswith(".completed"):
                final_text = evt.get("text") or evt.get("content") or ""
                await client_ws.send_text(json.dumps({"type": "transcript", "text": final_text, "final": True}))
            elif etype == "error":
                await client_ws.send_text(json.dumps({"type": "error", "message": evt.get("error", evt)}))
            else:
                # 디버그용: 필요한 경우 전체 이벤트 전달
                await client_ws.send_text(json.dumps({"type": "oai_event", "event": evt}))

    except ConnectionClosed:
        pass
    except Exception as e:
        await client_ws.send_text(json.dumps({"type": "error", "message": f"OAI relay error: {e}"}))


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

# 필요 import
import contextlib
