# server.py
import asyncio
import json
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    # 간단한 핑 루프(옵션)
    ping_task = asyncio.create_task(ping(ws))

    try:
        while True:
            msg = await ws.receive()  # text 또는 bytes 모두 수신 가능
            if "text" in msg and msg["text"] is not None:
                data = json.loads(msg["text"])
                # 예: {"type":"input_audio_buffer.append","audio":"base64"}

                if data.get("type") == "hello":
                    await ws.send_text(json.dumps({
                        "type": "hello_ack",
                        "payload": {"server_time": datetime.utcnow().isoformat()}
                    }))
                elif data.get("type") == "text":
                    txt = data["payload"]["text"]
                    await ws.send_text(json.dumps({
                        "type": "echo",
                        "payload": {"text": f"server echo: {txt}"}
                    }))
                elif data.get("type") == "input_audio_buffer.append":
                    audio = data["audio"]
                    await ws.send_text(json.dumps({
                        "type": "script",
                        "delta": "wwsssd"
                    }))
                elif data.get("type") == "session.close":
                    await ws.send_text(json.dumps({
                        "type": "session.close",
                        "payload": {"status": "closed"}
                    }))
            
            elif "bytes" in msg and msg["bytes"] is not None:
                buf: bytes = msg["bytes"]
                # 여기서 PCM/Opus 등 처리 가능. 일단 길이만 회신
                await ws.send_text(json.dumps({
                    "type": "binary_ack",
                    "payload": {"received_bytes": len(buf)}
                }))
    except WebSocketDisconnect:
        pass
    finally:
        ping_task.cancel()


async def ping(ws: WebSocket):
    while True:
        try:
            await ws.send_text(json.dumps({"type": "ping"}))
            await asyncio.sleep(5)  # 15초마다 핑
        except Exception:
            break
