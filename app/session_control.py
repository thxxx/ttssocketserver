import contextlib
from app.session import Session
import asyncio
from fastapi import WebSocket
import json

def jdumps(o): return json.dumps(o).decode()

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