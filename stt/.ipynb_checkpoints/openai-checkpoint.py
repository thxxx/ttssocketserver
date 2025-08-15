import os
from websockets.asyncio.client import connect as ws_connect  # pip install websockets

OPENAI_KEY = os.environ.get("OPENAI_KEY")

async def open_openai_ws():
    """
    OpenAI Realtime(Transcribe) WS 연결 생성.
    문서 기준:
    - URL: wss://api.openai.com/v1/realtime?intent=transcription
        후 initMsg에서 model 선택 후 전송
    - Headers:
        Authorization: Bearer <OPENAI_KEY>
        OpenAI-Beta: realtime=v1
    """
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_KEY not set")

    url = "wss://api.openai.com/v1/realtime?intent=transcription"
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }
    # websockets.connect 에서는 'additional_headers' 인자 사용
    return await ws_connect(url, additional_headers=headers)