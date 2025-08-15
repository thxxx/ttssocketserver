import asyncio, json, base64, websockets
import os

VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "wj5ree7FcgKDPFphpPWQ")
DEFAULT_MODEL_ID = os.environ.get("ELEVEN_MODEL_ID", "eleven_flash_v2_5")
DEFAULT_FORMAT   = os.environ.get("ELEVEN_OUT_FMT", "mp3_22050_32")

API_KEY = os.environ.get("ELEVENLABS_API_KEY")
URI = f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input?output_format={DEFAULT_FORMAT}"

async def main():
    async with websockets.connect(
        URI,
        extra_headers={"xi-api-key": API_KEY}
    ) as elws:
        # 1) 초기 설정 전송
        await elws.send(json.dumps({
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8, "speed": 1.0},
            "xi_api_key": API_KEY
        }))

        # 2) 단어(또는 짧은 구절) 단위로 입력
        words = ["Hello", " ", "world", "! "]
        for w in words:
            await elws.send(json.dumps({
                "text": w,
                "try_trigger_generation": True  # 바로 합성 시작
            }))

        # 3) (선택) 입력 종료 알림
        await elws.send(json.dumps({"text": ""}))

        # 4) 오디오 수신 루프
        while True:
            msg = await elws.recv()
            data = json.loads(msg)
            if "audio" in data:
                chunk = base64.b64decode(data["audio"])  # mp3 바이트
                # 여기서 MediaSource/SourceBuffer 또는 소켓으로 바로 플레이/전달
            if data.get("isFinal") is True:
                break

asyncio.run(main())
