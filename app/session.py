# app/ws/session.py
import asyncio, time, numpy as np
from typing import Optional, List
from collections import deque
from asyncio import Queue
import queue

class Session:
    translate_q: Queue[str]
    translator_task: Optional[asyncio.Task]
    running: bool = True
    
    def __init__(self, input_sr: int, input_channels: int):
        self.translate_q = asyncio.Queue(maxsize=2)   # 역압: 최신 2개까지만 허용
        self.translator_task = None
        
        # self.oai_ws = None
        # self.oai_task: Optional[asyncio.Task] = None
        self.running = True

        self.audio_buf = bytearray()
        self.audios = np.empty(0, dtype=np.float32)

        self.input_sample_rate = input_sr
        self.input_channels = input_channels

        self.current_transcript: str = ''
        self.transcripts: List[str] = []
        self.current_translated: str = ''
        self.translateds: List[str] = []

        self.out_q: asyncio.Queue[str] = asyncio.Queue()
        self.sender_task: Optional[asyncio.Task] = None

        self.tts_ws = None
        self.tts_task: Optional[asyncio.Task] = None
        self.tts_in_q: asyncio.Queue[str] = asyncio.Queue(maxsize=256)

        self.tts_buf: list[str] = []
        self.tts_debounce_task: Optional[asyncio.Task] = None
        self.buf_count = 0

        # 로깅/통계
        self.start_scripting_time = 0
        self.end_scripting_time = 0
        self.end_translation_time = 0
        self.first_translated_token_output_time = 0
        self.end_tts_time = 0
        self.end_audio_input_time = 0

        self.connection_start_time = 0
        self.llm_cached_token_count = 0
        self.llm_input_token_count = 0
        self.llm_output_token_count = 0
        self.stt_output_token_count = 0

        self.is_network_logging = False
        self.current_audio_state = "none"
        self.new_speech_start = 0

        self.transcript = ""
        self.translated = ""
        self.end_count = -1
        self.count_after_last_translation = 0
        self.pre_roll = deque(maxlen=3)

        self.is_use_filler = False

        self.ref_audios = queue.Queue()

