# -*- coding: utf-8 -*-
"""QwenRealtimePipeline — 封装 Qwen-Omni-Realtime SDK 为 VoicePipeline 接口。"""

from __future__ import annotations

import base64
import threading
import time

import numpy as np

from dashscope.audio.qwen_omni import (
    OmniRealtimeConversation, AudioFormat, MultiModality,
)

from voice.pipeline.base import VoicePipeline
from voice.pipeline.events import EventType, PipelineEvent
from voice.qwen_realtime.adapter import QwenCallbackAdapter
from voice.config import MODEL, VOICE, CONNECT_TIMEOUT_S
from voice.state import log


class QwenRealtimePipeline(VoicePipeline):
    """Qwen-Omni-Realtime 端到端后端 — 通过 WebSocket 连接 DashScope。"""

    def __init__(self, model: str = MODEL, voice: str = VOICE) -> None:
        super().__init__()
        self._model = model
        self._voice = voice
        self._conv: OmniRealtimeConversation | None = None
        self._adapter = QwenCallbackAdapter(self)
        self._session_ready = threading.Event()
        self._tools: list = []
        self._last_connect_at = 0.0
        self._min_connect_gap = 1.0

    # ── 会话生命周期 ──

    def open_session(self, instructions: str, tools: list,
                     timeout: float = CONNECT_TIMEOUT_S) -> bool:
        gap = time.monotonic() - self._last_connect_at
        if gap < self._min_connect_gap:
            time.sleep(self._min_connect_gap - gap)
        self._last_connect_at = time.monotonic()
        self._tools = tools
        self._session_ready.clear()

        c = OmniRealtimeConversation(
            model=self._model, callback=self._adapter)
        holder: dict = {"err": None}

        def _connect():
            try:
                c.connect()
                log(f"📤 update_session(初始化) instructions前80字: "
                    f"{instructions[:80]}... tools数: {len(tools)}")
                c.update_session(
                    output_modalities=[
                        MultiModality.AUDIO, MultiModality.TEXT],
                    voice=self._voice,
                    input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
                    output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                    enable_input_audio_transcription=True,
                    enable_turn_detection=True,
                    turn_detection_type="semantic_vad",
                    instructions=instructions,
                    tools=tools,
                )
            except Exception as e:
                holder["err"] = e

        threading.Thread(target=_connect, daemon=True).start()

        if self._session_ready.wait(timeout):
            self._conv = c
            return True

        log(f"⚠ 连接失败/超时(>{timeout:.1f}s) err={holder['err']}")
        try:
            c.close()
        except Exception:
            pass
        return False

    def close_session(self) -> None:
        c = self._conv
        if c is not None:
            try:
                c.close()
            except Exception:
                pass
        self._conv = None

    # ── 音频输入 ──

    def feed_audio(self, pcm_16k_mono: np.ndarray) -> None:
        c = self._conv
        if c is None:
            return
        b64 = base64.b64encode(pcm_16k_mono.tobytes()).decode()
        c.append_audio(b64)

    # ── 工具 ──

    def submit_tool_result(self, call_id: str, output: str) -> None:
        c = self._conv
        if c is None:
            return
        c.create_item({
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        })

    # ── 响应控制 ──

    def cancel_response(self) -> None:
        c = self._conv
        if c is not None:
            c.cancel_response()

    def trigger_response(self, instructions: str | None = None) -> None:
        c = self._conv
        if c is None:
            return
        if instructions:
            c.create_response(instructions=instructions)
        else:
            c.create_response()

    # ── 指令 / 记忆 ──

    def update_instructions(self, instructions: str,
                            tools: list | None = None) -> None:
        c = self._conv
        if c is None:
            raise RuntimeError("Not connected")
        c.update_session(
            output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
            voice=self._voice,
            input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
            output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            enable_input_audio_transcription=True,
            enable_turn_detection=True,
            turn_detection_type="semantic_vad",
            instructions=instructions,
            tools=tools or self._tools,
        )

    def inject_system_message(self, text: str) -> None:
        c = self._conv
        if c is None:
            raise RuntimeError("Not connected")
        c.create_item({
            "type": "message", "role": "system",
            "content": [{"type": "input_text", "text": text}],
        })
        c.create_response()

    # ── 状态 ──

    @property
    def is_connected(self) -> bool:
        return self._conv is not None
