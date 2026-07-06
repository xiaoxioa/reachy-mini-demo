# -*- coding: utf-8 -*-
"""QwenCallbackAdapter — Qwen Omni Realtime 事件 → PipelineEvent 翻译器。"""

from __future__ import annotations

import base64

import numpy as np
from scipy.signal import resample_poly

from dashscope.audio.qwen_omni import OmniRealtimeCallback

from voice.pipeline.events import EventType, PipelineEvent
from voice.config import OUT_SR, PLAY_SR
from voice.state import log, _record_event


class QwenCallbackAdapter(OmniRealtimeCallback):
    """将 Qwen SDK 回调事件翻译为 PipelineEvent 并通过 pipeline._emit 分发。"""

    def __init__(self, pipeline) -> None:
        self._pipeline = pipeline

    def on_open(self) -> None:
        log("✅ WebSocket 已连接 dashscope.aliyuncs.com")

    def on_close(self, close_status_code, close_msg) -> None:
        log(f"🔌 连接关闭:code={close_status_code} msg={close_msg}")

    def on_event(self, event) -> None:
        try:
            etype = event.get("type", "")
            _record_event(etype, event)

            if etype == "session.created":
                log(f"✅ 会话已建立 "
                    f"session_id={event['session']['id']}")

            elif etype == "session.updated":
                self._pipeline._session_ready.set()
                self._pipeline._emit(
                    PipelineEvent(EventType.SESSION_READY))

            elif etype == "input_audio_buffer.speech_started":
                self._pipeline._emit(
                    PipelineEvent(EventType.USER_SPEECH_START))

            elif etype == "input_audio_buffer.speech_stopped":
                self._pipeline._emit(
                    PipelineEvent(EventType.USER_SPEECH_END))

            elif etype == ("conversation.item."
                           "input_audio_transcription.completed"):
                text = (event.get("transcript") or "").strip()
                self._pipeline._emit(PipelineEvent(
                    EventType.USER_TRANSCRIPT,
                    {"text": text, "is_final": True},
                ))

            elif etype == "response.created":
                self._pipeline._emit(
                    PipelineEvent(EventType.RESPONSE_START))

            elif etype == "response.function_call_arguments.done":
                self._pipeline._emit(PipelineEvent(
                    EventType.TOOL_CALL,
                    {
                        "name": event.get("name", ""),
                        "call_id": event.get("call_id", ""),
                        "arguments": event.get("arguments", "{}"),
                    },
                ))

            elif etype == "response.audio_transcript.delta":
                self._pipeline._emit(PipelineEvent(
                    EventType.RESPONSE_TEXT_DELTA,
                    {"delta": event.get("delta", "")},
                ))

            elif etype == "response.audio_transcript.done":
                self._pipeline._emit(PipelineEvent(
                    EventType.RESPONSE_TEXT_DONE,
                    {"text": event.get("transcript", "")},
                ))

            elif etype == "response.audio.delta":
                b64 = event.get("delta") or event.get("audio") or ""
                if not b64:
                    return
                pcm_24k = np.frombuffer(
                    base64.b64decode(b64), dtype=np.int16)
                pcm_16k = resample_poly(
                    pcm_24k.astype(np.float32) / 32768.0,
                    PLAY_SR, OUT_SR,
                ).astype(np.float32)
                self._pipeline._emit(PipelineEvent(
                    EventType.RESPONSE_AUDIO_DELTA,
                    {"pcm_16k": pcm_16k},
                ))

            elif etype == "response.done":
                delay = None
                c = self._pipeline._conv
                if c is not None:
                    try:
                        delay = c.get_last_first_audio_delay()
                    except Exception:
                        pass
                self._pipeline._emit(PipelineEvent(
                    EventType.RESPONSE_DONE,
                    {"first_audio_delay": delay},
                ))

            elif etype == "error":
                self._pipeline._emit(PipelineEvent(
                    EventType.ERROR,
                    {"message": str(event)},
                ))

        except Exception as e:
            log(f"❌ QwenCallbackAdapter.on_event 异常:"
                f"{type(e).__name__}: {e}")
