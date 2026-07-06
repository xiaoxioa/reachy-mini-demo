# -*- coding: utf-8 -*-
"""CascadedPipeline — 半级联语音管线：本地 VAD+ASR + 云端 LLM+TTS。"""

from __future__ import annotations

import queue
import threading
import time
from typing import Iterator

import numpy as np

from voice.pipeline.base import VoicePipeline
from voice.pipeline.events import EventType, PipelineEvent
from voice.pipeline.providers import (
    VADProvider, ASRProvider, LLMProvider, TTSProvider, LLMChunk,
)
from voice.state import log


class CascadedPipeline(VoicePipeline):
    """半级联后端：本地 VAD+ASR → 云端 LLM → 云端 TTS。

    feed_audio() 同步喂 VAD（<1ms），语音段结束后在独立线程
    执行 ASR → LLM(streaming) → TTS(streaming) 链。
    """

    def __init__(
        self,
        vad: VADProvider,
        asr: ASRProvider,
        llm: LLMProvider,
        tts: TTSProvider,
        voice: str = "default",
    ) -> None:
        super().__init__()
        self._vad = vad
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._voice = voice

        self._connected = False
        self._instructions: str = ""
        self._tools: list = []
        self._messages: list[dict] = []

        self._responding = False
        self._cancelled = threading.Event()
        self._tool_result_q: queue.Queue[tuple[str, str]] = queue.Queue()
        self._resp_thread: threading.Thread | None = None

    # ── 会话生命周期 ──

    def open_session(
        self, instructions: str, tools: list, timeout: float = 10.0
    ) -> bool:
        self._instructions = instructions
        self._tools = tools
        self._messages = [{"role": "system", "content": instructions}]
        self._connected = True
        self._vad.reset()
        self._emit(PipelineEvent(type=EventType.SESSION_READY))
        log("✅ CascadedPipeline session ready")
        return True

    def close_session(self) -> None:
        self._connected = False
        self.cancel_response()
        if self._resp_thread and self._resp_thread.is_alive():
            self._resp_thread.join(timeout=3.0)
        self._vad.reset()

    # ── 音频输入 ──

    def feed_audio(self, pcm_16k_mono: np.ndarray) -> None:
        if not self._connected:
            return
        events = self._vad.feed(pcm_16k_mono)
        for ev in events:
            if ev.type == "speech_start":
                if self._responding:
                    self.cancel_response()
                self._emit(PipelineEvent(type=EventType.USER_SPEECH_START))
            elif ev.type == "speech_end":
                self._emit(PipelineEvent(type=EventType.USER_SPEECH_END))
                if ev.audio is not None and ev.audio.size > 0:
                    self._start_response(speech_audio=ev.audio)

    # ── 工具 ──

    def submit_tool_result(self, call_id: str, output: str) -> None:
        self._tool_result_q.put((call_id, output))

    # ── 响应控制 ──

    def cancel_response(self) -> None:
        self._cancelled.set()
        self._llm.cancel()
        self._tts.cancel()

    def trigger_response(self, instructions: str | None = None) -> None:
        self._start_response(instructions_override=instructions)

    # ── 指令 / 记忆 ──

    def update_instructions(
        self, instructions: str, tools: list | None = None
    ) -> None:
        self._instructions = instructions
        if tools is not None:
            self._tools = tools
        if self._messages and self._messages[0]["role"] == "system":
            self._messages[0]["content"] = instructions

    def inject_system_message(self, text: str) -> None:
        self._messages.append({"role": "system", "content": text})
        self._start_response()

    # ── 状态 ──

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── 内部 ──

    def _start_response(
        self,
        speech_audio: np.ndarray | None = None,
        instructions_override: str | None = None,
    ) -> None:
        if self._resp_thread and self._resp_thread.is_alive():
            self.cancel_response()
            self._resp_thread.join(timeout=2.0)
        self._cancelled.clear()
        self._responding = True
        self._resp_thread = threading.Thread(
            target=self._response_loop,
            args=(speech_audio, instructions_override),
            daemon=True,
        )
        self._resp_thread.start()

    def _response_loop(
        self,
        speech_audio: np.ndarray | None,
        instructions_override: str | None,
    ) -> None:
        try:
            self._do_response(speech_audio, instructions_override)
        except Exception as e:
            log(f"❌ CascadedPipeline response error: {type(e).__name__}: {e}")
            self._emit(PipelineEvent(
                type=EventType.ERROR,
                data={"message": str(e)},
            ))
        finally:
            self._responding = False

    def _do_response(
        self,
        speech_audio: np.ndarray | None,
        instructions_override: str | None,
    ) -> None:
        user_text: str | None = None

        # ASR
        if speech_audio is not None:
            user_text = self._asr.transcribe(speech_audio)
            if not user_text:
                log("⚠ ASR 返回空文本，跳过响应")
                return
            self._emit(PipelineEvent(
                type=EventType.USER_TRANSCRIPT,
                data={"text": user_text, "final": True},
            ))
            self._messages.append({"role": "user", "content": user_text})

        if instructions_override:
            self._messages.append(
                {"role": "system", "content": instructions_override}
            )

        self._emit(PipelineEvent(type=EventType.RESPONSE_START))

        # LLM → TTS loop (may iterate for tool calls)
        full_text = self._run_llm_turn()

        # TTS
        if full_text and not self._cancelled.is_set():
            self._run_tts(full_text)

        self._emit(PipelineEvent(type=EventType.RESPONSE_DONE))

    def _run_llm_turn(self) -> str:
        """Run LLM, handle tool calls in a loop, return final assistant text."""
        while not self._cancelled.is_set():
            full_text, tool_calls = self._stream_llm()
            if not tool_calls:
                return full_text

            # Handle tool calls then re-enter LLM
            for tc in tool_calls:
                self._emit(PipelineEvent(
                    type=EventType.TOOL_CALL,
                    data={
                        "name": tc["name"],
                        "call_id": tc["id"],
                        "arguments": tc["args"],
                    },
                ))

            # Wait for all tool results
            for tc in tool_calls:
                try:
                    call_id, result = self._tool_result_q.get(timeout=30)
                except queue.Empty:
                    log(f"⚠ tool result 超时: {tc['name']}")
                    call_id, result = tc["id"], '{"error": "timeout"}'
                self._messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["args"],
                        },
                    }],
                })
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                })

        return ""

    def _stream_llm(self) -> tuple[str, list[dict]]:
        """Single LLM streaming call. Returns (full_text, tool_calls)."""
        full_text = ""
        tool_calls: list[dict] = []

        for chunk in self._llm.chat_stream(self._messages, self._tools):
            if self._cancelled.is_set():
                break
            if chunk.type == "text_delta":
                full_text += chunk.text or ""
                self._emit(PipelineEvent(
                    type=EventType.RESPONSE_TEXT_DELTA,
                    data={"delta": chunk.text},
                ))
            elif chunk.type == "tool_call":
                tool_calls.append({
                    "id": chunk.tool_call_id or "",
                    "name": chunk.tool_name or "",
                    "args": chunk.tool_arguments or "",
                })
            elif chunk.type == "text_done":
                if full_text:
                    self._messages.append(
                        {"role": "assistant", "content": full_text}
                    )
                    self._emit(PipelineEvent(
                        type=EventType.RESPONSE_TEXT_DONE,
                        data={"text": full_text},
                    ))

        return full_text, tool_calls

    def _run_tts(self, text: str) -> None:
        """Stream TTS audio, emit AUDIO_DELTA events."""
        for pcm_int16 in self._tts.synthesize_stream(text, self._voice):
            if self._cancelled.is_set():
                break
            pcm_f32 = pcm_int16.astype(np.float32) / 32768.0
            self._emit(PipelineEvent(
                type=EventType.RESPONSE_AUDIO_DELTA,
                data={"pcm_16k": pcm_f32},
            ))
