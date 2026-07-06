# -*- coding: utf-8 -*-
"""级联管线组件 Provider 抽象基类。

本地 VAD / ASR（本地或 API）/ LLM API / TTS API — 每个可独立替换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np


# ── VAD ──

@dataclass
class VADEvent:
    type: str  # "speech_start" | "speech_end"
    audio: np.ndarray | None = None  # speech_end 时附带完整语音段


class VADProvider(ABC):
    """本地 VAD — 必须逐帧实时，CPU 即可。"""

    @abstractmethod
    def feed(self, pcm_16k_mono: np.ndarray) -> list[VADEvent]:
        """喂入音频帧，返回 0~N 个事件。"""

    @abstractmethod
    def reset(self) -> None:
        """重置内部状态（新会话时调用）。"""


# ── ASR ──

class ASRProvider(ABC):
    """ASR — 本地模型或云端 API，统一接口。"""

    @abstractmethod
    def transcribe(self, pcm_16k_mono: np.ndarray) -> str:
        """整段识别（VAD 切出的完整语句）。"""

    def transcribe_streaming(self, audio_iter: Iterator[np.ndarray]) -> Iterator[str]:
        """流式识别（可选，默认 fallback 到攒完整段再识别）。"""
        import itertools
        chunks = list(audio_iter)
        if not chunks:
            return
        full = np.concatenate(chunks)
        text = self.transcribe(full)
        if text:
            yield text


# ── LLM ──

@dataclass
class LLMChunk:
    type: str  # "text_delta" | "text_done" | "tool_call" | "done"
    text: str = ""
    tool_name: str = ""
    tool_call_id: str = ""
    tool_arguments: str = ""


class LLMProvider(ABC):
    """文本 LLM — 走 API，支持流式 + 工具调用。"""

    @abstractmethod
    def chat_stream(self, messages: list, tools: list | None = None,
                    tool_choice: str = "auto") -> Iterator[LLMChunk]:
        """流式生成，yield LLMChunk。"""

    @abstractmethod
    def cancel(self) -> None:
        """取消当前生成。"""


# ── TTS ──

class TTSProvider(ABC):
    """TTS — 流式返回 16kHz mono PCM16 音频块。"""

    @abstractmethod
    def synthesize_stream(self, text: str,
                          voice: str = "default") -> Iterator[np.ndarray]:
        """流式合成，yield 16kHz PCM16 音频块。"""

    @abstractmethod
    def cancel(self) -> None:
        """取消当前合成。"""
