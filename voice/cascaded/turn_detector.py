# -*- coding: utf-8 -*-
"""TurnDetector — 端点检测 + 打断判定。

替代 Qwen-Omni 内置的语义 VAD，用规则引擎实现:
- 端点检测: 静音时长 + 句末标点
- 打断判定: TTS 播放中用户开口 + 区分 backchannel("嗯") vs 真打断
"""

from __future__ import annotations

import re
from typing import Literal


_SENTENCE_ENDERS = re.compile(r"[。！？!?；;]$")
_BACKCHANNEL = re.compile(
    r"^(嗯|哦|啊|哈|嗯嗯|哦哦|嗯哼|好的?|对|是|行|ok|okay)$",
    re.IGNORECASE,
)

Decision = Literal["respond", "barge_in", "backchannel", "wait"]


class TurnDetector:
    """规则引擎式端点检测 + 打断判定。

    参数均可通过环境变量或构造函数调整。
    """

    def __init__(
        self,
        silence_respond_ms: int = 800,
        silence_short_ms: int = 400,
        barge_in_min_chars: int = 4,
        barge_in_max_silence_ms: int = 50,
    ) -> None:
        self.silence_respond_ms = silence_respond_ms
        self.silence_short_ms = silence_short_ms
        self.barge_in_min_chars = barge_in_min_chars
        self.barge_in_max_silence_ms = barge_in_max_silence_ms

    def should_respond(
        self,
        partial_text: str,
        silence_ms: int,
        is_tts_playing: bool,
    ) -> Decision:
        text = partial_text.strip()

        if is_tts_playing:
            if silence_ms < self.barge_in_max_silence_ms:
                if len(text) <= self.barge_in_min_chars and _BACKCHANNEL.match(text):
                    return "backchannel"
                if len(text) >= self.barge_in_min_chars:
                    return "barge_in"
                return "wait"
            if silence_ms >= self.silence_respond_ms:
                return "respond"
            return "wait"

        if silence_ms >= self.silence_respond_ms:
            return "respond"

        if (silence_ms >= self.silence_short_ms
                and text
                and _SENTENCE_ENDERS.search(text)):
            return "respond"

        return "wait"
