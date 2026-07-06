# -*- coding: utf-8 -*-
"""管线事件类型与事件数据类。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(Enum):
    SESSION_READY = "session.ready"
    USER_SPEECH_START = "user.speech_start"
    USER_SPEECH_END = "user.speech_end"
    USER_TRANSCRIPT = "user.transcript"
    RESPONSE_START = "response.start"
    RESPONSE_TEXT_DELTA = "response.text_delta"
    RESPONSE_TEXT_DONE = "response.text_done"
    RESPONSE_AUDIO_DELTA = "response.audio_delta"
    RESPONSE_DONE = "response.done"
    TOOL_CALL = "tool.call"
    ERROR = "error"


@dataclass
class PipelineEvent:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
