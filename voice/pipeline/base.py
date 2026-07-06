# -*- coding: utf-8 -*-
"""VoicePipeline 抽象基类 — 端到端/级联后端均实现此协议。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from voice.pipeline.events import PipelineEvent


class VoicePipeline(ABC):

    def __init__(self) -> None:
        self._event_handler: Callable[[PipelineEvent], None] | None = None

    def set_event_handler(self, handler: Callable[[PipelineEvent], None]) -> None:
        self._event_handler = handler

    def _emit(self, event: PipelineEvent) -> None:
        if self._event_handler is not None:
            self._event_handler(event)

    # ── 会话生命周期 ──

    @abstractmethod
    def open_session(self, instructions: str, tools: list,
                     timeout: float = 10.0) -> bool:
        """建立会话，成功返回 True。"""

    @abstractmethod
    def close_session(self) -> None:
        """关闭连接。"""

    # ── 音频输入 ──

    @abstractmethod
    def feed_audio(self, pcm_16k_mono: np.ndarray) -> None:
        """喂入 16kHz mono PCM16 音频帧。"""

    # ── 工具 ──

    @abstractmethod
    def submit_tool_result(self, call_id: str, output: str) -> None:
        """回传工具执行结果。"""

    # ── 响应控制 ──

    @abstractmethod
    def cancel_response(self) -> None:
        """取消当前响应（barge-in）。"""

    @abstractmethod
    def trigger_response(self, instructions: str | None = None) -> None:
        """主动触发模型回复（唤醒招呼、工具后补话）。"""

    # ── 指令 / 记忆 ──

    @abstractmethod
    def update_instructions(self, instructions: str,
                            tools: list | None = None) -> None:
        """更新会话指令（记忆注入）。"""

    @abstractmethod
    def inject_system_message(self, text: str) -> None:
        """注入系统消息并触发回复（安全删除工作流等）。"""

    # ── 状态 ──

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """当前是否已连接。"""
