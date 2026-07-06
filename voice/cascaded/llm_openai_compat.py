# -*- coding: utf-8 -*-
"""OpenAICompatLLM — OpenAI 兼容 API 流式 LLM，实现 LLMProvider 接口。"""

from __future__ import annotations

import threading
from typing import Iterator

from openai import OpenAI

from voice.pipeline.providers import LLMProvider, LLMChunk
from voice.state import log


class OpenAICompatLLM(LLMProvider):
    """基于 OpenAI 兼容 API（DashScope compatible-mode）的流式 LLM。

    chat_stream() 流式返回 LLMChunk，支持 text_delta + tool_call 累积。
    """

    def __init__(self, client: OpenAI, model: str = "qwen-plus") -> None:
        self._client = client
        self._model = model
        self._cancel_flag = threading.Event()

    def chat_stream(
        self,
        messages: list,
        tools: list | None = None,
        tool_choice: str = "auto",
    ) -> Iterator[LLMChunk]:
        self._cancel_flag.clear()

        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": t} for t in tools
            ]

        try:
            comp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            log(f"❌ LLM chat_stream 失败: {type(e).__name__}: {e}")
            yield LLMChunk(type="done")
            return

        full_text = ""
        pending_tcs: dict[int, dict] = {}

        for chunk in comp:
            if self._cancel_flag.is_set():
                break
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue

            if delta.content:
                full_text += delta.content
                yield LLMChunk(type="text_delta", text=delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in pending_tcs:
                        pending_tcs[idx] = {"id": "", "name": "", "args": ""}
                    if tc.id:
                        pending_tcs[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        pending_tcs[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        pending_tcs[idx]["args"] += tc.function.arguments

        for tc_info in pending_tcs.values():
            yield LLMChunk(
                type="tool_call",
                tool_name=tc_info["name"],
                tool_call_id=tc_info["id"],
                tool_arguments=tc_info["args"],
            )

        if full_text:
            yield LLMChunk(type="text_done", text=full_text)

        yield LLMChunk(type="done")

    def cancel(self) -> None:
        self._cancel_flag.set()
