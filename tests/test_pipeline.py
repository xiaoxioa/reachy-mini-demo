#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VoicePipeline 抽象层测试。

用例:
  1. PipelineEvent 构造与字段
  2. EventType 枚举完备性
  3. Mock VoicePipeline 实现 — 验证接口契约
  4. 事件回调 set_event_handler / _emit
  5. Provider ABCs — 子类必须实现抽象方法
  6. ASRProvider.transcribe_streaming 默认 fallback
  7. VADEvent / LLMChunk 数据类

运行:
  cd reachy-mini-demo && .venv/bin/python -m pytest tests/test_pipeline.py -v
"""

import os
import sys
import time
import unittest

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from voice.pipeline.events import EventType, PipelineEvent
from voice.pipeline.base import VoicePipeline
from voice.pipeline.providers import (
    VADEvent, VADProvider,
    ASRProvider,
    LLMChunk, LLMProvider,
    TTSProvider,
)


# ── Mock 实现 ──

class MockPipeline(VoicePipeline):
    def __init__(self):
        super().__init__()
        self._connected = False
        self._instructions = ""
        self._audio_frames = []
        self._tool_results = []

    def open_session(self, instructions, tools, timeout=10.0):
        self._connected = True
        self._instructions = instructions
        self._emit(PipelineEvent(type=EventType.SESSION_READY))
        return True

    def close_session(self):
        self._connected = False

    def feed_audio(self, pcm_16k_mono):
        self._audio_frames.append(pcm_16k_mono)

    def submit_tool_result(self, call_id, output):
        self._tool_results.append((call_id, output))

    def cancel_response(self):
        pass

    def trigger_response(self, instructions=None):
        pass

    def update_instructions(self, instructions, tools=None):
        self._instructions = instructions

    def inject_system_message(self, text):
        pass

    @property
    def is_connected(self):
        return self._connected


class MockVAD(VADProvider):
    def __init__(self):
        self._events = []

    def feed(self, pcm_16k_mono):
        return self._events

    def reset(self):
        self._events = []


class MockASR(ASRProvider):
    def __init__(self, result="你好"):
        self._result = result

    def transcribe(self, pcm_16k_mono):
        return self._result


class MockLLM(LLMProvider):
    def __init__(self):
        self._cancelled = False

    def chat_stream(self, messages, tools=None, tool_choice="auto"):
        yield LLMChunk(type="text_delta", text="你")
        yield LLMChunk(type="text_delta", text="好")
        yield LLMChunk(type="text_done", text="你好")
        yield LLMChunk(type="done")

    def cancel(self):
        self._cancelled = True


class MockTTS(TTSProvider):
    def __init__(self):
        self._cancelled = False

    def synthesize_stream(self, text, voice="default"):
        yield np.zeros(1600, dtype=np.int16)

    def cancel(self):
        self._cancelled = True


# ── 测试 ──

class TestPipelineEvent(unittest.TestCase):
    def test_event_construction(self):
        e = PipelineEvent(type=EventType.USER_TRANSCRIPT, data={"text": "你好"})
        self.assertEqual(e.type, EventType.USER_TRANSCRIPT)
        self.assertEqual(e.data["text"], "你好")
        self.assertIsInstance(e.timestamp, float)
        self.assertGreater(e.timestamp, 0)

    def test_event_default_data(self):
        e = PipelineEvent(type=EventType.SESSION_READY)
        self.assertEqual(e.data, {})

    def test_event_custom_timestamp(self):
        t = 1234567890.0
        e = PipelineEvent(type=EventType.ERROR, data={"message": "fail"}, timestamp=t)
        self.assertEqual(e.timestamp, t)


class TestEventType(unittest.TestCase):
    def test_all_types_exist(self):
        expected = [
            "SESSION_READY", "USER_SPEECH_START", "USER_SPEECH_END",
            "USER_TRANSCRIPT", "RESPONSE_START", "RESPONSE_TEXT_DELTA",
            "RESPONSE_TEXT_DONE", "RESPONSE_AUDIO_DELTA", "RESPONSE_DONE",
            "TOOL_CALL", "ERROR",
        ]
        for name in expected:
            self.assertTrue(hasattr(EventType, name), f"Missing EventType.{name}")

    def test_values_are_strings(self):
        for e in EventType:
            self.assertIsInstance(e.value, str)


class TestVoicePipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = MockPipeline()

    def test_open_close(self):
        self.assertFalse(self.pipeline.is_connected)
        ok = self.pipeline.open_session("你是机器人", [], timeout=5.0)
        self.assertTrue(ok)
        self.assertTrue(self.pipeline.is_connected)
        self.pipeline.close_session()
        self.assertFalse(self.pipeline.is_connected)

    def test_event_handler(self):
        events = []
        self.pipeline.set_event_handler(lambda e: events.append(e))
        self.pipeline.open_session("test", [])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, EventType.SESSION_READY)

    def test_no_handler_no_crash(self):
        self.pipeline.open_session("test", [])

    def test_feed_audio(self):
        self.pipeline.open_session("test", [])
        frame = np.zeros(160, dtype=np.int16)
        self.pipeline.feed_audio(frame)
        self.pipeline.feed_audio(frame)
        self.assertEqual(len(self.pipeline._audio_frames), 2)

    def test_submit_tool_result(self):
        self.pipeline.open_session("test", [])
        self.pipeline.submit_tool_result("call_123", '{"result": "ok"}')
        self.assertEqual(len(self.pipeline._tool_results), 1)
        self.assertEqual(self.pipeline._tool_results[0], ("call_123", '{"result": "ok"}'))

    def test_update_instructions(self):
        self.pipeline.open_session("original", [])
        self.pipeline.update_instructions("updated")
        self.assertEqual(self.pipeline._instructions, "updated")

    def test_abc_prevents_instantiation(self):
        with self.assertRaises(TypeError):
            VoicePipeline()


class TestVADProvider(unittest.TestCase):
    def test_mock_vad(self):
        vad = MockVAD()
        events = vad.feed(np.zeros(160, dtype=np.int16))
        self.assertEqual(events, [])
        vad.reset()

    def test_vad_event_dataclass(self):
        e = VADEvent(type="speech_start")
        self.assertEqual(e.type, "speech_start")
        self.assertIsNone(e.audio)

        audio = np.ones(1600, dtype=np.int16)
        e2 = VADEvent(type="speech_end", audio=audio)
        self.assertEqual(e2.type, "speech_end")
        np.testing.assert_array_equal(e2.audio, audio)

    def test_abc_prevents_instantiation(self):
        with self.assertRaises(TypeError):
            VADProvider()


class TestASRProvider(unittest.TestCase):
    def test_transcribe(self):
        asr = MockASR(result="你好世界")
        text = asr.transcribe(np.zeros(16000, dtype=np.int16))
        self.assertEqual(text, "你好世界")

    def test_streaming_fallback(self):
        asr = MockASR(result="测试结果")
        chunks = [np.zeros(1600, dtype=np.int16) for _ in range(3)]
        results = list(asr.transcribe_streaming(iter(chunks)))
        self.assertEqual(results, ["测试结果"])

    def test_streaming_empty(self):
        asr = MockASR(result="")
        results = list(asr.transcribe_streaming(iter([])))
        self.assertEqual(results, [])

    def test_abc_prevents_instantiation(self):
        with self.assertRaises(TypeError):
            ASRProvider()


class TestLLMProvider(unittest.TestCase):
    def test_chat_stream(self):
        llm = MockLLM()
        chunks = list(llm.chat_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0].type, "text_delta")
        self.assertEqual(chunks[0].text, "你")
        self.assertEqual(chunks[2].type, "text_done")
        self.assertEqual(chunks[2].text, "你好")
        self.assertEqual(chunks[3].type, "done")

    def test_cancel(self):
        llm = MockLLM()
        self.assertFalse(llm._cancelled)
        llm.cancel()
        self.assertTrue(llm._cancelled)

    def test_llm_chunk_tool_call(self):
        c = LLMChunk(type="tool_call", tool_name="nod", tool_call_id="c1", tool_arguments="{}")
        self.assertEqual(c.type, "tool_call")
        self.assertEqual(c.tool_name, "nod")

    def test_abc_prevents_instantiation(self):
        with self.assertRaises(TypeError):
            LLMProvider()


class TestTTSProvider(unittest.TestCase):
    def test_synthesize(self):
        tts = MockTTS()
        chunks = list(tts.synthesize_stream("你好"))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].dtype, np.int16)
        self.assertEqual(len(chunks[0]), 1600)

    def test_cancel(self):
        tts = MockTTS()
        self.assertFalse(tts._cancelled)
        tts.cancel()
        self.assertTrue(tts._cancelled)

    def test_abc_prevents_instantiation(self):
        with self.assertRaises(TypeError):
            TTSProvider()


if __name__ == "__main__":
    unittest.main()
