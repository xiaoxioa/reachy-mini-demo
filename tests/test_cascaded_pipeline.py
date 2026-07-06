# -*- coding: utf-8 -*-
"""CascadedPipeline 单元测试 — mock 全部 4 个 provider。"""

from __future__ import annotations

import queue
import threading
import time
import unittest
from unittest.mock import MagicMock

import numpy as np

from voice.pipeline.events import EventType, PipelineEvent
from voice.pipeline.providers import VADEvent, LLMChunk
from voice.cascaded.pipeline import CascadedPipeline


def _make_pipeline(
    vad_events=None, asr_text="你好", llm_chunks=None, tts_chunks=None
):
    """Create a CascadedPipeline with mocked providers."""
    vad = MagicMock()
    vad.feed.return_value = vad_events or []
    vad.reset.return_value = None

    asr = MagicMock()
    asr.transcribe.return_value = asr_text

    llm = MagicMock()
    if llm_chunks is None:
        llm_chunks = [
            LLMChunk(type="text_delta", text="回复"),
            LLMChunk(type="text_done", text="回复"),
            LLMChunk(type="done"),
        ]
    llm.chat_stream.return_value = iter(llm_chunks)
    llm.cancel.return_value = None

    tts = MagicMock()
    if tts_chunks is None:
        tts_chunks = [np.zeros(1600, dtype=np.int16)]
    tts.synthesize_stream.return_value = iter(tts_chunks)
    tts.cancel.return_value = None

    p = CascadedPipeline(vad=vad, asr=asr, llm=llm, tts=tts)
    return p, vad, asr, llm, tts


class TestSessionLifecycle(unittest.TestCase):
    def test_open_close(self):
        p, *_ = _make_pipeline()
        events = []
        p.set_event_handler(lambda e: events.append(e))

        self.assertFalse(p.is_connected)
        ok = p.open_session("instructions", [])
        self.assertTrue(ok)
        self.assertTrue(p.is_connected)
        self.assertEqual(events[-1].type, EventType.SESSION_READY)

        p.close_session()
        self.assertFalse(p.is_connected)

    def test_feed_audio_before_connect(self):
        p, vad, *_ = _make_pipeline()
        p.feed_audio(np.zeros(160, dtype=np.int16))
        vad.feed.assert_not_called()


class TestFullPipeline(unittest.TestCase):
    def test_speech_end_triggers_response(self):
        speech_audio = np.random.randint(-1000, 1000, 16000, dtype=np.int16)
        vad_events = [
            VADEvent(type="speech_start"),
            VADEvent(type="speech_end", audio=speech_audio),
        ]
        p, vad, asr, llm, tts = _make_pipeline(vad_events=vad_events)

        collected = []
        p.set_event_handler(lambda e: collected.append(e))
        p.open_session("你是助手", [])

        vad.feed.return_value = vad_events
        p.feed_audio(np.zeros(160, dtype=np.int16))

        # Wait for response thread to finish
        time.sleep(0.5)

        types = [e.type for e in collected]
        self.assertIn(EventType.USER_SPEECH_START, types)
        self.assertIn(EventType.USER_SPEECH_END, types)
        self.assertIn(EventType.USER_TRANSCRIPT, types)
        self.assertIn(EventType.RESPONSE_START, types)
        self.assertIn(EventType.RESPONSE_TEXT_DELTA, types)
        self.assertIn(EventType.RESPONSE_TEXT_DONE, types)
        self.assertIn(EventType.RESPONSE_AUDIO_DELTA, types)
        self.assertIn(EventType.RESPONSE_DONE, types)

        asr.transcribe.assert_called_once()
        llm.chat_stream.assert_called_once()
        tts.synthesize_stream.assert_called_once()

    def test_empty_asr_skips_response(self):
        speech_audio = np.zeros(1600, dtype=np.int16)
        vad_events = [
            VADEvent(type="speech_start"),
            VADEvent(type="speech_end", audio=speech_audio),
        ]
        p, vad, asr, llm, tts = _make_pipeline(
            vad_events=vad_events, asr_text=""
        )

        collected = []
        p.set_event_handler(lambda e: collected.append(e))
        p.open_session("test", [])

        vad.feed.return_value = vad_events
        p.feed_audio(np.zeros(160, dtype=np.int16))
        time.sleep(0.3)

        types = [e.type for e in collected]
        self.assertNotIn(EventType.RESPONSE_START, types)
        llm.chat_stream.assert_not_called()


class TestCancelResponse(unittest.TestCase):
    def test_cancel_stops_generation(self):
        def slow_llm(*a, **kw):
            yield LLMChunk(type="text_delta", text="a")
            time.sleep(2.0)
            yield LLMChunk(type="text_done", text="ab")
            yield LLMChunk(type="done")

        p, vad, asr, llm, tts = _make_pipeline()
        llm.chat_stream.side_effect = slow_llm

        collected = []
        p.set_event_handler(lambda e: collected.append(e))
        p.open_session("test", [])

        speech = np.zeros(1600, dtype=np.int16)
        vad.feed.return_value = [
            VADEvent(type="speech_start"),
            VADEvent(type="speech_end", audio=speech),
        ]
        p.feed_audio(np.zeros(160, dtype=np.int16))
        time.sleep(0.2)

        p.cancel_response()
        llm.cancel.assert_called()
        tts.cancel.assert_called()


class TestTriggerResponse(unittest.TestCase):
    def test_trigger_without_asr(self):
        p, vad, asr, llm, tts = _make_pipeline()

        collected = []
        p.set_event_handler(lambda e: collected.append(e))
        p.open_session("你是助手", [])

        p.trigger_response(instructions="说你好")
        time.sleep(0.5)

        types = [e.type for e in collected]
        self.assertIn(EventType.RESPONSE_START, types)
        self.assertIn(EventType.RESPONSE_DONE, types)
        asr.transcribe.assert_not_called()


class TestToolCalls(unittest.TestCase):
    def test_tool_call_and_result(self):
        llm_call_count = 0

        def multi_turn_llm(messages, tools=None):
            nonlocal llm_call_count
            llm_call_count += 1
            if llm_call_count == 1:
                yield LLMChunk(type="tool_call", tool_name="nod",
                               tool_call_id="tc_1", tool_arguments="{}")
                yield LLMChunk(type="done")
            else:
                yield LLMChunk(type="text_delta", text="好的")
                yield LLMChunk(type="text_done", text="好的")
                yield LLMChunk(type="done")

        p, vad, asr, llm, tts = _make_pipeline()
        llm.chat_stream.side_effect = multi_turn_llm

        collected = []
        p.set_event_handler(lambda e: collected.append(e))
        p.open_session("test", [{"name": "nod"}])

        def feed_tool_result():
            time.sleep(0.3)
            p.submit_tool_result("tc_1", '{"success": true}')

        threading.Thread(target=feed_tool_result, daemon=True).start()

        speech = np.zeros(1600, dtype=np.int16)
        vad.feed.return_value = [
            VADEvent(type="speech_start"),
            VADEvent(type="speech_end", audio=speech),
        ]
        p.feed_audio(np.zeros(160, dtype=np.int16))
        time.sleep(1.0)

        types = [e.type for e in collected]
        self.assertIn(EventType.TOOL_CALL, types)
        self.assertIn(EventType.RESPONSE_TEXT_DELTA, types)
        self.assertEqual(llm.chat_stream.call_count, 2)


class TestInjectSystemMessage(unittest.TestCase):
    def test_inject_triggers_response(self):
        p, vad, asr, llm, tts = _make_pipeline()

        collected = []
        p.set_event_handler(lambda e: collected.append(e))
        p.open_session("test", [])

        p.inject_system_message("安全提示")
        time.sleep(0.5)

        types = [e.type for e in collected]
        self.assertIn(EventType.RESPONSE_START, types)
        asr.transcribe.assert_not_called()


class TestUpdateInstructions(unittest.TestCase):
    def test_instructions_update(self):
        p, *_ = _make_pipeline()
        p.open_session("原始指令", [])
        self.assertEqual(p._messages[0]["content"], "原始指令")

        p.update_instructions("新指令")
        self.assertEqual(p._messages[0]["content"], "新指令")
        self.assertEqual(p._instructions, "新指令")


class TestAudioDeltaFormat(unittest.TestCase):
    def test_audio_is_float32(self):
        speech = np.zeros(1600, dtype=np.int16)
        vad_events = [
            VADEvent(type="speech_start"),
            VADEvent(type="speech_end", audio=speech),
        ]
        tts_out = [np.array([16384, -16384], dtype=np.int16)]
        p, vad, asr, llm, tts = _make_pipeline(
            vad_events=vad_events, tts_chunks=tts_out
        )

        audio_deltas = []
        def handler(e):
            if e.type == EventType.RESPONSE_AUDIO_DELTA:
                audio_deltas.append(e.data["pcm_16k"])
        p.set_event_handler(handler)
        p.open_session("test", [])

        vad.feed.return_value = vad_events
        p.feed_audio(np.zeros(160, dtype=np.int16))
        time.sleep(0.5)

        self.assertTrue(len(audio_deltas) > 0)
        self.assertEqual(audio_deltas[0].dtype, np.float32)
        np.testing.assert_allclose(
            audio_deltas[0], np.array([0.5, -0.5], dtype=np.float32), atol=0.01
        )


if __name__ == "__main__":
    unittest.main()
