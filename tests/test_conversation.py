#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ConversationHandler 业务逻辑测试。

用例:
  1. 事件分发到正确 handler
  2. SESSION_READY 设置 session_updated
  3. barge-in: 清 play_q、cancel_response、clear_player
  4. 无播放/无 in_flight 时不触发 barge-in
  5. 工具分发: motion、end_session、snapshot、identify_pointed_object
  6. transcript 录制: 用户/助手/空文本
  7. 标签泄漏兜底: <nod> → motion_q + 从 transcript 剥离
  8. audio delta: play_q 入队、drop_audio、thinking 清除
  9. response done: follow-up 触发条件、in_flight 递减
  10. session 管理: open/close/state 清理

运行:
  cd reachy-mini-demo && .venv/bin/python tests/test_conversation.py -v
"""

import os
import sys
import json
import queue
import threading
import time
import unittest

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from voice.pipeline.base import VoicePipeline
from voice.pipeline.events import EventType, PipelineEvent
from voice.conversation import (
    ConversationHandler, _extract_tag_action, _ACTION_TAG_RE,
)


# ── Test doubles ──

class RecordingPipeline(VoicePipeline):
    """记录所有调用的 mock pipeline。"""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[str, dict]] = []
        self._connected = False

    def open_session(self, instructions, tools, timeout=10.0):
        self.calls.append(("open_session",
                           {"instructions": instructions, "tools": tools}))
        self._connected = True
        return True

    def close_session(self):
        self.calls.append(("close_session", {}))
        self._connected = False

    def feed_audio(self, pcm_16k_mono):
        self.calls.append(("feed_audio", {"pcm": pcm_16k_mono}))

    def submit_tool_result(self, call_id, output):
        self.calls.append(("submit_tool_result",
                           {"call_id": call_id, "output": output}))

    def cancel_response(self):
        self.calls.append(("cancel_response", {}))

    def trigger_response(self, instructions=None):
        self.calls.append(("trigger_response",
                           {"instructions": instructions}))

    def update_instructions(self, instructions, tools=None):
        self.calls.append(("update_instructions",
                           {"instructions": instructions, "tools": tools}))

    def inject_system_message(self, text):
        self.calls.append(("inject_system_message", {"text": text}))

    @property
    def is_connected(self):
        return self._connected


def _make_handler(**overrides):
    """创建测试用 ConversationHandler，所有依赖用 mock。"""
    from voice.state import State
    st = overrides.pop("st", None) or State()
    pipeline = overrides.pop("pipeline", None) or RecordingPipeline()
    return ConversationHandler(
        st=st,
        pipeline=pipeline,
        play_q=overrides.pop("play_q", queue.Queue()),
        motion_q=overrides.pop("motion_q", queue.Queue()),
        snap_q=overrides.pop("snap_q", queue.Queue()),
        clear_player_fn=overrides.pop("clear_player_fn", lambda: None),
        memory_mgr=overrides.pop("memory_mgr", None),
        owner_mgr=overrides.pop("owner_mgr", None),
        id_recognizer=overrides.pop("id_recognizer", None),
        oai_client=overrides.pop("oai_client", None),
        instructions=overrides.pop("instructions", "test instructions"),
        tools=overrides.pop("tools", []),
        no_memory=overrides.pop("no_memory", True),
        face_pipeline=overrides.pop("face_pipeline", None),
    )


def _calls_of(pipeline, method_name):
    return [c for c in pipeline.calls if c[0] == method_name]


# ── Tests ──

class TestEventDispatch(unittest.TestCase):
    def test_all_event_types_handled_without_crash(self):
        handler = _make_handler()
        for et in EventType:
            handler.handle_event(PipelineEvent(et, {}))

    def test_exception_in_handler_caught(self):
        handler = _make_handler()
        handler._on_error = lambda e: (_ for _ in ()).throw(ValueError("boom"))
        handler.handle_event(PipelineEvent(EventType.ERROR, {}))


class TestSessionReady(unittest.TestCase):
    def test_sets_session_updated(self):
        from voice.state import State
        st = State()
        handler = _make_handler(st=st)
        st.session_updated.clear()
        handler.handle_event(PipelineEvent(EventType.SESSION_READY))
        self.assertTrue(st.session_updated.is_set())


class TestBargeIn(unittest.TestCase):
    def test_barge_in_when_in_flight(self):
        from voice.state import State
        st = State()
        pipeline = RecordingPipeline()
        play_q = queue.Queue()
        play_q.put((0, np.zeros(160)))
        handler = _make_handler(st=st, pipeline=pipeline, play_q=play_q)
        st.in_flight = 1
        handler.handle_event(PipelineEvent(EventType.USER_SPEECH_START))
        self.assertEqual(len(_calls_of(pipeline, "cancel_response")), 1)
        self.assertTrue(play_q.empty())
        self.assertTrue(st.drop_audio)

    def test_barge_in_when_playing(self):
        from voice.state import State
        st = State()
        pipeline = RecordingPipeline()
        play_q = queue.Queue()
        play_q.put((0, np.zeros(160)))
        handler = _make_handler(st=st, pipeline=pipeline, play_q=play_q)
        st.in_flight = 0
        handler.handle_event(PipelineEvent(EventType.USER_SPEECH_START))
        self.assertEqual(len(_calls_of(pipeline, "cancel_response")), 0)
        self.assertTrue(play_q.empty())

    def test_no_barge_in_when_idle(self):
        from voice.state import State
        st = State()
        pipeline = RecordingPipeline()
        handler = _make_handler(st=st, pipeline=pipeline)
        st.in_flight = 0
        st.playback_end_estimate = 0.0
        handler.handle_event(PipelineEvent(EventType.USER_SPEECH_START))
        self.assertEqual(len(_calls_of(pipeline, "cancel_response")), 0)
        self.assertFalse(st.drop_audio)

    def test_clear_player_called(self):
        from voice.state import State
        st = State()
        play_q = queue.Queue()
        play_q.put((0, np.zeros(160)))
        cleared = []
        handler = _make_handler(
            st=st, play_q=play_q,
            clear_player_fn=lambda: cleared.append(True))
        st.in_flight = 0
        handler.handle_event(PipelineEvent(EventType.USER_SPEECH_START))
        self.assertEqual(len(cleared), 1)

    def test_user_speaking_set(self):
        from voice.state import State
        st = State()
        handler = _make_handler(st=st)
        handler.handle_event(PipelineEvent(EventType.USER_SPEECH_START))
        self.assertTrue(st.user_speaking)


class TestSpeechEnd(unittest.TestCase):
    def test_thinking_set(self):
        from voice.state import State
        st = State()
        handler = _make_handler(st=st)
        handler.handle_event(PipelineEvent(EventType.USER_SPEECH_END))
        self.assertTrue(st.thinking)
        self.assertFalse(st.user_speaking)


class TestToolDispatch(unittest.TestCase):
    def test_motion_tool(self):
        pipeline = RecordingPipeline()
        motion_q = queue.Queue()
        handler = _make_handler(pipeline=pipeline, motion_q=motion_q)
        handler.handle_event(PipelineEvent(EventType.TOOL_CALL, {
            "name": "nod", "call_id": "test-call-123", "arguments": "{}",
        }))
        self.assertFalse(motion_q.empty())
        self.assertEqual(motion_q.get()["name"], "nod")
        submits = _calls_of(pipeline, "submit_tool_result")
        self.assertEqual(len(submits), 1)
        self.assertIn("nod", submits[0][1]["output"])

    def test_end_session(self):
        from voice.state import State
        st = State()
        pipeline = RecordingPipeline()
        handler = _make_handler(st=st, pipeline=pipeline)
        handler.handle_event(PipelineEvent(EventType.TOOL_CALL, {
            "name": "end_session", "call_id": "end-123", "arguments": "{}",
        }))
        self.assertTrue(st.exit_request)
        self.assertEqual(len(_calls_of(pipeline, "submit_tool_result")), 1)

    def test_snapshot_tool(self):
        from voice.state import State
        st = State()
        snap_q = queue.Queue()
        handler = _make_handler(st=st, snap_q=snap_q)
        handler.handle_event(PipelineEvent(EventType.TOOL_CALL, {
            "name": "take_snapshot", "call_id": "snap-123",
            "arguments": "{}",
        }))
        self.assertFalse(snap_q.empty())
        item = snap_q.get()
        self.assertEqual(item["call_id"], "snap-123")
        self.assertEqual(item["mode"], "scene")
        self.assertEqual(st.snapshot_pending, 1)

    def test_snapshot_pointing_mode(self):
        from voice.state import State
        st = State()
        st.finger_ext_at = time.monotonic()
        snap_q = queue.Queue()
        handler = _make_handler(st=st, snap_q=snap_q)
        handler.handle_event(PipelineEvent(EventType.TOOL_CALL, {
            "name": "take_snapshot", "call_id": "snap-456",
            "arguments": "{}",
        }))
        self.assertEqual(snap_q.get()["mode"], "judge")

    def test_identify_pointed_object(self):
        from voice.state import State
        st = State()
        snap_q = queue.Queue()
        handler = _make_handler(st=st, snap_q=snap_q)
        handler.handle_event(PipelineEvent(EventType.TOOL_CALL, {
            "name": "identify_pointed_object", "call_id": "point-123",
            "arguments": "{}",
        }))
        self.assertEqual(snap_q.get()["mode"], "judge")
        self.assertEqual(st.snapshot_pending, 1)

    def test_fc_seen_set(self):
        from voice.state import State
        st = State()
        handler = _make_handler(st=st)
        handler.handle_event(PipelineEvent(EventType.TOOL_CALL, {
            "name": "nod", "call_id": "c1", "arguments": "{}",
        }))
        self.assertTrue(st.fc_seen_this_resp)


class TestTranscript(unittest.TestCase):
    def test_user_transcript_recorded(self):
        from voice.state import State
        st = State()
        handler = _make_handler(st=st)
        handler.handle_event(PipelineEvent(
            EventType.USER_TRANSCRIPT, {"text": "你好世界"}))
        self.assertEqual(len(st.display_transcript), 1)
        self.assertEqual(st.display_transcript[0]["role"], "user")
        self.assertEqual(st.display_transcript[0]["text"], "你好世界")

    def test_empty_transcript_ignored(self):
        from voice.state import State
        st = State()
        handler = _make_handler(st=st)
        handler.handle_event(PipelineEvent(
            EventType.USER_TRANSCRIPT, {"text": ""}))
        self.assertEqual(len(st.display_transcript), 0)

    def test_assistant_text_recorded(self):
        from voice.state import State
        st = State()
        handler = _make_handler(st=st)
        handler.handle_event(PipelineEvent(
            EventType.RESPONSE_TEXT_DONE, {"text": "你好！"}))
        self.assertEqual(len(st.display_transcript), 1)
        self.assertEqual(st.display_transcript[0]["role"], "assistant")

    def test_transcript_overflow_trimmed(self):
        from voice.state import State
        st = State()
        handler = _make_handler(st=st)
        for i in range(110):
            handler.handle_event(PipelineEvent(
                EventType.USER_TRANSCRIPT, {"text": f"msg{i}"}))
        self.assertLess(len(st.display_transcript), 100)


class TestTagLeaking(unittest.TestCase):
    def test_tag_extraction(self):
        self.assertEqual(_extract_tag_action("<nod>"), "nod")
        self.assertEqual(_extract_tag_action("(点头)"), "nod")
        self.assertEqual(_extract_tag_action("*shake_head*"), "shake_head")

    def test_tag_triggers_motion(self):
        motion_q = queue.Queue()
        handler = _make_handler(motion_q=motion_q)
        handler.handle_event(PipelineEvent(
            EventType.RESPONSE_TEXT_DONE, {"text": "好的<nod>我知道了"}))
        self.assertFalse(motion_q.empty())
        self.assertEqual(motion_q.get()["name"], "nod")

    def test_tag_stripped_from_transcript(self):
        from voice.state import State
        st = State()
        handler = _make_handler(st=st)
        handler.handle_event(PipelineEvent(
            EventType.RESPONSE_TEXT_DONE, {"text": "好的<nod>我知道了"}))
        self.assertEqual(st.display_transcript[0]["text"], "好的我知道了")

    def test_multiple_tags(self):
        motion_q = queue.Queue()
        handler = _make_handler(motion_q=motion_q)
        handler.handle_event(PipelineEvent(
            EventType.RESPONSE_TEXT_DONE,
            {"text": "<nod>好<shake_head>"},
        ))
        actions = []
        while not motion_q.empty():
            actions.append(motion_q.get()["name"])
        self.assertIn("nod", actions)
        self.assertIn("shake_head", actions)


class TestAudioDelta(unittest.TestCase):
    def test_audio_put_in_play_q(self):
        from voice.state import State
        st = State()
        play_q = queue.Queue()
        handler = _make_handler(st=st, play_q=play_q)
        pcm = np.zeros(160, dtype=np.float32)
        handler.handle_event(PipelineEvent(
            EventType.RESPONSE_AUDIO_DELTA, {"pcm_16k": pcm}))
        self.assertFalse(play_q.empty())
        gen, audio = play_q.get()
        self.assertEqual(gen, 0)
        np.testing.assert_array_equal(audio, pcm)

    def test_audio_dropped_when_flag_set(self):
        from voice.state import State
        st = State()
        st.drop_audio = True
        play_q = queue.Queue()
        handler = _make_handler(st=st, play_q=play_q)
        handler.handle_event(PipelineEvent(
            EventType.RESPONSE_AUDIO_DELTA,
            {"pcm_16k": np.zeros(160, dtype=np.float32)}))
        self.assertTrue(play_q.empty())

    def test_thinking_cleared_on_audio(self):
        from voice.state import State
        st = State()
        st.thinking = True
        handler = _make_handler(st=st)
        handler.handle_event(PipelineEvent(
            EventType.RESPONSE_AUDIO_DELTA,
            {"pcm_16k": np.zeros(160, dtype=np.float32)}))
        self.assertFalse(st.thinking)

    def test_resp_audio_count_incremented(self):
        from voice.state import State
        st = State()
        handler = _make_handler(st=st)
        handler.handle_event(PipelineEvent(
            EventType.RESPONSE_AUDIO_DELTA,
            {"pcm_16k": np.zeros(160, dtype=np.float32)}))
        handler.handle_event(PipelineEvent(
            EventType.RESPONSE_AUDIO_DELTA,
            {"pcm_16k": np.zeros(160, dtype=np.float32)}))
        self.assertEqual(st.resp_audio_count, 2)


class TestResponseDone(unittest.TestCase):
    def test_followup_triggered(self):
        from voice.state import State
        st = State()
        pipeline = RecordingPipeline()
        handler = _make_handler(st=st, pipeline=pipeline)
        st.in_flight = 1
        st.fc_seen_this_resp = True
        st.resp_audio_count = 0
        st.fc_gen = 0
        st.play_gen = 0
        st.snapshot_pending = 0
        handler.handle_event(PipelineEvent(EventType.RESPONSE_DONE, {}))
        self.assertEqual(len(_calls_of(pipeline, "trigger_response")), 1)

    def test_no_followup_when_audio_present(self):
        from voice.state import State
        st = State()
        pipeline = RecordingPipeline()
        handler = _make_handler(st=st, pipeline=pipeline)
        st.in_flight = 1
        st.fc_seen_this_resp = True
        st.resp_audio_count = 5
        handler.handle_event(PipelineEvent(EventType.RESPONSE_DONE, {}))
        self.assertEqual(len(_calls_of(pipeline, "trigger_response")), 0)

    def test_no_followup_when_no_fc(self):
        from voice.state import State
        st = State()
        pipeline = RecordingPipeline()
        handler = _make_handler(st=st, pipeline=pipeline)
        st.in_flight = 1
        st.fc_seen_this_resp = False
        st.resp_audio_count = 0
        handler.handle_event(PipelineEvent(EventType.RESPONSE_DONE, {}))
        self.assertEqual(len(_calls_of(pipeline, "trigger_response")), 0)

    def test_no_followup_when_snapshot_pending(self):
        from voice.state import State
        st = State()
        pipeline = RecordingPipeline()
        handler = _make_handler(st=st, pipeline=pipeline)
        st.in_flight = 1
        st.fc_seen_this_resp = True
        st.resp_audio_count = 0
        st.fc_gen = 0
        st.play_gen = 0
        st.snapshot_pending = 1
        handler.handle_event(PipelineEvent(EventType.RESPONSE_DONE, {}))
        self.assertEqual(len(_calls_of(pipeline, "trigger_response")), 0)

    def test_in_flight_decremented(self):
        from voice.state import State
        st = State()
        st.in_flight = 2
        handler = _make_handler(st=st)
        handler.handle_event(PipelineEvent(EventType.RESPONSE_DONE, {}))
        self.assertEqual(st.in_flight, 1)

    def test_in_flight_not_negative(self):
        from voice.state import State
        st = State()
        st.in_flight = 0
        handler = _make_handler(st=st)
        handler.handle_event(PipelineEvent(EventType.RESPONSE_DONE, {}))
        self.assertEqual(st.in_flight, 0)


class TestResponseStart(unittest.TestCase):
    def test_in_flight_incremented(self):
        from voice.state import State
        st = State()
        handler = _make_handler(st=st)
        handler.handle_event(PipelineEvent(EventType.RESPONSE_START, {}))
        self.assertEqual(st.in_flight, 1)

    def test_drop_audio_cleared(self):
        from voice.state import State
        st = State()
        st.drop_audio = True
        handler = _make_handler(st=st)
        handler.handle_event(PipelineEvent(EventType.RESPONSE_START, {}))
        self.assertFalse(st.drop_audio)

    def test_snapshot_pid_captured(self):
        from voice.state import State
        st = State()
        st.current_person_id = "test-pid"
        st.current_person_name = "Alice"
        handler = _make_handler(st=st)
        handler.handle_event(PipelineEvent(EventType.RESPONSE_START, {}))
        self.assertEqual(st.resp_snapshot_pid, "test-pid")
        self.assertEqual(st.resp_snapshot_name, "Alice")


class TestSessionManagement(unittest.TestCase):
    def test_open_session(self):
        pipeline = RecordingPipeline()
        handler = _make_handler(pipeline=pipeline,
                                instructions="hello", tools=["t1"])
        result = handler.open_session()
        self.assertTrue(result)
        self.assertEqual(pipeline.calls[0][0], "open_session")
        self.assertEqual(pipeline.calls[0][1]["instructions"], "hello")

    def test_close_session_clears_state(self):
        from voice.state import State
        st = State()
        pipeline = RecordingPipeline()
        handler = _make_handler(st=st, pipeline=pipeline)
        st.current_person_id = "test-pid"
        st.identity_injected = True
        st.clear_workflow = {"phase": "verifying"}
        st.clear_lock = True
        handler.close_session()
        self.assertIsNone(st.current_person_id)
        self.assertFalse(st.identity_injected)
        self.assertIsNone(st.clear_workflow)
        self.assertFalse(st.clear_lock)
        self.assertEqual(len(_calls_of(pipeline, "close_session")), 1)

    def test_close_session_clears_conv_log(self):
        from voice.state import State
        st = State()
        handler = _make_handler(st=st)
        st.conversation_log["pid1"] = [("user", "hello")]
        handler.close_session()
        self.assertEqual(len(st.conversation_log), 0)


if __name__ == "__main__":
    unittest.main()
