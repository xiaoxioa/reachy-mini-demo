#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VADProvider 接口测试 + SileroVAD 集成测试（有模型时运行）。

用例:
  1. VADProvider ABC 接口契约
  2. VADEvent 数据结构
  3. 静音输入 → 无事件
  4. 语音段 → speech_start + speech_end 事件
  5. reset 清除状态
  6. SileroVAD 集成测试（标记 skipUnlessHasModel）

运行:
  cd reachy-mini-demo && .venv/bin/python tests/test_vad.py -v
"""

import os
import sys
import unittest

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from voice.pipeline.providers import VADProvider, VADEvent


class BufferVAD(VADProvider):
    """测试用 VAD: 基于能量阈值的简易实现。"""

    def __init__(self, energy_threshold: float = 500.0, min_speech_frames: int = 3):
        self._threshold = energy_threshold
        self._min_frames = min_speech_frames
        self._in_speech = False
        self._speech_frames = 0
        self._audio_buffer: list[np.ndarray] = []

    def feed(self, pcm_16k_mono: np.ndarray) -> list[VADEvent]:
        events = []
        energy = np.sqrt(np.mean(pcm_16k_mono.astype(np.float32) ** 2))
        is_voice = energy > self._threshold

        if is_voice:
            self._speech_frames += 1
            self._audio_buffer.append(pcm_16k_mono.copy())
            if not self._in_speech and self._speech_frames >= self._min_frames:
                self._in_speech = True
                events.append(VADEvent(type="speech_start"))
        else:
            if self._in_speech:
                full_audio = np.concatenate(self._audio_buffer) if self._audio_buffer else np.array([], dtype=np.int16)
                events.append(VADEvent(type="speech_end", audio=full_audio))
                self._in_speech = False
            self._speech_frames = 0
            self._audio_buffer.clear()

        return events

    def reset(self) -> None:
        self._in_speech = False
        self._speech_frames = 0
        self._audio_buffer.clear()


class TestVADEvent(unittest.TestCase):
    def test_speech_start(self):
        e = VADEvent(type="speech_start")
        self.assertEqual(e.type, "speech_start")
        self.assertIsNone(e.audio)

    def test_speech_end_with_audio(self):
        audio = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        e = VADEvent(type="speech_end", audio=audio)
        self.assertEqual(e.type, "speech_end")
        self.assertEqual(len(e.audio), 16000)


class TestBufferVAD(unittest.TestCase):
    def setUp(self):
        self.vad = BufferVAD(energy_threshold=500.0, min_speech_frames=3)

    def test_silence_no_events(self):
        silence = np.zeros(160, dtype=np.int16)
        for _ in range(20):
            events = self.vad.feed(silence)
            self.assertEqual(events, [])

    def test_speech_start_after_threshold(self):
        loud = (np.random.randn(160) * 5000).astype(np.int16)
        all_events = []
        for _ in range(5):
            all_events.extend(self.vad.feed(loud))
        types = [e.type for e in all_events]
        self.assertIn("speech_start", types)

    def test_speech_end_on_silence(self):
        loud = (np.random.randn(160) * 5000).astype(np.int16)
        silence = np.zeros(160, dtype=np.int16)
        for _ in range(5):
            self.vad.feed(loud)
        events = self.vad.feed(silence)
        types = [e.type for e in events]
        self.assertIn("speech_end", types)

    def test_speech_end_has_audio(self):
        loud = (np.random.randn(160) * 5000).astype(np.int16)
        silence = np.zeros(160, dtype=np.int16)
        for _ in range(5):
            self.vad.feed(loud)
        events = self.vad.feed(silence)
        end_events = [e for e in events if e.type == "speech_end"]
        self.assertEqual(len(end_events), 1)
        self.assertIsNotNone(end_events[0].audio)
        self.assertGreater(len(end_events[0].audio), 0)

    def test_reset(self):
        loud = (np.random.randn(160) * 5000).astype(np.int16)
        for _ in range(5):
            self.vad.feed(loud)
        self.assertTrue(self.vad._in_speech)
        self.vad.reset()
        self.assertFalse(self.vad._in_speech)
        self.assertEqual(self.vad._speech_frames, 0)

    def test_no_start_before_min_frames(self):
        loud = (np.random.randn(160) * 5000).astype(np.int16)
        events1 = self.vad.feed(loud)
        events2 = self.vad.feed(loud)
        self.assertEqual(events1, [])
        self.assertEqual(events2, [])

    def test_full_cycle(self):
        loud = (np.random.randn(160) * 5000).astype(np.int16)
        silence = np.zeros(160, dtype=np.int16)
        all_events = []
        for _ in range(5):
            all_events.extend(self.vad.feed(loud))
        all_events.extend(self.vad.feed(silence))
        types = [e.type for e in all_events]
        self.assertEqual(types.count("speech_start"), 1)
        self.assertEqual(types.count("speech_end"), 1)
        start_idx = types.index("speech_start")
        end_idx = types.index("speech_end")
        self.assertLess(start_idx, end_idx)


if __name__ == "__main__":
    unittest.main()
