#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TTSProvider 接口测试。

用例:
  1. TTSProvider ABC 接口契约
  2. synthesize_stream 返回音频块
  3. cancel 停止合成
  4. 音频块格式验证（dtype, 采样率）
  5. 空文本处理
  6. 多次合成独立性

运行:
  cd reachy-mini-demo && .venv/bin/python tests/test_tts.py -v
"""

import os
import sys
import unittest

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from voice.pipeline.providers import TTSProvider


class FakeTTS(TTSProvider):
    """测试用 TTS: 根据文本长度生成固定模式音频。"""

    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1600  # 100ms per chunk

    def __init__(self):
        self._cancelled = False

    def synthesize_stream(self, text: str, voice: str = "default") -> iter:
        self._cancelled = False
        if not text.strip():
            return
        num_chunks = max(1, len(text))
        for i in range(num_chunks):
            if self._cancelled:
                return
            t = np.linspace(0, 0.1, self.CHUNK_SIZE, endpoint=False)
            freq = 440 + i * 10
            audio = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
            yield audio

    def cancel(self):
        self._cancelled = True


class TestTTSInterface(unittest.TestCase):
    def test_abc_prevents_instantiation(self):
        with self.assertRaises(TypeError):
            TTSProvider()


class TestFakeTTS(unittest.TestCase):
    def setUp(self):
        self.tts = FakeTTS()

    def test_synthesize_returns_audio(self):
        chunks = list(self.tts.synthesize_stream("你好"))
        self.assertGreater(len(chunks), 0)

    def test_audio_format(self):
        chunks = list(self.tts.synthesize_stream("测试"))
        for chunk in chunks:
            self.assertEqual(chunk.dtype, np.int16)
            self.assertEqual(len(chunk), FakeTTS.CHUNK_SIZE)

    def test_chunk_count_proportional_to_text(self):
        short = list(self.tts.synthesize_stream("你"))
        long = list(self.tts.synthesize_stream("你好世界测试"))
        self.assertLess(len(short), len(long))

    def test_empty_text_no_output(self):
        chunks = list(self.tts.synthesize_stream(""))
        self.assertEqual(chunks, [])
        chunks2 = list(self.tts.synthesize_stream("   "))
        self.assertEqual(chunks2, [])

    def test_cancel_stops_generation(self):
        gen = self.tts.synthesize_stream("这是一个很长的测试文本用于验证取消功能是否正常工作")
        first = next(gen)
        self.assertIsNotNone(first)
        self.tts.cancel()
        remaining = list(gen)
        full = list(self.tts.synthesize_stream("这是一个很长的测试文本用于验证取消功能是否正常工作"))
        self.assertLess(len(remaining) + 1, len(full))

    def test_cancel_resets_on_new_call(self):
        self.tts.cancel()
        chunks = list(self.tts.synthesize_stream("你好"))
        self.assertGreater(len(chunks), 0)

    def test_multiple_calls_independent(self):
        c1 = list(self.tts.synthesize_stream("第一次"))
        c2 = list(self.tts.synthesize_stream("第二次"))
        self.assertGreater(len(c1), 0)
        self.assertGreater(len(c2), 0)

    def test_voice_parameter_accepted(self):
        chunks = list(self.tts.synthesize_stream("测试", voice="Ethan"))
        self.assertGreater(len(chunks), 0)

    def test_audio_is_not_silence(self):
        chunks = list(self.tts.synthesize_stream("测试"))
        for chunk in chunks:
            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
            self.assertGreater(rms, 100)


if __name__ == "__main__":
    unittest.main()
