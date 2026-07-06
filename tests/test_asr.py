#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ASRProvider 接口测试。

用例:
  1. ASRProvider ABC 接口契约
  2. transcribe 基本调用
  3. transcribe_streaming 默认 fallback（攒完整段再识别）
  4. transcribe_streaming 空输入
  5. 自定义 streaming 实现覆盖默认 fallback

运行:
  cd reachy-mini-demo && .venv/bin/python tests/test_asr.py -v
"""

import os
import sys
import unittest
from typing import Iterator

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from voice.pipeline.providers import ASRProvider


class EchoASR(ASRProvider):
    """测试用 ASR: 根据音频长度返回固定文本。"""

    def __init__(self):
        self.call_count = 0

    def transcribe(self, pcm_16k_mono: np.ndarray) -> str:
        self.call_count += 1
        duration_s = len(pcm_16k_mono) / 16000
        if duration_s < 0.1:
            return ""
        return f"识别结果_{self.call_count}_{duration_s:.1f}s"


class StreamingASR(ASRProvider):
    """测试用: 自定义 streaming 实现。"""

    def transcribe(self, pcm_16k_mono: np.ndarray) -> str:
        return "完整识别"

    def transcribe_streaming(self, audio_iter: Iterator[np.ndarray]) -> Iterator[str]:
        for i, chunk in enumerate(audio_iter):
            yield f"partial_{i}"
        yield "final"


class TestASRInterface(unittest.TestCase):
    def test_abc_prevents_instantiation(self):
        with self.assertRaises(TypeError):
            ASRProvider()

    def test_transcribe(self):
        asr = EchoASR()
        result = asr.transcribe(np.zeros(16000, dtype=np.int16))
        self.assertEqual(result, "识别结果_1_1.0s")
        self.assertEqual(asr.call_count, 1)

    def test_transcribe_short_audio(self):
        asr = EchoASR()
        result = asr.transcribe(np.zeros(800, dtype=np.int16))
        self.assertEqual(result, "")

    def test_transcribe_call_count(self):
        asr = EchoASR()
        asr.transcribe(np.zeros(16000, dtype=np.int16))
        asr.transcribe(np.zeros(32000, dtype=np.int16))
        self.assertEqual(asr.call_count, 2)


class TestASRStreamingFallback(unittest.TestCase):
    def test_fallback_concatenates_chunks(self):
        asr = EchoASR()
        chunks = [np.zeros(4000, dtype=np.int16) for _ in range(4)]
        results = list(asr.transcribe_streaming(iter(chunks)))
        self.assertEqual(len(results), 1)
        self.assertIn("1.0s", results[0])
        self.assertEqual(asr.call_count, 1)

    def test_fallback_empty_input(self):
        asr = EchoASR()
        results = list(asr.transcribe_streaming(iter([])))
        self.assertEqual(results, [])
        self.assertEqual(asr.call_count, 0)

    def test_fallback_single_chunk(self):
        asr = EchoASR()
        chunks = [np.zeros(16000, dtype=np.int16)]
        results = list(asr.transcribe_streaming(iter(chunks)))
        self.assertEqual(len(results), 1)


class TestASRCustomStreaming(unittest.TestCase):
    def test_custom_streaming(self):
        asr = StreamingASR()
        chunks = [np.zeros(1600, dtype=np.int16) for _ in range(3)]
        results = list(asr.transcribe_streaming(iter(chunks)))
        self.assertEqual(results, ["partial_0", "partial_1", "partial_2", "final"])

    def test_transcribe_still_works(self):
        asr = StreamingASR()
        result = asr.transcribe(np.zeros(16000, dtype=np.int16))
        self.assertEqual(result, "完整识别")


if __name__ == "__main__":
    unittest.main()
