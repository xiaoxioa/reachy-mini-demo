# -*- coding: utf-8 -*-
"""DashScopeASR 单元测试 — mock DashScope SDK。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class TestDashScopeASR(unittest.TestCase):
    @patch("voice.cascaded.asr_dashscope.Recognition")
    def test_transcribe_returns_text(self, MockRecognition):
        mock_result = MagicMock()
        mock_result.output.sentence = [
            {"text": "你好"},
            {"text": "世界"},
        ]
        mock_instance = MagicMock()
        mock_instance.call.return_value = mock_result
        MockRecognition.return_value = mock_instance

        from voice.cascaded.asr_dashscope import DashScopeASR
        asr = DashScopeASR(model="paraformer-realtime-v2")

        audio = np.random.randint(-1000, 1000, 16000, dtype=np.int16)
        text = asr.transcribe(audio)

        self.assertEqual(text, "你好世界")
        mock_instance.call.assert_called_once()
        call_args = mock_instance.call.call_args
        self.assertTrue(call_args[0][0].endswith(".wav"))

    @patch("voice.cascaded.asr_dashscope.Recognition")
    def test_transcribe_empty_result(self, MockRecognition):
        mock_result = MagicMock()
        mock_result.output.sentence = []
        mock_instance = MagicMock()
        mock_instance.call.return_value = mock_result
        MockRecognition.return_value = mock_instance

        from voice.cascaded.asr_dashscope import DashScopeASR
        asr = DashScopeASR()

        audio = np.zeros(1600, dtype=np.int16)
        text = asr.transcribe(audio)
        self.assertEqual(text, "")

    def test_transcribe_empty_audio(self):
        from voice.cascaded.asr_dashscope import DashScopeASR
        asr = DashScopeASR.__new__(DashScopeASR)
        asr._model = "test"
        text = asr.transcribe(np.array([], dtype=np.int16))
        self.assertEqual(text, "")

    @patch("voice.cascaded.asr_dashscope.Recognition")
    def test_transcribe_sdk_error(self, MockRecognition):
        mock_instance = MagicMock()
        mock_instance.call.side_effect = RuntimeError("API error")
        MockRecognition.return_value = mock_instance

        from voice.cascaded.asr_dashscope import DashScopeASR
        asr = DashScopeASR()

        audio = np.zeros(1600, dtype=np.int16)
        text = asr.transcribe(audio)
        self.assertEqual(text, "")


if __name__ == "__main__":
    unittest.main()
