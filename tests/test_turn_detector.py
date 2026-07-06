#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TurnDetector 端点检测 + 打断判定测试。

用例:
  1. 长静音 → respond
  2. 短静音 + 句末标点 → respond
  3. 短静音 + 无标点 → wait
  4. TTS 播放中 + 长文本 → barge_in
  5. TTS 播放中 + 短回应("嗯") → backchannel
  6. TTS 播放中 + 短文本非回应词 → wait
  7. 参数可调
  8. 各种 backchannel 词

运行:
  cd reachy-mini-demo && .venv/bin/python tests/test_turn_detector.py -v
"""

import os
import sys
import unittest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from voice.cascaded.turn_detector import TurnDetector


class TestEndpointDetection(unittest.TestCase):
    def setUp(self):
        self.td = TurnDetector()

    def test_long_silence_respond(self):
        self.assertEqual(self.td.should_respond("你好", 800, False), "respond")
        self.assertEqual(self.td.should_respond("你好", 1200, False), "respond")

    def test_short_silence_with_punctuation_respond(self):
        self.assertEqual(self.td.should_respond("你叫什么？", 400, False), "respond")
        self.assertEqual(self.td.should_respond("很好。", 500, False), "respond")
        self.assertEqual(self.td.should_respond("真的吗！", 450, False), "respond")
        self.assertEqual(self.td.should_respond("然后呢;", 400, False), "respond")

    def test_short_silence_no_punctuation_wait(self):
        self.assertEqual(self.td.should_respond("你好", 400, False), "wait")
        self.assertEqual(self.td.should_respond("我想", 300, False), "wait")
        self.assertEqual(self.td.should_respond("嗯", 200, False), "wait")

    def test_very_short_silence_wait(self):
        self.assertEqual(self.td.should_respond("你好吗？", 100, False), "wait")
        self.assertEqual(self.td.should_respond("你好吗？", 399, False), "wait")

    def test_empty_text_long_silence(self):
        self.assertEqual(self.td.should_respond("", 800, False), "respond")
        self.assertEqual(self.td.should_respond("  ", 900, False), "respond")

    def test_empty_text_short_silence(self):
        self.assertEqual(self.td.should_respond("", 400, False), "wait")


class TestBargeIn(unittest.TestCase):
    def setUp(self):
        self.td = TurnDetector()

    def test_long_text_barge_in(self):
        self.assertEqual(self.td.should_respond("等一下我想说", 0, True), "barge_in")
        self.assertEqual(self.td.should_respond("不对不对", 30, True), "barge_in")

    def test_short_backchannel(self):
        self.assertEqual(self.td.should_respond("嗯", 0, True), "backchannel")
        self.assertEqual(self.td.should_respond("哦", 10, True), "backchannel")
        self.assertEqual(self.td.should_respond("好", 20, True), "backchannel")
        self.assertEqual(self.td.should_respond("对", 0, True), "backchannel")
        self.assertEqual(self.td.should_respond("嗯嗯", 0, True), "backchannel")
        self.assertEqual(self.td.should_respond("ok", 0, True), "backchannel")
        self.assertEqual(self.td.should_respond("OK", 0, True), "backchannel")
        self.assertEqual(self.td.should_respond("好的", 0, True), "backchannel")

    def test_short_non_backchannel_wait(self):
        self.assertEqual(self.td.should_respond("停", 0, True), "wait")
        self.assertEqual(self.td.should_respond("等", 0, True), "wait")
        self.assertEqual(self.td.should_respond("不", 0, True), "wait")

    def test_tts_playing_silence_medium_wait(self):
        # TTS 播放中,静音 50-800ms — 不够长触发 respond,barge-in 窗口也过了
        self.assertEqual(self.td.should_respond("等一下我想说", 100, True), "wait")

    def test_tts_playing_silence_long_respond(self):
        # TTS 播放中,静音 >= 800ms — 用户停止说话足够久,切换回应
        self.assertEqual(self.td.should_respond("等一下我想说", 800, True), "respond")

    def test_not_playing_ignores_barge_in(self):
        self.assertEqual(self.td.should_respond("等一下我想说", 0, False), "wait")


class TestCustomParams(unittest.TestCase):
    def test_custom_silence_threshold(self):
        td = TurnDetector(silence_respond_ms=500)
        self.assertEqual(td.should_respond("你好", 500, False), "respond")
        self.assertEqual(td.should_respond("你好", 499, False), "wait")

    def test_custom_barge_in_chars(self):
        td = TurnDetector(barge_in_min_chars=2)
        self.assertEqual(td.should_respond("等一下", 0, True), "barge_in")

    def test_custom_barge_in_silence(self):
        td = TurnDetector(barge_in_max_silence_ms=100)
        # 静音 80ms < 100ms 阈值,仍在 barge-in 窗口内
        self.assertEqual(td.should_respond("等一下我想说", 80, True), "barge_in")
        # 静音 150ms > 100ms,超出 barge-in 窗口,进入 wait(不够 800ms respond)
        self.assertEqual(td.should_respond("等一下我想说", 150, True), "wait")


if __name__ == "__main__":
    unittest.main()
