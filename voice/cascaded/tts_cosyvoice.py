# -*- coding: utf-8 -*-
"""CosyVoiceTTS — DashScope CosyVoice2 流式 TTS，实现 TTSProvider 接口。"""

from __future__ import annotations

import queue
import threading
from typing import Iterator

import numpy as np
from dashscope.audio.tts_v2 import AudioFormat, ResultCallback, SpeechSynthesizer

from voice.pipeline.providers import TTSProvider
from voice.state import log


class CosyVoiceTTS(TTSProvider):
    """基于 DashScope CosyVoice2 的流式 TTS。

    synthesize_stream() 使用回调模式收集 PCM 音频块，
    yield int16 @ 16kHz（CascadedPipeline 负责转 float32）。
    """

    def __init__(
        self,
        model: str = "cosyvoice-v2",
        default_voice: str = "longxiaochun",
    ) -> None:
        self._model = model
        self._default_voice = default_voice
        self._cancel = threading.Event()

    def synthesize_stream(
        self, text: str, voice: str = "default"
    ) -> Iterator[np.ndarray]:
        if not text.strip():
            return
        self._cancel.clear()
        audio_q: queue.Queue[bytes | None] = queue.Queue()

        class _Cb(ResultCallback):
            def on_data(_, data: bytes) -> None:
                audio_q.put(data)

            def on_complete(_) -> None:
                audio_q.put(None)

            def on_error(_, message) -> None:
                log(f"❌ CosyVoice TTS error: {message}")
                audio_q.put(None)

        v = voice if voice != "default" else self._default_voice
        synth = SpeechSynthesizer(
            model=self._model,
            voice=v,
            format=AudioFormat.PCM_16000HZ_MONO_16BIT,
            callback=_Cb(),
        )

        t = threading.Thread(
            target=self._run_synth, args=(synth, text), daemon=True
        )
        t.start()

        while True:
            try:
                data = audio_q.get(timeout=30)
            except queue.Empty:
                break
            if data is None or self._cancel.is_set():
                break
            pcm = np.frombuffer(data, dtype=np.int16)
            if pcm.size > 0:
                yield pcm

    @staticmethod
    def _run_synth(synth: SpeechSynthesizer, text: str) -> None:
        try:
            synth.call(text)
        except Exception as e:
            log(f"❌ CosyVoice synth.call 异常: {type(e).__name__}: {e}")

    def cancel(self) -> None:
        self._cancel.set()
