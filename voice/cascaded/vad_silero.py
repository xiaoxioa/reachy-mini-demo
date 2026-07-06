# -*- coding: utf-8 -*-
"""SileroVAD — sherpa-onnx Silero VAD 封装，实现 VADProvider 接口。"""

from __future__ import annotations

import numpy as np
import sherpa_onnx

from voice.pipeline.providers import VADProvider, VADEvent
from voice.state import log


class SileroVAD(VADProvider):
    """基于 sherpa-onnx VadModel（Silero ONNX）的帧级 VAD。

    feed() 接受 int16 或 float32 音频，内部按 window_size 分帧，
    维护 speaking 状态机，返回 speech_start / speech_end 事件。
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        min_silence_ms: int = 300,
        min_speech_ms: int = 250,
    ) -> None:
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = model_path
        config.silero_vad.threshold = threshold
        config.silero_vad.min_silence_duration = min_silence_ms / 1000.0
        config.silero_vad.min_speech_duration = min_speech_ms / 1000.0
        config.sample_rate = 16000
        config.num_threads = 1
        config.provider = "cpu"
        self._model = sherpa_onnx.VadModel.create(config)
        self._window = self._model.window_size
        self._buf = np.array([], dtype=np.float32)
        self._speaking = False
        self._speech_chunks: list[np.ndarray] = []
        log(f"✅ SileroVAD 就绪 (window={self._window}, thr={threshold})")

    def feed(self, pcm_16k_mono: np.ndarray) -> list[VADEvent]:
        audio = pcm_16k_mono.astype(np.float32)
        if audio.size > 0 and np.abs(audio).max() > 1.5:
            audio = audio / 32768.0
        self._buf = np.concatenate([self._buf, audio])

        events: list[VADEvent] = []
        while len(self._buf) >= self._window:
            frame = self._buf[: self._window]
            self._buf = self._buf[self._window :]
            is_speech = self._model.is_speech(frame.tolist())

            if is_speech and not self._speaking:
                self._speaking = True
                self._speech_chunks = [frame.copy()]
                events.append(VADEvent(type="speech_start"))
            elif is_speech and self._speaking:
                self._speech_chunks.append(frame.copy())
            elif not is_speech and self._speaking:
                self._speaking = False
                full_audio = np.concatenate(self._speech_chunks)
                self._speech_chunks = []
                events.append(VADEvent(type="speech_end", audio=full_audio))

        if self._speaking and len(self._buf) > 0:
            pass

        return events

    def reset(self) -> None:
        self._model.reset()
        self._buf = np.array([], dtype=np.float32)
        self._speaking = False
        self._speech_chunks.clear()
