# -*- coding: utf-8 -*-
"""SherpaASR — sherpa-onnx Paraformer 离线 ASR，实现 ASRProvider 接口。"""

from __future__ import annotations

import numpy as np
import sherpa_onnx

from voice.pipeline.providers import ASRProvider
from voice.state import log


class SherpaASR(ASRProvider):
    """本地 Paraformer ASR（sherpa-onnx OfflineRecognizer）。

    transcribe() 接受 VAD 切出的完整语音段（int16 或 float32），
    返回识别文本。
    """

    def __init__(
        self,
        model_path: str,
        tokens_path: str,
        num_threads: int = 2,
    ) -> None:
        self._rec = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=model_path,
            tokens=tokens_path,
            num_threads=num_threads,
            sample_rate=16000,
            provider="cpu",
        )
        log(f"✅ SherpaASR 就绪 (Paraformer, threads={num_threads})")

    def transcribe(self, pcm_16k_mono: np.ndarray) -> str:
        audio = pcm_16k_mono.astype(np.float32)
        if audio.size > 0 and np.abs(audio).max() > 1.5:
            audio = audio / 32768.0
        stream = self._rec.create_stream()
        stream.accept_waveform(16000, audio)
        self._rec.decode(stream)
        return stream.result.text.strip()
