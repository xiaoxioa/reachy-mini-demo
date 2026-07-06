# -*- coding: utf-8 -*-
"""DashScopeASR — DashScope Paraformer API ASR，实现 ASRProvider 接口。"""

from __future__ import annotations

import os
import struct
import tempfile

import numpy as np
from dashscope.audio.asr import Recognition

from voice.pipeline.providers import ASRProvider
from voice.state import log


class DashScopeASR(ASRProvider):
    """基于 DashScope Paraformer API 的远程 ASR。

    transcribe() 将 PCM 音频写入临时 WAV 文件，
    调用 Recognition.call() 获取识别结果。
    """

    def __init__(self, model: str = "paraformer-realtime-v2") -> None:
        self._model = model
        log(f"✅ DashScopeASR 就绪 (model={model})")

    def transcribe(self, pcm_16k_mono: np.ndarray) -> str:
        audio = pcm_16k_mono.astype(np.float32)
        if audio.size > 0 and np.abs(audio).max() > 1.5:
            audio = audio / 32768.0
        pcm_int16 = (audio * 32767).astype(np.int16)

        if pcm_int16.size == 0:
            return ""

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        try:
            raw = pcm_int16.tobytes()
            self._write_wav(fd, raw, sample_rate=16000)

            rec = Recognition(
                model=self._model,
                format="wav",
                sample_rate=16000,
                callback=None,
            )
            result = rec.call(tmp_path)

            if result and hasattr(result, "output"):
                sentences = getattr(result.output, "sentence", []) or []
                return "".join(s.get("text", "") for s in sentences).strip()
            return ""
        except Exception as e:
            log(f"❌ DashScopeASR 识别失败: {type(e).__name__}: {e}")
            return ""
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _write_wav(fd: int, raw_pcm: bytes, sample_rate: int = 16000) -> None:
        """Write a minimal WAV header + PCM data to an open file descriptor."""
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(raw_pcm)
        file_size = 36 + data_size

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", file_size, b"WAVE",
            b"fmt ", 16, 1, num_channels,
            sample_rate, byte_rate, block_align, bits_per_sample,
            b"data", data_size,
        )
        os.write(fd, header)
        os.write(fd, raw_pcm)
        os.close(fd)
