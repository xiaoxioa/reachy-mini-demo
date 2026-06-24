# -*- coding: utf-8 -*-
"""音频 I/O 线程：DOA 传感器轮询 + 播放队列。"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
import urllib.request
from collections import deque

import numpy as np
from reachy_mini import ReachyMini

from voice.config import (
    DOA_URL, DOA_POLL_HZ, SND_WIN_S, SND_MIN_SAMPLES, SND_RESID_MIN,
    DOA_DEBUG, DOA_WIN_S, DOA_MIN_SAMPLES, GATE_SPREAD,
    PLAY_SR, JITTER_S, JITTER_WALL_S,
)
from voice.state import State, log


def _read_doa(opener) -> tuple[float, bool] | None:
    try:
        with opener.open(DOA_URL, timeout=2.0) as r:
            d = json.loads(r.read().decode("utf-8"))
        return math.degrees(float(d["angle"])), bool(d["speech_detected"])
    except Exception:
        return None


def doa_sensor_loop(st: State, stop: threading.Event) -> None:
    """DOA 纯传感器:10Hz 轮询 → 中值窗口 → 置信的视场外残差发布到 st.sound_resid。
    机器人自己说话期间的读数不入窗(防自声/扬声器反射污染)。behavior_loop 消费,不动头。"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if _read_doa(opener) is None:
        log("⚠ DOA 端点不可用,本次无声源转向(其余功能不受影响)")
        return
    log("👂 声源传感器就绪(DOA REST 10Hz)")
    buf: deque[tuple[float, float]] = deque()
    buf2: deque[tuple[float, float]] = deque()

    def _med_iqr(samples):
        a = sorted(samples)
        n = len(a)
        return a[n // 2], a[(3 * n) // 4] - a[n // 4]

    while not stop.is_set():
        time.sleep(1.0 / DOA_POLL_HZ)
        r = _read_doa(opener)
        now = time.monotonic()
        with st.lock:
            robot_speaking = now < st.playback_end_estimate + 0.4
            by = st.body_yaw_deg
        if r is not None and r[1]:
            buf.append((now, r[0]))
            buf2.append((now, r[0]))
            if DOA_DEBUG:
                log(f"🎧 raw={r[0]:+6.0f}° vad=1 speaking={int(robot_speaking)} body_yaw={by:+.0f}° (n={len(buf2)})")

        while buf and now - buf[0][0] > SND_WIN_S:
            buf.popleft()
        if len(buf) >= SND_MIN_SAMPLES:
            med, spread = _med_iqr(a for _, a in buf)
            with st.lock:
                st.sound_resid = 90.0 - med
                st.sound_at = now
                st.sound_spread = spread

        while buf2 and now - buf2[0][0] > DOA_WIN_S:
            buf2.popleft()
        if len(buf2) >= DOA_MIN_SAMPLES:
            med2, iqr2 = _med_iqr(a for _, a in buf2)
            resid2 = 90.0 - med2
            confident = iqr2 < GATE_SPREAD
            with st.lock:
                st.doa_resid_stable = resid2
                st.doa_confident = confident
                st.doa_at = now
            if DOA_DEBUG:
                log(f"🎧→ resid_stable={resid2:+.0f}° IQR={iqr2:.0f}° confident={confident} "
                    f"speaking={int(robot_speaking)} body_yaw={by:+.0f}° n={len(buf2)}")
        else:
            with st.lock:
                st.doa_confident = False


def _fresh_sound(st: State) -> float | None:
    """读取新鲜(<0.6s)且偏离够大(>25°)的声源残差;否则 None。"""
    now = time.monotonic()
    with st.lock:
        if st.sound_resid is None or (now - st.sound_at) > 0.6:
            return None
        return st.sound_resid if abs(st.sound_resid) >= SND_RESID_MIN else None


def player_loop(mini: ReachyMini, st: State, play_q: queue.Queue, stop: threading.Event) -> None:
    def current_gen() -> int:
        with st.lock:
            return st.play_gen

    def push(chunk: np.ndarray) -> None:
        try:
            mini.media.push_audio_sample(chunk)
        except Exception as e:
            log(f"⚠ push_audio_sample 失败: {type(e).__name__}: {e}")
            return
        with st.lock:
            base = max(st.playback_end_estimate, time.monotonic())
            st.playback_end_estimate = base + len(chunk) / PLAY_SR

    buffering = True
    while not stop.is_set():
        try:
            gen, chunk = play_q.get(timeout=0.1)
        except queue.Empty:
            buffering = True
            continue
        if gen != current_gen():
            continue
        if buffering:
            stash = [(gen, chunk)]
            dur = len(chunk) / PLAY_SR
            t_start = time.monotonic()
            while dur < JITTER_S and time.monotonic() - t_start < JITTER_WALL_S:
                try:
                    g2, c2 = play_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if g2 != current_gen():
                    continue
                stash.append((g2, c2))
                dur += len(c2) / PLAY_SR
            g_now = current_gen()
            valid = [c for g, c in stash if g == g_now]
            if not valid:
                continue
            for c in valid:
                push(c)
            buffering = False
        else:
            push(chunk)
