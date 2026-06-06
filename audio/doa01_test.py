# -*- coding: utf-8 -*-
"""DOA-01:听声辨位实测(ReSpeaker XVF3800 板载 DOA,经 daemon REST)。

链路:XVF3800 4 麦线性阵列板载 DOA → daemon AudioDoA(USB 控制传输,独占)
  → GET /api/state/doa → {"angle": rad, "speech_detected": bool}

角度约定(SDK audio_doa.py):0 rad = 左,π/2 = 前/后,π = 右。
线性阵列(mic0 右天线侧 … mic3 左天线侧)→ 只有 180° 半平面,前后不分。

本测试只走 REST,不 acquire 媒体、不连 ReachyMini 客户端:
  与对话脚本可并存,也避开 daemon 媒体重取坑。

协议(70s,固定时序,用户配合在不同方位说话):
  t  0–10s 安静基线
  t 10–25s 在机器人【左侧】说话
  t 25–40s 在机器人【正前方】说话
  t 40–55s 在机器人【右侧】说话
  t 55–70s 在机器人【正后方】说话(验证前后歧义:预期读数 ≈ 前方)

运行:$env:PYTHONUTF8=1; python audio\\doa01_test.py
"""

import json
import math
import os
import sys
import time
import urllib.request

# 本机代理直通会劫持 localhost(老坑):走无代理 opener
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

URL = "http://127.0.0.1:8000/api/state/doa"
POLL_HZ = 10.0
PHASES = [
    ("BASELINE_QUIET", 0.0, 10.0),
    ("LEFT", 10.0, 25.0),
    ("FRONT", 25.0, 40.0),
    ("RIGHT", 40.0, 55.0),
    ("BACK", 55.0, 70.0),
]
# 期望角(度):0=左,90=前/后,180=右
EXPECT = {"LEFT": 0.0, "FRONT": 90.0, "RIGHT": 180.0, "BACK": 90.0}

T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[t+{time.monotonic() - T0:6.2f}s] {msg}", flush=True)


def read_doa() -> tuple[float, bool] | None:
    try:
        with OPENER.open(URL, timeout=2.0) as r:
            d = json.loads(r.read().decode("utf-8"))
        if d is None:
            return None
        return math.degrees(float(d["angle"])), bool(d["speech_detected"])
    except Exception as e:
        log(f"⚠ REST 读取失败:{type(e).__name__}: {e}")
        return None


def main() -> int:
    print("=== DOA-01:听声辨位实测(XVF3800 板载 DOA / daemon REST)===", flush=True)
    probe = read_doa()
    if probe is None:
        log("❌ /api/state/doa 不可用(daemon 没跑或无 DoA 设备),中止")
        return 1
    log(f"✅ 端点就绪:angle={probe[0]:.1f}° speech={probe[1]}")
    log("READY_FOR_DOA(开始 65s 分相位采集)")

    samples: list[tuple[float, str, float, bool]] = []  # (t, phase, deg, speech)
    cur_phase = ""
    t_start = time.monotonic()
    while True:
        t = time.monotonic() - t_start
        if t >= PHASES[-1][2]:
            break
        phase = next((n for n, a, b in PHASES if a <= t < b), "?")
        if phase != cur_phase:
            cur_phase = phase
            log(f"────── 相位切换 → {phase} ──────")
        r = read_doa()
        if r is not None:
            deg, speech = r
            samples.append((t, phase, deg, speech))
            log(f"{phase:14s} angle={deg:6.1f}°  speech={'YES' if speech else 'no '}")
        time.sleep(1.0 / POLL_HZ)

    # ── 汇总 ──
    print("\n========== 汇总(只统计 speech_detected=True 的样本)==========", flush=True)
    print(f"{'相位':14s} {'样本':>5s} {'有声':>5s} {'中位角':>8s} {'均值':>8s} {'范围':>16s} {'期望':>6s}", flush=True)
    ok_phases = 0
    for name, a, b in PHASES:
        ph = [s for s in samples if s[1] == name]
        sp = sorted(s[2] for s in ph if s[3])
        if sp:
            med = sp[len(sp) // 2]
            mean = sum(sp) / len(sp)
            rng = f"[{sp[0]:.0f}°,{sp[-1]:.0f}°]"
            exp = EXPECT.get(name)
            mark = ""
            if exp is not None:
                err = abs(med - exp)
                mark = f"{exp:.0f}°" + ("✓" if err <= 30 else "✗")
                if err <= 30:
                    ok_phases += 1
            print(f"{name:14s} {len(ph):5d} {len(sp):5d} {med:7.1f}° {mean:7.1f}° {rng:>16s} {mark:>7s}", flush=True)
        else:
            print(f"{name:14s} {len(ph):5d} {0:5d} {'—':>8s} {'—':>8s} {'—':>16s}", flush=True)

    n_speech = sum(1 for s in samples if s[3])
    print(f"\n总样本 {len(samples)},有声样本 {n_speech};方向相位达标 {ok_phases}/4(中位角与期望差 ≤30°)", flush=True)
    print(f"=== {'DOA 可用' if ok_phases >= 3 else 'DOA 表现不及预期,看明细'} ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
