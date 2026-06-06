# -*- coding: utf-8 -*-
"""DOA 单方位采样器:python _doa_sample.py <标签> [秒数=12]

用户已就位并持续说话时运行;10Hz 轮询 /api/state/doa,
只统计 speech_detected=True 样本,打印每条读数 + 汇总。
角度约定(SDK 文档):0=左,π/2(90°)=前/后,π(180°)=右。
"""

import json
import math
import sys
import time
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
URL = "http://127.0.0.1:8000/api/state/doa"

label = sys.argv[1] if len(sys.argv) > 1 else "UNLABELED"
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0

print(f"=== DOA 采样:{label}({secs:.0f}s)===", flush=True)
t0 = time.monotonic()
speech_angles: list[float] = []
n = 0
while time.monotonic() - t0 < secs:
    try:
        with OPENER.open(URL, timeout=2.0) as r:
            d = json.loads(r.read().decode("utf-8"))
        deg = math.degrees(float(d["angle"]))
        sp = bool(d["speech_detected"])
        n += 1
        if sp:
            speech_angles.append(deg)
        print(f"[{time.monotonic() - t0:5.1f}s] angle={deg:6.1f}°  speech={'YES' if sp else 'no '}", flush=True)
    except Exception as e:
        print(f"⚠ {type(e).__name__}: {e}", flush=True)
    time.sleep(0.1)

print(f"\n—— {label} 汇总 ——", flush=True)
if speech_angles:
    s = sorted(speech_angles)
    print(f"总样本 {n},有声 {len(speech_angles)}({100 * len(speech_angles) / max(1, n):.0f}%)", flush=True)
    print(f"中位角 {s[len(s) // 2]:.1f}°|均值 {sum(s) / len(s):.1f}°|范围 [{s[0]:.0f}°,{s[-1]:.0f}°]", flush=True)
else:
    print(f"总样本 {n},有声 0 —— VAD 未触发(太远/太小声?)", flush=True)
