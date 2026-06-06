# -*- coding: utf-8 -*-
"""DOA 参考系测定:麦克风阵列跟头转?跟身体转?还是固定底座?

用户站定斜侧方(±45°附近)持续说话,机器人按脚本依次:
  S0 全中立 8s → S1 只转头 +20° 8s → S2 头回 0 8s → S3 只转身体 +30° 8s → S4 回 0 8s
比较各段 DOA 中位:
  跟头转   → S1 相对 S0 偏 ~+20°
  跟身体转 → S3 相对 S0 偏 ~+30°
  都不偏   → 阵列在固定底座
运行:$env:PYTHONUTF8=1; python audio\\_doa_frame_test.py
"""

import json
import math
import os
import sys
import time
import urllib.request

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
URL = "http://127.0.0.1:8000/api/state/doa"
T0 = time.monotonic()


def log(m: str) -> None:
    print(f"[t+{time.monotonic() - T0:6.2f}s] {m}", flush=True)


def head_pose(yaw_deg: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", [0.0, 0.0, yaw_deg], degrees=True).as_matrix()
    return T


def collect(seconds: float, tag: str) -> list[float]:
    out: list[float] = []
    n = 0
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        try:
            with OPENER.open(URL, timeout=2.0) as r:
                d = json.loads(r.read().decode("utf-8"))
            n += 1
            if d["speech_detected"]:
                out.append(math.degrees(float(d["angle"])))
        except Exception:
            pass
        time.sleep(0.1)
    med = sorted(out)[len(out) // 2] if out else float("nan")
    log(f"〔{tag}〕有声 {len(out)}/{n},DOA 中位 {med:.1f}°")
    return out


def med(a: list[float]) -> float:
    return sorted(a)[len(a) // 2] if a else float("nan")


print("=== DOA 参考系测定(站定斜侧方持续说话,机器人自己动)===", flush=True)
with ReachyMini(
    connection_mode="localhost_only",
    media_backend="no_media",
    automatic_body_yaw=False,
) as mini:
    try:
        mini.goto_target(np.eye(4), duration=1.0, body_yaw=0.0)
        time.sleep(1.0)
        log("READY_FOR_FRAME(请在斜侧方站定,持续说话 ~45s)")

        s0 = collect(8.0, "S0 全中立")
        log("→ 只转头 +20°")
        mini.goto_target(head_pose(20.0), duration=0.8, body_yaw=0.0)
        s1 = collect(8.0, "S1 头+20°")
        log("→ 头回 0")
        mini.goto_target(head_pose(0.0), duration=0.8, body_yaw=0.0)
        s2 = collect(8.0, "S2 头回0")
        log("→ 只转身体 +30°")
        mini.goto_target(head_pose(0.0), duration=1.2, body_yaw=math.radians(30))
        s3 = collect(8.0, "S3 身体+30°")
        log("→ 全部回 0")
        mini.goto_target(head_pose(0.0), duration=1.2, body_yaw=0.0)
        s4 = collect(8.0, "S4 回0")

        print("\n========== 判定 ==========", flush=True)
        base = np.nanmean([med(s0), med(s2), med(s4)])
        d_head = med(s1) - base
        d_body = med(s3) - base
        print(f"中立基准(S0/S2/S4 平均):{base:.1f}°", flush=True)
        print(f"头 +20° 时偏移:{d_head:+.1f}°(跟头转应≈+20°)", flush=True)
        print(f"身体 +30° 时偏移:{d_body:+.1f}°(跟身体转应≈+30°)", flush=True)
        verdict = []
        verdict.append(f"头:{'跟转' if abs(d_head) > 10 else '不跟转'}")
        verdict.append(f"身体:{'跟转' if abs(d_body) > 12 else '不跟转'}")
        print("判定:" + ",".join(verdict), flush=True)
    finally:
        try:
            mini.goto_target(np.eye(4), duration=1.0, body_yaw=0.0)
            time.sleep(1.0)
            mini.set_automatic_body_yaw(True)
        except Exception:
            pass
print("=== 完成 ===", flush=True)
