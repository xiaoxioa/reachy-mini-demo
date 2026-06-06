# -*- coding: utf-8 -*-
"""SOUND-TURN 诊断:区分"转不到位"的根因(用户全程站定一个位置持续说话)。

三段式(60s):
  A 0–15s  静止观测:机器人不动,采 DOA → 稳定性(中位/std/双峰占比)
  B 15–30s 开环单转:按 A 段中位角转一次 → 之后只测不转 → 残差(应≈90°)
  C 30–60s 闭环逐步逼近:残差中位 ≥8° 就微调(每次最多 30°)→ 看收敛轨迹

全程打印每个决策因子:窗口样本数/散布、阈值拦截、限幅截断、转后残差。
运行:$env:PYTHONUTF8=1; python audio\\_doa_diag.py
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

HEAD_YAW_MAX = 25.0
BODY_YAW_MAX = 65.0
T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[t+{time.monotonic() - T0:6.2f}s] {msg}", flush=True)


INIT_HEAD_POSE = np.eye(4)
INIT_ANTENNAS = [-0.1745, 0.1745]


def head_pose(yaw_deg: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", [0.0, 0.0, yaw_deg], degrees=True).as_matrix()
    return T


def read_doa() -> tuple[float, bool] | None:
    try:
        with OPENER.open(URL, timeout=2.0) as r:
            d = json.loads(r.read().decode("utf-8"))
        return math.degrees(float(d["angle"])), bool(d["speech_detected"])
    except Exception:
        return None


def collect(seconds: float, tag: str) -> list[float]:
    """采样 seconds 秒,返回有声样本角度列表;每条都打印。"""
    out: list[float] = []
    t_end = time.monotonic() + seconds
    n = 0
    while time.monotonic() < t_end:
        r = read_doa()
        if r is not None:
            n += 1
            deg, sp = r
            if sp:
                out.append(deg)
            print(f"  [{tag}] angle={deg:6.1f}° speech={'YES' if sp else 'no '}", flush=True)
        time.sleep(0.1)
    log(f"〔{tag}〕{seconds:.0f}s 采样:总 {n},有声 {len(out)}({100 * len(out) / max(1, n):.0f}%)")
    return out


def stats(a: list[float]) -> str:
    if not a:
        return "无有声样本"
    s = sorted(a)
    med = s[len(s) // 2]
    near = sum(1 for x in a if abs(x - med) <= 10)
    return (f"中位 {med:.1f}°|均值 {np.mean(a):.1f}°|std {np.std(a):.1f}°|"
            f"范围 [{s[0]:.0f},{s[-1]:.0f}]|中位±10°内占比 {100 * near / len(a):.0f}%")


def turn_to(mini: ReachyMini, target: float, cur: float) -> float:
    """转到世界系 target(度),返回实际(含截断)朝向。打印限幅情况。"""
    raw = target
    target = float(np.clip(target, -90.0, 90.0))
    if abs(raw - target) > 0.5:
        log(f"✂ 目标 {raw:+.0f}° 超范围,截断到 {target:+.0f}°")
    head = float(np.clip(target, -HEAD_YAW_MAX, HEAD_YAW_MAX))
    body = float(np.clip(target - head, -BODY_YAW_MAX, BODY_YAW_MAX))
    dur = 0.4 + 0.008 * abs(target - cur)
    log(f"🤖 转向 {target:+.0f}°(头 {head:+.0f}° + 身体 {body:+.0f}°,{dur:.1f}s)")
    mini.goto_target(head_pose(yaw_deg=head), duration=dur, body_yaw=math.radians(body))
    time.sleep(0.3)  # 让机械稳定
    return target


def main() -> int:
    print("=== SOUND-TURN 根因诊断(用户全程站定 + 持续说话)===", flush=True)
    if read_doa() is None:
        log("❌ /api/state/doa 不可用,中止")
        return 1

    with ReachyMini(
        connection_mode="localhost_only",
        media_backend="no_media",
        automatic_body_yaw=False,
    ) as mini:
        try:
            mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.0, body_yaw=0.0)
            time.sleep(1.0)
            cur = 0.0
            log("READY_FOR_DIAG(请站定一个位置,持续说话 60s 不要移动)")

            # ── A:静止观测 15s ──
            log("====== A 段:静止观测(机器人不动)======")
            a = collect(15.0, "A")
            log(f"A 段稳定性:{stats(a)}")
            if not a:
                log("❌ A 段无有声样本,无法继续(太远/太小声)")
                return 1
            sa = sorted(a)
            med_a = sa[len(sa) // 2]

            # ── B:开环单转 + 残差观测 ──
            log("====== B 段:开环转一次 → 只测不转 ======")
            target = cur + (90.0 - med_a)
            cur = turn_to(mini, target, cur)
            b = collect(12.0, "B")
            log(f"B 段残差(对准时应≈90°):{stats(b)}")

            # ── C:闭环逐步逼近 ──
            log("====== C 段:闭环微调(残差中位≥8° 就修)======")
            for i in range(6):
                c = collect(4.0, f"C{i}")
                if not c:
                    log(f"C{i}:无有声样本,跳过")
                    continue
                sc = sorted(c)
                med_c = sc[len(sc) // 2]
                resid = 90.0 - med_c   # 正=声源在阵列左侧
                log(f"C{i}:中位 {med_c:.1f}° → 残差 {resid:+.1f}°|{stats(c)}")
                if abs(resid) < 8.0:
                    log(f"✅ C{i}:残差 {resid:+.1f}° < 8°,已对准,停止微调")
                    break
                step = float(np.clip(resid, -30.0, 30.0))
                cur = turn_to(mini, cur + step, cur)
            log(f"最终朝向:{cur:+.1f}°")

        except KeyboardInterrupt:
            log("收到 Ctrl+C,提前结束")
        finally:
            try:
                mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.0, body_yaw=0.0)
                time.sleep(1.0)
                mini.set_automatic_body_yaw(True)
            except Exception:
                pass
    print("=== 诊断完成 ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
