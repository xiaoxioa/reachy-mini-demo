# -*- coding: utf-8 -*-
"""SOUND-TURN-01:听声转头(独立 demo,DOA → 转头/转身闭环,不接对话)。

链路:daemon REST /api/state/doa(10Hz 轮询,XVF3800 板载 DOA + VAD)
  → 0.5s 中值滤波(只取 speech_detected=True 样本,压反射双峰)
  → DOA 角 → 目标朝向角 → 头部 yaw(±25°)+ 余量给 body_yaw → goto_target 转过去

角度映射(DOA-01 实测标定,CALIBRATION §11):
  DOA:0°=阵列左,90°=阵列正前/正后,180°=阵列右(线性阵列,前后不分)
  机器人:yaw+ = 看左,body_yaw+ = 身体左转(弧度)
  ⭐ 阵列长在头上(mic0 靠右天线),DOA 是【相对头当前世界朝向】的偏角:
  → 目标朝向(世界系)= 当前朝向 + (90° − DOA角)
  (首版把 DOA 当世界角,转完读数跟着变 → 来回扭振荡,已修)
  ⭐⭐ head pose 是【世界系】姿态(参考系实测 2026-06-06):身体转动会被
  Stewart 反向补偿,头世界朝向不变 → head 必须直接给完整目标角,
  body_yaw 只是分担量(保证颈相对量 ≤25°)。首版给"头25°+身体余量"
  导致永远最多转 25° = "转不到位"的真正根因。

闭环逐步逼近(根因诊断 2026-06-06:开环单转残差 ~13° 且无核对 = "不到位"主因):
  - 首转:窗口 ≥5 个有声样本、中值滤波 → 大转向目标
  - 锁定:之后只要还在说话,持续核对残差(90° − DOA 中位),
    |残差| ≥ 8° 就微调(单步 ≤30°)→ 实测 0~1 次微调即对准
  - DOA 静止噪声 std ~10°、15% 反射离群(诊断 A 段)→ 中值 + 样本下限压制

已知限制(物理,接受):
  - 前后不分:正后方说话读数 ≈90° → 它转向正前(预期内现象)
  - 反射双峰:中值滤波压制

运行:$env:PYTHONUTF8=1; python audio\\sound_turn.py [秒数=90]
"""

import json
import math
import os
import sys
import time
import urllib.request
from collections import deque

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
URL = "http://127.0.0.1:8000/api/state/doa"

POLL_HZ = 10.0
MEDIAN_WIN_S = 1.5        # 中值滤波窗口(VAD 触发率实测仅 11~57%,窗口要够长才凑得齐样本)
MIN_SPEECH_IN_WIN = 5     # 窗口内至少几个有声样本才出方向(3 太少双峰漏过;1.5s×10Hz 下 5≈33% 触发率)
TURN_THRESHOLD_DEG = 12.0  # 首转:目标与当前差超过此值才转
TRIM_THRESHOLD_DEG = 8.0  # 闭环微调:残差超过此值就修(诊断 C 段验证)
TURN_COOLDOWN_S = 1.2     # 两次动作最小间隔(让动作做完 + 新窗口积累)
HEAD_YAW_MAX = 25.0       # 头部承担的最大 yaw(CALIBRATION 跟随限幅)
BODY_YAW_MAX = 65.0       # 身体承担的最大 yaw(总计可达 ±90°)
TARGET_LIMIT_DEG = 90.0   # 世界系目标朝向限幅
TURN_DURATION_MIN = 0.4
TURN_DURATION_PER_DEG = 0.008  # 转得越多越慢:90° ≈ 1.1s

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


def main() -> int:
    run_seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0

    print("=== SOUND-TURN-01:听声转头(DOA → 头+身体)===", flush=True)
    if read_doa() is None:
        log("❌ /api/state/doa 不可用,中止")
        return 1
    log("✅ DOA 端点就绪")

    # (t, deg) 只存有声样本;窗口按时间裁剪
    speech_buf: deque[tuple[float, float]] = deque()
    cur_heading = 0.0      # 当前朝向(度,0=正前,+=左)
    last_turn_t = 0.0
    n_turns = 0

    log("连接 Reachy Mini(no_media,只动头/身)…")
    with ReachyMini(
        connection_mode="localhost_only",
        media_backend="no_media",
        automatic_body_yaw=False,
    ) as mini:
        try:
            mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.0, body_yaw=0.0)
            time.sleep(1.0)
            log(f"READY_FOR_SOUND(开始 {run_seconds:.0f}s:在左/前/右/后说话,它会转向你)")

            t_end = time.monotonic() + run_seconds
            while time.monotonic() < t_end:
                now = time.monotonic()
                r = read_doa()
                if r is not None:
                    deg, speech = r
                    if speech:
                        speech_buf.append((now, deg))
                # 裁剪窗口
                while speech_buf and now - speech_buf[0][0] > MEDIAN_WIN_S:
                    speech_buf.popleft()

                if len(speech_buf) >= MIN_SPEECH_IN_WIN and now - last_turn_t >= TURN_COOLDOWN_S:
                    angles = sorted(a for _, a in speech_buf)
                    med = angles[len(angles) // 2]           # 中值滤波:压反射双峰
                    resid = 90.0 - med                        # 残差:对准时应≈0
                    # DOA 是相对阵列(头)的偏角 → 世界系目标 = 当前朝向 + 残差
                    target = float(np.clip(cur_heading + resid,
                                           -TARGET_LIMIT_DEG, TARGET_LIMIT_DEG))
                    delta = target - cur_heading
                    # 闭环:首转 12° 阈值;之后残差 ≥8° 就修(诊断:0~1 次微调即对准)
                    thresh = TURN_THRESHOLD_DEG if n_turns == 0 else TRIM_THRESHOLD_DEG
                    if abs(delta) >= thresh:
                        new_heading = target
                        # ⭐ head pose 是世界系姿态(实测:身体转动被 Stewart 反向补偿,
                        #   头世界朝向不变)→ head 直接给完整目标角;
                        #   body_yaw 只是分担量,保证 Stewart 相对量(头−身)≤25° 不顶限
                        neck_rel = float(np.clip(new_heading, -HEAD_YAW_MAX, HEAD_YAW_MAX))
                        body_yaw_deg = float(np.clip(new_heading - neck_rel, -BODY_YAW_MAX, BODY_YAW_MAX))
                        dur = TURN_DURATION_MIN + TURN_DURATION_PER_DEG * abs(delta)
                        kind = "转向" if abs(delta) >= TURN_THRESHOLD_DEG else "微调"
                        log(f"🔊 检测到说话 @ DOA {med:.0f}°(窗口 {len(angles)} 样本,"
                            f"范围 {angles[0]:.0f}~{angles[-1]:.0f}°,残差 {resid:+.0f}°)")
                        log(f"🤖 {kind} → 世界朝向 {new_heading:+.0f}°(身体分担 {body_yaw_deg:+.0f}°,颈 {new_heading - body_yaw_deg:+.0f}°,{dur:.1f}s)")
                        try:
                            mini.goto_target(
                                head_pose(yaw_deg=new_heading),   # 世界系完整目标
                                duration=dur,
                                body_yaw=math.radians(body_yaw_deg),
                            )
                            cur_heading = new_heading
                            n_turns += 1
                        except Exception as e:
                            log(f"⚠ 转向失败:{type(e).__name__}: {e}")
                        last_turn_t = time.monotonic()
                        speech_buf.clear()  # 转完重新积累,避免用转身前的旧角度
                time.sleep(1.0 / POLL_HZ)

        except KeyboardInterrupt:
            log("收到 Ctrl+C,提前结束")
        finally:
            try:
                mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.0, body_yaw=0.0)
                time.sleep(1.0)
                mini.set_automatic_body_yaw(True)
            except Exception:
                pass

    print(f"\n========== 汇总 ==========", flush=True)
    print(f"总转向次数:{n_turns}", flush=True)
    print("=== 听声转头闭环完成,准不准请肉眼确认 ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
