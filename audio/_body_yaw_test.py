# -*- coding: utf-8 -*-
"""body_yaw 单独验证:身体到底转不转?(用户肉眼确认 + 读回 daemon 状态)

依次:身体左转 30°(0.52rad)→ 停 3s → 右转 30° → 停 3s → 回 0。
每步读 GET /api/state/full 回报的 body_yaw 实际值。
运行:$env:PYTHONUTF8=1; python audio\\_body_yaw_test.py
"""

import json
import math
import os
import time
import urllib.request

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import numpy as np
from reachy_mini import ReachyMini

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def state_body_yaw() -> str:
    try:
        with OPENER.open("http://127.0.0.1:8000/api/state/full", timeout=2.0) as r:
            d = json.loads(r.read().decode("utf-8"))
        # 尽量多挖几个可能的字段
        keys = {k: d[k] for k in d if "yaw" in k.lower() or "body" in k.lower()}
        return json.dumps(keys, ensure_ascii=False, default=str)[:300]
    except Exception as e:
        return f"读取失败 {type(e).__name__}"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


print("=== body_yaw 单独验证 ===", flush=True)
with ReachyMini(
    connection_mode="localhost_only",
    media_backend="no_media",
    automatic_body_yaw=False,
) as mini:
    try:
        mini.goto_target(np.eye(4), duration=1.0, body_yaw=0.0)
        time.sleep(0.5)
        log(f"初始状态:{state_body_yaw()}")

        log("→ 指令:身体左转 30°(body_yaw=+0.524)")
        mini.goto_target(np.eye(4), duration=1.5, body_yaw=math.radians(30))
        time.sleep(0.5)
        log(f"状态:{state_body_yaw()}")
        log("【请看:身体/头整体有没有左转约30°?】")
        time.sleep(3.0)

        log("→ 指令:身体右转 30°(body_yaw=-0.524)")
        mini.goto_target(np.eye(4), duration=2.0, body_yaw=math.radians(-30))
        time.sleep(0.5)
        log(f"状态:{state_body_yaw()}")
        log("【请看:身体有没有转到右边30°?】")
        time.sleep(3.0)

        log("→ 回 0")
        mini.goto_target(np.eye(4), duration=1.5, body_yaw=0.0)
        time.sleep(0.5)
        log(f"状态:{state_body_yaw()}")
    finally:
        try:
            mini.goto_target(np.eye(4), duration=1.0, body_yaw=0.0)
            mini.set_automatic_body_yaw(True)
        except Exception:
            pass
print("=== 完成 ===", flush=True)
