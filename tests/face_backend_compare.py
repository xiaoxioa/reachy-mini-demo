# -*- coding: utf-8 -*-
"""MediaPipe vs YuNet 人脸检测精度对比。

用 tests/fixtures/ 下的夹具图，同时跑两个后端，输出对比表。
运行: python tests/face_backend_compare.py
"""
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
MANIFEST_PATH = os.path.join(FIXTURES_DIR, "manifest.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FACE_MODEL = os.path.join(ROOT, "models", "face_landmarker.task")
YUNET_MODEL = os.path.join(ROOT, "models", "face_detection_yunet_2023mar.onnx")


# ── MediaPipe 后端 ──
def mp_detect(rgb, face_lm, ts_ms):
    import mediapipe as mp
    from perception.vision_worker import pick_main_face
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    t0 = time.monotonic()
    res = face_lm.detect_for_video(mp_img, ts_ms)
    ms = (time.monotonic() - t0) * 1000.0
    face = pick_main_face(res)
    n = len(res.face_landmarks) if res.face_landmarks else 0
    return {"hit": face is not None, "n": n, "ms": ms,
            "h": face[2] if face else 0.0,
            "u": face[0] if face else 0.0,
            "v": face[1] if face else 0.0}


# ── YuNet 后端 ──
def yunet_detect(rgb):
    H, W = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    yunet = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (W, H), 0.5, 0.3, 10)
    t0 = time.monotonic()
    _, faces = yunet.detect(bgr)
    ms = (time.monotonic() - t0) * 1000.0
    if faces is None or len(faces) == 0:
        return {"hit": False, "n": 0, "ms": ms, "h": 0.0, "u": 0.0, "v": 0.0, "conf": 0.0}
    # pick largest face by bbox area
    best_idx = 0
    best_area = 0
    for i, f in enumerate(faces):
        area = f[2] * f[3]
        if area > best_area:
            best_area = area
            best_idx = i
    f = faces[best_idx]
    x, y, w, h = f[0], f[1], f[2], f[3]
    conf = f[-1]
    u = (x + w / 2) / W
    v = (y + h / 2) / H
    h_ratio = h / H
    return {"hit": True, "n": len(faces), "ms": ms, "h": h_ratio, "u": u, "v": v, "conf": conf}


# ── YuNet 多阈值测试 ──
def yunet_detect_multi(rgb, thresholds=(0.3, 0.5, 0.65, 0.8)):
    H, W = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    results = {}
    for thr in thresholds:
        yunet = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (W, H), thr, 0.3, 10)
        _, faces = yunet.detect(bgr)
        hit = faces is not None and len(faces) > 0
        n = len(faces) if faces is not None else 0
        conf = 0.0
        if hit:
            conf = max(f[-1] for f in faces)
        results[thr] = {"hit": hit, "n": n, "conf": conf}
    return results


def main():
    if not os.path.isfile(MANIFEST_PATH):
        print(f"❌ 找不到 {MANIFEST_PATH}，先运行 python tests/_vision_capture_fixtures.py")
        return 1

    manifest = json.loads(open(MANIFEST_PATH, encoding="utf-8").read())
    face_cases = [c for c in manifest if c.get("expect", {}).get("face")]

    # init MediaPipe
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    face_lm = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL),
            running_mode=mp_vision.RunningMode.VIDEO, num_faces=2,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3))

    print("=" * 90)
    print(f"{'文件':<28s} {'期望':>4s}  {'MP命中':>6s} {'MP_ms':>6s} {'MP_h':>6s}"
          f"  {'YN命中':>6s} {'YN_ms':>6s} {'YN_h':>6s} {'YN_conf':>7s}")
    print("-" * 90)

    mp_hits = yn_hits = total = 0
    for i, case in enumerate(face_cases):
        fname = case["file"]
        expect = case.get("expect", {})
        fpath = os.path.join(FIXTURES_DIR, fname)
        if not os.path.isfile(fpath):
            print(f"  ⏭️  {fname} — 文件缺失")
            continue

        rgb = np.array(Image.open(fpath).convert("RGB"))
        total += 1

        mp_res = mp_detect(rgb, face_lm, ts_ms=(i + 1) * 100)
        yn_res = yunet_detect(rgb)

        mp_mark = "✅" if mp_res["hit"] else "❌"
        yn_mark = "✅" if yn_res["hit"] else "❌"
        if mp_res["hit"]:
            mp_hits += 1
        if yn_res["hit"]:
            yn_hits += 1

        print(f"{fname:<28s} n≥{expect.get('n_faces',1):d}   "
              f"{mp_mark} n={mp_res['n']:d}  {mp_res['ms']:5.1f}  {mp_res['h']:.3f}"
              f"   {yn_mark} n={yn_res['n']:d}  {yn_res['ms']:5.1f}  {yn_res['h']:.3f}  {yn_res['conf']:.3f}")

    print("=" * 90)
    print(f"人脸用例 {total} 个:")
    print(f"  MediaPipe: {mp_hits}/{total} ({100*mp_hits/max(total,1):.0f}%)")
    print(f"  YuNet:     {yn_hits}/{total} ({100*yn_hits/max(total,1):.0f}%)")

    # YuNet 多阈值扫描
    print(f"\n{'=' * 70}")
    print("YuNet 多阈值对比 (0.3 / 0.5 / 0.65 / 0.8):")
    print(f"{'文件':<28s}  {'thr=0.3':>8s} {'thr=0.5':>8s} {'thr=0.65':>9s} {'thr=0.8':>8s}")
    print("-" * 70)
    thr_hits = {0.3: 0, 0.5: 0, 0.65: 0, 0.8: 0}
    for case in face_cases:
        fname = case["file"]
        fpath = os.path.join(FIXTURES_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        rgb = np.array(Image.open(fpath).convert("RGB"))
        multi = yunet_detect_multi(rgb)
        parts = []
        for thr in (0.3, 0.5, 0.65, 0.8):
            r = multi[thr]
            mark = "✅" if r["hit"] else "❌"
            if r["hit"]:
                thr_hits[thr] += 1
            parts.append(f"{mark} {r['conf']:.2f}")
        print(f"{fname:<28s}  {'  '.join(parts)}")
    print("-" * 70)
    for thr in (0.3, 0.5, 0.65, 0.8):
        print(f"  thr={thr}: {thr_hits[thr]}/{total} ({100*thr_hits[thr]/max(total,1):.0f}%)")

    # 负样本
    neg_cases = [c for c in manifest if not c.get("expect", {}).get("face")]
    if neg_cases:
        print(f"\n{'=' * 70}")
        print("负样本（期望无脸）:")
        for case in neg_cases:
            fname = case["file"]
            fpath = os.path.join(FIXTURES_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            rgb = np.array(Image.open(fpath).convert("RGB"))
            mp_res = mp_detect(rgb, face_lm, ts_ms=90000 + i)
            yn_res = yunet_detect(rgb)
            mp_mark = "✅正确" if not mp_res["hit"] else f"❌误检n={mp_res['n']}"
            yn_mark = "✅正确" if not yn_res["hit"] else f"❌误检n={yn_res['n']} conf={yn_res['conf']:.2f}"
            print(f"  {fname:<24s}  MP: {mp_mark}  YN: {yn_mark}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
