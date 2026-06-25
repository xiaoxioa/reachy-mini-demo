# Memory

> Chronological action log. Old sessions archived as one-line summaries.

## Archived Sessions (2026-06-23 ~ 2026-06-24)

| Date | Summary | Key Files | ~Tokens |
|------|---------|-----------|---------|
| 06-23 16:00 | YuNet 后端切换; vision_worker 双后端支持; debug_server float32 JSON 修复; forget_fact 工具; 唤醒方向门控修复 | perception/vision_worker.py, voice/config.py, start_mac.sh, memory/manager.py, voice/d01_realtime_chat.py | ~40k |
| 06-24 09:00 | Face DB 碎片化修复(质心匹配+auto_merge); GestureRecognizer 替换规则; 认主机制; 记忆注入 update_session 修复 | identity/recognizer.py, memory/manager.py, identity/owner.py, voice/d01_realtime_chat.py | ~50k |
| 06-24 11:00 | 唤醒优先级(a_active替代_is_A); TRACKING 身体跟随; 人脸误识别迟滞; clear_memory confirmed 守卫; debug_server 身份标注 | voice/state.py, voice/d01_realtime_chat.py, voice/config.py, voice/debug_server.py, memory/manager.py | ~70k |
| 06-24 15:00 | 安全删除工作流(多步验证+备份); Dashboard log smart-scroll; test_realtime_model.py; face_db 迁移 shutil.copy2 | voice/d01_realtime_chat.py, voice/debug_server.py, test_realtime_model.py, identity/recognizer.py | ~55k |
| 06-24 17:00 | 分人对话摘要(per-pid conv_log); 音频闸门(切人+DOA偏移); CONV_SUMMARY_THRESHOLD 自动摘要; display_transcript 持久记录本; Dashboard 上下文调试 | voice/state.py, voice/d01_realtime_chat.py, memory/manager.py, voice/config.py, voice/debug_server.py | ~60k |
| 06-24 18:00 | 多人脸 DOA 选人(_select_face_by_doa); all_faces 输出; debug overlay 多人框渲染; 中文渲染 PIL 修复 | perception/vision_worker.py, voice/d01_realtime_chat.py, voice/debug_server.py | ~60k |

## Session: 2026-06-25

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 10:30 | d01 瘦身重构方案(领域驱动) — plan 编写 | flickering-brewing-puppy.md | 拆出 kws.py/fusion.py/safety.py/realtime.py | ~3k |
| 11:56 | 创建 4 个新模块 | voice/kws.py, perception/fusion.py, memory/safety.py, voice/realtime.py | 领域模块就位 | ~10k |
| 12:01 | d01 瘦身: 删除已拆出的代码(~763行) | voice/d01_realtime_chat.py | 编译通过 | ~5k |
| 14:17 | 方向门控白名单化: state==TRACKING时才关门 | voice/d01_realtime_chat.py, .wolf/cerebrum.md | 防止新状态默认被关门 | ~2k |
| 14:50 | fix: 唤醒竞态 — wake_ok 延迟到 set_state(ENGAGING) 之后清除 | voice/d01_realtime_chat.py | 消除 audio/behavior 竞态窗口,无需超时阈值 | ~300 |
| 14:50 | fix: conv=None 时 KWS 命中可重连 WS | voice/d01_realtime_chat.py | WS 断连后不再卡死在对话态 | ~200 |
| 14:43 | Edited voice/d01_realtime_chat.py | 33→36 lines | ~530 |
| 14:44 | Created todo.md | — | ~494 |
| 14:45 | Created PROJECT_STATE.md | — | ~426 |
| 14:46 | Session end: 11 writes across 3 files (d01_realtime_chat.py, todo.md, PROJECT_STATE.md) | 6 reads | ~30463 tok |
| 14:47 | Session end: 11 writes across 3 files (d01_realtime_chat.py, todo.md, PROJECT_STATE.md) | 6 reads | ~30463 tok |
| 14:48 | Edited todo.md | expanded (+9 lines) | ~58 |
| 14:48 | Session end: 12 writes across 3 files (d01_realtime_chat.py, todo.md, PROJECT_STATE.md) | 6 reads | ~28419 tok |
| 14:51 | Edited todo.md | — | ~0 |
| 14:51 | Edited todo.md | expanded (+12 lines) | ~139 |
| 14:52 | Session end: 14 writes across 3 files (d01_realtime_chat.py, todo.md, PROJECT_STATE.md) | 6 reads | ~28568 tok |

## Session: 2026-06-25 14:59

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|

## Session: 2026-06-25 15:28

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|

## Session: 2026-06-25 15:29

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 15:30 | Edited todo.md | expanded (+16 lines) | ~146 |
| 15:30 | Session end: 1 writes across 1 files (todo.md) | 2 reads | ~7334 tok |
| 15:39 | Edited voice/realtime.py | modified open_session() | ~146 |
| 15:39 | Edited voice/debug_server.py | 4→4 lines | ~72 |
| 15:39 | Edited voice/d01_realtime_chat.py | 2→3 lines | ~38 |
| 15:39 | Edited voice/d01_realtime_chat.py | 5→8 lines | ~90 |
| 16:00 | fix: vis_ready日志刷屏→仅首次log | voice/d01_realtime_chat.py | 20+行→1行 | ~200 |
| 16:00 | fix: Dashboard角色混淆→assistant不显示人名 | voice/debug_server.py | 🔊无[名]标签 | ~100 |
| 16:00 | fix: WS快速重连死→open_session 1s最小间隔 | voice/realtime.py | 防服务端限流 | ~200 |
| 16:01 | buglog: bug-048/049/050 | .wolf/buglog.json | 3个bug记录 | ~100 |
| 15:41 | Session end: 5 writes across 4 files (todo.md, realtime.py, debug_server.py, d01_realtime_chat.py) | 5 reads | ~52757 tok |
| 15:48 | Edited voice/debug_server.py | 2→2 lines | ~44 |
| 15:49 | Session end: 6 writes across 4 files (todo.md, realtime.py, debug_server.py, d01_realtime_chat.py) | 5 reads | ~52801 tok |
| 15:53 | Session end: 6 writes across 4 files (todo.md, realtime.py, debug_server.py, d01_realtime_chat.py) | 6 reads | ~52801 tok |
| 15:57 | Session end: 6 writes across 4 files (todo.md, realtime.py, debug_server.py, d01_realtime_chat.py) | 6 reads | ~52849 tok |
