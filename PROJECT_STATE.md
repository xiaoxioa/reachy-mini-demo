# PROJECT_STATE

## 已完成事项

### 架构
- 6 模块拆分 → d01 瘦身(领域驱动): 拆出 kws.py / fusion.py / safety.py / realtime.py
- 方向门控白名单化(仅 TRACKING 关门)

### 核心特性
- YuNet+ArcFace 身份识别 + auto_merge 碎片修复
- GestureRecognizer 手势(模型优先+规则 fallback)
- 认主机制(OwnerManager) + 记忆权限矩阵
- 记忆注入 update_session 替代 create_item
- 分人对话摘要(per-pid) + 音频闸门 + CONV_SUMMARY_THRESHOLD 自动摘要
- 多人脸 DOA 说话人选择 + all_faces 输出
- 唤醒优先级(a_active) + TRACKING 身体跟随 + 人脸误识别迟滞
- 安全删除工作流(多步验证+备份)
- display_transcript 持久记录本 + Dashboard 上下文调试
- Intel Mac 兼容(mediapipe<0.10.15 + onnxruntime<1.20)

## 当前架构状态

```
voice/
  config.py        — 常量 + 工具元数据 + prompt
  state.py         — State 类 + log + OneEuroFilter
  d01_realtime_chat.py — 主程序 (~600 行，已瘦身)
  debug_server.py  — Dashboard
  kws.py           — 唤醒词门控
  realtime.py      — Qwen-Omni-Realtime 协议层
perception/
  vision_worker.py — Face(YuNet/MediaPipe) + Hand(GestureRecognizer)
  fusion.py        — 声源-视觉融合
identity/
  recognizer.py    — ArcFace 身份识别 + auto_merge
  owner.py         — 主人认定
memory/
  manager.py       — 个人记忆管理 + 对话摘要
  safety.py        — 安全删除工作流
```

- 9 状态 FSM: ARMED/IDLE_CENTER/ENGAGING/TRACKING/SEARCHING/RETURNING/POINTING/PLAYING
- 5 层运动仲裁: Primary > Playing > SoundTurn > Tracking > Idle

## 遗留问题

1. **YuNet 无 blendshapes**: smile/frown 恒 0.0, 可用 insightface 2D106 估算
2. **auto_merge 未同步 MemoryManager**: 合并碎片后 drop_pid 的 memories 残留
3. **多人同框介绍**: 指着他人说"这是XX" → 关联名字(方案见 docs/MULTI_PERSON_INTRO_PLAN.md)
4. **end_session 乱码**: 模型偶尔把 function_call_output 当文字朗读

## 下一步建议

1. 真机测试验证检出率
2. 继续 todo.md 未完成项(#1 DOA / #7 身份优化 / #9 对话质量)
3. 记忆格式重构: facts {key:value} → 自然语言分类存储
