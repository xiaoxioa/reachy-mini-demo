# DOA 唤醒好奇寻声方案（草案，待讨论）

> 归档日期: 2026-06-22 | 状态: 待优化

## 当前问题

唤醒后经常转到相反方向或不动：
1. **DOA confident ≠ correct**: XVF3800 存在镜像翻转——稳定地指向错误方向，IQR 很小所以 `confident=True`
2. **唤醒时未重置 FaceSelector**: `sticky_reset` 只在 M1.5-b 二次唤醒时发送，首次唤醒时残留人脸锁定干扰
3. **"不动"场景**: 无 DOA 时全场扫从 sin(0)=0 起，开头位移小

## 草案方向：好奇寻声

去掉当前 3 阶段系统（direct → nearby → full），替换为**单一连续弧线扫描**——像小动物循着声音好奇地转头寻找。

### 行为流程（概念）

```
唤醒 "小艺" → 天线微抬 cue (已有)
→ 以 DOA sign 定起扫方向（+1=左 / -1=右，若无 DOA 默认+1）
→ 从中心出发，缓慢弧线扫向 DOA 那一侧
   - 速度：~50°/s（比当前 90°/s 慢，更从容）
   - pitch 微微起伏（好奇抬头/低头看）
   - 全程 breathing 不断
→ 途中看到脸 → 自然过渡到 TRACKING
→ 扫到一侧极限（±80°）没找到 → 平滑减速、掉头，继续扫另一侧
   - 不跳跃，弧线自然反向
→ 扫完两侧没找到（~7s）→ gentle giveup cue → 回 ARMED
```

### 初步实现思路

```python
# 新常量
SEEK_CRUISE_DPS = 50.0       # 寻声巡航速度（°/s）
SEEK_RANGE = 80.0            # 单侧扫描范围（°）
SEEK_TIMEOUT_S = 7.0         # 总超时
SEEK_PITCH_CURIOUS_F = 0.35  # 好奇抬低头频率
SEEK_PITCH_CURIOUS_A = 5.0   # 好奇抬低头幅度

# 状态变量
seek_pos = 0.0               # 当前目标位置
seek_vel = seek_dir * SEEK_CRUISE_DPS  # 当前速度

# 每帧逻辑
seek_pos += seek_vel * dt
if abs(seek_pos) > SEEK_RANGE:
    seek_pos = math.copysign(SEEK_RANGE, seek_pos)
    seek_vel = -seek_vel  # 到边界自然反向

curious_pitch = SEEK_PITCH_UP + SEEK_PITCH_CURIOUS_A * math.sin(
    2 * math.pi * SEEK_PITCH_CURIOUS_F * (now - phase_t))

approach(seek_pos, clip(seek_pos, -BODY_LIMIT), curious_pitch)
```

### 其他配套修改

1. **sticky_reset on wake**: ARMED→ENGAGING 时向 vis_frame_q 发送 `"sticky_reset"`
2. **DOA 只用 sign**: 不再区分 confident/不 confident，只取左右方向提示
3. **去掉 seek_suppress**: 因为扫描速度慢，途中锁脸大概率正确

## 待讨论

- 扫描速度是否合适？太快不优雅，太慢用户等不及
- 到边界的反向应该是瞬间还是有减速过渡？
- 是否需要在 DOA 目标角度附近减速（"仔细看看"效果）？
- 与 M1.5-b 二次唤醒切换的交互：切换也应该用同样的风格吗？
- seek_suppress 去掉后，短距离唤醒（人就在旁边）会不会误锁到更远的人？
