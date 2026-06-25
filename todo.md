# Reachy Mini Demo — TODO 清单

> 更新: 2026-06-25 | 仅保留待办项，已完成项已归档到 git 历史

---

## 1. DOA 能力优化 ⚠️

唤醒二次优先级已修复(a_active 替代 _is_A)。剩余：

- [ ] 唤醒时 DOA 是否被视觉跟踪覆盖
- [ ] 多人脸 FaceSelector sticky 是否干扰 DOA 转向
- [ ] DOA→head_control 优先级仲裁
- [ ] 嘈杂环境: DOA+视觉联合判定(DOA 指向有人脸方向时加成，无人脸方向降权)

---

## 3. 空闲表情增强 ⚠️

8 态呼吸已完成。剩余：

- [ ] 长时间无交互随机小动作(偶尔抬头、左右看)
- [ ] 官方 blendshape 表情库接入
- [ ] idle 动作变化感

---

## 5. 视觉寻物 Observer ❌

- [ ] 主动寻物流程("帮我找X" → 系统化搜索 → 汇报)
- [ ] 头部扫描策略
- [ ] VLM 针对特定物品的 prompt 模板

---

## 6. RAG 接入 ❌

- [ ] 知识源选型(本地文档/网络搜索/产品知识库)
- [ ] 本地 embedding + FAISS / 远端向量数据库
- [ ] query → retrieve → inject prompt 流程

---

## 7. 身份识别优化 ⚠️

YuNet+ArcFace 链+auto_merge 已就位。剩余核心问题: 入库不应默认触发。

- [ ] 说话人判定: 视觉确认(朝向) + 语音关联(VAD+DOA) → 判断谁在说话
- [ ] 入库时机: 对话发生后才记忆，非检测到人脸就入库
- [ ] 多人场景 DOA+视觉综合判断

---

## 8. 渐进式记忆加载 ❌

前置依赖 #7。

- [ ] 渐进加载(名字 → 关键记忆 → 细节)
- [ ] 记忆权重/优先级排序
- [ ] 动态 prompt 更新机制

---

## 9. 对话质量 ⚠️

HTML/XML 标签泄漏已修复。剩余：

- [ ] 动作通过 function call 触发(注册 perform_action 工具)
- [ ] prompt 策略对 Qwen Omni 输出格式的影响测试

---

## 12. 手势识别优化 ⚠️

GestureRecognizer 已集成(模型优先+规则 fallback)。剩余：

- [ ] 更新 `tests/vision_model_test.py`: 增加 GestureRecognizer 测试，对比模型 vs 规则一致性
- [ ] Model Maker 自定义训练扩展手势(如 ok/three/four 目前靠规则)

---

## 13. 指向+说名字关联记忆 ❌

说"这个人叫xxx"时有指向手势 → 更新对应人的记忆。

---

## 18. 视觉工具调用报错 ❌

take_snapshot/identify_pointed_object 调用有 error，需复现排查。

---

## 20. take_snapshot 时延 ❌

VLM 推理时间长，返回时环境已变。考虑异步 snapshot / 降级模型。

---

## 23. 记忆索引重名冲突 ❌

**问题**：当前 memory 以 person_id 为主键，但不同人可能被分配到相同或相似的 id，导致记忆索引错误、取到别人的记忆。

**现状**：
- `identity/recognizer.py` 中 person_id 由 ArcFace embedding 生成（hash）
- `memory/manager.py` 以 person_id 为 key 存取 `data/memories/<pid>.json`
- `remember_fact(key="name")` 更新 face_db 中的 display name，但 pid 本身不含 name 信息
- 如果两个人的 embedding 碰撞或被误合并，记忆会串

**方案方向**：
- [ ] 索引主键改为复合键（如 name + embedding hash），确保唯一性
- [ ] 或在 pid 生成时引入更多区分度（embedding 维度、时间戳等）
- [ ] 排查 auto_merge 是否在双命名保护下仍有误合并风险
- [ ] 考虑加 pid 冲突检测：新人入库时校验是否与已有 pid 重复

---

## 24. 已知人重复调用 remember_fact(name) ❌

**问题**：模型识别到已知人(如 sim=0.92 的"坤坤")后，仍然调用 `remember_fact` 尝试记名字，且缺少 key/value 参数导致报错。

**现状**：
- 记忆注入时 `_update_memory_instructions()` 已将该人的 facts(含 name)写入 session instructions
- 但 prompt 中没有明确告诉模型"如果已经知道名字就不要再调 remember_fact"
- 模型看到人脸识别结果后条件反射式调用 remember_fact，浪费一轮工具调用且报错

**方案方向**：
- [ ] prompt 中加入规则: "如果 session instructions 中已包含该人的记忆(name/facts)，不要重复调用 remember_fact"
- [ ] 或在 `remember_fact` handler 中做幂等检查: 已存在相同 key=value 时直接返回"已知"而非报错
- [ ] 检查记忆注入的 instructions 格式，确保模型能清晰看出"这个人的信息我已经知道了"

---

## 其他

- [ ] `perception/__init__.py` 工厂函数(get_vision_worker)

---

## 优先级

| 优先级 | 项目 | 原因 |
|--------|------|------|
| P0 | #9 对话质量 | 基础交互体验 |
| P0 | #1 DOA 优化 | 唤醒第一印象 |
| P1 | #7 身份识别 | #8 前置依赖 |
| P1 | #8 渐进式记忆 | 个性化核心 |
| P2 | #3 空闲表情 | 体验丰富度 |
| P2 | #5 寻物 | 新能力 |
| P3 | #6 RAG | 知识增强 |
