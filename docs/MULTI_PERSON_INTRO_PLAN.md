# 多人同框介绍朋友 — 方案文档

> 调研日期：2026-06-23 ｜ 状态：DRAFT（仅调研，未改代码）
> 涉及模块：`perception/vision_worker.py`、`identity/recognizer.py`、`memory/manager.py`、`voice/d01_realtime_chat.py`、`voice/config.py`、`voice/state.py`

---

## 1. 需求描述

场景：多人同时出现在摄像头画面里，用户用手指指向其中一个人，口头说出关系/名字，例如：

- “这是我的朋友小红”
- “这是我妈妈，李阿姨”

机器人需要：

1. **空间匹配**：把用户的指向方向，匹配到画面中对应的那张人脸（多张人脸里挑一张）。
2. **身份关联**：给该人脸在 `face_db.json` 里的条目写入名字 + 关系标签（`name`、`relation` 等 fact）。
3. **记忆复用**：下次该人单独或同框再出现时，身份识别命中后注入这个人的记忆（名字、关系），让机器人能主动认出并称呼。

关键差异点（与现有“记住自己”能力的区别）：

- 现有 `remember_fact` 记的是**当前对话对象自己**（`st.current_person_id`，第一人称“我叫…”）。
- 新需求是**第三人称介绍**：被介绍的人不是说话者，且要在多张脸里靠“指向”选中目标脸。

---

## 2. 现有能力盘点

### 2.1 已有、可直接复用

| 能力 | 位置 | 说明 |
|---|---|---|
| 多人脸检测 | `vision_worker.py:298` `yunet.detect()` | YuNet 一次返回 N×15 数组，每行 `[x,y,w,h, kp0x,kp0y,...,kp4x,kp4y, conf]`。**天然支持多脸**。 |
| 主脸 bbox + 5 关键点输出 | `vision_worker.py:302-306` | 但**只输出选中的主脸**（`face_sel.selected_face`），其余脸的 box/kps 被丢弃。 |
| 多脸独立识别 | `recognizer.py:282` `detect_and_recognize_all()` | 已实现：从整帧检测并识别**所有**人脸，返回 `[(pid, name, sim, is_new, box, kps), ...]`。**目前 d01 没调用它**，d01 只对主脸调 `recognize()`。 |
| ArcFace embedding + 特征库匹配 | `recognizer.py:80-202` | `ArcFaceONNX.get_embedding()` + `FaceDB.match()`。完整可用。 |
| 给某 person 设名字 | `recognizer.py:170` `FaceDB.set_name()` | 已有。 |
| per-person facts 存储 | `recognizer.py:141` `add_person()` 的 `facts:[]` 字段 + `memory/manager.py` | **注意有两套存储**：①`face_db.json` 里每个 person 有个 `facts:[]`（list，目前没人写）；②`data/memories/<pid>.json` 是 `MemoryManager` 维护的 `{facts:{}, history:[]}`（dict，真正在用的）。 |
| 记忆注入 prompt | `manager.py:170` `get_prompt()` | 生成“你面前的人叫X、你记得关于ta：…”。**目前只对 `current_person_id` 注入**（d01:1964-1976、1992-2004）。 |
| 指向方向（食指角度） | `vision_worker.py:146` `index_dir()` → `st.finger_angle`（画面系角度）、`st.finger_ext_at`（伸指时刻） | 已发布到 state（d01:766-770）。但**只对单手**（`num_hands=1`），且只有角度+指尖，无“指向落点”与脸的匹配。 |
| 指向转头 FSM（POINTING） | `d01:1097-1126` + `1361` | 两段式：VLM judge 出粗方向 → 转头 → 重拍。是“看物体”流程，**不是“选脸”流程**。 |

### 2.2 需要新增 / 改造

| 缺口 | 影响 |
|---|---|
| vision_worker 只输出主脸 box/kps | 多人同框时拿不到“每张脸的 box”，无法做指向-人脸空间匹配。**必须改造**：增加 `faces_all` 字段输出全部脸的 box+kps。 |
| 没有“指尖坐标 + 指向射线”发布 | `index_dir()` 算了 `tip(u,v)` 和 angle，但 d01 只取 angle，没存指尖归一化坐标。指向-脸匹配需要指尖坐标 + 方向向量。 |
| 没有“介绍朋友”工具 | 需要新增 `introduce_person` 工具定义（`config.py:BASE_TOOLS`）。 |
| 没有“第三人称写记忆”路径 | 现有 `remember_fact` 写 `current_person_id`（说话者自己）。介绍朋友要写**被指向的那个人**的 pid。 |
| 多人记忆注入 | 现仅注入 `current_person_id`。多人在场时需考虑是否注入在场所有已知人的简短记忆（见 §6）。 |
| 指向-脸匹配算法 | 全新，见 §5。 |

---

## 3. 技术方案

### 3.1 整体数据流

```
摄像头帧
  └─> vision_worker 子进程
        ├─ YuNet detect() → faces_all（N张脸 box+kps）  [新增输出]
        ├─ FaceSelector → 主脸 face/face_box（保持不变，跟随用）
        └─ HandLandmarker → hand{angle, tip(u,v), extended, ...}  [新增 tip 发布]
  └─> result_q → integrate_loop（主进程）
        ├─ 主脸身份识别（保持不变，更新 current_person_id）
        ├─ faces_all → 缓存到 st.faces_all（带 box + 已识别 pid）  [新增]
        └─ hand.tip → st.finger_tip（归一化坐标）  [新增]

用户说“这是我朋友小红” + 指向
  └─> Qwen Realtime 识别意图 → function_call: introduce_person(name, relation)  [新增工具]
        └─> ChatCallback 处理（d01:368 区块内新增 elif）
              ├─ 取 st.finger_tip + st.finger_angle（最近 POINT_FRESH_S 内）
              ├─ 取 st.faces_all（最近一帧的全部脸 + box）
              ├─ 指向-脸匹配算法 → 选中 target_pid（见 §5）
              ├─ FaceDB.set_name(target_pid, name)
              ├─ MemoryManager.save_fact(target_pid, "name", name)
              │                .save_fact(target_pid, "relation", relation)
              └─ function_call_output：成功/失败描述 → 机器人语音确认
```

### 3.2 模块改动清单

#### A. `perception/vision_worker.py`

**改动点 1**：输出全部人脸（不止主脸）。在 `out` dict 增加字段：

```python
out["faces_all"] = None   # YuNet: list[{"box":(x,y,w,h), "kps":[(x,y)*5], "conf":float}]
```

在 YuNet 分支（`vision_worker.py:292-306`）里，遍历 `faces_raw` 全部行，组装 `faces_all`（box+kps+conf），与现有主脸选择逻辑并存。MediaPipe 分支可后做（项目默认 `FACE_BACKEND=yunet`）。

**改动点 2**：手部指尖坐标已在 `hand["tip"]`（`vision_worker.py:374`），无需改 worker，只需 d01 侧发布（见 B）。

> 性能注意：`faces_all` 只是把已有 `faces_raw` 数组转 dict，**不增加推理开销**（YuNet 本来就检了全部脸）。kps 已在数组里。

#### B. `voice/d01_realtime_chat.py` — `integrate_loop`

**改动点 3**（`d01:755-770` 附近）：发布指尖归一化坐标到 state：

```python
if _valid_pos and hand.get("score",1.0) >= PLAY_SCORE_MIN:
    st.finger_angle = hand["angle"]
    st.finger_tip = hand.get("tip")     # 新增 (u,v) 归一化
    st.finger_extended = hand["extended"]
    st.finger_at = now
```

**改动点 4**（`d01:732` 身份识别区块附近）：缓存“全部脸 + 各自 pid”。

当前每 `IDENTITY_COOLDOWN_S` 秒只识别**主脸一张**。介绍朋友时需要在场每张脸都有 pid（被指向的那张可能不是主脸）。方案：

- 维持现有主脸识别（跟随/记忆注入用）不变。
- 新增：把 `msg["faces_all"]`（仅 box+kps，**不立即识别**）缓存到 `st.faces_all`，每帧更新（轻量，只存坐标）。
- 在 `introduce_person` 触发时，**按需**对“被指向选中的那张脸”单独跑一次 `recognizer.recognize(rgb, box, kps)`（注册/匹配），避免每帧全脸 ArcFace 的开销。

> 这样平时零额外开销，只在“介绍”这个偶发事件时对**一张**脸做 embedding。

#### C. `voice/config.py` — 新增工具定义

在 `BASE_TOOLS` 末尾加：

```python
{"type": "function", "name": "introduce_person",
 "description": (
   "当用户指着画面中另一个人，介绍这个人是谁时调用。"
   "例如『这是我朋友小红』『这位是我妈妈李阿姨』『他叫小明，是我同事』。"
   "name=被介绍人的名字/称呼；relation=与说话者的关系(如 朋友/妈妈/同事，没有就留空)。"
   "注意：这是介绍【别人】，不是说话者自己；说话者介绍自己用 remember_fact。"),
 "parameters": {
   "type": "object",
   "properties": {
     "name": {"type": "string", "description": "被介绍人的名字或称呼"},
     "relation": {"type": "string", "description": "与说话者的关系，可空"},
   },
   "required": ["name"],
 }},
```

> 注意 `TOOLS = BASE_TOOLS + QWEN_TOOLS`（d01:177）。`introduce_person` 放 BASE_TOOLS 即可。但 `active_tools` 在某些状态会过滤掉记忆类工具（d01:1762 过滤 `remember_fact/clear_memory/forget_fact`），需确认 `introduce_person` 是否也要随 `no_memory` 一起过滤——建议**随记忆工具一起过滤**（无记忆模式不介绍）。

#### D. `voice/d01_realtime_chat.py` — 工具处理

在 `function_call_arguments.done` 处理区（`d01:368-458`）新增 `elif name == "introduce_person":` 分支：

```python
elif name == "introduce_person":
    args_dict = _parse_args(event)
    intro_name = args_dict.get("name","").strip()
    relation = args_dict.get("relation","").strip()
    result = _handle_introduce(st, intro_name, relation)  # 见下，含匹配+写库
    self.conv.create_item({"type":"function_call_output","call_id":call_id,
        "output": json.dumps({"result": result}, ensure_ascii=False)})
```

`_handle_introduce()` 逻辑：

1. 读 `st.finger_tip`、`st.finger_angle`、`st.finger_at`（判断 `POINT_FRESH_S` 内是否有有效指向）。
2. 读 `st.faces_all`（含 box+kps）。
3. 调指向-脸匹配（§5）选 `target_face`。
4. 若匹配失败 → 返回提示语（“我没看清你指的是谁，再指一下？”），机器人口头追问。
5. 匹配成功 → 对 `target_face` 跑 `_id_recognizer.recognize(rgb, box, kps)` 拿 `target_pid`（命中已知人则复用，否则新建）。
6. `_id_recognizer.db.set_name(target_pid, intro_name)`；`_memory_mgr.save_fact(target_pid,"name",intro_name)`；若有 relation：`save_fact(target_pid,"relation",relation)`。
7. 返回成功语（“好的，我记住了，这是你朋友小红”）。

#### E. `voice/state.py` — 新增字段

```python
self.finger_tip = None          # (u,v) 归一化指尖坐标
self.faces_all = []             # [{"box":(x,y,w,h),"kps":[...]}]
self.faces_all_t = 0.0          # 最近更新时刻（判新鲜）
```

#### F. 记忆注入扩展（多人在场）—— 见 §6，建议二期。

---

## 4. face DB 注册流程现状（关键）

当前 `recognize()`（`recognizer.py:218-270`）的注册逻辑：

- 命中已知人（`sim ≥ COSINE_THRESHOLD=0.35`）→ 复用 pid，追加 embedding。
- 未命中 → 进 `_pending_new`，**连续 `NEW_PERSON_CONFIRM_FRAMES=3` 帧**才 `add_person()` 新建。

**对“介绍朋友”的影响**：介绍是一次性事件，只有一帧。若被介绍人是新人，单帧调 `recognize()` 会返回 `(None,...)`（pending 中，未确认），**当场拿不到 pid**。

解决方案（二选一）：

- **方案 A（推荐）**：`_handle_introduce` 里**绕过 pending 机制**，直接对目标脸 `get_embedding()` → `db.match()`，命中就用，没命中就**立即** `db.add_person(emb, name=intro_name)`。一次性介绍场景下，单帧建人是可接受的（用户明确指认，比自动注册更可信）。
- 方案 B：连拍 3 帧再确认（增加交互延迟与复杂度，不推荐）。

→ 建议在 `IdentityRecognizer` 新增一个方法 `register_named(rgb, box, kps, name) -> pid`，专供介绍场景：匹配→命中复用/未命中立即建人并命名。

---

## 5. 指向-人脸匹配算法设计

输入：
- 指尖坐标 `tip=(tu,tv)`（归一化，画面系，来自 `hand["tip"]`）。
- 食指方向角 `angle`（画面系度数，`index_dir()`：0°=右，-90°=上，+90°=下，±180°=左）。
- 候选脸 `faces_all`：每张脸 box `(x,y,w,h)`（**像素坐标**，注意要除以 W/H 归一化对齐到指尖坐标系）。

> ⚠ 坐标系一致性陷阱：`hand["tip"]` 是 MediaPipe 归一化（0~1），`face box` 是 YuNet 像素坐标，且身份识别用的是 `DECIMATE` 降采样帧。匹配前必须统一到同一归一化坐标系（除以各自的 W/H）。这是最易出 bug 的点。

**算法（射线-夹角评分）**：

```
方向向量 d = (cos(angle), sin(angle))     # 画面系，y 向下
对每张候选脸 i：
    脸中心 c_i = ((x+w/2)/W, (y+h/2)/H)
    从指尖到脸中心向量 r_i = c_i - tip
    若 |r_i| < eps：跳过（脸就在指尖上，无意义）
    cosθ_i = dot(d, r_i/|r_i|)            # 方向与“指尖→脸”的夹角余弦
    score_i = cosθ_i                       # 越接近 1 越对齐
选 score 最大且 score ≥ COS_MIN(建议 0.5，约 ±60°锥角) 的脸为 target。
若最高分 < COS_MIN → 匹配失败（指向没对准任何脸）。
```

**加分项（可选，提升鲁棒性）**：
- 距离惩罚：太远的脸轻微降权（`score - λ·|r_i|`），避免选到画面边缘恰好对齐的远脸。
- 唯一性检查：若前两名 score 很接近（差 < 0.1），判为“歧义”→ 让机器人追问“你指的是左边还是右边那位？”

**边界**：单人在场时直接选唯一脸（不需要指向也能匹配，宽容处理）。

---

## 6. 多人记忆注入（二期）

现状：只注入 `current_person_id` 的记忆（d01:1964）。

多人扩展思路（建议作为二期、不阻塞主功能）：

- 把 `faces_all` 识别出的所有命中 pid 收集，对每个有名字/关系的人生成一句极简记忆（“画面里还有你朋友小红”），合并成一条 system message 注入。
- 风险：注入过多会污染上下文、增加 token、可能让模型啰嗦。建议**仅注入“有名字的在场者”**，且每人一句话以内。
- 触发时机沿用现有 `identity_injected` 门控，避免重复注入；多人场景需把单一 `identity_injected` 布尔改为“已注入 pid 集合”。

---

## 7. 边界条件与风险点

| 风险 | 说明 | 缓解 |
|---|---|---|
| 坐标系不统一 | tip 归一化 vs face box 像素 vs DECIMATE 降采样帧 | §5 强制归一化；写匹配函数时加断言/日志打印两边坐标范围。 |
| 单手限制 | `num_hands=1`（vision_worker.py:259）。说话者自己的手可能不是举着指的那只 | 介绍场景通常只有一只手在指，可接受；若误检需 `score`/`size` 过滤（已有）。 |
| 指向新鲜度 | 用户说完话才触发工具，此时手可能已放下 | 用 `POINT_FRESH_S=1.2s` 粘滞窗（已有机制）；窗口外则失败追问。 |
| 单帧建新人不稳 | 介绍时只一帧，embedding 质量可能差（角度/模糊） | §4 方案 A 接受单帧建人；可加最小人脸像素门（`MIN_FACE_PX=60`）拒绝太小的脸。 |
| 被指向人=说话者自己 | 用户指自己（少见但可能） | 匹配出的脸若 ≈ `current_person` 的主脸，可提示“这是你自己呀？”或仍按介绍处理。 |
| 歧义（两人挨着） | 指向锥角内有多张脸 | §5 唯一性检查 → 追问。 |
| 关系字段语义 | relation 自由文本，注入时如何用 | `get_prompt` 已是“你记得关于ta：relation: 朋友”，自然语言够用；无需枚举。 |
| 工具误触发 | 模型把“我叫小明”错路由到 introduce_person | 工具 description 明确区分第一/第三人称（见 §3.2-C）；必要时加 few-shot。 |
| FaceDB 两套 facts | `face_db.json.facts[]`(list,没用) vs `memories/<pid>.json`(dict,在用) | **统一走 `MemoryManager`**，忽略 face_db 的 facts 字段（或后续清理）。 |

---

## 8. 分步实施建议

**P0（核心闭环，先做）**

1. `vision_worker.py`：输出 `faces_all`（box+kps+conf）。
2. `state.py`：加 `finger_tip` / `faces_all` / `faces_all_t` 字段。
3. `d01 integrate_loop`：发布 `st.finger_tip`，缓存 `st.faces_all`。
4. `recognizer.py`：新增 `register_named(rgb, box, kps, name)`（绕过 pending，单帧匹配/建人）。
5. `config.py`：加 `introduce_person` 工具定义。
6. `d01`：加 `introduce_person` 处理分支 + `_handle_introduce()`（含 §5 匹配算法）。
7. 写匹配算法单测（坐标系是重灾区）：构造“指尖+多脸 box”的 fixture，验证选中正确脸（参考 `tests/test_identity.py` 风格）。

**P1（体验完善）**

8. 歧义追问（两脸都在锥角内 → 让机器人问“左边还是右边？”）。
9. relation fact 注入后的自然称呼验证（端到端跑一次）。

**P2（多人记忆，可后做）**

10. 在场多人记忆注入（`identity_injected` 改集合，注入有名字的在场者）。

---

## 9. 关键文件 / 函数索引

- 多脸检测源头：`perception/vision_worker.py:298`（`yunet.detect`）、`vision_worker.py:99`（`FaceSelector.select_yunet`，主脸）
- 全脸识别现成方法：`identity/recognizer.py:282` `detect_and_recognize_all()`
- 单脸识别 + 注册：`identity/recognizer.py:218` `recognize()`，注册门控 `recognizer.py:30-34`
- 设名字：`identity/recognizer.py:170` `set_name()`
- 写 fact：`memory/manager.py:107` `save_fact()`；注入 `manager.py:170` `get_prompt()`
- 工具定义：`voice/config.py:224` `BASE_TOOLS`；`memory/manager.py:26` `QWEN_TOOLS`；组装 `d01:177`
- 工具处理：`voice/d01_realtime_chat.py:368-458`（function_call 分发）；记忆工具分支 `d01:412-447`
- 指向角度发布：`d01:755-770`（`st.finger_angle/finger_ext_at`）；`index_dir` `vision_worker.py:146`
- 指向转头 FSM：`d01:1097-1126`（进 POINTING）、`d01:1361`（POINTING 处理）
- 身份识别集成：`d01:732-754`（主脸 cooldown 识别）
- 记忆注入：`d01:1964-1976`、`d01:1992-2004`
- State 字段：`voice/state.py:244-277`
