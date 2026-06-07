# 中文唤醒词(KWS)方案调研 — Reachy Mini「小艺」

> 调研日期:2026-06-07 · 平台:Windows 11 + Python 3.12 · CPU(RTX 5060 可选不依赖)
> 目标:待命态只听唤醒词「小艺」(xiǎo yì),喊到才进对话;KWS 须能消费**程序内传入的 16kHz PCM chunk**(不能自己开麦,Reachy 麦克风设备独占)。

---

## 1. 结论先行

**可行,强烈推荐 `sherpa-onnx` 的关键词检测器(KeywordSpotter)。**

- 有现成中文 zipformer KWS 预训练模型(wenetspeech,10000h 训练),**自定义关键词只需写拼音 token,完全不用训练、不用采样、不用 GPU**。
- 模型自带的关键词词库里**已经内置了「小艺小艺」**(`x iǎo y ì x iǎo y ì`),我们要的词几乎是开箱即用。
- Windows + Python 3.12 **一行 pip 装上**(`sherpa-onnx`,纯 CPU 轮子),无编译、无系统依赖。
- 已在本机实测:**唤醒检出 4/4,单 chunk(100ms 音频)推理均值 ~2.5ms,实时率 RTF≈0.025**——CPU 占用极低,与实时对话 + MediaPipe 视觉 + 动作并行毫无压力。
- API 就是「喂 PCM chunk → 问有没有命中」的流式接口,**天然匹配 Reachy 的 16kHz 流**,只要把已有上行麦克风 chunk 多分发一路给它即可。
- 离线、Apache-2.0 许可、可商用。

**唯一注意点:**单字「小艺」太短(就两个音节 `xiǎo yì`),容易被近音词「小义气 / 小一 / 小议」等命中(实测「小义气」会误触)。
**生产配置建议用叠词「小艺小艺」做唤醒词**(实测叠词误触 0/6),与「小爱同学 / 小度小度 / 天猫精灵」等商业唤醒词同样是 4 音节设计,这是行业通行的鲁棒性做法。若产品上必须单字「小艺」,则叠加一个二次确认(命中后短窗口内 ASR/能量复核),见 §3。

次选:**Porcupine(Picovoice)**——支持中文普通话、离线、检出质量工业级,但自定义词要在其云端 Console 训练并下载 `.ppn`,免费层限 3 个活跃用户/月且需账号,模型不在我们手里。适合"想要最稳、能接受云训练 + 账号绑定"的场景。

不推荐:openWakeWord(中文要自己训练、官方自动训练只支持 Linux/Piper)、Vosk(是完整 ASR 不是轻量 KWS,资源更重)、snowboy(已停止维护,基本不可用)。

---

## 2. 方案对比表

| 方案 | 中文支持 | 自定义「小艺」成本 | Windows 安装 | 资源占用(待命) | 离线 | 许可证 |
|---|---|---|---|---|---|---|
| **sherpa-onnx KWS**(k2-fsa)**【推荐】** | ✅ 原生(wenetspeech 中文专用模型 + 中英双语模型) | **零训练**:写一行拼音 `x iǎo y ì` 即可;模型自带词库甚至已含「小艺小艺」 | ✅ `pip install sherpa-onnx`,纯 CPU 轮子,无编译 | 极低:单 chunk ~2.5ms,RTF≈0.025,单线程;模型 enc 12M(int8 4.6M) | ✅ 全离线 | Apache-2.0,可商用 |
| **Porcupine**(Picovoice) | ✅ 中文普通话(官方支持语言之一) | 云端 Console 输入中文词训练 → 下载 `.ppn`;不用自己采样,但**词在云端生成** | ✅ `pip install pvporcupine`,有官方 Win 轮子 | 极低(为 MCU 设计,比 sherpa 更轻) | ✅ 推理离线(训练在云) | 商业;免费层限 3 活跃用户/月,需账号 AccessKey |
| **openWakeWord** | ⚠️ 需自训(有人验证可训中文);内置词全英文 | **要训练**:合成 N 千条 TTS 正样本 + 负样本训小模型;**官方自动训练只支持 Linux(Piper)**,Windows 需 WSL/手搓 | ⚠️ 推理(onnxruntime)能在 Win 跑;训练在 Win 麻烦 | 低(onnx 小模型) | ✅ | Apache-2.0 |
| **Vosk small zh** | ✅(完整 ASR) | 可用 grammar 限定词表近似 KWS,但本质是跑 ASR | ✅ `pip install vosk` | 中等偏重(完整声学+语言模型,CPU 占用明显高于专用 KWS) | ✅ | Apache-2.0 |
| **FunASR**(阿里) | ✅(有 KWS 分支) | 偏向云/服务化、依赖较重(PyTorch),Win 上工程量大 | ⚠️ 依赖重(torch 等) | 重(PyTorch 栈) | 部分 | MIT/Apache(看模型) |
| **snowboy** | ⚠️ 历史上支持 | 已停止维护(2020 关停官方训练服务),社区 fork 不稳 | ❌ 老旧,Py3.12 基本装不动 | 低 | ✅ | 已弃 |

---

## 3. 推荐方案集成草图(sherpa-onnx KeywordSpotter)

### 3.1 现有音频链路(已读 `voice/d01_realtime_chat.py`)

上行主循环(约 1393–1417 行)是**唯一的麦克风消费点**:

```python
chunk = mini.media.get_audio_sample()      # float 帧, shape=(N, ch), 16kHz
mono  = chunk[:, 0]                          # 取声道 0
pcm16 = np.clip(mono*32767, -32768, 32767).astype(np.int16)
conv.append_audio(base64.b64encode(pcm16.tobytes()).decode("ascii"))  # → Qwen 上行
```

KWS 需要的就是这同一份 `mono`(float32, 16kHz, [-1,1])。**不另开声卡流**,在这里"分发两路"即可。

### 3.2 同一份 PCM 分发两路(共流)

```
mini.media.get_audio_sample() ──► mono(float32 16kHz)
                                   ├─► (对话态) conv.append_audio(...)   # 现有,送 Qwen
                                   └─► (待命态) kws_stream.accept_waveform(16000, mono)  # 新增
```

用一个状态开关 `armed`(待命) / `engaged`(对话中)。待命态**只喂 KWS、不喂 Qwen**(省上行带宽和云费用);命中后切到 `engaged`,开始喂 Qwen 并(可选)继续喂 KWS 监听"睡眠词"。

### 3.3 待命态省资源做法

- **待命态根本不连 Qwen Realtime WebSocket**(或连了但不 append_audio)——省云费 + 省网络。检测到唤醒词再 `connect()` / 开始上行。
- KWS 单线程 CPU 占用实测 RTF≈0.025(即处理 1 秒音频只花 25ms CPU),**待命可常驻、几乎不占资源**。视觉/动作此时可降频甚至挂起。
- 命中 → 进对话;`NO_INTERACT_S`(现有 15s 无互动)或显式"睡眠词"→ 回待命态、断开/暂停上行。

### 3.4 关键词配置(已实测可用)

模型自带的 `keywords.txt` 已含「小艺小艺」。自定义只需:

```bash
# 用 sherpa-onnx-cli 把汉字转拼音 token(需 pip install click sentencepiece jieba pypinyin)
echo 小艺小艺 > raw.txt
sherpa-onnx-cli text2token --tokens <model>/tokens.txt --tokens-type ppinyin raw.txt keywords.txt
# 产出:  x iǎo y ì x iǎo y ì
```

也可直接手写 `keywords.txt`(每行 `拼音token... @显示名`,可带 `:score :threshold` 调灵敏度)。

### 3.5 引擎构造与喂流(实测 API,sherpa-onnx 1.13.2)

```python
import sherpa_onnx, numpy as np

kws = sherpa_onnx.KeywordSpotter(
    tokens   = f"{M}/tokens.txt",
    encoder  = f"{M}/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",  # int8 即可
    decoder  = f"{M}/decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    joiner   = f"{M}/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    num_threads     = 1,           # 单线程足矣,不抢对话/视觉
    keywords_file   = "keywords.txt",
    keywords_threshold = 0.25,     # 默认;叠词时可保持,单字时调高减误触
    provider = "cpu",
)
stream = kws.create_stream()

# 在上行循环里,待命态时:
def feed_kws(mono_f32_16k):                 # mono: float32 [-1,1]
    stream.accept_waveform(16000, mono_f32_16k)
    while kws.is_ready(stream):
        kws.decode_stream(stream)
    r = kws.get_result(stream)
    if r:                                    # 命中!
        kws.reset_stream(stream)
        on_wake()                            # → 切 engaged,开始喂 Qwen
```

### 3.6 单字「小艺」鲁棒化(若产品必须单字)

实测单字会被「小义气」等近音误触。两种缓解(任选其一或叠加):
1. **改叠词「小艺小艺」**(首选,实测误触 0/6);
2. 单字命中后做**二次确认**:命中后开 ~0.8s 窗口,把这段音频快速过一遍(能量门 + 可选轻量 ASR / 或直接连 Qwen 让它确认是否被叫),不通过则回待命。代价是首字延迟略增。

---

## 4. 最小验证结果(本机实跑)

环境:`tools/_kws_venv`(从机器人 venv 的 Python 3.12.13 新建的独立 venv,**未污染机器人 venv**);pip 全程清华镜像。
模型:`sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01`(GitHub release **直连下载成功**,32MB,无需代理)。
样本:pyttsx3 + **Microsoft Huihui Desktop (zh-CN)** 合成,22050Hz→16kHz,前后补静音,**按 100ms/块**喂入(模拟 Reachy 流)。**全程离线 wav,未打开任何麦克风/摄像头。**

### 配置 A:关键词 = 单字「小艺」+ 叠词「小艺小艺」,threshold=0.25(fp32)

| 文件 | 说的内容 | 期望 | 检出 | 结果 |
|---|---|---|---|---|
| wake_xiaoyi_1 | 小艺 | 唤醒 | ✅ | 检出 |
| wake_xiaoyi_2 | 小艺小艺 | 唤醒 | ✅(2 次) | 检出 |
| wake_xiaoyi_3 | 小艺,你在吗 | 唤醒 | ✅ | 检出 |
| wake_xiaoyi_4 | 嘿,小艺 | 唤醒 | ✅ | 检出 |
| neg_nihao | 你好 | 不触发 | — | ✅ |
| neg_xiaomi | 小米小米 | 不触发 | — | ✅ |
| neg_xiaoai | 小爱同学 | 不触发 | — | ✅ |
| neg_xiaoyu | 小鱼 | 不触发 | — | ✅ |
| neg_random | 今天天气怎么样 | 不触发 | — | ✅ |
| **neg_xiaoyiqi** | **小义气** | 不触发 | iǎoyì | ❌ **误触** |

**唤醒检出 4/4;误触 1/6**(误触来自「小义气」,含完整 `xiǎo yì` 子串,是真·难例)。

### 配置 B:关键词 = 仅叠词「小艺小艺」,threshold=0.25(fp32)

- **误触 0/6**(含「小义气」也不触发);
- 唤醒侧只有真正说出"小艺小艺"的 wake_xiaoyi_2 命中;其余只说一遍"小艺"的样本不触发(**符合预期**——若把叠词定为唤醒词,本就该说两遍才唤醒)。

### 配置 C:int8 量化模型(单+叠词,threshold=0.25)

- 准确率与 fp32 **完全一致**(4/4 检出,1 误触);
- 推理更省:模型体积 enc 4.6M(vs fp32 12M)。

### 性能(全部配置一致量级,单线程 CPU)

| 指标 | 数值 |
|---|---|
| 单 chunk(100ms 音频)推理耗时 | 均值 **~2.5ms** / 中位 ~0.2ms / p95 ~12ms / max ~17ms |
| 实时率 RTF | **≈0.025**(处理 1s 音频耗 25ms CPU) |
| 线程 | 1 |

> p95/max 偏高是因为含语音的块才真正解码(静音块几乎 0ms),即便如此 12–17ms 远低于 100ms 块长 → CPU 余量极大,**与 Realtime 对话 + MediaPipe 视觉子进程 + 动作并行无压力**。

**结论:配置 B(叠词「小艺小艺」)零误触、性能极佳,作为生产首选;int8 模型用于部署。**

---

## 5. 工作量估计(从现在到"喊小艺能唤醒")

| 步 | 内容 | 工作量 |
|---|---|---|
| 1 | 机器人 venv 装 `sherpa-onnx`(清华镜像,纯 CPU 轮子);模型已下载在 `tools/`,拷到 repo 模型目录 | 10 分钟 |
| 2 | 生成/确认 `keywords.txt`(叠词「小艺小艺」,已验证) | 5 分钟 |
| 3 | 在 `d01_realtime_chat.py` 上行循环加"分发两路":封一个 `KwsGate` 类(构造引擎 + `feed()` 返回是否命中) | 0.5–1 天 |
| 4 | 加 `armed`/`engaged` 状态机:待命态只喂 KWS 不连 Qwen;命中 `on_wake()` 切对话;`NO_INTERACT_S` 或睡眠词回待命 | 0.5–1 天 |
| 5 | 真机联调:用 Reachy 真实麦克风流验证检出率/误触/唤醒延迟,按需微调 threshold;评估是否要 §3.6 二次确认 | 0.5 天 |
| 6(可选) | 若坚持单字「小艺」:加命中后二次确认窗口 | +0.5 天 |

**合计 ~2 天**到可用;叠词方案最快。
**前置已全部验证清楚:模型可下、API 可用、Win/Py3.12 可装、CPU 占用可忽略、中文「小艺」可检出。**

---

## 附:本次调研产物(均在 `tools/`,与 voice/ 现有代码无任何改动)

- `tools/_kws_venv/` — 独立测试 venv(**未碰机器人 venv**)
- `tools/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/` — 中文 KWS 模型(含自带 `小艺小艺`)
- `tools/_kws_keywords.txt` — 生成的关键词(`x iǎo y ì` / `x iǎo y ì x iǎo y ì`)
- `tools/_kws_synth.py` — pyttsx3 合成测试 wav(Huihui zh-CN)
- `tools/_kws_test.py` — 离线喂 chunk 的 KWS 验证脚本(支持 `--int8 --thr=`)
- `tools/_kws_wavs/` — 合成的 10 条测试音频

> 这些是验证临时产物,可在确认后清理(`tools/_kws_venv`、`*.tar.bz2`、`_kws_wavs` 体积较大)。

---

## 参考来源

- sherpa-onnx KWS 文档与预训练模型:https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html
- sherpa-onnx KWS 总览:https://k2-fsa.github.io/sherpa/onnx/kws/index.html
- sherpa-onnx Python KWS 示例:https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/keyword-spotter.py
- openWakeWord:https://github.com/dscripka/openWakeWord
- Porcupine 支持语言/FAQ:https://picovoice.ai/docs/faq/porcupine/ ;免费层:https://picovoice.ai/blog/introducing-picovoices-free-tier/
