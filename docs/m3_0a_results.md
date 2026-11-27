# M3-0a 措辞判别器验收结果

> 派单 §7.10。本单只做 M3 的前置：一台能实时给医生问句打四维标签的判别器。
> 结论先行：**四条判据未全过**。三维核心能力基本可用（multi 97% / openness 91% / leadingness 88%），但 leadingness 差 2pp、follow_up 降级不可用、置信度无区分度、延迟 2 秒超标。**同时暴露一个超出派单预期的重大发现：M0 标注在三维上存在系统性偏差**，这导致「边缘分布 vs M0 ≤5pp」判据被误触。

---

## 1. 同源声明（路径 A，开工前书面）

| 项 | 选择 | 是否同源 |
|---|---|---|
| 判别器 | **qwen-plus**（`reviewer` 角色） | 非 M0（qwen-flash）✅ |
| 真值来源 1 | M0 标注（qwen-flash，`annot_all_flash.jsonl`） | — |
| 真值来源 2 | deepseek-chat 独立标注（找分歧） | 非判别器 ✅ |
| 分歧仲裁 | 执行侧逐条语义仲裁，理由见 `disagreements.json` | — |

判别器（qwen-plus）与两个真值标注源（qwen-flash / deepseek-chat）均不同源，未犯「自己判自己」。

## 2. 评测集（100 条，分层抽样）

从 holdout（seed=42，20%，270 对话 / 1011 轮）分层抽，固定 seed=12345 可复现：

| 层 | 要求 | 实际纳入 | 全库量 |
|---|---|---|---|
| strong leading | 30 | 30 | 203 |
| multi_barreled | 30 | 31 | 264 |
| surface follow_up | 30 | 32 | 315 |
| 其余随机 | 10 | 补齐至 100 | — |

> 术语对齐：派单表格的 `leadingness: leading` = 数据实际取值 `mild`；`follow_up: follow_up` = 数据实际取值 `deepen`。抽样不受影响，已注明别名。

产出：`outputs/m3_0a/eval_set.jsonl`（含 M0 四维）、`eval_set_meta.json`。

## 3. 两模型原始一致率（任务难度上界）

M0（qwen-flash）vs deepseek-chat，100 条评测集：

| 维度 | 一致率 | 主要分歧模式 |
|---|---|---|
| openness | **93.9%** | semi↔closed 边界 |
| leadingness | **78.8%** | M0 过度「引导化」（strong/mild → 对方判 neutral） |
| multi_barreled | **83.8%** | 单向：M0 把 A-or-B 双问句漏判 single（16 条全此向） |
| follow_up | **43.4%** | surface/deepen → deepseek 判 new（系统性） |

**follow_up 一致率仅 43.4%**，是本单第一块「任务难度」照妖镜——follow_up 定义本身模糊，两模型都判不准。

## 4. 仲裁（分歧项 → 真值）

103 条分歧项，执行侧逐条语义仲裁（理由见 `disagreements.json`）：

| 维度 | 分歧数 | 仲裁方向 | 理由 |
|---|---|---|---|
| openness | 6（+1 deepseek 非法） | 采纳 deepseek | M0 把 wh 判 closed、yes/no 判 semi，6/6 语义复核 deepseek 对 |
| leadingness | 21（+1） | 采纳 deepseek | M0 过度「引导化」，把普通问句判 strong/mild |
| multi_barreled | 16（+1） | 采纳 deepseek | A-or-B 双问句是 multi，M0 漏判 single |
| follow_up | 56（+1） | **降级，不仲裁点真值** | 两模型 43.4%，真值暂用 M0 并标 low_conf |

> 关键：三维分歧项执行侧按 M0 定义逐条复核，结论是 deepseek 全对、M0 错——不是盲从 deepseek，是 M0 三维确有系统性偏差（详见 §6）。

产出：`gold_labels.jsonl`（100 条仲裁后真值）、`disagreements.json`、`inter_model_agreement.json`。

## 5. 判别器验收四项数字

判别器 = qwen-plus，独立 prompt（聚焦四维 + 置信度，表述与 M0 标注 prompt 不同），缓存键 =（规范化问句, 规范化 prev）。

### 5.1 逐条一致率（pred vs 仲裁后真值）——判据 1

| 维度 | 一致率 | 阈值 ≥0.90 | 判定 |
|---|---|---|---|
| openness | **91.0%** | 过 | ✅ |
| leadingness | **88.0%** | 过 | ❌ 差 2pp |
| multi_barreled | **97.0%** | 过 | ✅ |
| follow_up | **61.0%** | — | ⬇️ 降级 |

### 5.2 混淆矩阵（gold × pred）

```
openness      closed→semi 6, closed→open 1, open→semi 2
leadingness   neutral→mild 8, mild→neutral 3, mild→strong 1
multi         multi→single 2, single→multi 1
follow_up     surface→new 20, surface→deepen 11, surface→n/a 1, deepen→new 7
```

三维相邻档混淆率：openness closed↔semi 6.8%、leadingness neutral↔mild ≤25%、multi 1.9%——**均 <30%，三维不触发降级**。follow_up 的 surface→new/deepen 混淆率 97%，**触发降级**（§7.10 判定条款）。

### 5.3 边缘分布对照——判据 2（≤5pp）

| 维度 | pred vs M0 最大差 | pred vs gold 最大差 | 性质 |
|---|---|---|---|
| openness | 6.0pp | 8.0pp | 判别器有真实偏移（closed→semi 6 条） |
| leadingness | 11.0pp | **3.0pp** | 假漂移：pred≈gold，偏离的是 M0 |
| multi_barreled | 15.0pp | **1.0pp** | 假漂移：pred≈gold，偏离的是 M0 |
| follow_up | 32.0pp | 32.0pp | 降级（gold=M0） |

**判据 2 在形式上全部超 5pp，但 leadingness / multi 的「漂移」是 M0 三维偏差的镜像，不是判别器漂移**（pred vs gold 仅 1–3pp）。openness 的 6pp 里有约 8pp 是判别器真实偏移（把「Any…列出」类 closed 判 semi），需正视。

### 5.4 置信度-准确率校准曲线——交付 4

判别器自评置信度**全部落在 [0.85, 1.00]**，均值 0.95–0.99，唯一值仅 3–8 个。五个分箱只有 [0.8,1.0] 有样本，其余四箱为空。

**结论：置信度无区分度，是摆设**（违反派单「不得让置信度成为摆设」）。无法据此给出「低于某阈值拒答」——模型对自己判错的那 9–12% 同样报 0.9+ 置信度。补救方向（本单未做，属调优）：改 logprobs top-token 概率，或 temperature>0 多次采样取一致性。

### 5.5 延迟——判据 4（<500ms）

| 模型 | 判别单次延迟（mean） | 裸请求延迟（"reply OK"） |
|---|---|---|
| qwen-plus（判别器） | **2008ms** | 318ms |
| deepseek-chat（对照） | 795ms | — |
| qwen-flash（对照） | 766ms | 229ms |

**判据 4 不满足**。延迟瓶颈 = 生成四维 JSON（约 150 token）的推理时间，非网络（裸延迟仅 229–318ms）。**500ms 与同源约束存在结构性冲突**：非 flash 的可用模型（plus/deepseek）均超 500ms，只有同源的 flash 相对快（766ms 也超）。要 <500ms 需流式提前截断或本地小模型。

### 5.6 缓存——判据 3

✅ 已实现（键 = 规范化问句 + prev），同输入同输出，断点可复用（`disc_cache.jsonl`）。

## 6. 重大发现：M0 标注三维系统性偏差（超出派单预期）

两模型一致率 + 逐条语义复核共同坐实：**M0（qwen-flash）标注在三维有系统性偏差**：

1. **openness**：把 wh 问句判 closed、yes/no 问句判 semi（6/6 分歧 deepseek 对）。
2. **multi_barreled**：把 A-or-B 双问句系统性漏判 single（16/16 分歧 deepseek 对；M0 全库 multi 仅 264 条，真实占比被低估约 16pp）。
3. **leadingness**：过度「引导化」，把普通问句判 strong/mild（21 条分歧里 M0 方向偏「强引导」）。

**影响**：M0 标注是条件概率表（`simulator_probs_v3`）的校准基础，三维偏差会沿校准传导进模拟器。这不是本单职责，但必须上报——「边缘分布 vs M0 ≤5pp」判据在此前提下会**误伤判别器**（判别器 pred≈gold 反而被判定「漂移」）。

## 7. 失败模式自评

- **openness 边界**：`Any medications I should know about?` 语义是「列药物」（semi）但 M0 定义「Any…」=closed；`How is your pain today?` 是自由描述（open）判别器判 semi。closed/semi/open 灰色地带占 8pp 偏移。
- **leadingness 边界**：`Does this make the pain worse?`（暗示更糟）、`So is your head hurting?`（暗示头疼）——「是否暗示答案」的强度判断主观性强，neutral↔mild 是主失败带。
- **follow_up surface 档判不出**：判别器对 surface 命中 0/32。surface（重复/换说法重问）与 deepen（深入患者刚说的）、new（新话题）在「系统问诊连续问句」场景下边界塌陷——这是定义问题，不是模型问题。
- **置信度**：模型自评置信度恒 ≥0.85，无拒答价值。

## 8. 四条判据汇总

| # | 判据 | 阈值 | 结果 | 判定 |
|---|---|---|---|---|
| 1 | 逐条一致率 | ≥0.90 | open 91 / lead 88 / multi 97 / fu 61 | ❌（lead 差 2pp，fu 降级） |
| 2 | 边缘分布 vs M0 | ≤5pp | 6 / 11 / 15 / 32 pp | ❌ 形式超（lead/multi 为假漂移） |
| 3 | 缓存同输入同输出 | — | 已实现 | ✅ |
| 4 | 单次延迟 | <500ms | 2008ms | ❌ |

## 9. 需主控裁定（5 项）

1. **leadingness 88%（差 2pp）**：接受 0.88，还是换新评测集复评调优？派单禁止在本 100 条上调 prompt/阈值。
2. **follow_up 降级后**：M3 的 follow_up 相关判据如何改？（判据条款「相邻档混淆率>30% 降级」已触发）
3. **边缘分布判据基准**：M0 三维有偏差，「vs M0 ≤5pp」会误伤判别器（pred≈gold 仍超）。是否改为「vs 仲裁后真值」？
4. **延迟 500ms**：非 flash 模型均超（plus 2008 / deepseek 795 / flash 766）。是否放宽，或换流式/本地小模型？
5. **置信度无区分度**：是否改用 logprobs / 多次采样一致性重构置信度？（否则「拒答阈值」交付空转）

## 10. 交付物清单

| 文件 | 内容 |
|---|---|
| `src/discriminator.py` | 判别器实现（qwen-plus + 缓存 + 延迟记录） |
| `scripts/m3_0a_eval_set.py` | 评测集构建（分层抽 100 条） |
| `scripts/m3_0a_label_2nd.py` | deepseek 独立标注 |
| `scripts/m3_0a_adjudicate.py` | 分歧仲裁 → 真值 |
| `scripts/m3_0a_eval.py` | 判别器验收（四项数字） |
| `outputs/m3_0a/eval_set.jsonl` | 100 条评测集 |
| `outputs/m3_0a/gold_labels.jsonl` | 仲裁后真值 |
| `outputs/m3_0a/disagreements.json` | 103 条分歧项 + 逐条理由 |
| `outputs/m3_0a/inter_model_agreement.json` | 两模型一致率 |
| `outputs/m3_0a/eval_result.json` | 判别器四项数字全量 |
