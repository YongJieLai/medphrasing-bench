# 患者模拟器契约 v1

> 状态：**待评审**（主控起草 2026-11-27）
> 适用：把本项目的患者模拟器接入 EID / AgentClinic 式问诊任务
> 一句话：**这份文档定义「患者拿到什么输入、吐什么输出、中间由谁算」，执行侧照此实现，不再各自解释。**

---

## 0. 这份文档要解决什么

前几轮反复出现同一类问题：模拟器接进新任务后，各环节对"患者该说什么"的理解不一致，导致结果无法归因——分不清是措辞效应、是实现 bug、还是任务设定。

这份契约把所有可争议的点**在实现之前固定下来**：字段名、取值域、由哪一层产生、哪些是校准过的、哪些是先验假设。

---

## 1. 设计目标（唯一的）

> 让「医生的询问方式」能够**逐环传导**到「诊断是否正确」，且每一环都可观测、可证伪。

四环：

```
① 医生问法 → ② 患者反应 → ③ 医生掌握的关键证据 → ④ 诊断正确率
```

**任何设计选择，只要削弱其中任一环的可观测性，就不采用。**

### 1.1 已有的两个致命退化，本契约必须避开

| 退化 | 表现 | 根因 | 本契约的对策 |
|---|---|---|---|
| **精确患者** | 问什么答什么，从不多说、从不回避 | EID 的 `PATIENT_SYSTEM` 写死 "if you are asked" / "unsure or do not recall" | `stance` 字段 + `salience` 字段 |
| **教科书患者** | 优先说对诊断最关键的事实 | 选择器若按 `is_key` 打分就会这样 | **选择器禁止读 `is_key`**（§7.3） |

---

## 2. 接口总图

```
┌──────────────────────────────┐
│ 输入 A  事实池状态             │
│   池级：剩余条数 / 最高代价 / 连续封闭轮数 │
│   条目级：每条事实的 5 个属性    │
└──────────────┬───────────────┘
               │
┌──────────────┴───────────────┐
│ 输入 B  医生这一句的标签        │
│   openness / leadingness / multi_barreled │
│   question_form / back_chunk / empathy    │
└──────────────┬───────────────┘
               │
┌──────────────┴───────────────┐
│ 输入 C  上一轮的状态            │
│   last_stance / consecutive_closed        │
└──────────────┬───────────────┘
               ▼
        ┌─────────────┐
        │  权重层      │  三层分工（§7）
        │ 表 / 规则 / 排序 │
        └──────┬──────┘
               ▼
┌──────────────────────────────┐
│ 输出  反应标签                 │
│   n_items / definiteness / completeness │
│   relevance / stance / affect            │
│   items_said / items_withheld            │
└──────────────┬───────────────┘
               │  ＋ 选中事实的第一人称文本
               ▼
        ┌─────────────┐
        │  渲染层 LLM  │ → 口语 → 医生
        └──────┬──────┘
               ▼
        回抽验证（必须，F1 ≥ 0.90）
```

---

## 3. 输入 A：事实池状态

### 3.1 池级（每轮重算）

| 字段 | 类型 | 含义 |
|---|---|---|
| `n_total` | int | 事实总条数 |
| `n_remaining` | int | 尚未披露的条数 |
| `n_key_remaining` | int | 尚未披露的**关键**事实条数（仅统计用） |
| `max_cost_remaining` | `low/mid/high` | 未披露事实里的最高披露代价 |
| `consecutive_closed` | int | 连续被封闭式提问的轮数 |
| `turns_elapsed` | int | 已进行轮数 |

### 3.2 条目级（每条事实一个对象）

```python
{
  "id": "F03",
  "text_raw": "The patient reports a 1-month history of double vision.",   # EID 原文
  "text_first_person": "I've been seeing double for about a month.",       # 改写后，渲染层只用这个
  "is_key": true,          # 来自 key_facts.json，只用于统计，选择器禁读
  "cost": "low",           # 披露代价：low / mid / high
  "elicitation": "spontaneous",  # spontaneous（患者会主动说）/ prompted（必须问）
  "salience": 4,           # 患者主观重要性 1–5
  "disclosed": false       # 运行时状态
}
```

### 3.3 `cost` 的三档定义（**标注前必须冻结**）

| 档 | 定义 | 例子 |
|---|---|---|
| `low` | 说出来没有负面后果 | 症状、年龄、职业、病程 |
| `mid` | 说出来可能被评价，但不严重 | 睡眠差、饮食不规律、工作压力 |
| `high` | 说出来可能招致负面判断或后果 | 吸烟饮酒量、用药不依从、情绪/精神症状、性史、体重、经济困难、家庭关系 |

> **注意**：`high` 不等于"污名病"。判定标准是「患者主观上认为说出来对自己不利」，不是「社会污名度」。
>
> 依据：EID 200 例扫描显示，污名类事实（性病/吸毒/精神）仅占关键事实 **5.4%**，撑不起主效应；而"披露代价"覆盖面大得多。

### 3.4 `salience` 的定义

> 患者**自己**觉得这条有多重要，与它的诊断价值**无关**。

| 分值 | 含义 |
|---|---|
| 5 | 这次就诊最困扰我的事 |
| 3 | 有明显影响，但不是主要问题 |
| 1 | 老毛病，我都不当回事 |

**这是本设计最关键、也最容易被做错的字段。** 没有它，患者会变成教科书病人——有问必答、答必切题，与 EID 的精确患者无差别，整条链路的差异点消失。

### 3.5 `elicitation` 的定义

| 值 | 含义 | 例 |
|---|---|---|
| `spontaneous` | 患者会在开场或相关话题时**主动**说出 | 主诉、"最难受的是……" |
| `prompted` | 只有被明确问到才说 | 既往史、阴性症状、社会史 |

建议每例至少 2 条 `spontaneous`，且其中至少 1 条同时 `is_key=true`。否则开场话语权的效应无从体现（开放组开场能白拿关键证据，封闭组拿不到）。

---

## 4. 输入 B：医生这一句的标签

由判别器实时产出（M3-0a 已验收版本）。

```python
{
  "openness": "open | semi | closed",
  "leadingness": "neutral | strong",     # mild 已塌陷（档内一致率 5.9%），并入 neutral
  "multi_barreled": 0 | 1,
  "question_form": "yesno | wh | option | imperative",
  "back_chunk": 0 | 1,
  "empathy": 0 | 1
}
```

| 字段 | 状态 | 说明 |
|---|---|---|
| `openness` | ✅ 已验收（91%） | |
| `multi_barreled` | ✅ 已验收（97%） | |
| `leadingness` | ⚠️ 仅二分 | 不区分 mild/strong 的连续强度 |
| `question_form` | 🆕 **待实现** | 见下 |
| `back_chunk` | 🆕 **待实现** | 见下 |
| `empathy` | 🆕 **留接口，v1 不实现** | |
| `follow_up` | ❌ 已剔除 | 单句定义下不可判（61%），不纳入 |

### 4.1 `question_form` 为什么必须单列

`openness` 把两类完全不同的封闭问句混成了一种，而它们效果相反：

| 问句 | openness | 效果 |
|---|---|---|
| "疼吗？" | closed | 患者答一个字 |
| "刺痛、胀痛还是烧灼痛？" | closed | **患者能说出精确性质** |

选项列举**给了患者词汇**。没有词，患者只能说"就是疼"。这在诊断上差别巨大（压榨性 vs 针刺样指向不同疾病）。

> 这同时解释了预检的一个疑点：封闭组医生大量使用选项列举。若它实际提升了信息量，"封闭式 = 信息少"这个前提本身就是错的。

### 4.2 `back_chunk` 为什么必须单列

它是**唯一能触发"补充遗漏"的机制**。其它所有维度影响"这次说几条"，只有它影响"之前漏的还能不能补上"。

判定特征：医生复述患者已说的内容并请求确认（"所以你是说……对吗"）。可判，不需时间戳。

### 4.3 显式不做

| 维度 | 原因 |
|---|---|
| `silence` / `wait time` | 需要时间戳，对话记录没有 |
| `interruption` | 同上 |
| `politeness` / `hedging` | 与 `leadingness` 高度共线，收益低 |

**记为已知局限，不假装能做。**

---

## 5. 输入 C：上一轮状态

| 字段 | 来源 | 作用 |
|---|---|---|
| `last_stance` | 上一轮输出 | 刚回避过 → 本轮更易继续回避 |
| `last_was_back_chunk` | 上一轮输入 B | 刚被回述 → 本轮更易 `elaborate` |
| `consecutive_closed` | 池级累计 | 连续封闭 → 主动披露倾向下降 |

> 累积效应（被打断后不再回到原话题，Beckman & Frankel 1984）**我们没有数据校准**。v1 实现为可开关参数，默认关闭，用消融实验单独测贡献。

---

## 6. 输出：反应标签

```python
{
  "n_items": 0,                              # 本轮说几条
  "definiteness": "exact|vague|agree",
  "completeness": "full|partial|last_only",
  "relevance": "on_topic|spillover|off_topic",
  "stance": "cooperate|avoid|deny|elaborate",
  "affect": "calm|anxious|ashamed|impatient",
  "items_said": ["F03", "F07"],
  "items_withheld": [{"id": "F11", "reason": "cost_high"}]
}
```

### 6.1 `stance` —— 本契约最重要的新增

| 值 | 含义 | EID 的患者能做吗 |
|---|---|---|
| `cooperate` | 问了就答 | ✅ |
| `avoid` | **有这个信息，但不想说**（"没什么特别的"） | ❌ **不会** |
| `deny` | 防御性否认（"我没有"） | ❌ 不会 |
| `elaborate` | 主动补充之前漏的 | ❌ 不会 |

**"不知道"和"不想说"必须分开。** EID 的患者只会前者，而真实误诊的大量来源是后者。

### 6.2 `items_withheld` —— 缺失证据的直接度量

这是整条链路的**物理载体**。缺失必须分三类，混在一起就无法归因：

| `reason` | 含义 | 干预手段 |
|---|---|---|
| `not_asked` | 医生没问，患者也没主动说 | 换问法（更开放 / 回述） |
| `withheld` | **问了，患者回避** | 共情、降低代价、换封闭式直接问 |
| `exhausted` | 池子空了 | 无（任务设定问题） |

> 只有 `withheld` 是"询问方式"造成的。`not_asked` 是"没问到"。两者相加才是"缺失"，但归因完全不同。

---

## 7. 权重层：三层分工

**不能用同一机制算所有标签**，否则会重犯"LLM 自己判自己"的错。

| 层 | 算什么 | 机制 | 为什么必须是这个 |
|---|---|---|---|
| **统计层** | `n_items` / `definiteness` / `completeness` / `relevance` | **条件概率表 + 采样** | 查表**可证伪**（能算预期、能对不上）。换 LLM 就不可证伪 |
| **规则层** | `stance` | **if-else 规则** | 无校准数据，只能用先验。**必须标"假设"** |
| **选择层** | `items_said` | **打分排序** | 已有的 `aiu_selector` |

### 7.1 统计层（校准过，可证伪）

```python
n_items, definiteness, completeness, relevance = sample_behavior(
    openness, leadingness, multi_barreled,
    remaining_n=n_remaining,
    turn_index=turns_elapsed,
    rng=rng,
)
```

- 表来自 MTS-Dialog 校准（M2b），**不得用 LLM 替代**
- `remaining_n=0` 时 `n_items` 必须为 0（不得出现"剩 0 条还能说 2.56 个"的历史 bug）
- 非法取值必须**抛异常**，不得静默 fallback

### 7.2 规则层（**未校准先验，禁止写成"发现"**）

```python
def decide_stance(cost, openness, leadingness, empathy, back_chunk, pool, rng):
    if back_chunk and pool.has_undisclosed_related():
        return "elaborate"
    if cost == "high":
        if openness == "open":        return "avoid"      # 要自己主动承认，代价太大
        if leadingness == "strong":   return "deny"       # 防御性否认
        if empathy:                   return "cooperate"  # 共情打开了
        return "cooperate" if rng.random() < 0.6 else "avoid"
    if cost == "mid":
        return "cooperate" if rng.random() < 0.85 else "avoid"
    return "cooperate"
```

> 上面所有概率（0.6 / 0.85）都是**编的**。本节必须原样标注为 `PRIOR, UNCALIBRATED`，论文与报告中不得表述为"真实患者的回避率"。

### 7.3 选择层（防退化的关键约束）

```python
score = w1 * normalize(salience) + w2 * relevance_to_question(doctor_utt, fact)
```

**硬约束：选择器不得读 `is_key`。**

一旦读了，患者会优先说诊断最关键的事实——变成教科书病人，与 EID 的精确患者无差别，整条链路的差异点消失。`is_key` 只允许在**事后统计**中使用。

---

## 8. 渲染层约束

### 8.1 输入

```
要说的事实（只给选中的 n_items 条，原文照贴）：
  - I've been seeing double for about a month.
  - My eyelids feel heavy by evening.

反应标签：definiteness=vague, stance=avoid, affect=anxious

已说过、不要重复：F01, F02

硬约束：
  1. 只说上面列出的事实，一条不多、一条不少
  2. 不得新增任何未列出的症状、病史、检查结果
  3. 不得复述"已说过"列表中的内容
  4. 1–3 句，第一人称口语
```

### 8.2 必须回抽验证

渲染层会把 `n_items` 冲掉（LLM 天然倾向于"说全"）。M2c 已踩过一次。

| 检查 | 阈值 | 不过怎么办 |
|---|---|---|
| 回抽事实数 == `n_items` | 必须相等 | 报警并重渲染（最多 2 次） |
| 回抽事实集合 == `items_said` | F1 ≥ 0.90 | 记录偏差，进分析时作为噪声项 |
| 是否出现未授权事实 | 0 容忍 | 该轮作废重渲染 |

---

## 9. 字段字典（权威表）

| 字段 | 层 | 取值 | 来源 | 是否校准 |
|---|---|---|---|---|
| `openness` | 输入 B | open/semi/closed | 判别器 | ✅ 91% |
| `leadingness` | 输入 B | neutral/strong | 判别器 | ⚠️ 二分 |
| `multi_barreled` | 输入 B | 0/1 | 判别器 | ✅ 97% |
| `question_form` | 输入 B | yesno/wh/option/imperative | 判别器（新增） | 🆕 待验 |
| `back_chunk` | 输入 B | 0/1 | 判别器（新增） | 🆕 待验 |
| `empathy` | 输入 B | 0/1 | 未实现 | 🆕 v1 不做 |
| `cost` | 输入 A | low/mid/high | 标注（新增） | 🆕 待标 |
| `elicitation` | 输入 A | spontaneous/prompted | 标注（新增） | 🆕 待标 |
| `salience` | 输入 A | 1–5 | 标注（新增） | 🆕 待标 |
| `is_key` | 输入 A | bool | `key_facts.json` | ✅ 已有 |
| `n_items` | 输出 | 0–4 | 统计层 | ✅ M2b |
| `definiteness` | 输出 | exact/vague/agree | 统计层 | ✅ M2b |
| `completeness` | 输出 | full/partial/last_only | 统计层 | ✅ M2b |
| `relevance` | 输出 | on_topic/spillover/off_topic | 统计层 | ✅ M2c-b |
| `stance` | 输出 | cooperate/avoid/deny/elaborate | **规则层** | ❌ 先验 |
| `affect` | 输出 | calm/anxious/ashamed/impatient | 规则层 | ❌ 先验 |

---

## 10. 每轮必须落盘的量

```json
{
  "turn": 3,
  "doctor_utt": "...",
  "tags": {...},
  "state_before": {"n_remaining": 9, "consecutive_closed": 2},
  "behavior": {"n_items": 1, "stance": "avoid", "relevance": "on_topic"},
  "items_said": ["F05"],
  "items_withheld": [{"id": "F09", "reason": "withheld"}],
  "rendered": "...",
  "extraction": {"n": 1, "f1": 1.0, "unauthorized": []}
}
```

**缺任何一项，事后就无法做中介分析。**

---

## 11. 不做什么（硬性禁止）

| 禁止 | 理由 |
|---|---|
| 让 LLM 决定"说几条" | 不可证伪；统计层必须是表 |
| 选择器读 `is_key` | 退化成教科书病人，链路差异消失 |
| 渲染 prompt 里给全部事实 | LLM 会把 `n_items` 冲掉 |
| 把规则层的概率写成"发现" | 那是编的，不是估的 |
| 用 `remaining_n=0` 时仍返回 `n_items>0` | 历史 bug，会伪造"无中生有" |
| 跳过回抽验证 | M2c 已证明 LLM 会多说 |

---

## 12. 待校准项（明确列出，避免被当作已知）

| 项 | 现状 | 校准路径 |
|---|---|---|
| `cost` 三档边界 | 未标 | 2546 条标注 + 人工抽 30 条 |
| `salience` 评分标准 | 未标 | 同上 |
| 规则层概率（0.6 / 0.85） | **编的** | 无法用现有语料校准（MTS-Dialog 无此标注）；只能做敏感性分析 |
| 累积封闭效应 | 无数据 | 实现为开关，默认关，消融测 |
| 外推比例 | 未知 | 条件表来自 3.9 轮短对话，本任务是 16+ 轮长对话。须报"落在观测区间外的格子占比"，>40% 则结论只能表述为"在该参数族预测下" |

---

## 13. 本契约如何支撑「证明询问方式影响诊断」

四段观测，缺一不可：

| 段 | 观测 | 通过标准 |
|---|---|---|
| ① 操纵有效 | 两组医生问句的 `openness` 分布 | 差 ≥ 40pp（预检已验：85pp ✅） |
| ② 患者反应不同 | `n_items` 均值、`stance=avoid` 比例 | 显著差异 |
| ③ 关键证据不同 | **`withheld_key` 条数**（有关键事实但被回避） | 显著差异 |
| ④ 诊断不同 | 准确率 | 显著差异，且 ③ 能解释 ④ 的差异 |

**核心指标是 `withheld_key`**，不是总披露量。

理由：预检已经证明"总披露量"被饱和抹平（12 轮问完 10 条，两组都能问全）。而 `withheld_key` 不受饱和影响——它是"有但没说"，与池子大小无关。

### 必须做的消融

抹平统计层里 `openness` 的依赖（各档用同一披露分布），重跑。若 ②③④ 的差异一起消失，才能说"是问法造成的"。

---

## 附：本契约与既有实现的关系

| 既有模块 | 本契约中的位置 | 改动 |
|---|---|---|
| `src/conditional_simulator.py` | 统计层 | 加 `question_form` 维度（可选） |
| `src/disclosure_ledger.py` | 输入 A 条目级状态 | 加 `cost` / `salience` / `elicitation` |
| `src/aiu_selector.py` | 选择层 | **移除 `is_key` 依赖** |
| `src/renderer.py` | 渲染层 | 加硬约束与回抽 |
| 新增 `stance_rules.py` | 规则层 | 全新 |
