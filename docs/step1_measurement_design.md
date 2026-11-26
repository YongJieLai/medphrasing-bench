# 第一步设计：在真实医患对话上测量「提问措辞 → 患者披露」的效应量

> 目标：把当前「溢出」从 LLM prompt 的循环论证，升级为**真实数据上的经验命题**。
> 即：真实临床对话里，医生的提问措辞（怎么问）是否、以及在多大程度上，改变患者自发披露的信息（量 / 准确性 / 完整性 / 相关性）。

---

## 1. 数据源决策（已实测）

| 候选 | 结论 | 原因 |
|---|---|---|
| MedDialog（2501.09484 同款） | ❌ 沙箱不可用 | 数据在 Google Drive（沙箱直连 HTTP 000），HF 镜像无 parquet 缓存 |
| 2501.09484 开源仓库 | ❌ 空壳 | `LIO-H-ZEN/PatientSimulator` 仅 3KB，只给了 LoRA 权重（Qwen2.5-72B adapter，本地跑不动），标注管线代码不可复用 |
| **MTS-Dialog** | ✅ **已下载，采用** | 见下 |

**MTS-Dialog**（Ben Abacha et al., 2021，英文真实医患对话）实测规模：

| 指标 | 值 |
|---|---|
| 总对话数 | **1501**（train 1201 / val 100 / test 200） |
| 总 utterance | 13763（医生 7104 + 患者 6659） |
| 平均每对话轮数 | 9.17 |
| **医生提问句** | **5425 条（占医生 utterance 76.4%）** |
| 措辞规则粗分 | open 1207 / closed 2076 / leading 181 / 复合 324 / other 1961 |

> 关键点：76.4% 的医生 utterance 是提问，措辞天然覆盖开/闭/引导/复合/追问全部形态，且逐轮 `Doctor: / Patient:` 对齐 —— 这是四维标注的理想底料。

---

## 2. 变量定义

### 2.1 自变量：医生提问的四个措辞维度（每条医生提问标 4 个）

| 维度 | 档位 | 判据 |
|---|---|---|
| **开放度 open-endedness** | open / semi / closed | open=要求自由作答（What/How/Describe）；semi=限定领域但自由作答（What meds are you taking?）；closed=是/否/确认（Do you have…?） |
| **引导性 leadingness** | neutral / mild / strong | neutral=中立；mild=带预设但可否认（Is the pain sharp?）；strong=暗示答案（…, correct? / …right?） |
| **复合性 multi-barreled** | single / multi | multi=一句话含 ≥2 个信息请求（Do you have fever, cough, or chest pain?） |
| **追问深度 follow-up** | surface / deepen / new | surface=换词重复；deepen=把患者上句的模糊描述具体化（"你说头疼，具体哪种疼？"）；new=新话题 |

### 2.2 因变量：患者回答的四个披露属性（每条患者回答标 4 个）

| 属性 | 度量 | 操作化 |
|---|---|---|
| **披露量 quantity** | 连续 | 回答中的信息单元（AIU）数；词数作粗代理 |
| **披露准确性 definiteness** | 序数 3 档 | exact（确切陈述） / vague（模糊） / agree（附和医生预设，无新信息） |
| **披露完整性 completeness** | 0/1（或比例） | 是否回答了医生问到的**所有**子问题（复合提问时尤其关键） |
| **披露相关性 relevance** | 序数 3 档 | on-topic（只答所问）/ spillover（主动多说相关但未问信息）/ off-topic |

### 2.3 核心待验证假设（效应方向）

| 医生维度 | 预期作用于 | 假设方向 |
|---|---|---|
| 开放度 ↑ | 披露量 ↑ | open 的 AIU 数 > closed |
| 引导性 ↑ | 披露准确性 ↓ | strong leading 更易得 "agree"（附和，无独立信息） |
| 复合性 ↑ | 披露完整性 ↓ | multi-barreled 漏答率 ↑（患者只答最后一个） |
| 追问深度 ↑ | 披露准确性 ↑ / 深度 ↑ | deepen 后患者回答更 exact |

---

## 3. 标注方案

- **标注单元**：`(医生 utterance, 下一句患者 utterance)` 有序对，共 5425 条提问 + 对应回答。
- **执行方式**：qwen-flash 结构化标注（输出 JSON），人工抽检一致性。
- **一致性门槛**：抽 200 条，LLM vs 人工 双标，逐维度算 Cohen's κ / 加权 κ，≥ 0.7 通过；不通过的维度细化 rubric 重标。
- **防泄漏**：标注时不给诊断金标准，只看对话文本本身。

---

## 4. 效应量测量方案（统计）

- **披露量（连续）**：按维度分组的均值差 + 效应量 **Cohen's d**；开放度用 3 档做趋势（ANOVA + 线性对照）。
- **准确性 / 相关性（序数）**：比例差异 + **OR / Cliff's delta**；引导性强弱 → "agree" 占比的 Cochran–Armitage 趋势检验。
- **完整性（二值）**：multi vs single 的漏答率差 + χ² / 风险差。
- **关键：对话内相关性** —— 同一对话的多条提问不独立，必须用**混合效应模型**（`accuracy ~ openness + (1|dialogue_id)`，logistic/linear mixed），否则会低估标准误、虚假显著。至少用 cluster-robust SE（按 dialogue_id 聚类）。
- **预算内先做**：qwen-flash 标 5425 条 ≈ 5400 次调用，几元以内。

---

## 5. 执行计划

| 阶段 | 内容 | 产出 |
|---|---|---|
| P0 | 解析 MTS-Dialog（已完成） | `parsed_dialogues.json` |
| P1 | 抽 200 条做金标准双标 + 一致性 | rubric 定稿 + κ 报告 |
| P2 | 全量 LLM 标注 5425 条 | 标注数据集 |
| P3 | 效应量测量 + 混合效应模型 | 「四维 × 四属性」效应量表 |
| P4 | 与现有工作对齐写结论 | 是否支撑「溢出=提问质量的函数」 |

---

## 6. 风险与待决策点

1. **语言**：MTS-Dialog 是英文（与 DDxPlus 对齐，可直接开干）；中文 MedDialog 需你手动下载上传。默认用 MTS。
2. **引导性样本偏少**（规则粗分仅 181 条）：全量 LLM 标注后会更准，但可能不足统计功效，届时需补采样或放宽到 "mild+strong" 合并。
3. **披露准确性/完整性的标注主观性**：需在 P1 用精细 rubric + 人工金标准锁定。
4. **MTS-Dialog 许可**：原始论文公开，使用前需确认 license（当前按学术研究使用，不商用）。
5. **"other" 1961 条**（规则未识别）：必须靠 LLM 标注补全，不能丢弃——它们可能含大量隐性提问。

> 待你拍板：① 默认用英文 MTS-Dialog 直接开干？② 金标准标注的 200 条人工核对，你有临床背景的人可协助吗（没有就先 LLM 双模型交叉标注 + 我抽检）？
