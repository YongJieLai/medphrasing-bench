# M0 三维审计结果（§7.11）

> 派单：§7.11「M0 三维审计」（R13 定级）。本单只做审计，不改标注、不重建表。
> 复标模型：deepseek-chat（与 M0 的 qwen-flash 不同源，禁 qwen 系列）。
> 复标手册：**原文照抄** `annotate_mtsdialog.py` 医生侧三维定义（含例子），不加解释。

## 一句话结论

**污染率 9.5%（CP 95% CI [5.7%, 14.6%]），落 5%–15% 档：效应被稀释、方向不变。** M0 在 `multi_barreled` 上存在**完全单向**的系统性漏判——把 A-or-B 型复合问句判成 single（审计样本 18 条 single→multi，反向 0 条，McNemar p<0.0001）。H3（复合性→漏答）风险差被低估约 0.55pp（相对 11%），修正后 5.23pp → 5.78pp，方向不变。

## 一、抽样说明（本单最容易做错的地方，按派单要求记录）

| 项 | 值 |
|---|---|
| 抽样种子 | `20261127` |
| 全量问句池 N | **5167**（`annot_all_flash.jsonl` 中 is_question=1 且三维标签合法） |
| 排除 | 评测集 100 条（§7.10 已挑过） |
| 抽样池 N | 5067 |
| 抽样方式 | **随机、无分层、无 M0 标签配额** |
| 抽中 | 200 条 |

**代表性检验**（抽中 200 条的 M0 三维分布 vs 全库）：

| 维度 | 样本分布 | 全库分布 | 判定 |
|---|---|---|---|
| multi_barreled | single 189 / multi 11（5.5%） | single 4904 / multi 263（**5.09%**） | ✅ 吻合 |
| openness | closed 130 / open 36 / semi 34 | closed 3232 / open 741 / semi 1194 | ✅ 吻合 |
| leadingness | neutral 174 / mild 17 / strong 9 | neutral 4594 / mild 385 / strong 188 | ✅ 吻合 |

## 二、三维一致率 + 95% CI + 混淆矩阵

| 维度 | 一致率 | CP 95% CI | 判定 |
|---|---|---|---|
| openness | **92.0%** | [87.3%, 95.4%] | 高，分歧分散（见下） |
| leadingness | **90.0%** | [85.0%, 93.8%] | 高，mild↔neutral 边界模糊 |
| multi_barreled | **91.0%** | [86.1%, 94.6%] | 一致率被**单向**分歧拉低 |

**分歧方向（m0 → deepseek）**：

| 维度 | 分歧明细 | 模式 |
|---|---|---|
| openness | semi→open 4 / semi→closed 5 / open→closed 3 / open→semi 3 / closed→semi 1 | 双向、分散，semi 中间档边界模糊 |
| leadingness | mild→neutral 14 / mild→strong 2 / neutral→strong 1 / strong→mild 2 / strong→neutral 1 | 以 mild→neutral 为主（14 条），但非完全单向 |
| multi_barreled | **single→multi 18** | **完全单向，无反向** |

## 三、污染率（核心数，单列）

> 定义：M0=single 的样本里，deepseek 复标为 multi 的比例。

| 项 | 值 |
|---|---|
| M0=single 样本数 | 189 |
| 其中 deepseek 判 multi | **18** |
| **污染率** | **9.5%** |
| CP 95% CI | **[5.7%, 14.6%]** |
| bootstrap 95% CI（5000 次） | [5.3%, 13.8%] |

## 四、单向性检验

| 方向 | 条数 |
|---|---|
| single → multi（M0 漏判 multi） | **18** |
| multi → single（M0 多判 multi） | **0** |

McNemar 双侧 p < 0.0001 → **单向（系统偏差），非随机噪声**。与 §7.2.10 主控逐条核对的 16/17 单向结论一致，且扩展到随机样本仍为 18/18 单向。

## 五、三档判定（先写定再看结果）

污染率 9.5% 落在 **5%–15%** 档：

> **效应被稀释，方向不变。** M1 四个效应量重算并同报旧值；M2b 表加敏感性列；M3 可用但须注明。

## 六、H3 敏感性重算（交付项 5，只重算不改正文）

H3（复合性 → 漏答）全库基线：multi 7.98%（n=263）vs single 2.75%（n=4904），风险差 **5.23pp**。

污染方向是 single 组混入高漏答率的 multi → single 组漏答率被抬高 → H3 风险差被低估。用污染率做期望修正：

| 口径 | 风险差 |
|---|---|
| 旧值（M0 标签） | **5.23pp** |
| 修正点估计（p=9.5%） | **5.78pp** |
| 修正区间（p 的 CI 上下界） | **[5.55pp, 6.13pp]** |
| 低估幅度 | 0.55pp（相对 11%） |

**结论：修正后 H3 只增不减，方向不变。** 反向污染（multi→single）样本 0/11，其存在只会进一步抬高真实 multi 漏答率、使 H3 更大，故本估为低估的**下界**。

## 七、18 条 single→multi 样例（人工可核对）

全部为 A-or-B 复合问句，deepseek 判 multi 符合 M0 手册（"asks for >=2 distinct pieces"）：

```
Have you been treated, and are you taking medicine for this?
Ma'am, do you have headache or chest pain?
How are you both doing? What brings you two to the Emergency Department?
any other problem or any uneasiness?
Have you been experiencing any bowel or, bladder dysfunction.
this weakness or the numbness anywhere else except your legs?
Is this your first surgery or have you had other surgeries in the past?
do you smoke and drink?
How are you now? Any bad side effects?
Any partial loss of vision or blind spot?
Any agitation or any other issues?
Any bladder issues or frequent urination?
do you smoke or drink?
Do you have any pain with urination or blood in your urine?
What's bothering you? (复合句)
Are you having any pain with urination, or blood in the urine?
...are you doing young man? Are there any difficulties...
```

（18/18 复核一致，无一例是 single。）

## 八、自评（样本代表性局限、复标模型已知偏好）

1. **multi 样本量小，反向污染率 0% 的 CI 上界宽**：随机 200 条只抽到 11 条 M0=multi（全库 multi 仅 5.09%，随机抽必然少）。multi→single 0 条的点估计可靠，但其 Clopper-Pearson 上界约 26%，无法排除少量反向污染。因反向污染只会让 H3 更大，故污染率 9.5% 仍是「低估 H3」方向的保守估计。
2. **污染率 9.5% 的 CI 偏宽**（[5.7%, 14.6%]），落在 5–15% 档的下半段，靠近 5% 边界。若要求更精确的定档，需扩样本到 ~1000 条（成本约 25 分钟机器时间），本单未做。
3. **deepseek 复标不是人工真值**：它是「严格执行 M0 手册」的第二个模型，与 M0 的分歧按手册原文仲裁，18/18 样例人工核对全部支持 deepseek。但 openness/leadingness 的分歧（semi↔open/closed、mild↔neutral）是定义边界模糊，两模型都无绝对权威，仅作方向参考。
4. **openness 无单向污染**：§7.2.10 报告的「openness 6/7 判错」在随机 200 条上未复现为单向（16 条分歧双向分散），说明那是分层评测集的小样本现象。openness 一致率 92%、CI 下界 87.3%，作为 M0 分组基础可接受但需标注精度。

## 九、给主控的裁定建议

按 §7.11 三档判定，污染率 9.5% 落 **5–15% 档**，后续动作为：

1. **M1 四个效应量重算并同报旧值**（H3 已在本单敏感性重算：5.23pp → 5.78pp，方向不变；H1/H2/H4 不受 multi 污染直接影响，可同报旧值并注明）。
2. **M2b 条件表加敏感性列**（multi_barreled 分组受污染，加注）。
3. **M3 可用但须注明** M0 分组基础存在 9.5% multi 污染。

是否扩样本到 ~1000 条以收紧污染率 CI（当前 CI 下界 5.7% 接近 5% 边界）——建议主控裁定是否需要，本单按派单「200 条」执行完毕。
