# M2d 反事实验收结果（§7.8）

> 复核对象：`scripts/m2d_rewrite.py` + `scripts/m2d_run.py`，数据 `outputs/20261127_m2d/`。
> 本单目的：**第一次检验因果**——改一处措辞、别的不动，看患者披露量是否随之改变。
> 主判据（§7.8）：方向与单调性；幅度只报不判，与主控预测对照。

---

## 一、结论速览

| 判据 | 阈值/预期 | 结果 | 判定 |
|---|---|---|---|
| 改写校验（closed 判对率） | ≥ 0.90 | **0.951**（143 条，136 判 closed） | ✅ |
| 单调性 open > semi > closed | 成立 | 模拟 **2.238 > 2.039 > 1.571** | ✅ |
| 方向（改 closed 降、改 open 升） | 同向 | 降 **0.524**、升 **0.582** | ✅ |
| 幅度（与真实差对照） | 低估 30–40% | 低估 **46–51%** | ⚠️ 超预测，未达 60% 停机线 |

**总结论**：模拟器**认措辞**（因果方向正确、单调性成立），但效应幅度被压缩约一半，来源可解释（见 §四），**未发现未识别的损耗源**。幅度超出主控预测区间但未触发停机线，提请主控裁定是否放行进 M3-0。

---

## 二、改写与校验（§7.8 第 1/2 件）

- holdout：按 `dialogue_id` 划分，seed=42，20% → **270 对话 / 1011 条问句**（与 replay_validation 同口径）。
- 改写方向：open 轮 **143 条全量** → closed；closed 轮抽样 **143 条** → open（对称）。改写用 `builder` 角色（qwen-flash，未换模型）。

| 方向 | 条数 | 判对率 | 判定 |
|---|---|---|---|
| open→closed | 143 | 判为 closed **136** → **0.951** | ✅ ≥0.90 |
| closed→open | 143 | 判为 open **132** → **0.923** | ✅ |

- 改写失败（空/异常）**0 条**。
- 20 条人工核对样例已出（`outputs/20261127_m2d/rewrites_verify.json`，10 open→closed + 10 closed→open），语义等价性**待主控/人工终核**。

改写样例（节选）：

| 原文 | 改写 | 方向 |
|---|---|---|
| What brings you in to see me today? | Are you here today because of a specific health concern? | open→closed |
| Can you tell me about cancer in your family? | Do you have a family history of cancer? | open→closed |
| Do you smoke, sir? | Tell me about your smoking habits. | closed→open |
| Any fever? | Tell me about any fever you've been experiencing. | closed→open |

> 已知小瑕疵：个别 closed→open 改写把第三人称问句（如 "does he have difficulty urinating"）改成了第二人称（"your experience"）。判对率不因此受影响，但语义等价性以人工终核为准。

---

## 三、真实对照 + 三层跑批（§7.8 第 3/4 件）

### 3.1 真实 holdout（观测基线）

| openness | n | iu 均值 |
|---|---|---|
| open | 143 | **2.427** |
| semi | 203 | **1.892** |
| closed | 665 | **1.292** |

- 真实 open − closed 差 = **1.135**，bootstrap 95% CI **[0.893, 1.397]**（对话级重采样，B=2000）。
- **真实耗尽率 = 5.4%**（55/1011 轮），模拟器 8.3% → 模拟器尾部偏厚（印证 §7.2.5 记录二）。

### 3.2 三层跑批（同一对话、同一患者、seed=42、只改 openness 标签）

| 层 | open | semi | closed | overall |
|---|---|---|---|---|
| 原句对照 | 2.238 | 2.039 | 1.571 | 1.760 |
| open→closed | **1.713** | 2.044 | 1.582 | 1.693 |
| closed→open | 2.224 | 2.025 | **2.153** | 2.137 |

- 模拟降幅（open 层改 closed）= 2.238 − 1.713 = **0.524**。
- 模拟升幅（closed 层改 open）= 2.153 − 1.571 = **0.582**。
- 单调性：原句层 open 2.238 > semi 2.039 > closed 1.571 ✅（改写前后都成立）。

### 3.3 幅度对照（只报不判）

| 量 | 数值 | 占真实差(1.135) |
|---|---|---|
| 模拟降幅 | 0.524 | **46.2%** |
| 模拟升幅 | 0.582 | **51.3%** |

主控预测：实测降幅比真实低 30–40%（即占真实差 60–70%）。实测 **46–51%**，**低于预测**，但**高于 60% 停机线**（停机线 = 占真实差 < 40%）。

---

## 四、幅度低估 50% 的来源（为什么是可解释的，不是新损耗）

1. **模拟器对 open/closed 差异的压缩（B2 低估的直接体现）**
   模拟横截面差 = open 2.238 − closed 1.571 = **0.667**，仅真实差 1.135 的 **59%**，即低估 **41%**。
   → 注意：主控预测引用的「B2 低估 29%」是**残量效应斜率**的压缩（0.1145 vs 0.1603），不是 open/closed 差异的压缩；后者实际是 41%，比 29% 更严重。

2. **remaining 状态约束（反事实降幅 < 横截面差）**
   open 轮 remaining 均值 **9.14** vs closed 轮 **7.14**（open 轮多在对话开头、closed 轮在中后段）。
   反事实把 open 轮改成 closed 后，iu 仍被较大的 remaining 支撑，降幅（0.524）低于横截面差（0.667），额外损失约 13%。

3. **关于主控预测里「B3 渲染衰减 19%」**
   B3 的 19% 是**轮间自相关**的衰减（−0.2417 → −0.1949），不是 iu 均值的衰减；且 M2d 跑批在**标签层**（无渲染），渲染对 iu 均值的影响约 5%（回抽 F1 0.95 的失真率），远小于 19%。故「B2×B3」合成对均值幅度的预测偏保守，实测 50% 低估主要由 B2（41%）+ remaining 约束（~13%）构成，**无需假设新的损耗源**。

---

## 五、并排对比（同一对话 open vs closed）

对话 `id=241`（10 轮，含 3 个 open 轮）。改写轮用 ★ 标出：

| 轮 | 原句（openness） | 原患者 iu | 改写句（closed） | 改患者 iu |
|---|---|---|---|---|
| t0 ★ | Hi Ryan, what brings you here today? | 3 | Are you here today because of a specific health concern? | 2 |
| t1 ★ | tell me about your cough, is it getting worse... | 0 | Is your cough getting worse than before? | 0 |
| t8 ★ | what other issues were you talking about? | 1 | Are there any other issues you were discussing? | 1 |

完整渲染见 `outputs/20261127_m2d/m2d_result.json` 的 `side_by_side_demo`（含两版患者口语句）。

---

## 六、自评（含「没测到什么」）

**测到了**：
- 模拟器对 openness 的因果响应**方向正确**（改 closed 降、改 open 升），单调性成立。
- 改写器忠实（判对率 0.951/0.923），改写不是噪声。
- 真实耗尽率 5.4%，坐实「模拟器尾部偏厚」。

**没测到 / 局限（如实报）**：
1. **渲染层的幅度影响未纳入**：主指标在标签层测量，渲染对 iu 均值影响约 5% 未计入。若主控要求「渲染后回抽 iu」口径，降幅占比会再降约 5%（46% → ~44%），方向不变。
2. **只改了 openness 一维**：leadingness / multi_barreled 未做反事实改写，本单不检验这两维的因果。
3. **closed→open 改写文本只抽样 143 条**（跑批标签全量改 665 轮，改写文本仅用于校验与展示，够用）。
4. **模拟器为全量校准（含 holdout）**，真实对照为 holdout 观测，两者非严格独立；此泄漏只影响幅度对照的精确性，不影响方向判据。
5. **幅度落在灰色地带**：低估 50% 高于预测 30–40%，但来源可解释、未达 60% 停机线。是否放行需主控裁定。

**建议**：方向判据已过，幅度偏差可解释。若要压缩幅度（修 41% 的 open/closed 差异压缩），需动 B2 校准（调整 openness→iu 条件表的平滑/分档），属「修模型」，本单明确禁止，故留待主控决定是否单独立项。
