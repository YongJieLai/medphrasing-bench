# open-ended-elicitation

> **提问措辞（开放式 vs 封闭式）是否改变患者的自发信息溢出，进而改变可得诊断证据？**

在交互式医学诊断中，现有工作几乎都聚焦在"**问什么**"——即如何选择信息增益最高的问题（DOCTOR-R1、ProMed、MedClarify 等）。本项目研究被忽视的另一半："**怎么问**"——即**同一个信息点的提问，开放式措辞 vs 封闭式措辞，患者吐露的信息量是否不同**。

## 核心假设

**H1（措辞→溢出）**：给定相同的信息寻求意图，开放式提问比封闭式提问引发更多患者"信息溢出"（spillover）——患者会主动说出未被直接问到的、但相关的病情信息。

**H2（溢出→诊断，阶段 B）**：由措辞引发的溢出差异，最终传导为端到端诊断证据/准确率的差异。

## 与现有工作的关键分野

| 维度 | 现有工作（医生侧） | 本项目（患者侧） |
|---|---|---|
| 研究问题 | 问**什么**（选哪个问题信息价值高） | 怎么**问**（措辞的开放度） |
| 患者模型 | 门控：问 A 只答 A（MAI-DxO Gatekeeper、MAQuE Disclosure Control、Q4Dx 固定隐藏） | **溢出**：回答 A 时主动补充相关 B、C |
| 信息量来源 | 问题数量的函数 | **提问质量的函数** |
| 代表工作 | DOCTOR-R1 (ICLR'26)、ProMed (ACL'26)、MedExAgent、MAI-DxO | 本项目 |

- 2501.09484（百川+人大，李比希定律）已证明"问询质量是独立瓶颈"，其患者模拟器虽含 `[Provide Information]` 行为，但溢出由**外生随机采样的策略流**决定，与提问质量无关。
- MAQuE 的 Disclosure Control 方向甚至相反——它限制患者"最多说 N 条"，管的是**上限**而非"好问题触发多溢出"。

本项目的增量：**把"溢出"从外生剧本变成内生于提问措辞的函数**，并把它做成可量化、可复现的机制。

## 实验设计

### 阶段 A（核心 MVP）——措辞对照

固定"想问的信息点"，仅改措辞，测同一患者回答中的溢出 AIU 数。

- **自变量**：同一信息点的两种措辞（开放式 / 封闭式）
- **因变量**：溢出 AIU 数（回答中被披露、但不在该问题意图内的原子信息单元数）
- **控制变量**：患者 system prompt 在所有条件下**完全一致**——任何差异只能归因于措辞

### 阶段 B（端到端）——自由问诊

医生 agent 分两组，仅 system prompt 注入"开放式提问"vs"封闭式提问"策略，测端到端：溢出总量、信息覆盖率、最终诊断准确率。

## MVP 结果（阶段 A）

2 个样例病例 × 6 信息点 × 2 措辞 × 3 采样（n=36 配对）：

| 措辞 | 平均溢出 AIU | 零溢出占比 |
|---|---|---|
| 开放式 | **2.64** | 6% (2/36) |
| 封闭式 | **0.81** | 64% (23/36) |

配对 t 检验：差值均值 1.83，**t = 4.32，p < 0.001**。方向验证成功。

> ⚠️ 当前为样例病例占位验证，用于证明管线可行与信号存在；结论需在更大真实病例集上复现。

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env        # 填入 API key（.env 已 gitignore）
python experiments/phaseA_phrasing.py --cases cap-01 --repeat 3
```

## 目录结构

```
config/models.yaml            # 模型/端点/角色配置
src/cases.py                  # 病例 + AIU + 信息点措辞（含 targets）
src/patient.py                # 患者模拟器（溢出机制在此）
src/doctor.py                 # 医生 agent（阶段 B）
src/phrasing.py               # 措辞展开
src/metrics.py                # AIU 提取 + 溢出测量
src/llm.py                    # 统一 LLM 客户端
experiments/phaseA_phrasing.py  # 阶段 A：措辞对照
outputs/                      # 实验输出（gitignore）
```

## 下一步

- [ ] 扩充真实病例集（DDxPlus / MedQA 子集，或人工标注 AIU）
- [ ] 溢出机制 grounding：用 MedDialog 真实对话验证"开放式→多披露"的临床依据
- [ ] 防泄题约束（溢出不得直接点破诊断）
- [ ] 阶段 B 端到端 + 诊断准确率
- [ ] 多模型鲁棒性（患者/医生换不同 LLM 是否稳健）

## 参考

- 2501.09484 — Question-Answering vs Diagnosis（李比希定律）
- MAQuE (2509.24958) — AIU 表示 + Disclosure Control
- MedExAgent (2605.07058) — POMDP 三动作 + 患者噪声
- MAI-DxO / SDBench (2506.22405) — 序贯诊断 + Gatekeeper
- DOCTOR-R1 (ICLR'26)、ProMed (ACL'26) — 医生侧信息增益
