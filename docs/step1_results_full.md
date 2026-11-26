# 第一步正式结果：MTS-Dialog 真实对话上的效应量测量

> 数据：MTS-Dialog 真实医患对话，5325 个「医生提问 → 患者回答」对（经提问规则过滤）。
> 标注：qwen-flash 主标注（P1 双标一致性 8 维 AC1 ≥ 0.77，已达标）。
> 统计：混合效应模型（`info_units ~ openness + (1|dialogue_id)`）+ 卡方检验（比例类）。

---

## 一、总览：5 个假设全部成立

| # | 假设 | 自变量 | 因变量 | 效应量 | 显著性 |
|---|---|---|---|---|---|
| H1 | 开放度 ↑ → 披露量 ↑ | open vs closed | info_units（信息单元数） | 1.32 → 2.48（d=-0.979） | p=4.9e-123 |
| H2 | **开放度 ↑ → 溢出 ↑** | open vs closed | spillover 率 | 4.9% → 13.2%（RD -8.3pp） | p=3.6e-16 |
| H3 | 引导性 ↑ → 附和 ↑ | strong vs neutral | agree 率（附和，无新信息） | 14.2% → 51.6%（RD -37.4pp） | p=5.4e-43 |
| H4 | 复合性 → 漏答 ↑ | multi vs single | 漏答率 | 2.8% → 8.0%（RD -5.2pp） | p=3.7e-06 |
| H5 | 追问 → 更精确 | deepen vs surface | exact 率（精确可验证） | 23.2% → 74.6%（RD -51.5pp） | p=4.2e-76 |

**最关键的一条（H2）**：开放式提问引发患者「自发溢出未被问到的相关信息」的概率，是封闭式的 **2.7 倍**（13.2% vs 4.9%），p=3.6e-16。这正是本项目核心 setting 的实证根基。

---

## 二、逐假设详细结果

### H1 开放度 → 披露量（info_units / 词数）

| 措辞 | n | info_units（mean±SE） | 词数（mean±SE） |
|---|---|---|---|
| closed | 3233 | 1.32 ± 0.04 | 8.45 ± 0.30 |
| semi | 1194 | 1.93 ± 0.06 | 12.97 ± 0.59 |
| open | 742 | 2.48 ± 0.10 | 17.54 ± 1.01 |

- Cohen's d (closed vs open)：**-0.979**（大效应）
- 混合效应模型：open p=4.9e-123，semi p=3.3e-42

### H2 开放度 → 溢出（spillover 阳性率）

| 措辞 | n | spillover 率 |
|---|---|---|
| closed | 3233 | 4.9% |
| semi | 1194 | 8.1% |
| open | 742 | **13.2%** |

- 风险差 closed - open = -8.3pp，χ² p=3.6e-16

### H3 引导性 → 附和（agree 阳性率）

| 引导性 | n | agree 率 |
|---|---|---|
| neutral | 4595 | 14.2% |
| mild | 385 | 36.1% |
| strong | 188 | **51.6%** |

- 风险差 neutral - strong = -37.4pp，χ² p=5.4e-43
- 含义：强引导性问题下，超过一半的患者回答只是「附和」，不带新信息——信息可靠性显著下降。

### H4 复合性 → 漏答（completeness=0 阳性率）

| 复合性 | n | 漏答率 |
|---|---|---|
| single | 4905 | 2.8% |
| multi | 264 | **8.0%** |

- 风险差 = -5.2pp，χ² p=3.7e-06

### H5 追问深度 → 精确（exact 阳性率）

| 追问 | n | exact 率 |
|---|---|---|
| surface | 311 | 23.2% |
| new | 1318 | 69.6% |
| deepen | 2545 | **74.6%** |

- 风险差 surface - deepen = -51.5pp，χ² p=4.2e-76
- 含义：顺着患者模糊线索追问（deepen），把回答从「模糊」变「精确」的概率是 surface 的 3.2 倍。

### 附：spillover 确实伴随更多披露

| relevance | n | info_units |
|---|---|---|
| on-topic | 4811 | 1.51 |
| spillover | 355 | 3.16 |

- t=-21.08，p=2.8e-66，Cohen's d=-1.406
- 溢出不是「换话题」，而是「在回答之外额外吐露了更多原子信息」。

---

## 三、这说明了什么（相对之前 LLM prompt 版本的意义）

1. **打破了循环论证**：之前 DDxPlus 50 例的「开放式→多溢出」是 prompt 里写出来的（循环论证）。现在这个效应在**真实医患对话数据**上被独立测量出来，成为可引用的**经验命题**。

2. **四维提问 × 四属性披露的框架得到完整验证**：不只是「披露量」一个标量，四个正交属性（量、准确性、完整性、相关性）分别被四个提问维度显著影响。

3. **为患者模拟器提供了真实行为开关**：后续构建患者模拟器时，不再用 prompt 命令「开放式多说」，而是用这里测得的**真实条件概率**（如 open→spillover 13.2%、strong leading→agree 51.6%）。

---

## 四、诚实的局限

1. **标注是 LLM 做的**（qwen-flash），虽与 qwen-plus 双标 AC1≥0.77，但仍非人工金标准（人工核对待补）。
2. **效应量是「观察性相关」**，未做因果干预——对话是自然发生的，不能排除混杂（如开放式问题本身可能出现在更复杂的病例中）。
3. **单一语料**（MTS-Dialog 英文），未在中文（MedDialog）或其他语料上复现。
4. **spillover 基线低**（4.9%~13.2%），绝对效应量不大，但相对倍数（2.7×）和显著性都很强。
5. 引导性维度中「strong」样本仅 188 条，需注意小样本稳定性（但效应方向与量级一致）。

---

## 五、可复现命令

```bash
# 数据解析
python3 scripts/parse_mtsdialog.py --parquet-dir data/raw --out data/mtsdialog_parsed.json
# 全量标注（8 路并行分片）
for i in $(seq 0 7); do
  python3 scripts/annotate_mtsdialog.py --role annotator --shard $i --nshards 8 \
    --out data/annot_all_flash.shard$i.jsonl &
done; wait
cat data/annot_all_flash.shard*.jsonl > data/annot_all_flash.jsonl
# 效应量分析
python3 scripts/analyze_effects.py data/annot_all_flash.jsonl
```
