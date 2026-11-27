# M2c-b 结果：内容相关性 + vague 规则（§7.7，M2d 前置）

> 日期 2026-11-27 · 状态：**四项数字三项通过，一项「耗尽率」需主控知悉（纠正我此前的误诊）**
> 对应派单 §7.7。复现命令见文末。

## 一句话结论

`relevance` 维第一次有了**内容落点**：`spillover` 现在真的说出「医生没问的事实」，`on-topic` 说出「医生问的事实」。vague 短板修复生效（0.792 → 0.9284）。**但「真值集提前耗尽」不是顺序选的锅，换相关性选不会下降**——这是我上一单（M2c 发现二）的误诊，本单把它纠正掉。

## 四项数字（§7.7 第 4 件）

| # | 指标 | 判据 | 结果 | 判定 |
|---|---|---|---|---|
| a | 集合级 F1 | ≥ 0.90 | **0.9518**（轮级均值 0.9570） | ✅ |
| b | vague 轮 F1 | ≥ 0.88（修前 0.792） | **0.9284**（n=47） | ✅ |
| c | 构念溢出率 轮级差 | spillover − on-topic ≥ 15pp | **35.0pp** | ✅ |
| d | 真值集耗尽轮占比 | 预期显著下降 | 8.3% → **8.3%**（不降） | ⚠️ 见发现三 |

辅助：B3 标签 lag-1 −0.2417 → 渲染后 −0.1885（未抹平）；渲染失真率 11.95%；exact 轮 F1 0.9437。

## 构念溢出率：现在有内容，且有阴性对照

判定用的是「轮级」（任一说未问事实即算），spillover 组 87.5% vs on-topic 组 52.5%，**差 35.0pp**，远超 15pp。事实级口径差 **24.2pp**（94.2% vs 70.0%）。

| 口径 | 顺序选（修前，阴性对照） | 相关性选（修后） |
|---|---|---|
| 轮级差 | 18.3pp | **35.0pp** |
| 事实级差 | **3.4pp**（≈噪声） | **24.2pp** |

**事实级对照是关键证据（D11 失败路径成立）**：顺序选下 spillover 与 on-topic 挑的是**同一批**事实（差仅 3.4pp），证明「说没问的事实」完全由相关性选择带来，不是测量恒等式。轮级顺序选也有 18.3pp 的残余，来自一个真实混杂——`spillover` 与 `open` 问句强相关（M1 H5），开放问句的 `info_units` 更高、说的事实更多，于是「至少说一个没问的事实」的概率更高。相关性选择把轮级差从 18.3pp 放大到 35.0pp，把事实级差从 3.4pp 拉到 24.2pp。

阈值 `θ=0.2`（事实内容词被问句覆盖 < 20% 即判「未问」），轮级差在 θ∈[0.05, 0.5] 全区间稳定在 29–36pp（见 `relevance_cache.json`）。

## vague 修复（§7.7 第 3 件）

`denied` 事实（如 `no hypertension`）在 `vague` 语气下不再被渲染成「大概没有 / 说不好有没有」，改由 `_facts_block` 强制明确否定 + 系统提示加硬规则。效果：

| 指标 | 修前（M2c） | 修后（M2c-b） |
|---|---|---|
| vague 轮 F1 | 0.792 | **0.9284** |
| vague 失真轮 | 20/47（43%） | **7/47（14.9%）** |

剩余 7 轮失真分两类，如实记录：① 少量 `denied` 仍被 qwen-flash 弱化（如「chronic illnesses」轮把 `no osteoarthritis` 说成「说不好有没有关节炎」）——qwen-flash 对否定+含糊的服从不完美；② `vague` 语气对 `definite` 事实的固有语义漂移（「说不好」本身被回抽成一条 spurious「不确定」事实）。①可换更强模型缓解，②是「含糊」维度的固有代价，非本单能消。

## 发现三：耗尽率不降，纠正我此前的误诊

**顺序选 8.3% → 相关性选 8.3%，完全不变。** 原因：`remaining` 是否归零只由「每轮 `info_units` 的累计和 vs 真值集大小」决定，与「选哪几件事」无关——`info_units` 是标签层采样的（本单禁止改），相关性选择不改变每轮披露**数量**，只改变披露**内容**。

我上一单 `docs/m2c_results.md`「发现二·第 2 点」把 demo 对话「后 7 轮只能说『别的没有了』」归因于「顺序选把事实提前说完了」，**这个归因是错的**。真实原因是：demo 对话（id=38，真值集 33 条）前几轮被标签层采到 `iu=4/10/6/7`（重尾），8 轮就把 33 条说完了。换相关性选，同样的 `iu` 序列照样耗尽。

所以 §7.7 里「长对话不再提前耗尽 / 耗尽轮占比显著下降」这条预期**不成立**。它的正确修法不在 M2c-b 范围：
- 演示层：挑真值集足够大的对话（本单 demo 改用 id=408，60 条真值集，15 轮全程未耗尽，见下）；
- 标签层：`r9p` 档（剩余 ≥9）的 `info_units` 分布是否重尾失衡，属标签层校准问题，需主控另行裁定是否派单。

## demo 对话（id=408，15 轮，全程无「别的没有了」）

```
[医生] Any sudden weight gain or loss?
[患者] 没有突然发胖，也没有突然瘦。
[医生] No fever or chills, right?
[患者] 没有发冷，也没有发烧，皮肤颜色正常，没发现皮疹，指甲也没断裂。
[医生] Any chest pain, shortness of breath or shortness of breath while sleeping?
[患者] 睡觉时没有气短。
[医生] Any heart burn, nausea or vomiting?
[患者] 没有呕吐，没有恶心，没有心烧，没有脱发，也没有异常掉发。
[医生] Loss of appetite or blood in stool or abdominal pain?
[患者] 胃口很好，没有食欲不振。
[医生] Constipation?
[患者] 没有便秘，也没有自杀念头。
[医生] Did you experience any blackouts or seizures lately?
[患者] 没有癫痫发作。
（中间含 2 轮附和「对，没错」、1 轮「也没什么要补充的」，均为 iu=0 的正当低披露，非耗尽）
```

全文 15 轮在 `outputs/20261127_m2cb/m2cb_result.json` 的 `demo_dialogue`。

## 20 条人工可核对样例（打分函数校验）

高分段（问「有没有发烧/寒战/咳嗽…」→ `no fever` 1.00 / `no cough` 1.00）；低分段（同一问句 → `age seventy six` 0.00 / `identifies as white` 0.00）。完整 20 条见 `m2cb_result.json` 的 `verify_samples_20`。打分是纯词法（unigram 覆盖 + bigram + 医学同义词归一），确定性、零 API、可复现，缓存于 `outputs/20261127_m2cb/relevance_cache.json`。

## 自评（200 字）

**测到了什么**：`relevance` 维从「只有语气」变成「有内容」——spillover 事实级未问占比 94.2% vs on-topic 70.0%（差 24.2pp），顺序选对照仅 3.4pp，证明是相关性选择在起作用；vague 失真从 43% 降到 14.9%，F1 0.792→0.9284；集合级 F1 0.9518 未因改内容选择而掉。

**没测到什么**：① 相关性打分的语义盲区——「how old are you」与 `age seventy six` 词法不同源，仅靠同义词「old→age」救回一部分，开放问句（如「what brings you back」）对主诉事实打 0 分，on-topic 在这些轮会退化为随机选（拉高了 on-topic 的未问率）；② 中文渲染 vs 英文真值集的同源回抽噪声仍在。

**哪步可能错**：θ=0.2 是我定的「未问」门槛，虽在宽区间稳健，但「未问」本身仍靠词法判定，跨语言/跨表述的语义等价没测。若要收紧，需在 M2d 用人类盲标核 20 条打分样例（§9.3 人类盲标风险的第二次兑现）。

## 需主控知悉 / 裁定（2 件）

1. **耗尽率「不降」是预期的修正，不是失败**。§7.7 第 4 件的「预期显著下降」基于我上单的误诊（顺序选→提前耗尽），本单已证伪：耗尽只由 `info_units` 累计 vs 真值集大小决定，与选择顺序无关。若主控希望降低耗尽率，需派单校准标签层 `r9p` 档 `info_units` 分布（M2c-b 禁止改标签层，未越权）。
2. **相关性打分的语义盲区是否接受为 P1 限制**。开放问句（无内容词）对主诉事实打 0 分，on-topic 退化为随机选。不影响本单判据（35.0pp 达标），但 M2d 若要在开放问句上测「open→closed 改写」，这块会漏。可选升级：对低分事实补一次 qwen 语义打分（§7.7 明确允许「qwen 批量打分」），本单用纯词法先跑通了单调性，是否升级待裁定。

## 复现命令（D14）

```bash
python3 -m pytest tests/test_draw.py -q
python3 scripts/m2cb_render.py data/annot_all_flash.jsonl \
    --truthset data/aiu_truthset_v2_20261126.json \
    --probs data/simulator_probs_v3_20261127.json \
    --out outputs/20261127_m2cb --limit 100
```

产出：`outputs/20261127_m2cb/m2cb_result.json` + `relevance_cache.json`。
代码：`src/relevance_scorer.py`（新）、`src/aiu_selector.py`（relevance 选择）、`src/renderer.py`（vague 规则）、`scripts/m2cb_render.py`（新）。
