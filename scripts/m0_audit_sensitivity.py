"""§7.11 交付项 5：污染率 > 5% 时的 H3 敏感性重算（只重算，不改正文）。

H3（复合性 → 漏答）：multi vs single 的漏答率（completeness==0）风险差。
污染方向：M0 把 multi 系统性误判为 single（single->multi 18 条，multi->single 0 条）。
后果：single 组混入高漏答率的 multi → single 组漏答率被抬高 → H3 风险差被低估。

用污染率 p 做全库期望修正：
  实测 single 漏答率 s_obs = (1-p)*r_s + p*r_m  （r_s=真single漏答, r_m=真multi漏答）
  实测 multi  漏答率 m_obs = r_m                （假设反向污染≈0，见自评）
  => 修正后 r_s(p) = (s_obs - p*m_obs)/(1-p)； 修正后 RD(p) = m_obs - r_s(p)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.calibrate_simulator import valid_rows


def miss_rate(rs):
    return sum(1 for r in rs if r.get("completeness") == 0) / len(rs)


def main():
    result = json.load(open("outputs/m0_audit/result.json", encoding="utf-8"))
    p = result["contamination_rate"]["rate"]
    p_lo = result["contamination_rate"]["ci_95_clopper_pearson"][0]
    p_hi = result["contamination_rate"]["ci_95_clopper_pearson"][1]

    rows = [json.loads(l) for l in open("data/annot_all_flash.jsonl", encoding="utf-8")]
    valid = valid_rows(rows)
    multi = [r for r in valid if r["multi_barreled"] == "multi"]
    single = [r for r in valid if r["multi_barreled"] == "single"]
    m_obs = miss_rate(multi)
    s_obs = miss_rate(single)
    rd_obs = m_obs - s_obs  # 旧值

    def corrected(p_):
        r_s = (s_obs - p_ * m_obs) / (1 - p_)
        return m_obs - r_s

    rd_new = corrected(p)
    rd_new_lo = corrected(p_hi)  # 污染越重，single 组被抬得越多，修正后 RD 越大
    rd_new_hi = corrected(p_lo)

    sens = {
        "baseline_old": {
            "multi_n": len(multi), "multi_miss_rate": m_obs,
            "single_n": len(single), "single_miss_rate": s_obs,
            "rd_pp": rd_obs * 100,
        },
        "contamination_p": p,
        "contamination_ci_95": [p_lo, p_hi],
        "corrected_rd_pp": rd_new * 100,
        "corrected_rd_range_pp": [rd_new_hi * 100, rd_new_lo * 100],
        "underestimate_pp": (rd_new - rd_obs) * 100,
        "underestimate_relative": (rd_new - rd_obs) / rd_obs,
        "note": (
            "修正后 H3 风险差只增不减（方向不变，效应被低估）。"
            "反向污染(multi->single)样本 0/11，其存在只会进一步抬高真实 multi 漏答率、"
            "使 H3 更大，故本估为低估的下界。"),
    }
    json.dump(sens, open("outputs/m0_audit/h3_sensitivity.json", "w",
                         encoding="utf-8"), ensure_ascii=False, indent=2)

    print("=== H3 敏感性重算（复合性 → 漏答） ===")
    print(f"旧值(M0标签): multi {m_obs:.2%} (n={len(multi)}) vs single {s_obs:.2%} "
          f"(n={len(single)})  风险差 {rd_obs*100:.2f}pp")
    print(f"污染率 p={p:.1%}  CI [{p_lo:.1%}, {p_hi:.1%}]")
    print(f"修正后风险差 = {rd_new*100:.2f}pp  (区间 [{rd_new_hi*100:.2f}, {rd_new_lo*100:.2f}]pp)")
    print(f"低估幅度 = {(rd_new-rd_obs)*100:.2f}pp  (相对 {((rd_new-rd_obs)/rd_obs)*100:.0f}%)")
    print(f"方向：{'不变（效应被低估）' if rd_new >= rd_obs else '反转'}")


if __name__ == "__main__":
    main()
