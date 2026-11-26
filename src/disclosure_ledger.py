"""双状态账本：T_d（患者知道的事实）与 D_d(t)（医生真正获得的信息）。

契约见 PROJECT_CHARTER.md §2.2：
  - T_d：患者【知道】的事实全集（含被确认的 yeah→有高血压；含 denied no→无糖尿病），
    由 M2a 真值集给定，全对话不变。
  - D_d(t)：医生在 t 时刻【真正获得】的信息。仅 fidelity=definite 且
    definiteness≠agree 且 completeness=1 的 AIU 入账。

两条硬约束（§2.2）：
  1. 采样披露内容时，只能从 T_d \\ D_d(t) 中抽（不重复说已获取的）；
  2. 计算信息获取量只能用 |D_d(t)|，不得用 |T_d| - |R_d| 之类近似。

M2b 阶段使用「计数版」：T_d 与 D_d 各记一个整数大小（真值集大小 vs 已披露量）。
M2c 升级为「AIU 文本集版」（记录具体 AIU 内容，用于 B4 内容正确性判据）。
"""
from __future__ import annotations


class DisclosureLedger:
    """计数版双状态账本。

    - truth_size: |T_d|（该患者知道的事实总数）
    - disclosed:  |D_d(t)|（已真正披露、入账的事实数）

    remaining() 即 |T_d \\ D_d(t)|。record(n) 把本轮披露的 n 个事实入账，
    但入账数不超过剩余量（不重复获取已披露的）。
    """

    def __init__(self, truth_size: int):
        if truth_size < 0:
            raise ValueError("truth_size 必须 ≥ 0")
        self.truth_size = int(truth_size)
        self.disclosed = 0

    def remaining(self) -> int:
        """|T_d \\ D_d(t)| = 还能再披露多少。"""
        return max(0, self.truth_size - self.disclosed)

    def record(self, n: int) -> int:
        """把本轮披露的 n 个事实入账，返回实际入账数（不超过剩余量）。

        对应 §2.2 约束：不重复获取已披露的。若 n 超过剩余量，只入账剩余量。
        """
        n = int(n)
        if n <= 0:
            return 0
        added = min(n, self.remaining())
        self.disclosed += added
        return added

    @property
    def t_size(self) -> int:
        return self.truth_size

    @property
    def d_size(self) -> int:
        return self.disclosed

    def snapshot(self) -> dict:
        """报告 §2.2 要求的三个量：|T_d|、|D_d|、|T_d \\ D_d|。"""
        return {
            "T_d": self.truth_size,
            "D_d": self.disclosed,
            "T_d_minus_D_d": self.remaining(),
        }
