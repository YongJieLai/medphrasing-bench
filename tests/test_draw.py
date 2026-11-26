"""M2c 第 0 件的单元测试：_draw / draw_1d 遇未知取值必须抛 ValueError。

§7.2.5 记录五：静默等权 fallback 会把非法取值（leadingness="none"）变成
「沉默过多的坏模型」而不报错，让整批实验变噪声。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conditional_simulator import ConditionalSimulator

REPO = Path(__file__).resolve().parent.parent
PROBS = REPO / "data" / "simulator_probs_v3_20261127.json"


@pytest.fixture(scope="module")
def sim():
    return ConditionalSimulator.load(PROBS)


def test_3d_unknown_leadingness_raises(sim):
    """3 维模式：leadingness="none"（合法值 neutral/mild/strong）必须抛 ValueError。"""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sim.sample_behavior(
            "open", "none", "single", rng, remaining_n=5, turn_index=1)


def test_3d_unknown_openness_raises(sim):
    """3 维模式：openness 非法取值必须抛 ValueError。"""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sim.sample_behavior(
            "foo", "neutral", "single", rng, remaining_n=5, turn_index=1)


def test_3d_legal_values_do_not_raise(sim):
    """合法取值不应抛异常，且返回四个维度标签。"""
    rng = np.random.default_rng(0)
    b = sim.sample_behavior(
        "open", "neutral", "single", rng, remaining_n=5, turn_index=1)
    assert b["relevance"] in ("on-topic", "spillover", "off-topic")
    assert b["definiteness"] in ("exact", "vague", "agree")
    assert b["completeness"] in (1, 0)
    assert isinstance(b["info_units"], int)


if __name__ == "__main__":
    # 直接运行（不依赖 pytest 也可自检）
    s = ConditionalSimulator.load(PROBS)
    rng = np.random.default_rng(0)
    try:
        s.sample_behavior("open", "none", "single", rng, remaining_n=5, turn_index=1)
        print("FAIL: 未抛异常")
        sys.exit(1)
    except ValueError as e:
        print(f"OK: 非法取值已抛 ValueError\n{e}")
