import numpy as np
import pandas as pd
import pytest

from subscription_analytics import ab_test as ab


def experiment(n: int, p_control: float, p_treatment: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    version = np.array(["gate_30", "gate_40"] * (n // 2))
    p = np.where(version == "gate_30", p_control, p_treatment)
    return pd.DataFrame(
        {
            "version": version,
            "retention_7": rng.random(n) < p,
            "retention_1": rng.random(n) < 0.45,
            "sum_gamerounds": rng.integers(0, 100, n),
        }
    )


def test_srm_detects_a_broken_split():
    assert ab.srm_check(pd.Series({"a": 5_000, "b": 5_000}))["status"] == "ok"
    assert ab.srm_check(pd.Series({"a": 5_000, "b": 5_600}))["status"] == "mismatch"


def test_analysis_stops_on_sample_ratio_mismatch():
    df = experiment(20_000, 0.2, 0.2)
    df = df[~((df["version"] == "gate_40") & (df.index % 5 == 0))]  # lose 20 % of one arm
    result = ab.analyse(df)
    assert result["srm"]["status"] == "mismatch"
    assert "metrics" not in result
    assert result["decision"].startswith("stop")


def test_detects_a_real_drop_and_keeps_control():
    result = ab.analyse(experiment(40_000, 0.20, 0.17))
    m = result["metrics"]["retention_7"]
    assert m["diff_pp"] == pytest.approx(-3, abs=1)
    assert m["ci_high_pp"] < 0 and m["p_value"] < 0.05
    assert m["bootstrap_ci_pp"][1] < 0
    assert result["decision"].startswith("gate_40 lowers")


def test_no_effect_is_not_called_a_win():
    result = ab.analyse(experiment(10_000, 0.2, 0.2, seed=3))
    assert result["decision"].startswith("no clear effect")


def test_sample_size_matches_textbook_value():
    # 50 % -> 55 %, alpha 0.05, power 0.8: about 1,565 per group
    assert ab.sample_size_per_group(0.50, 5.0) == pytest.approx(1_565, abs=5)
    assert ab.sample_size_per_group(0.85, 2.0) > ab.sample_size_per_group(0.85, 5.0)
