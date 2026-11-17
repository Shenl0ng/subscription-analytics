"""Analyse a randomised A/B test on a retention metric, and size the next one.

The order of the checks is the order that protects the decision:

1. **Sample ratio mismatch (SRM):** if the split is not the one that was planned, the
   randomisation or the logging is broken and no metric can be trusted.
2. **Primary metric fixed in advance** (here 7-day retention), secondary metrics reported
   but not used to decide, so the test is not "won" by picking whichever metric looks best.
3. **Effect with its uncertainty:** difference in proportions with a normal-approximation
   confidence interval and a bootstrap interval as a cross-check, plus the p-value.
4. **Power:** how many units per group the next test needs to detect a given effect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ABConfig:
    group_col: str = "version"
    control: str = "gate_30"
    treatment: str = "gate_40"
    primary: str = "retention_7"
    secondary: tuple[str, ...] = ("retention_1",)
    alpha: float = 0.05
    srm_alpha: float = 0.001  # strict: an SRM alarm stops the analysis
    srm_warn: float = 0.01  # between warn and alpha: read the metrics, but investigate
    n_bootstrap: int = 5_000
    random_state: int = 7


def srm_check(
    counts: pd.Series, expected_share: float = 0.5, alpha: float = 0.001, warn: float = 0.01
) -> dict[str, Any]:
    """Chi-square goodness of fit of the observed split against the planned split.

    ``status`` is ``mismatch`` (stop), ``warning`` (unlikely by chance: check assignment and
    logging before trusting the result) or ``ok``.
    """
    total = int(counts.sum())
    expected = [total * expected_share, total * (1 - expected_share)]
    _, p_value = stats.chisquare(counts.to_numpy(), f_exp=expected)
    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "share_treatment": round(float(counts.iloc[1] / total), 4),
        "p_value": round(float(p_value), 4),
        "status": "mismatch" if p_value < alpha else "warning" if p_value < warn else "ok",
    }


def diff_in_proportions(
    control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05
) -> dict[str, float]:
    """Treatment minus control, in percentage points, with a two-sided z-test."""
    n_c, n_t = len(control), len(treatment)
    p_c, p_t = control.mean(), treatment.mean()
    diff = p_t - p_c
    se = math.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    pooled = (control.sum() + treatment.sum()) / (n_c + n_t)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    z = diff / se_pooled
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return {
        "control_rate": round(float(p_c), 4),
        "treatment_rate": round(float(p_t), 4),
        "diff_pp": round(float(diff) * 100, 2),
        "relative_change": round(float(diff / p_c), 4),
        "ci_low_pp": round(float(diff - z_crit * se) * 100, 2),
        "ci_high_pp": round(float(diff + z_crit * se) * 100, 2),
        "p_value": round(float(2 * stats.norm.sf(abs(z))), 4),
    }


def bootstrap_diff_ci(
    control: np.ndarray, treatment: np.ndarray, n: int, alpha: float, seed: int
) -> tuple[float, float]:
    """Percentile bootstrap of the difference in means (a check that needs no normality)."""
    rng = np.random.default_rng(seed)
    p_c = rng.binomial(len(control), control.mean(), n) / len(control)
    p_t = rng.binomial(len(treatment), treatment.mean(), n) / len(treatment)
    low, high = np.quantile(p_t - p_c, [alpha / 2, 1 - alpha / 2])
    return round(float(low) * 100, 2), round(float(high) * 100, 2)


def sample_size_per_group(
    baseline: float, mde_pp: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Units per group to detect an absolute change of ``mde_pp`` points (two-sided)."""
    p1, p2 = baseline, baseline + mde_pp / 100
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    num = z_a * math.sqrt(2 * p_bar * (1 - p_bar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil((num / (p2 - p1)) ** 2)


def analyse(df: pd.DataFrame, cfg: ABConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ABConfig()
    counts = df[cfg.group_col].value_counts().reindex([cfg.control, cfg.treatment])
    srm = srm_check(counts, alpha=cfg.srm_alpha, warn=cfg.srm_warn)
    result: dict[str, Any] = {"n": len(df), "srm": srm}
    if srm["status"] == "mismatch":
        result["decision"] = "stop: sample ratio mismatch, fix assignment before reading metrics"
        return result

    groups = {g: d for g, d in df.groupby(cfg.group_col)}
    metrics: dict[str, dict[str, Any]] = {}
    for metric in (cfg.primary, *cfg.secondary):
        c = groups[cfg.control][metric].to_numpy(dtype=float)
        t = groups[cfg.treatment][metric].to_numpy(dtype=float)
        m: dict[str, Any] = dict(diff_in_proportions(c, t, cfg.alpha))
        m["bootstrap_ci_pp"] = bootstrap_diff_ci(c, t, cfg.n_bootstrap, cfg.alpha, cfg.random_state)
        metrics[metric] = m
    result["metrics"] = metrics

    primary = metrics[cfg.primary]
    if primary["p_value"] >= cfg.alpha:
        decision = f"no clear effect on {cfg.primary}: keep {cfg.control}"
    elif primary["diff_pp"] < 0:
        decision = f"{cfg.treatment} lowers {cfg.primary}: keep {cfg.control}"
    else:
        decision = f"{cfg.treatment} raises {cfg.primary}: ship {cfg.treatment}"
    result["decision"] = decision
    return result


def engagement_outliers(df: pd.DataFrame, col: str = "sum_gamerounds") -> dict[str, Any]:
    """Describe the heavy tail: means are misleading here, medians are not."""
    q = df[col].quantile([0.5, 0.99]).to_dict()
    return {
        "median": float(q[0.5]),
        "p99": float(q[0.99]),
        "max": int(df[col].max()),
        "zero_rounds_share": round(float((df[col] == 0).mean()), 4),
    }
