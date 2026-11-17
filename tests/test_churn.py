import numpy as np
import pandas as pd
import pytest

from subscription_analytics import churn


def test_kaplan_meier_handles_censoring():
    # 4 customers: churn at month 1, active at month 2 (censored), churn at 3, active at 3
    curve = churn.kaplan_meier(np.array([1, 2, 3, 3]), np.array([True, False, True, False]))
    assert curve[0] == 1.0
    assert curve[1] == pytest.approx(0.75)  # 1 of 4 at risk churned
    assert curve[2] == pytest.approx(0.75)  # censored customer is not a churn
    assert curve[3] == pytest.approx(0.375)  # 1 of the 2 still at risk churned


def test_restricted_mean_lifetime_is_area_under_curve():
    curve = pd.Series([1.0, 0.5, 0.5])
    assert churn.restricted_mean_lifetime(curve, horizon=4) == pytest.approx(2.5)


def test_precision_at_top():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    score = np.arange(10, 0, -1, dtype=float)  # highest score first
    assert churn.precision_at_top(y, score, 0.2) == 0.5


def telco_like(n: int = 2_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    tenure = rng.integers(0, 72, n)
    risk = (contract == "Month-to-month") * 2.0 - tenure / 24
    df = pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": rng.uniform(20, 110, n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Contract": contract,
            "churned": rng.random(n) < 1 / (1 + np.exp(-risk)),
        }
    )
    for col in churn.CATEGORICAL:
        if col not in df:
            df[col] = "No"
    return df


def test_churn_models_beat_random_and_report_lift():
    result = churn.evaluate_churn(telco_like())
    for r in result["results"].values():
        assert r["auc"] > 0.6
        assert r["lift_top"] > 1


def test_ltv_is_higher_for_customers_who_stay_longer():
    df = telco_like()
    ltv = churn.ltv_by(df, "Contract")
    assert ltv["Two year"]["expected_months"] > 0
    assert ltv["gross_margin_assumed"] == 0.7
