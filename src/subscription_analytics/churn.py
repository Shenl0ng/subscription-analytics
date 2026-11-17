"""Retention curves, churn model and LTV for a subscription business.

* **Retention (Kaplan-Meier):** share of customers still subscribed after *m* months, by
  contract type. Customers still active are *censored* (we know they lasted at least
  ``tenure`` months), so a plain "churned / total" would understate how long they stay.
* **Churn model:** who is most likely to cancel, compared with a one-line business rule
  ("month-to-month contract = at risk"). What matters for a retention team is how many of
  the customers it can call (the top 10 % of scores) really churn.
* **LTV:** monthly revenue x gross margin x expected lifetime, where the lifetime comes from
  the retention curve. The margin is an assumption and is reported as one.

The Telco data is a snapshot (no dates), so the model is evaluated on a stratified random
hold-out, not on a later period. With dated data the split should be by time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure", "MonthlyCharges", "SeniorCitizen"]
CATEGORICAL = [
    "Contract",
    "PaymentMethod",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "PaperlessBilling",
    "Partner",
    "Dependents",
    "StreamingTV",
    "MultipleLines",
]


@dataclass(frozen=True)
class ChurnConfig:
    test_size: float = 0.3
    top_share: float = 0.10  # share of customers a retention team can contact
    gross_margin: float = 0.70  # assumption for LTV
    horizon_months: int = 72  # longest tenure observed in the data
    random_state: int = 7


# ------------------------------------------------------------------ retention curves
def kaplan_meier(durations: np.ndarray, events: np.ndarray) -> pd.Series:
    """Survival probability S(t) at each month t = 0..max(durations).

    ``events`` is True when the customer churned at ``durations`` months, False when still
    active (censored). S(t) = prod over months <= t of (1 - churned_t / at_risk_t).
    """
    max_t = int(durations.max())
    survival = []
    s = 1.0
    for t in range(max_t + 1):
        at_risk = int((durations >= t).sum())
        churned = int(((durations == t) & events).sum())
        if at_risk:
            s *= 1 - churned / at_risk
        survival.append(s)
    return pd.Series(survival, index=range(max_t + 1), name="survival")


def restricted_mean_lifetime(survival: pd.Series, horizon: int) -> float:
    """Expected months subscribed within ``horizon`` = area under the survival curve."""
    return float(survival.reindex(range(horizon), method="ffill").sum())


def retention_by(df: pd.DataFrame, col: str, months: tuple[int, ...]) -> dict[str, Any]:
    out = {}
    for value, g in df.groupby(col):
        curve = kaplan_meier(g["tenure"].to_numpy(), g["churned"].to_numpy())
        out[str(value)] = {
            f"month_{m}": round(float(curve.reindex(range(m + 1), method="ffill").iloc[-1]), 3)
            for m in months
        }
    return out


# ------------------------------------------------------------------ churn model
def _preprocessor(scale: bool) -> ColumnTransformer:
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )


def models(random_state: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [("prep", _preprocessor(True)), ("clf", LogisticRegression(max_iter=1_000))]
        ),
        "gradient_boosting": Pipeline(
            [
                ("prep", _preprocessor(False)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=200, learning_rate=0.05, random_state=random_state
                    ),
                ),
            ]
        ),
    }


def precision_at_top(y: np.ndarray, score: np.ndarray, share: float) -> float:
    """Churn rate among the ``share`` of customers with the highest score."""
    k = max(1, round(len(y) * share))
    top = np.argsort(-score, kind="stable")[:k]
    return float(y[top].mean())


def evaluate_churn(df: pd.DataFrame, cfg: ChurnConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ChurnConfig()
    train, test = train_test_split(
        df, test_size=cfg.test_size, stratify=df["churned"], random_state=cfg.random_state
    )
    y_test = test["churned"].to_numpy()
    base_rate = float(y_test.mean())

    # Business rule baseline: monthly contracts first, then shortest tenure.
    rule = (test["Contract"].eq("Month-to-month") * 1_000 - test["tenure"]).to_numpy(float)
    results: dict[str, Any] = {
        "rule_month_to_month": {
            "auc": round(roc_auc_score(y_test, rule), 3),
            "precision_top": round(precision_at_top(y_test, rule, cfg.top_share), 3),
        }
    }
    coefficients: list[dict[str, Any]] = []
    for name, model in models(cfg.random_state).items():
        model.fit(train[NUMERIC + CATEGORICAL], train["churned"])
        score = model.predict_proba(test[NUMERIC + CATEGORICAL])[:, 1]
        results[name] = {
            "auc": round(roc_auc_score(y_test, score), 3),
            "precision_top": round(precision_at_top(y_test, score, cfg.top_share), 3),
        }
        if name == "logistic_regression":
            names = model.named_steps["prep"].get_feature_names_out()
            coefs = model.named_steps["clf"].coef_[0]
            ranked = sorted(zip(names, coefs, strict=True), key=lambda kv: -abs(kv[1]))
            coefficients = [
                {"feature": str(n).split("__", 1)[1], "odds_ratio": round(float(np.exp(c)), 2)}
                for n, c in ranked[:8]
            ]
    for r in results.values():
        r["lift_top"] = round(r["precision_top"] / base_rate, 2)
    return {
        "n_train": len(train),
        "n_test": len(test),
        "churn_rate_test": round(base_rate, 3),
        "top_share": cfg.top_share,
        "results": results,
        "logistic_top_drivers": coefficients,
    }


# ------------------------------------------------------------------ LTV
def ltv_by(df: pd.DataFrame, col: str, cfg: ChurnConfig | None = None) -> dict[str, Any]:
    """LTV = median monthly revenue x margin x expected months (restricted to the horizon)."""
    cfg = cfg or ChurnConfig()
    out = {}
    for value, g in df.groupby(col):
        curve = kaplan_meier(g["tenure"].to_numpy(), g["churned"].to_numpy())
        months = restricted_mean_lifetime(curve, cfg.horizon_months)
        arpu = float(g["MonthlyCharges"].median())
        out[str(value)] = {
            "customers": len(g),
            "monthly_revenue": round(arpu, 2),
            "expected_months": round(months, 1),
            "ltv": round(arpu * cfg.gross_margin * months, 0),
        }
    return {"gross_margin_assumed": cfg.gross_margin, "horizon_months": cfg.horizon_months, **out}
