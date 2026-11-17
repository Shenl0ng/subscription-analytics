"""Command line entry point.

suban ab        analyse the Cookie Cats A/B test and size a test for agencies
suban churn     retention curves, churn model vs. business rule, LTV by contract
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from subscription_analytics import ab_test, churn, data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="suban")
    parser.add_argument("command", choices=["ab", "churn"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args(argv)

    result: dict[str, Any]
    if args.command == "ab":
        df = data.load_cookie_cats(args.data_dir)
        result = {
            "test": ab_test.analyse(df),
            "engagement_tail": ab_test.engagement_outliers(df),
            # Sizing for a subscription business: e.g. 85 % renewal, detect +2 points.
            "sample_size_per_group": {
                f"baseline_{b:.0%}_mde_{m}pp": ab_test.sample_size_per_group(b, m)
                for b, m in ((0.85, 2.0), (0.85, 5.0), (0.19, 1.0))
            },
        }
    else:
        df = data.load_telco(args.data_dir)
        result = {
            "customers": len(df),
            "churn_rate": round(float(df["churned"].mean()), 3),
            "retention_by_contract": churn.retention_by(df, "Contract", (1, 6, 12, 24, 48)),
            "model": churn.evaluate_churn(df),
            "ltv_by_contract": churn.ltv_by(df, "Contract"),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
