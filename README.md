# subscription-analytics

How I approach the core questions of a subscription business with data: **did a change
work?** (A/B test), **who is about to cancel?** (churn model), **how long do customers
stay and what are they worth?** (retention curves and LTV).

Personal learning project on **public data**. The A/B test is a real randomised experiment
run by a mobile game; the churn data is IBM's fictional telco sample. No client data.

```bash
make install   # uv sync
make ab        # A/B test analysis + sample sizes            (~2 s)
make churn     # retention curves, churn model, LTV           (~5 s)
make test      # unit tests;  make lint: ruff + mypy
```

Data files are downloaded on first run and verified by SHA-256, so an analysis cannot
silently run on a different file.

## 1. A/B test: does moving a gate hurt retention?

[Cookie Cats](https://github.com/robguilarr/ab_testing_cookie_cats): 90,189 players randomly
assigned to a progress gate at level 30 (control) or level 40 (treatment).
**Primary metric fixed in advance: 7-day retention.** 1-day retention is secondary and
does not decide.

| Check | Result |
|---|---|
| Sample ratio (50/50 planned) | 44,700 vs 45,489 (50.4 % treatment), χ² p = 0.009 → **warning**, not a stop (alarm threshold 0.001): read the metrics, but check assignment and logging |
| **7-day retention** | 19.02 % → 18.20 %: **−0.82 pp** (−4.3 % relative), 95 % CI [−1.33, −0.31], p = 0.002; bootstrap CI [−1.33, −0.30] |
| 1-day retention | 44.82 % → 44.23 %: −0.59 pp, CI [−1.24, +0.06], p = 0.07 (not significant) |
| Engagement tail | median 16 rounds, p99 493, max 49,854: means would be driven by a few players |

**Decision: keep the gate at level 30.** The interval for the primary metric is entirely
below zero, and two independent methods (normal approximation and bootstrap) agree.

**Sizing the next test** (alpha 0.05, power 80 %, two-sided), e.g. for a subscription
renewal rate:

| Baseline | Smallest effect worth detecting | Units per group |
|---|---|---|
| 85 % renewal | +2 pp | 4,724 |
| 85 % renewal | +5 pp | 686 |
| 19 % retention | +1 pp | 24,641 |

With a few thousand paying customers, only large effects can be detected in one cycle:
the test design (metric, duration, unit) has to be decided with that in mind.

## 2. Retention, churn model and LTV

IBM Telco sample: 7,043 customers, 26.5 % churned. The data is a snapshot without dates,
so customers still active are treated as **censored** in the retention curves, and the
model is evaluated on a stratified 30 % hold-out (with dated data the split should be by
time).

**Retention by contract (Kaplan–Meier)**

| Contract | Month 1 | Month 6 | Month 12 | Month 24 | Month 48 |
|---|---|---|---|---|---|
| Month-to-month | 90 % | 79 % | 70 % | 59 % | 40 % |
| One year | 100 % | 99.7 % | 99 % | 98 % | 92 % |
| Two year | 100 % | 100 % | 100 % | 100 % | 99.6 % |

**Churn model vs. a business rule** (hold-out of 2,113 customers; "top 10 %" = the
customers a retention team could contact):

| Approach | AUC | Churn rate in top 10 % | Lift |
|---|---|---|---|
| Rule: month-to-month first, then shortest tenure | 0.794 | 61 % | 2.3× |
| **Logistic regression** | **0.853** | **75 %** | **2.8×** |
| Gradient boosting | 0.837 | 74 % | 2.8× |

The simple, explainable model wins. Main drivers (odds ratios, numeric features per
standard deviation): +2 years of tenure ×0.44, two-year contract ×0.47, month-to-month
×1.61, +30 in monthly charges ×1.51, online security ×0.72.

**LTV by contract** = median monthly revenue × gross margin (**assumed 70 %**) × expected
months within the 72-month horizon (area under the retention curve):

| Contract | Customers | Monthly revenue | Expected months | LTV |
|---|---|---|---|---|
| Month-to-month | 3,875 | 73.25 | 36.3 | 1,861 |
| One year | 1,473 | 68.75 | 66.4 | 3,196 |
| Two year | 1,695 | 64.35 | 71.5 | 3,223 |

Month-to-month customers pay the most per month but are worth ~40 % less: moving them to
annual plans is worth testing, with an A/B test sized as in section 1. Churn prediction
says *who* is at risk; only an experiment shows whether contacting them *keeps* them.

## Layout

```
src/subscription_analytics/  data (download + checksum) · ab_test · churn (KM, model, LTV) · cli
tests/                       synthetic-data unit tests (no download needed)
```
