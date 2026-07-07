# Salon Churn & Customer Value Intelligence (v2)

An evolution of the [original churn dashboard](../v1), rebuilt around a real supervised
machine learning model instead of hardcoded threshold rules, and extended with revenue/lifetime-value
analytics so churn can be prioritized by dollar impact, not just headcount.

## Why v2

The original project classified every guest as Active / At Risk / Churned using a fixed rule:
`Churned = 180+ days since last visit`, with 1x/2x personal-average-gap bands in between. It worked, but
it's a heuristic, not a model — there's no learning, no probability, no evaluation against held-out data,
and no way to say what a churned guest is actually worth. This version keeps that heuristic (it's a
genuinely reasonable baseline) but adds:

1. **A trained churn classifier** that outputs a probability per guest, evaluated out-of-time.
2. **Customer lifetime value & revenue-at-risk**, so outreach can be prioritized by dollars, not just count.
3. **RFM segmentation** (Recency/Frequency/Monetary) for a business-familiar view of the guest base.

## The core methodology problem (and its solution)

You can't train a model on `Churned = DaysSinceLastVisit >= 180 (as of today)` — recency both defines the
label and would be the single most predictive feature, so the model would trivially "solve" a circular
problem. The fix is **point-in-time snapshotting**:

- Pick a cutoff date `T` (this project uses the 1st of each month across the dataset's history).
- For every guest active before `T`, compute features using **only** appointments strictly before `T`.
- Label the guest **churned (1)** if they have no visit in `(T, T + 90 days]`, else **0**.
- Train on the earliest snapshots; evaluate on the most recent snapshots the model never saw — a genuine
  out-of-time test, not a random shuffle.

This turns "will this guest come back" into an honest supervised learning problem. See `features.py` for
the snapshot builder and `tests/test_pipeline.py::test_build_snapshots_features_do_not_leak_future_visits`
for a regression test that would fail if a future code change accidentally let features see beyond `T`.

## Model results (out-of-time test set)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Gradient Boosting (used for live predictions) | 0.776 | 0.811 | 0.849 |
| Logistic Regression (interpretable baseline) | 0.765 | 0.800 | 0.832 |
| Rules baseline (v1's heuristic, reimplemented) | 0.670 | 0.653 | n/a |

The trained model beats the hardcoded heuristic by ~10 points of accuracy and ~16 points of F1 on data it
was never trained on. Numbers regenerate with `python train.py`; exact values will shift slightly as new
appointment data is added. Feature importances and ROC/PR curves are in the **Advanced** tab.

## What it does

- **Churn Predictions** — every current guest scored with a churn probability by the trained model, ranked
  by **revenue at risk** (probability x expected future value), for a prioritized outreach list.
- **Customer Segments (RFM)** — guests bucketed into Champions / Regulars / At-Risk Loyalists / Slipping
  Away / Hibernating / New based on recency, frequency, and (if revenue data is available) spend.
- **Advanced** — ROC curve, feature importance, and a head-to-head metrics table against the rules
  baseline, so the modeling claims are checkable, not just asserted.
- **Executive Overview, Stylist Retention, Cohort Retention, Refresh Data** — carried over from v1, with
  the trend chart's month/week mislabeling bug fixed (`app.py`'s v1 called it "Monthly" but grouped by week).

## Graceful degradation — works today, gets richer later

The model and every core feature train and run from the **appointments export alone** (what v1 already
had). Revenue/CLV, and richer Monetary-based RFM, activate automatically the moment a Zenoti
**Sales-Accrual** export is dropped at `data/sales.csv` — no code changes required. Without it,
"revenue at risk" falls back to a frequency-weighted proxy (labeled accordingly in the UI via the
`ValueUnit` column). See `ingest.py` for the expected columns and alias-matching.

This project's `data/sales.csv` is a real Sales-Accrual export (7 stitched 10-month windows, since
Zenoti caps a single export's date range, covering April 2021 through today with no gaps or overlaps).
It's line-item level (one row per service/product per invoice), so `ingest.try_load_sales()` collapses
it to one row per invoice before it reaches `features.py`/`clv.py` — otherwise `AvgTicket` would average
individual line items instead of whole-visit totals. It also normalizes guest name casing
(`ELIZABETH MARTINEZ` → `Elizabeth Martinez`) to match the appointments export, which raised the guest
join rate from 24% to 100%.

## Architecture

```
Salon Internship 2/
├── config.py      # all thresholds/paths in one place (v1 had these hardcoded across files)
├── pipeline.py    # cleaning/dedup, reused verbatim from v1
├── ingest.py       # defensive loaders for Zenoti report exports
├── features.py    # point-in-time snapshot builder + feature engineering
├── model.py       # model training/evaluation/scoring
├── clv.py         # CLV, revenue-at-risk, RFM segmentation
├── train.py       # CLI: build snapshots -> train -> evaluate -> save model.pkl + metrics.json
├── app.py         # Shiny dashboard
└── tests/         # pytest: cleaning, gap computation, no-leakage, RFM
```

## Running it

`data/` is gitignored — it holds real customer records and is never committed, so it won't exist yet in a
fresh clone. Create it yourself and drop in your Zenoti exports before running anything:

```bash
mkdir data
# copy your Zenoti Appointments export to data/appointments_clean.csv (required)
# optionally copy a Sales-Accrual export to data/sales.csv (enables revenue/CLV features)
```

```bash
pip install -r requirements.txt
python train.py        # builds snapshots, trains the model, writes data/model.pkl + data/metrics.json
pytest tests/           # verify cleaning, labeling, and no-leakage behavior
shiny run app.py        # launch the dashboard
```

## Honest limitations

- The churn horizon (90 days) and snapshot cadence (monthly) are configurable in `config.py` but not yet
  tuned against business cost of a false positive/negative (e.g., cost of an unnecessary promo vs. a
  missed at-risk guest).
- Revenue (`TotalSpend`/`AvgTicket`) is now a real, active feature, but its model importance is modest
  (~0.5% combined vs. ~68% for `Recency`) — it doesn't change *who* the model flags much, but it does
  make revenue-at-risk and CLV numbers throughout the app real dollars instead of a visit-count proxy.
- No SHAP/partial-dependence explanations yet — feature importance is model-native
  (`feature_importances_` / coefficients), which is directional but not causal.
- Predictions use whatever model was last trained via `train.py`; the "Refresh Data" upload updates the
  appointment history immediately but does not auto-retrain the model (by design, to keep the dashboard
  responsive — retrain manually and periodically).

---

*Built as an extension of a Data Analytics & Operations internship project at Unique Threading Salon and
Spa, Summer 2026.*
