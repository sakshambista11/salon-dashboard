# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -r requirements.txt   # pandas, numpy, matplotlib, shiny, shinywidgets, scikit-learn, plotly, joblib, pytest
python train.py                   # build snapshots -> train models -> write data/model.pkl + data/metrics.json
pytest tests/                     # run the full suite
pytest tests/test_pipeline.py::test_build_snapshots_features_do_not_leak_future_visits  # run a single test
shiny run app.py --launch-browser # launch the dashboard locally
```

`train.py` must be run at least once (it produces `data/model.pkl` and `data/metrics.json`) before the
Churn Predictions, Model Insights, and Customer Segments tabs in `app.py` will show real data — without
it they render their empty-state fallback.

## Architecture

This is v2 of a salon customer-churn project (v1 lives in the sibling `../v1` folder and
is a simpler rules-only dashboard). v2 replaces the hardcoded churn heuristic with a trained ML model
while keeping the heuristic around as a benchmark baseline, and adds a revenue/CLV layer.

### Module dependency chain

```
config.py  -> single source of truth for every threshold, path, and constant used anywhere else
ingest.py  -> defensive loaders (appointments required; sales/feedback/memberships optional)
pipeline.py -> cleaning/dedup/gap-computation (copied verbatim from v1, UI-agnostic)
features.py -> point-in-time snapshot builder (the core ML data prep, depends on pipeline.py)
model.py   -> trains/evaluates/scores against features.py's output
clv.py     -> CLV + RFM, consumes model.py's live predictions
train.py   -> CLI orchestrator: ingest -> features -> model -> persist to data/
app.py     -> Shiny UI, orchestrates all of the above reactively
```

Changes to `config.py` constants (e.g. `CHURN_HORIZON_DAYS`, `ELIGIBILITY_LOOKBACK_DAYS`) ripple through
`features.py`'s snapshot building and therefore change what `train.py` produces — always re-run
`train.py` after touching those values.

### The point-in-time snapshot pattern (read before touching features.py or model.py)

The old rule (`DaysSinceLastVisit >= 180` as of *today*) can't be turned into an ML label directly:
recency both defines the label and would be the dominant feature, which is circular and leaks the
answer into the input. `features.build_snapshots()` avoids this by generating monthly historical
cutoffs `T` and, for each one, computing every feature from appointments **strictly before `T`**
(`history = clean_df[clean_df["Appointment Date"] < T]`), then labeling `Churned=1` only if there is no
visit in `(T, T + CHURN_HORIZON_DAYS]`. `model.temporal_split()` then trains on the earliest snapshots
and evaluates on the most recent ones — never a random shuffle, since this is a time series.

Any change to feature engineering must preserve this invariant: a feature for snapshot `T` must never
depend on data at or after `T`. `tests/test_pipeline.py::test_build_snapshots_features_do_not_leak_future_visits`
enforces this by comparing features computed on two datasets that differ only after the snapshot dates
being compared — it should fail loudly if a change breaks the boundary.

`features.build_current_features()` reuses the identical feature logic to score *today's* guests for
live predictions in `app.py` — keep the two entry points (`build_snapshots` for training,
`build_current_features` for inference) using the same underlying `_behavioral_features()` helper rather
than duplicating feature logic.

### Graceful degradation for optional Zenoti data

Only the Zenoti appointments export (`data/appointments_clean.csv`) is required. Revenue data
(`data/sales.csv`), feedback, and memberships are optional exports whose presence is detected at
`ingest.py` load time (column-alias matching per report; returns `None` if the file is absent or doesn't
match expected columns). Downstream code branches on `is not None` rather than assuming the data exists:
- `features.py` only attaches `TotalSpend`/`AvgTicket` columns when `sales_df` is provided.
- `clv.py`'s `forward_value()`/`build_value_table()` fall back to a frequency-weighted proxy (visits,
  not dollars) when `AvgTicket` isn't present, and label the output's `ValueUnit` column accordingly
  (`"USD"` vs `"expected visits"`) so the UI can display the right unit.
- `model.feature_columns()` dynamically includes `TotalSpend`/`AvgTicket` only if they exist in the
  snapshot frame, so the model trains on whatever columns are actually available.

When adding a new optional data source, follow this same three-part pattern (ingest alias-matching ->
conditional feature attachment -> UI label reflecting what's actually available) rather than assuming
the file exists.

### Model comparison, not just a single model

`model.py` always trains and evaluates three things side by side on the same out-of-time test set:
Gradient Boosting (used for live scoring), Logistic Regression (interpretable baseline), and
`rules_baseline_predict()` (v1's original heuristic, reimplemented against the snapshot column names so
it can be scored on identical data). Keep this three-way comparison intact when modifying `model.py` —
it's the evidence that the ML approach is actually better than the rule it replaced, not just different.

### Shiny + Plotly gotcha

`app.py` renders interactive Plotly charts (ROC curve, feature importance, RFM segment charts) using
`shinywidgets`' `output_widget`/`render_widget`, not `ui.output_ui`/`@render.ui` with `fig.to_html()`.
The latter looks like it works (no errors) but renders a permanently blank chart: Shiny updates reactive
UI via dynamic DOM injection, and browsers do not execute `<script>` tags inserted that way, so
Plotly's chart-drawing script silently never runs. Any new interactive chart must go through
`output_widget`/`render_widget`. Static matplotlib charts (`trend_chart`, `stylist_chart`,
`cohort_chart`) correctly use `ui.output_plot`/`@render.plot` instead and don't have this issue.

### Data privacy

`data/` and all `*.csv`/`*.pkl` are gitignored — customer records should never be committed. Sample/real
data lives only in `data/` on the local machine.
