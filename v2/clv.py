"""Customer lifetime value, revenue-at-risk, and RFM segmentation.

Everything here degrades gracefully without a revenue export: CLV and
revenue-at-risk fall back to a frequency-weighted proxy (measured in expected
visits rather than dollars), and RFM's Monetary axis falls back to Frequency.
Drop a Zenoti Sales-Accrual export at `data/sales.csv` (see ingest.py) to
activate the dollar-denominated versions with no code changes.
"""
import pandas as pd

import config


def historical_clv(sales_df):
    return sales_df.groupby("Guest Name")["Sale Amount"].sum().rename("HistoricalCLV")


def forward_value(guest_df):
    """Expected value of a guest's typical spend over the next `CLV_FORECAST_YEARS`
    year(s) — annual spend rate (their historical visit frequency x average ticket)
    projected forward and discounted once as a simple uncertainty haircut.
    Denominated in dollars if AvgTicket is available, otherwise in raw
    expected-visit units."""
    tenure_years = (guest_df["Tenure"] / 365.0).clip(lower=0.25)
    visits_per_year = (guest_df["Frequency"] / tenure_years).clip(upper=52)
    ticket = guest_df["AvgTicket"].fillna(guest_df["AvgTicket"].median()) if "AvgTicket" in guest_df.columns else 1.0
    return (
        ticket
        * visits_per_year
        * config.CLV_FORECAST_YEARS
        / (1 + config.CLV_DISCOUNT_RATE)
    ).rename("ForwardValue")


def _rfm_scores(guest_df):
    df = guest_df.copy()
    q = config.RFM_QUANTILES
    max_score = q - 1

    r_rank = pd.qcut(df["Recency"].rank(method="first"), q, labels=False, duplicates="drop")
    df["R_score"] = max_score - r_rank  # recent visit -> high score
    df["F_score"] = pd.qcut(df["Frequency"].rank(method="first"), q, labels=False, duplicates="drop")
    if "TotalSpend" in df.columns:
        df["M_score"] = pd.qcut(df["TotalSpend"].rank(method="first"), q, labels=False, duplicates="drop")
    else:
        df["M_score"] = df["F_score"]  # no revenue data: frequency proxies monetary value

    df["RFM_Score"] = df["R_score"] + df["F_score"] + df["M_score"]
    return df


def _segment(row, max_score):
    r, f = row["R_score"], row["F_score"]
    if r >= max_score - 1 and f >= max_score - 1:
        return "Champions"
    if f >= max_score - 1 and r <= 1:
        return "At-Risk Loyalists"
    if r >= max_score - 1:
        return "New / Recent"
    if r <= 1 and f <= 1:
        return "Hibernating"
    if r <= 1:
        return "Slipping Away"
    return "Regulars"


def build_value_table(guest_df, sales_df=None):
    """guest_df: output of features.build_current_features(...) merged with
    model.predict_live(...)'s ChurnProbability column. Returns guest_df enriched
    with CLV, revenue-at-risk, and RFM segment, sorted by risk descending."""
    df = guest_df.copy()

    if sales_df is not None:
        df = df.merge(historical_clv(sales_df), on="Guest Name", how="left")

    df["ForwardValue"] = forward_value(df)
    df["ValueUnit"] = "USD" if "AvgTicket" in df.columns else "expected visits"
    df["RevenueAtRisk"] = df["ChurnProbability"] * df["ForwardValue"]

    df = _rfm_scores(df)
    max_score = config.RFM_QUANTILES - 1
    df["Segment"] = df.apply(lambda row: _segment(row, max_score), axis=1)

    return df.sort_values("RevenueAtRisk", ascending=False).reset_index(drop=True)
