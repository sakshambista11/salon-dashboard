"""Point-in-time feature engineering for the churn model.

The core idea: a label built from "DaysSinceLastVisit as of today" (v1's approach)
can't be used to train a model — recency both defines the label and would be the
top predictive feature, which is circular. Instead we build many historical
snapshots: for each cutoff date T, compute features using ONLY visits strictly
before T, and label the guest churned if they have no visit in (T, T + horizon].
Training on early snapshots and evaluating on later ones (see model.py) makes
this an honest, leakage-free supervised learning problem.
"""
import numpy as np
import pandas as pd

import config
import pipeline


def _snapshot_dates(clean_df):
    min_date = clean_df["Appointment Date"].min()
    max_date = clean_df["Appointment Date"].max()
    start = min_date + pd.Timedelta(days=config.MIN_HISTORY_DAYS)
    end = max_date - pd.Timedelta(days=config.CHURN_HORIZON_DAYS)
    if start >= end:
        return pd.DatetimeIndex([])
    return pd.date_range(start=start, end=end, freq=config.SNAPSHOT_FREQUENCY)


def _behavioral_features(history_df, snapshot_date):
    """Per-guest features computed only from appointments strictly before snapshot_date."""
    visits = pipeline.compute_visits(history_df)
    visits = visits.sort_values("Appointment Date")
    g = visits.groupby("Guest Name")

    first_visit = g["Appointment Date"].min()
    last_visit = g["Appointment Date"].max()
    frequency = g.size().rename("Frequency")
    mean_gap = g["Gap"].mean().rename("MeanGap")
    median_gap = g["Gap"].median().rename("MedianGap")
    last_gap = g["Gap"].last().rename("LastGap")

    recency = (snapshot_date - last_visit).dt.days.rename("Recency")
    tenure = (snapshot_date - first_visit).dt.days.rename("Tenure")

    gap_trend = (last_gap / mean_gap).replace([np.inf, -np.inf], np.nan).rename("GapTrend")

    service_diversity = (
        history_df.groupby("Guest Name")["Service Name"].nunique().rename("ServiceDiversity")
    )

    stylist_counts = history_df.groupby(["Guest Name", "Stylist"]).size()
    top_stylist_count = stylist_counts.groupby("Guest Name").max()
    total_appt_count = history_df.groupby("Guest Name").size()
    stylist_loyalty = (top_stylist_count / total_appt_count).rename("StylistLoyalty")

    lead_days = (history_df["Appointment Date"] - history_df["Booked Date"]).dt.days
    booking_lead = (
        lead_days.groupby(history_df["Guest Name"]).mean().rename("AvgBookingLeadDays")
    )

    is_weekend = history_df["Appointment Date"].dt.dayofweek.isin([5, 6])
    weekend_share = (
        is_weekend.groupby(history_df["Guest Name"]).mean().rename("WeekendShare")
    )

    features = pd.concat(
        [
            frequency,
            recency,
            tenure,
            mean_gap,
            median_gap,
            gap_trend,
            service_diversity,
            stylist_loyalty,
            booking_lead,
            weekend_share,
        ],
        axis=1,
    )
    return features.reset_index()


def _label(guest_names, snapshot_date, clean_df):
    horizon_end = snapshot_date + pd.Timedelta(days=config.CHURN_HORIZON_DAYS)
    future = clean_df[
        (clean_df["Appointment Date"] > snapshot_date)
        & (clean_df["Appointment Date"] <= horizon_end)
    ]
    returned = set(future["Guest Name"].unique())
    return pd.Series(
        [0 if name in returned else 1 for name in guest_names],
        index=guest_names,
        name="Churned",
    )


def build_snapshots(clean_df, sales_df=None):
    """Build the long (guest x snapshot) training frame with features + leakage-free labels.

    Optional `sales_df` (columns: "Guest Name", "Sale Date", "Sale Amount") adds
    revenue features when available; omitted entirely if not provided.
    """
    clean_df = clean_df.copy()
    clean_df["Appointment Date"] = pd.to_datetime(clean_df["Appointment Date"])
    clean_df["Booked Date"] = pd.to_datetime(clean_df["Booked Date"])

    snapshots = []
    for snapshot_date in _snapshot_dates(clean_df):
        history = clean_df[clean_df["Appointment Date"] < snapshot_date]
        if history.empty:
            continue

        feats = _behavioral_features(history, snapshot_date)

        eligible = (
            (feats["Tenure"] >= config.MIN_HISTORY_DAYS)
            & (feats["Recency"] <= config.ELIGIBILITY_LOOKBACK_DAYS)
        )
        feats = feats[eligible]
        if feats.empty:
            continue

        labels = _label(feats["Guest Name"].tolist(), snapshot_date, clean_df)
        feats["Churned"] = feats["Guest Name"].map(labels)
        feats["SnapshotDate"] = snapshot_date

        if sales_df is not None:
            feats = _attach_sales_features(feats, sales_df, snapshot_date)

        snapshots.append(feats)

    if not snapshots:
        return pd.DataFrame()
    return pd.concat(snapshots, ignore_index=True)


def _attach_sales_features(feats, sales_df, snapshot_date):
    sales_history = sales_df[sales_df["Sale Date"] < snapshot_date]
    agg = sales_history.groupby("Guest Name")["Sale Amount"].agg(
        TotalSpend="sum", AvgTicket="mean"
    )
    return feats.merge(agg, on="Guest Name", how="left")


FEATURE_COLUMNS = [
    "Frequency",
    "Recency",
    "Tenure",
    "MeanGap",
    "MedianGap",
    "GapTrend",
    "ServiceDiversity",
    "StylistLoyalty",
    "AvgBookingLeadDays",
    "WeekendShare",
]


def build_current_features(clean_df, as_of=None, sales_df=None):
    """Features for TODAY's guests (for live scoring), using the same logic as training."""
    clean_df = clean_df.copy()
    clean_df["Appointment Date"] = pd.to_datetime(clean_df["Appointment Date"])
    clean_df["Booked Date"] = pd.to_datetime(clean_df["Booked Date"])
    as_of = as_of or pipeline.today()

    feats = _behavioral_features(clean_df, as_of)
    eligible = feats["Tenure"] >= 0  # no snapshot-eligibility gate for live scoring
    feats = feats[eligible]
    if sales_df is not None:
        feats = _attach_sales_features(feats, sales_df, as_of)
    return feats
