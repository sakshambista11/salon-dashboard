import pandas as pd
import pytest

import clv
import config
import features
import pipeline


def _base_row(overrides=None):
    row = {
        "Appointment Date": pd.Timestamp("2024-01-10"),
        "Booked Date": pd.Timestamp("2024-01-01"),
        "Invoice No": "INV1",
        "Guest Name": "Jane Doe",
        "Service Name": "Haircut",
        "Center Name": "Main",
        "Start Time": "10:00",
        "End Time": "10:30",
        "Scheduled Service Duration": 30,
        "Scheduled Service and Recovery Duration": 30,
        "Recovery Time": 0,
        "Stylist": "Alex",
        "Room": "1",
        "Status": "Closed",
        "Queue appointment type": "",
    }
    row.update(overrides or {})
    return row


# --- cleaning / dedup ---

def test_resolve_status_conflicts_prefers_closed_over_checkin():
    df = pd.DataFrame([_base_row({"Status": "Checkin"}), _base_row({"Status": "Closed"})])
    resolved = pipeline.resolve_status_conflicts(df)
    assert len(resolved) == 1
    assert resolved.iloc[0]["Status"] == "Closed"


def test_resolve_status_conflicts_deleted_outranks_closed():
    df = pd.DataFrame([_base_row({"Status": "Closed"}), _base_row({"Status": "Deleted"})])
    resolved = pipeline.resolve_status_conflicts(df)
    assert len(resolved) == 1
    assert resolved.iloc[0]["Status"] == "Deleted"


def test_filter_closed_drops_online_guest_and_non_closed():
    df = pd.DataFrame([
        _base_row({"Invoice No": "A", "Status": "Closed"}),
        _base_row({"Invoice No": "B", "Status": "Checkin"}),
        _base_row({"Invoice No": "C", "Status": "Closed", "Guest Name": "Online Guest"}),
    ])
    result = pipeline.filter_closed(df)
    assert len(result) == 1
    assert result.iloc[0]["Invoice No"] == "A"


# --- gap computation ---

def test_compute_visits_gap():
    df = pd.DataFrame([
        _base_row({"Appointment Date": pd.Timestamp("2024-01-01"), "Invoice No": "A"}),
        _base_row({"Appointment Date": pd.Timestamp("2024-01-11"), "Invoice No": "B"}),
        _base_row({"Appointment Date": pd.Timestamp("2024-02-10"), "Invoice No": "C"}),
    ])
    visits = pipeline.compute_visits(df).sort_values("Appointment Date")
    gaps = visits["Gap"].tolist()
    assert pd.isna(gaps[0])
    assert gaps[1] == 10
    assert gaps[2] == 30


# --- point-in-time labeling ---

def test_label_zero_when_visit_falls_within_horizon():
    df = pd.DataFrame([_base_row({"Appointment Date": pd.Timestamp("2024-03-15"), "Guest Name": "G1"})])
    labels = features._label(["G1"], pd.Timestamp("2024-03-01"), df)
    assert labels["G1"] == 0


def test_label_one_when_no_visit_within_horizon():
    df = pd.DataFrame([_base_row({"Appointment Date": pd.Timestamp("2024-08-01"), "Guest Name": "G1"})])
    # horizon end = 2024-03-01 + 90 days ~= 2024-05-30, well before the Aug visit
    labels = features._label(["G1"], pd.Timestamp("2024-03-01"), df)
    assert labels["G1"] == 1


def test_build_snapshots_features_do_not_leak_future_visits(monkeypatch):
    monkeypatch.setattr(config, "MIN_HISTORY_DAYS", 20)
    monkeypatch.setattr(config, "CHURN_HORIZON_DAYS", 60)
    monkeypatch.setattr(config, "ELIGIBILITY_LOOKBACK_DAYS", 3650)

    shared = [
        _base_row({"Appointment Date": pd.Timestamp("2024-01-01"), "Guest Name": "G1", "Invoice No": "A"}),
        _base_row({"Appointment Date": pd.Timestamp("2024-01-15"), "Guest Name": "G1", "Invoice No": "B"}),
        _base_row({"Appointment Date": pd.Timestamp("2024-05-01"), "Guest Name": "G1", "Invoice No": "C"}),
    ]
    df_base = pd.DataFrame(shared)
    df_with_future = pd.DataFrame(shared + [
        _base_row({"Appointment Date": pd.Timestamp("2024-05-20"), "Guest Name": "G1", "Invoice No": "D"}),
    ])

    snaps_base = features.build_snapshots(df_base)
    snaps_future = features.build_snapshots(df_with_future)

    common_dates = set(snaps_base["SnapshotDate"]) & set(snaps_future["SnapshotDate"])
    assert common_dates, "expected overlapping snapshot dates to compare"

    for d in common_dates:
        row_base = snaps_base[snaps_base["SnapshotDate"] == d].set_index("Guest Name").loc["G1"]
        row_future = snaps_future[snaps_future["SnapshotDate"] == d].set_index("Guest Name").loc["G1"]
        for col in features.FEATURE_COLUMNS:
            a, b = row_base[col], row_future[col]
            if pd.isna(a) and pd.isna(b):
                continue
            assert a == pytest.approx(b), f"leakage detected in {col!r} at snapshot {d}"


# --- RFM segmentation ---

def test_rfm_segmentation_assigns_valid_segments():
    df = pd.DataFrame({
        "Guest Name": [f"G{i}" for i in range(10)],
        "Recency": [5, 10, 20, 40, 60, 90, 150, 200, 300, 360],
        "Frequency": [20, 15, 10, 8, 6, 5, 3, 2, 1, 1],
        "Tenure": [400] * 10,
        "ChurnProbability": [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.95],
    })
    table = clv.build_value_table(df)
    valid_segments = {"Champions", "At-Risk Loyalists", "New / Recent", "Hibernating", "Slipping Away", "Regulars"}
    assert len(table) == 10
    assert set(table["Segment"]).issubset(valid_segments)
    assert (table["RevenueAtRisk"] >= 0).all()
