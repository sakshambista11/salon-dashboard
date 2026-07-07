"""Loaders for Zenoti report exports. Every optional report is loaded defensively:
if the file isn't present, or its columns don't match what we expect, we return
None and the rest of the pipeline degrades gracefully (that feature set just
never gets built). Drop matching exports into `data/` to activate them.
"""
import pandas as pd

import config

# Zenoti's exact column names vary by report version, so each loader tries a
# few common aliases before giving up.
_SALES_ALIASES = {
    "Guest Name": ["Guest Name", "Guest", "Customer Name"],
    "Sale Date": ["Sale Date", "Invoice Date", "Date", "Appointment Date"],
    "Invoice No": ["Invoice No", "Invoice Number", "Invoice"],
    "Sale Amount": [
        "Sale Amount", "Sales(Inc. Tax)", "Sales (Inc. Tax)", "Total",
        "Net Sales", "Amount", "Collected Amount",
    ],
}
_FEEDBACK_ALIASES = {
    "Guest Name": ["Guest Name", "Guest", "Customer Name"],
    "Rating": ["Rating", "Score", "Overall Rating"],
    "Feedback Date": ["Feedback Date", "Date", "Appointment Date"],
}
_MEMBERSHIP_ALIASES = {
    "Guest Name": ["Guest Name", "Guest", "Member Name", "Customer Name"],
    "Status": ["Status", "Membership Status"],
}


def _resolve_columns(df, aliases):
    rename = {}
    for canonical, options in aliases.items():
        for option in options:
            if option in df.columns:
                rename[option] = canonical
                break
        else:
            return None  # required column missing entirely
    return df.rename(columns=rename)


def _try_load(path, aliases, date_cols=()):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    df = _resolve_columns(df, aliases)
    if df is None:
        return None
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
    return df


def load_clean_appointments():
    df = pd.read_csv(config.APPOINTMENTS_CLEAN_PATH)
    df["Appointment Date"] = pd.to_datetime(df["Appointment Date"], format="mixed")
    df["Booked Date"] = pd.to_datetime(df["Booked Date"], format="mixed")
    return df


def try_load_sales():
    """Zenoti's Sales-Accrual export is line-item level (one row per service/product
    per invoice). Collapse to one row per invoice so downstream spend aggregations
    (TotalSpend, AvgTicket) reflect per-visit totals rather than per-line-item values."""
    df = _try_load(config.SALES_PATH, _SALES_ALIASES, date_cols=["Sale Date"])
    if df is None:
        return None
    # Zenoti's sales export capitalizes names differently than the appointments
    # export (e.g. "ELIZABETH MARTINEZ" vs "Elizabeth Martinez"); normalize to
    # match pipeline.clean_names() so guests join correctly downstream.
    df["Guest Name"] = df["Guest Name"].str.strip().str.title()
    invoice_totals = (
        df.groupby(["Guest Name", "Invoice No", "Sale Date"], as_index=False)["Sale Amount"].sum()
    )
    return invoice_totals


def try_load_feedback():
    return _try_load(config.FEEDBACK_PATH, _FEEDBACK_ALIASES, date_cols=["Feedback Date"])


def try_load_memberships():
    return _try_load(config.MEMBERSHIPS_PATH, _MEMBERSHIP_ALIASES)
