import pandas as pd
import numpy as np
CHURN_CEILING = 180
SINGLE_ACTIVE = 60
SINGLE_AT_RISK = 120
_STATUS_PRIORITY = {"Checkin": 0, "Closed": 1, "Deleted": 2}
_DEDUP_COLS=['Appointment Date', 'Booked Date', 'Invoice No', 'Guest Name',
       'Service Name', 'Center Name', 'Start Time', 'End Time',
       'Scheduled Service Duration', 'Scheduled Service and Recovery Duration',
       'Recovery Time', 'Stylist', 'Status']

def load_raw(filepath):
    df = pd.read_csv(filepath)
    df["Appointment Date"] = pd.to_datetime(df["Appointment Date"])
    df["Booked Date"] = pd.to_datetime(df["Booked Date"])
    df = df.sort_values("Appointment Date")
    return df

def clean_names(df):
    df = df.copy()
    df["Guest Name"] = df["Guest Name"].str.strip().str.title()
    return df

def resolve_status_conflicts(df):
    df = df.copy()
    df = df.drop_duplicates(subset=_DEDUP_COLS)
    df['_sp'] = df['Status'].map(_STATUS_PRIORITY).fillna(0)
    df = (df.sort_values('_sp')
            .drop_duplicates(subset=['Invoice No', 'Service Name'], keep='last')
            .drop(columns='_sp'))
    return df

def filter_closed(df):
    df = df.copy()
    df = df[df['Status'] == 'Closed']
    df = df[df['Guest Name'] != "Online Guest"]
    return df.sort_values(by = "Appointment Date").reset_index(drop=True)

def compute_visits(df):
    visits = df.drop_duplicates(subset=["Guest Name", "Appointment Date"]).copy()
    visits["Appointment Date"] = pd.to_datetime(visits["Appointment Date"])
    visits["Gap"] = (
        visits.sort_values("Appointment Date")
              .groupby("Guest Name")["Appointment Date"]
              .diff()
              .dt.days
    )
    return visits

def _classify(row):
    days = row["DaysSinceLastVisit"]
    gap = row["AverageGap"]
    if days >= CHURN_CEILING:
        return "Churned"

    if row["VisitCount"] == 1:
        if days < SINGLE_ACTIVE:
            return "Active"
        elif SINGLE_ACTIVE <= days < SINGLE_AT_RISK:
            return "At Risk"
        else:
            return "Churned"
    else:
        if days < gap:
            return "Active"
        elif gap <= days < 2 * gap:
            return "At Risk"
        else:
            return "Churned"

def build_guest_summary(df, visits):

    ref_date = pd.Timestamp("today").normalize()
 
    summary = visits.groupby("Guest Name").agg(
        AverageGap = ("Gap",              "mean"),
        LastVisit  = ("Appointment Date", "max"),
        VisitCount = ("Appointment Date", "count"),
    )
 
    summary["DaysSinceLastVisit"] = (ref_date - summary["LastVisit"]).dt.days
    summary["Status"] = summary.apply(_classify, axis=1)
 
    last_service = (
        df.sort_values("Appointment Date")
          .groupby("Guest Name")["Service Name"]
          .last()
          .rename("LastService")
    )
 
    preferred_stylist = (
        df.groupby(["Guest Name", "Stylist"])
          .size()
          .reset_index(name="_n")
          .sort_values("_n", ascending=False)
          .drop_duplicates(subset="Guest Name")
          .set_index("Guest Name")["Stylist"]
          .rename("PreferredStylist")
    )
 
    summary = summary.join(last_service).join(preferred_stylist)
    return summary.reset_index()

def run_pipeline(clean_df):
    visits = compute_visits(clean_df)
    guest_summary = build_guest_summary(clean_df, visits)
    return guest_summary

def merge_incremental(existing_clean_df, new_upload_path):
    new_df = load_raw(new_upload_path)
    new_df = clean_names(new_df)
    combined = pd.concat([existing_clean_df, new_df], ignore_index=True)
    combined = resolve_status_conflicts(combined)
    return filter_closed(combined)
    
def get_data_cutoff(df):
    return pd.to_datetime(df["Appointment Date"]).max().date()

def build_cohort_retention(df):
    df = df.copy()
    first_month_visit = df.groupby("Guest Name")['Appointment Date'].min().dt.to_period("M")
    df["cohort_month"] = df['Guest Name'].map(first_month_visit)
    df["appt_month"] = df["Appointment Date"].dt.to_period("M")
    df["months_offset"] = (df["appt_month"] - df["cohort_month"]).apply(lambda x: x.n)
    df = df[df["months_offset"].between(0, 12)]
    counts = df.groupby(["cohort_month", "months_offset"])['Guest Name'].nunique()
    matrix = counts.unstack(fill_value=0)
    retention = matrix.div(matrix[0], axis=0) * 100
    today_period = pd.Period(pd.Timestamp("today"), "M")
    for cohort in retention.index:
        max_offset = (today_period - cohort).n
        retention.loc[cohort, retention.columns > max_offset] = np.nan
    cutoff = pd.Period(pd.Timestamp("today"), "M") - 36
    retention = retention[retention.index >= cutoff]
    return retention