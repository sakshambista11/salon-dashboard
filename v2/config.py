from pathlib import Path

# --- paths ---
DATA_DIR = Path(__file__).parent / "data"
APPOINTMENTS_CLEAN_PATH = DATA_DIR / "appointments_clean.csv"
GUEST_SUMMARY_PATH = DATA_DIR / "guest_summary.csv"
SALES_PATH = DATA_DIR / "sales.csv"
FEEDBACK_PATH = DATA_DIR / "feedback.csv"
MEMBERSHIPS_PATH = DATA_DIR / "memberships.csv"
MODEL_PATH = DATA_DIR / "model.pkl"
METRICS_PATH = DATA_DIR / "metrics.json"

# --- rules baseline (v1 heuristic, kept for benchmarking) ---
CHURN_CEILING = 180
SINGLE_ACTIVE = 60
SINGLE_AT_RISK = 120

# --- ML snapshot / labeling ---
CHURN_HORIZON_DAYS = 90          # "did they NOT return within this many days after cutoff T"
SNAPSHOT_FREQUENCY = "MS"        # monthly snapshot cutoffs
MIN_HISTORY_DAYS = 30            # guest must have >=1 visit at least this long before T to be scored
ELIGIBILITY_LOOKBACK_DAYS = 365  # only score guests whose last visit before T was within this window
TEST_SNAPSHOT_FRACTION = 0.2     # most-recent snapshots held out for out-of-time evaluation

# --- stylist retention tab ---
STYLIST_MIN_GUESTS = 30
STYLIST_ACTIVE_WINDOW_DAYS = 90

# --- CLV ---
CLV_FORECAST_YEARS = 1     # forward-looking window: "what is this guest worth over the next N years"
CLV_DISCOUNT_RATE = 0.10   # mild discount applied once, as a simple uncertainty haircut

# --- RFM ---
RFM_QUANTILES = 5
