"""CLI entry point: build snapshots -> train models -> evaluate out-of-time -> persist.

Usage: python train.py
"""
import json

import joblib

import config
import features
import ingest
import model


def main():
    clean_df = ingest.load_clean_appointments()
    sales_df = ingest.try_load_sales()

    print(f"Loaded {len(clean_df):,} appointments "
          f"({clean_df['Appointment Date'].min().date()} to {clean_df['Appointment Date'].max().date()})")
    if sales_df is not None:
        print(f"Sales data found ({len(sales_df):,} rows) — revenue features enabled.")
    else:
        print("No sales data found — training on behavioral features only.")

    snapshots = features.build_snapshots(clean_df, sales_df=sales_df)
    if snapshots.empty:
        raise SystemExit("Not enough history to build any snapshots — need a longer date range.")

    print(f"Built {len(snapshots):,} (guest, snapshot) rows across "
          f"{snapshots['SnapshotDate'].nunique()} monthly snapshots.")

    train_df, test_df = model.temporal_split(snapshots)
    print(f"Temporal split: {len(train_df):,} train rows / {len(test_df):,} test rows "
          f"(test = most recent {config.TEST_SNAPSHOT_FRACTION:.0%} of snapshot dates).")

    cols = model.feature_columns(snapshots)
    X_train, y_train = train_df[cols], train_df["Churned"]
    X_test, y_test = test_df[cols], test_df["Churned"]

    models = model.build_models(cols)
    results, fitted = model.evaluate(models, X_train, y_train, X_test, y_test, test_df)

    print("\n--- Out-of-time test set performance ---")
    for name, m in results.items():
        auc = f"ROC-AUC={m['roc_auc']:.3f}" if "roc_auc" in m else "ROC-AUC=n/a"
        print(f"{name:22s} acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  {auc}")

    best_model = fitted["gradient_boosting"]
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, config.MODEL_PATH)
    with open(config.METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved model -> {config.MODEL_PATH}")
    print(f"Saved metrics -> {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
