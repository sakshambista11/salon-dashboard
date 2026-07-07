"""Train/evaluate/persist the churn classifier, and score current guests.

Three models are compared, from naive to sophisticated:
  1. Rules baseline  — v1's hardcoded threshold heuristic (`_classify`), reimplemented
                        here against the snapshot feature columns so it can be scored
                        on the exact same out-of-time test set as the ML models.
  2. Logistic Regression — interpretable linear baseline.
  3. Gradient Boosting    — the model used for live predictions.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
import features as feat_mod


def feature_columns(df):
    cols = [c for c in feat_mod.FEATURE_COLUMNS if c in df.columns]
    if "TotalSpend" in df.columns:
        cols += ["TotalSpend", "AvgTicket"]
    return cols


def temporal_split(snapshots_df):
    """Earliest snapshots train the model; the most recent ones are held out
    for out-of-time evaluation (never shuffled — this is a time series)."""
    dates = sorted(snapshots_df["SnapshotDate"].unique())
    n_test = max(1, int(round(len(dates) * config.TEST_SNAPSHOT_FRACTION)))
    test_dates = set(dates[-n_test:])
    train_df = snapshots_df[~snapshots_df["SnapshotDate"].isin(test_dates)]
    test_df = snapshots_df[snapshots_df["SnapshotDate"].isin(test_dates)]
    return train_df, test_df


def _make_pipeline(classifier, cols, scale):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer([("num", Pipeline(steps), cols)])
    return Pipeline([("pre", pre), ("clf", classifier)])


def build_models(cols):
    return {
        "gradient_boosting": _make_pipeline(
            GradientBoostingClassifier(random_state=42), cols, scale=False
        ),
        "logistic_regression": _make_pipeline(
            LogisticRegression(max_iter=1000, random_state=42, solver="liblinear"),
            cols,
            scale=True,
        ),
    }


def rules_baseline_predict(df):
    """v1's `_classify` heuristic, reimplemented against snapshot feature names,
    collapsed to a binary churned/not-churned call for a fair head-to-head."""

    def classify(row):
        days, gap, freq = row["Recency"], row["MeanGap"], row["Frequency"]
        if days >= config.CHURN_CEILING:
            return 1
        if freq == 1:
            return 1 if days >= config.SINGLE_AT_RISK else 0
        if pd.isna(gap) or days < 2 * gap:
            return 0
        return 1

    return df.apply(classify, axis=1)


def _binary_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate(models, X_train, y_train, X_test, y_test, test_df):
    results = {}
    fitted = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        # Apple Accelerate's BLAS emits a benign matmul RuntimeWarning on some
        # arm64 builds even when output is fully finite (verified separately).
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
            proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        fpr, tpr, _ = roc_curve(y_test, proba)
        precision, recall, _ = precision_recall_curve(y_test, proba)

        metrics = _binary_metrics(y_test, pred)
        metrics["roc_auc"] = roc_auc_score(y_test, proba)
        metrics["pr_auc"] = average_precision_score(y_test, proba)
        metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        metrics["pr_curve"] = {"precision": precision.tolist(), "recall": recall.tolist()}

        clf = pipe.named_steps["clf"]
        cols = feature_columns(test_df)
        if hasattr(clf, "feature_importances_"):
            metrics["feature_importance"] = dict(
                zip(cols, clf.feature_importances_.tolist())
            )
        elif hasattr(clf, "coef_"):
            metrics["feature_importance"] = dict(zip(cols, clf.coef_[0].tolist()))

        results[name] = metrics

    baseline_pred = rules_baseline_predict(test_df)
    results["rules_baseline"] = _binary_metrics(y_test, baseline_pred)

    return results, fitted


def predict_live(pipe, current_features_df):
    """Score today's active guests. Returns current_features_df with a ChurnProbability column."""
    cols = feature_columns(current_features_df)
    out = current_features_df.copy()
    out["ChurnProbability"] = pipe.predict_proba(current_features_df[cols])[:, 1]
    return out
