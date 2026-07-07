# Salon Churn Analysis Dashboard

A customer retention dashboard for a local threading salon, built on multi-year appointment data exported from Zenoti. Identifies at-risk customers using personalized visit-cycle thresholds and tracks cohort retention over time.

## The Problem

The salon had years of appointment history sitting in Zenoti but no visibility into customer retention — no way to know which regulars had quietly stopped coming, or whether first-time visitors were turning into repeat customers. Re-engagement was reactive at best.

This project turns that raw appointment data into an actionable dashboard the owner can use to spot churn early and prioritize outreach.

## What It Does

- **Classifies every customer** as Active, At Risk, or Churned based on their own personal visit cycle (not a one-size-fits-all cutoff)
- **Surfaces a filterable at-risk list** the owner can use as a weekly call/text list
- **Tracks cohort retention** — for each month's new customers, what percentage returned in subsequent months
- **Breaks down retention by stylist** to inform staffing and training decisions
- **Supports incremental data refresh** — new Zenoti exports merge into existing data without re-running the full pipeline from scratch
- **Self-bootstrapping** — the dashboard ships empty; the owner uploads Zenoti exports and downloads the cleaned data to keep locally between sessions

## Dataset

- ~92,000 appointment records across ~8,000 unique customers
- Date range: April 2021 – June 2026
- Source: Zenoti CSV exports (multiple overlapping files, deduplicated via a two-step status-priority strategy)

## Tech Stack

- **Python 3.13** — pandas, NumPy, matplotlib
- **Shiny for Python** — interactive dashboard
- File-based, no external database or API dependency

## Architecture

```
salon-dashboard/
├── app.py            # Shiny dashboard (entry point)
├── pipeline.py       # ETL + churn logic as importable functions
├── requirements.txt  # Pinned dependencies
├── README.md
└── .gitignore        # Excludes all *.csv from version control
```

The pipeline is fully separated from the UI layer — `pipeline.py` is import-clean and testable in isolation. The dashboard reads its outputs and renders them reactively.

### Stateless Deployment

The dashboard is designed to run on free-tier hosting where the server filesystem doesn't persist between restarts. Rather than depend on a database or paid persistent storage, the architecture treats the owner's local machine as the source of truth:

- The app ships empty
- The owner uploads raw Zenoti exports OR a previously-saved cleaned CSV via the Refresh Data tab
- The pipeline cleans, deduplicates, and computes churn classifications in-session
- Before closing, the owner downloads the current cleaned dataset and keeps it locally
- Next session: re-upload that file, continue working

This trades a small amount of user friction for zero hosting cost and zero customer data ever sitting on a third-party server.

## Methodology (High-Level)

Each customer is judged by their own visit history rather than a static threshold. A customer who comes in every 3 weeks and a customer who comes in every 3 months are both "regulars" by their own standards, and the model treats them that way:

- **Active** — fewer days since last visit than their personal average
- **At Risk** — between 1× and 2× their personal average
- **Churned** — beyond 2× their personal average (or a hard 180-day ceiling)

Single-visit customers fall back to static 60 / 120-day thresholds since no personal average can be computed.

## Privacy

All customer data (`*.csv`) is gitignored and never committed. The deployed instance starts with no data — customer records only exist in-session while the owner is using the dashboard, and the cleaned dataset lives on her local machine between sessions.

---

*Built as part of a Data Analytics & Operations internship at Unique Threading Salon and Spa, Summer 2026.*
