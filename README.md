# Salon Churn & Customer Value Dashboards

Two versions of a customer-retention dashboard for a local threading salon, built on appointment
data exported from Zenoti.

## [`v2/`](v2/) — Churn & Customer Value Intelligence (current)

Replaces v1's hardcoded churn rule with a trained ML model (evaluated out-of-time against that same
rule as a baseline), and adds a revenue/customer-lifetime-value layer so outreach can be prioritized
by dollar impact, not just headcount. See [`v2/README.md`](v2/README.md) for the full write-up.

## [`v1/`](v1/) — Original Churn Analysis Dashboard

The original rules-only dashboard: classifies customers as Active / At Risk / Churned using
personalized visit-cycle thresholds, with cohort retention tracking. See
[`v1/README.md`](v1/README.md) for details. Kept here as the baseline v2 is measured against.

---

*Built as part of a Data Analytics & Operations internship at Unique Threading Salon and Spa.*
