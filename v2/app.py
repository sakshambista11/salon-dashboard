import json

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

import clv
import config
import features
import ingest
import model
import pipeline

# ---------------------------------------------------------------------------
# Startup load: appointments, optional sales export, trained model, metrics.
# Everything falls back gracefully — the app runs (mostly empty) with no data
# at all, and predictive tabs light up once `python train.py` has been run.
# ---------------------------------------------------------------------------

def _load_appointments():
    try:
        return ingest.load_clean_appointments()
    except FileNotFoundError:
        cols = ["Appointment Date", "Booked Date", "Invoice No", "Guest Name",
                "Service Name", "Center Name", "Start Time", "End Time",
                "Scheduled Service Duration", "Stylist", "Status"]
        df = pd.DataFrame(columns=cols)
        df["Appointment Date"] = pd.to_datetime(df["Appointment Date"])
        df["Booked Date"] = pd.to_datetime(df["Booked Date"])
        return df


def _load_model():
    try:
        return joblib.load(config.MODEL_PATH)
    except FileNotFoundError:
        return None


def _load_metrics():
    try:
        with open(config.METRICS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


_init_df = _load_appointments()
_sales_df = ingest.try_load_sales()
_model_pipe = _load_model()
_metrics = _load_metrics()

_EMPTY_MSG = "Upload data on the Refresh Data tab, then run `python train.py`, to see this."


def _empty_plot_fig(msg=_EMPTY_MSG):
    fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="#999")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


def _empty_plotly_fig(msg=_EMPTY_MSG):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(color="#999", size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def _recommended_action(prob):
    if prob >= 0.7:
        return "Call this week"
    if prob >= 0.4:
        return "Send win-back promo"
    return "No action needed"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

app_ui = ui.page_fluid(
    ui.h1("Unique Threading — Churn & Customer Value Intelligence"),
    ui.navset_tab(
        ui.nav_panel("Executive Overview",
            ui.layout_columns(
                ui.value_box("Predicted At-Risk", ui.output_text("risk_count"),
                             ui.output_text("risk_pct"), theme="warning"),
                ui.value_box("Revenue at Risk", ui.output_text("revenue_at_risk_total"),
                             ui.output_text("revenue_at_risk_detail"), theme="danger"),
                ui.value_box("Guests Scored", ui.output_text("scored_count"),
                             ui.output_text("scored_detail"), theme="success"),
            ),
            ui.card(
                ui.h3(ui.span(ui.output_text("new_guest_stat"), style="color: #c00;")),
                ui.output_text("new_guest_detail"),
            ),
            ui.card(
                ui.h3("Monthly Active Customers"),
                ui.p("Unique customers who visited each calendar month since the salon opened on Zenoti."),
                ui.output_plot("trend_chart"),
            ),
        ),
        ui.nav_panel("Churn Predictions",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_slider("min_prob", "Minimum churn probability", 0.0, 1.0, 0.4, step=0.05),
                    ui.input_text("name_search_pred", "Search by name"),
                ),
                ui.p("Guests ranked by revenue at risk (churn probability x expected future value). "
                     "This is the priority call/outreach list."),
                ui.output_data_frame("predictions_table"),
            ),
        ),
        ui.nav_panel("Customer Segments",
            ui.card(
                ui.h3("RFM Segments"),
                ui.p("Recency / Frequency / Monetary segmentation of today's guest base."),
                output_widget("segment_bar"),
            ),
            ui.card(
                ui.h3("Recency vs. Frequency"),
                output_widget("segment_scatter"),
            ),
        ),
        ui.nav_panel("Stylist Retention",
            ui.card(
                ui.h3("Customer Retention by Stylist"),
                ui.p("What share of each stylist's customers are still active, at risk, or churned "
                     "(rules-based classification). Sorted by active rate."),
                ui.output_plot("stylist_chart"),
            ),
        ),
        ui.nav_panel("Cohort Retention",
            ui.card(
                ui.h3("New Customer Cohort Retention"),
                ui.p("Each row is a group of customers who first visited in that month. Columns show "
                     "what % of that group returned in subsequent months."),
                ui.output_plot("cohort_chart", height="700px"),
            ),
        ),
        ui.nav_panel("Refresh Data",
            ui.card(
                ui.h3("Update Dashboard Data"),
                ui.output_text("cutoff_message"),
                ui.input_file("new_data", "Upload Zenoti Export", accept=[".csv"]),
                ui.output_text("upload_status"),
                ui.p("Note: uploading refreshes appointment data and rescoring against the ",
                     "currently trained model. To retrain the model itself on the new data, ",
                     "run `python train.py` from the terminal."),
            ),
            ui.card(
                ui.h3("Save Your Progress"),
                ui.p("Important: this dashboard does not save your data permanently. "
                     "Before closing, download your cleaned data below and keep it on your computer. "
                     "Next time you open the dashboard, upload that file first to restore everything."),
                ui.download_button("download_data", "Download Cleaned Data"),
            ),
        ),
        ui.nav_panel("Advanced",
            ui.card(
                ui.h3("Model vs. Baseline (out-of-time test set)"),
                ui.p("Trained on early snapshots, evaluated on the most recent held-out snapshots "
                     "the model never saw during training — an honest measure of real-world performance."),
                ui.output_data_frame("metrics_table"),
            ),
            ui.card(ui.h3("ROC Curve"), output_widget("roc_plot")),
            ui.card(ui.h3("Feature Importance"), output_widget("importance_plot")),
        ),
    )
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def server(input, output, session):
    clean_df = reactive.Value(_init_df)
    upload_msg = reactive.Value("")

    @reactive.Calc
    def guest_summary():
        df = clean_df()
        if df.empty:
            return pd.DataFrame(columns=["Guest Name", "Status", "PreferredStylist"])
        return pipeline.run_pipeline(df)

    @reactive.Calc
    def scored_guests():
        df = clean_df()
        if df.empty or _model_pipe is None:
            return pd.DataFrame()
        feats = features.build_current_features(df, sales_df=_sales_df)
        if feats.empty:
            return pd.DataFrame()
        scored = model.predict_live(_model_pipe, feats)
        return clv.build_value_table(scored, sales_df=_sales_df)

    @reactive.Calc
    def cohort_retention():
        return pipeline.build_cohort_retention(clean_df())

    @reactive.Calc
    def monthly_customers():
        df = clean_df()
        mc = df.groupby(df["Appointment Date"].dt.to_period("M"))["Guest Name"].nunique()
        mc = mc[mc.index < pipeline.today().to_period("M")]
        return mc

    @reactive.Calc
    def stylist_retention():
        df = clean_df()
        gs = guest_summary()
        if df.empty or gs.empty:
            return pd.DataFrame()
        last_active = df.groupby("Stylist")["Appointment Date"].max()
        recent_cutoff = pipeline.today() - pd.Timedelta(days=config.STYLIST_ACTIVE_WINDOW_DAYS)
        current_stylists = last_active[last_active >= recent_cutoff].index
        stylist_counts = gs["PreferredStylist"].value_counts()
        main_stylists = stylist_counts[
            (stylist_counts >= config.STYLIST_MIN_GUESTS) & (stylist_counts.index.isin(current_stylists))
        ].index
        data = gs[gs["PreferredStylist"].isin(main_stylists)]
        sr = pd.crosstab(data["PreferredStylist"], data["Status"], normalize="index") * 100
        sr = sr.reindex(columns=["Active", "At Risk", "Churned"], fill_value=0)
        return sr.sort_values("Active")

    # --- Refresh Data ---
    @reactive.Effect
    @reactive.event(input.new_data)
    def handle_upload():
        file_info = input.new_data()
        if file_info is None:
            return
        path = file_info[0]["datapath"]
        try:
            updated_df = pipeline.merge_incremental(clean_df(), path)
            updated_df.to_csv(config.APPOINTMENTS_CLEAN_PATH, index=False)
            clean_df.set(updated_df)
            upload_msg.set("✓ Data updated successfully.")
        except Exception as e:
            print(f"Upload failed: {e}")
            upload_msg.set(
                "⚠️ Could not process that file. Make sure it's a Zenoti "
                "Appointments export (.csv) and try again."
            )

    @render.download(filename=lambda: f"appointments_clean_{pipeline.today().date()}.csv")
    def download_data():
        yield clean_df().to_csv(index=False)

    @render.text
    def upload_status():
        return upload_msg()

    @render.text
    def cutoff_message():
        df = clean_df()
        if df.empty:
            return "No data loaded yet. Upload a Zenoti Appointments CSV below to get started."
        date = pipeline.get_data_cutoff(df)
        return f"Your data is current through {date}. To refresh, export appointments from Zenoti starting {date} and upload below."

    # --- Executive Overview ---
    @render.text
    def risk_count():
        sg = scored_guests()
        if sg.empty:
            return "—"
        return f"{(sg['ChurnProbability'] >= 0.5).sum():,}"

    @render.text
    def risk_pct():
        sg = scored_guests()
        if sg.empty:
            return "Run train.py to enable predictions"
        pct = (sg["ChurnProbability"] >= 0.5).mean() * 100
        return f"{pct:.0f}% of scored guests"

    @render.text
    def revenue_at_risk_total():
        sg = scored_guests()
        if sg.empty:
            return "—"
        total = sg["RevenueAtRisk"].sum()
        unit = sg["ValueUnit"].iloc[0]
        return f"${total:,.0f}" if unit == "USD" else f"{total:,.0f} visits"

    @render.text
    def revenue_at_risk_detail():
        sg = scored_guests()
        if sg.empty:
            return ""
        unit = sg["ValueUnit"].iloc[0]
        note = "estimated" if unit == "USD" else "no revenue data yet — frequency-weighted proxy"
        return f"{note}"

    @render.text
    def scored_count():
        sg = scored_guests()
        return f"{len(sg):,}" if not sg.empty else "0"

    @render.text
    def scored_detail():
        return "guests with enough history to score"

    @render.text
    def new_guest_stat():
        gs = guest_summary()
        if gs.empty:
            return "No data yet"
        single_visit = (gs["VisitCount"] == 1).sum()
        pct = single_visit / len(gs) * 100
        return f"{pct:.0f}% of new guests never return"

    @render.text
    def new_guest_detail():
        gs = guest_summary()
        if gs.empty:
            return "Upload data to populate this dashboard."
        total = len(gs)
        single = (gs["VisitCount"] == 1).sum()
        return f"Out of {total:,} first-time visitors, only {total - single:,} came back."

    @render.plot
    def trend_chart():
        if clean_df().empty:
            return _empty_plot_fig()
        fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
        x_data = monthly_customers().index.to_timestamp()
        y_data = monthly_customers().values
        ax.plot(x_data, y_data, color="#2b5c8f", linewidth=2.5, label="Active Customers")
        ax.fill_between(x_data, y_data, color="#2b5c8f", alpha=0.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5, color='#e0e0e0')
        ax.set_ylabel("Unique Customers", fontsize=10, color='#444444', labelpad=10)
        cutoff = pipeline.get_data_cutoff(clean_df())
        ax.axvline(pd.Timestamp(cutoff), color='#999', linestyle='--', linewidth=1)
        return fig

    # --- Churn Predictions ---
    @render.data_frame
    def predictions_table():
        sg = scored_guests()
        if sg.empty:
            return render.DataGrid(pd.DataFrame(
                {"": ["No predictions available. Upload data and run `python train.py`."]}
            ))
        data = sg[sg["ChurnProbability"] >= input.min_prob()]
        if input.name_search_pred():
            data = data[data["Guest Name"].str.contains(
                input.name_search_pred(), case=False, na=False, regex=False
            )]
        unit = data["ValueUnit"].iloc[0] if not data.empty else "USD"
        fmt_value = (lambda v: f"${v:,.0f}") if unit == "USD" else (lambda v: f"{v:,.0f} visits")

        display = data[["Guest Name", "ChurnProbability", "ForwardValue", "RevenueAtRisk", "Segment"]].copy()
        display["ChurnProbability"] = (display["ChurnProbability"] * 100).round(0).astype(int).astype(str) + "%"
        display["ForwardValue"] = display["ForwardValue"].apply(fmt_value)
        display["RevenueAtRisk"] = display["RevenueAtRisk"].apply(fmt_value)
        display["Action"] = data["ChurnProbability"].apply(_recommended_action)
        display = display.rename(columns={
            "ChurnProbability": "Churn Probability",
            "ForwardValue": "Expected Value",
            "RevenueAtRisk": "Revenue at Risk",
        })
        return render.DataGrid(display)

    # --- Advanced (model insights) ---
    @render.data_frame
    def metrics_table():
        if _metrics is None:
            return render.DataGrid(pd.DataFrame(
                {"": ["Run `python train.py` to generate model metrics."]}
            ))
        rows = []
        for name, m in _metrics.items():
            rows.append({
                "Model": name,
                "Accuracy": round(m.get("accuracy", float("nan")), 3),
                "F1": round(m.get("f1", float("nan")), 3),
                "ROC-AUC": round(m["roc_auc"], 3) if "roc_auc" in m else "n/a",
            })
        return render.DataGrid(pd.DataFrame(rows))

    @render_widget
    def roc_plot():
        if _metrics is None:
            return _empty_plotly_fig()
        fig = go.Figure()
        for name in ("gradient_boosting", "logistic_regression"):
            m = _metrics.get(name)
            if not m or "roc_curve" not in m:
                continue
            fig.add_trace(go.Scatter(
                x=m["roc_curve"]["fpr"], y=m["roc_curve"]["tpr"],
                mode="lines", name=f"{name} (AUC={m['roc_auc']:.2f})",
            ))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(dash="dash", color="#ccc"), name="Random"))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                          height=420, margin=dict(l=40, r=20, t=20, b=40))
        return fig

    @render_widget
    def importance_plot():
        if _metrics is None or "feature_importance" not in _metrics.get("gradient_boosting", {}):
            return _empty_plotly_fig()
        imp = _metrics["gradient_boosting"]["feature_importance"]
        items = sorted(imp.items(), key=lambda kv: kv[1])
        fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h"))
        fig.update_layout(xaxis_title="Importance", yaxis_title="",
                          height=420, margin=dict(l=120, r=20, t=20, b=40))
        return fig

    # --- Customer Segments ---
    @render_widget
    def segment_bar():
        sg = scored_guests()
        if sg.empty:
            return _empty_plotly_fig()
        counts = sg["Segment"].value_counts()
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values))
        fig.update_layout(xaxis_title="Segment", yaxis_title="Guests",
                          height=420, margin=dict(l=40, r=20, t=20, b=40))
        return fig

    @render_widget
    def segment_scatter():
        sg = scored_guests()
        if sg.empty:
            return _empty_plotly_fig()
        fig = go.Figure()
        for seg, group in sg.groupby("Segment"):
            fig.add_trace(go.Scatter(
                x=group["Recency"], y=group["Frequency"], mode="markers", name=seg,
                marker=dict(size=6, opacity=0.6),
                text=group["Guest Name"],
            ))
        fig.update_layout(xaxis_title="Recency (days since last visit)", yaxis_title="Frequency (visits)",
                          height=500, margin=dict(l=40, r=20, t=20, b=40))
        return fig

    # --- Stylist Retention ---
    @render.plot
    def stylist_chart():
        sr = stylist_retention()
        if sr.empty:
            return _empty_plot_fig()
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        names = sr.index
        active, at_risk, churned = sr["Active"], sr["At Risk"], sr["Churned"]
        ax.barh(names, active, color="#2e7d32", label="Active")
        ax.barh(names, at_risk, left=active, color="#f5a623", label="At Risk")
        ax.barh(names, churned, left=active + at_risk, color="#c0392b", label="Churned")
        ax.set_xlabel("Percentage of Customers")
        ax.legend(loc="lower right")
        return fig

    # --- Cohort Retention ---
    @render.plot
    def cohort_chart():
        matrix = cohort_retention()
        if matrix.empty:
            return _empty_plot_fig()
        fig, ax = plt.subplots(figsize=(16, 16), tight_layout=True)
        im = ax.imshow(matrix.values, aspect="auto", cmap="YlGn", vmin=0, vmax=40)
        fig.colorbar(im, ax=ax, label="Retention %")
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels([str(p) for p in matrix.index], fontsize=9)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.iloc[i, j]
                if not pd.isna(val):
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8,
                             color="white" if val > 25 else "black")
        ax.set_xlabel("Months since first visit")
        ax.set_ylabel("Cohort (first visit month)")
        return fig


app = App(app_ui, server)
