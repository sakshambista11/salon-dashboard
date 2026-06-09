from shiny import App, ui, render, reactive
import pipeline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Load data once at the top, fall back to empty frames if files don't exist yet
def _load_or_empty():
    appt_cols = ["Appointment Date", "Booked Date", "Invoice No", "Guest Name",
                 "Service Name", "Center Name", "Start Time", "End Time",
                 "Scheduled Service Duration", "Stylist", "Status"]
    gs_cols = ["Guest Name", "AverageGap", "LastVisit", "VisitCount",
               "DaysSinceLastVisit", "Status", "LastService", "PreferredStylist"]
    try:
        df = pd.read_csv("appointments_clean.csv")
        df["Appointment Date"] = pd.to_datetime(df["Appointment Date"], format="mixed")
        df["Booked Date"] = pd.to_datetime(df["Booked Date"], format="mixed")
    except FileNotFoundError:
        df = pd.DataFrame(columns=appt_cols)
        df["Appointment Date"] = pd.to_datetime(df["Appointment Date"])
        df["Booked Date"] = pd.to_datetime(df["Booked Date"])
    try:
        gs = pd.read_csv("guest_summary.csv")
    except FileNotFoundError:
        gs = pd.DataFrame(columns=gs_cols)
    return df, gs

_init_df, _init_gs = _load_or_empty()
_init_stylists = sorted(_init_gs["PreferredStylist"].dropna().unique().tolist())

# UI definition
app_ui = ui.page_fluid(
    ui.h1("Unique Threading — Churn Dashboard"),
    ui.navset_tab(
        ui.nav_panel("Overview",
            ui.layout_columns(
    ui.value_box(
        "Active",
        ui.output_text("active_count"),
        ui.output_text("active_pct"),
        theme="success"   # green
        ),
    ui.value_box(
        "At Risk",
        ui.output_text("at_risk_count"),
        ui.output_text("at_risk_pct"),
        theme="warning"   # yellow
        ),
    ui.value_box(
        "Churned",
        ui.output_text("churned_count"),
        ui.output_text("churned_pct"),
        theme="danger"    # red
        ),
    ),
    
    ui.card(
    ui.h3(
    ui.span(ui.output_text("new_guest_stat"), style="color: #c00;")),
    ui.output_text("new_guest_detail"),
    ),

    ui.card(
    ui.h3("Monthly Active Customers"),
    ui.p("Unique customers who visited each month since the salon opened on Zenoti. The upward trend shows the business is growing."),
    ui.output_plot("trend_chart"),
    )
        ),
        ui.nav_panel("At Risk Customers",
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select(
                "status_filter", "Status",
                choices=["At Risk", "Churned", "Active", "All"],
                selected="At Risk"
            ),
            ui.input_select(
                "stylist_filter", "Stylist",
                choices=["All"] + _init_stylists,
                selected="All"
            ),
            ui.input_text("name_search", "Search by name"),
        ),
        ui.output_data_frame("customer_table"),
    )
),
        ui.nav_panel("Stylist Retention",
            ui.card(
        ui.h3("Customer Retention by Stylist"),
        ui.p("What share of each stylist's customers are still active, at risk, or churned. Sorted by active rate."),
        ui.output_plot("stylist_chart"),
    )
        ),
        ui.nav_panel("Cohort Retention",
            ui.card(
            ui.h3("New Customer Cohort Retention"),
            ui.p("Each row is a group of customers who first visited in that month. Columns show what % of that group returned in subsequent months."),
            ui.output_plot("cohort_chart", height="700px"),
    )
),
        ui.nav_panel("Refresh Data",
            ui.card(
                ui.h3("Update Dashboard Data"),
                ui.output_text("cutoff_message"),
                ui.input_file("new_data", "Upload Zenoti Export", accept=[".csv"]),
                ui.output_text("upload_status"),
            ),
            ui.card(
                ui.h3("Save Your Progress"),
                ui.p("Important: this dashboard does not save your data permanently. "
                     "Before closing, download your cleaned data below and keep it on your computer. "
                     "Next time you open the dashboard, upload that file first to restore everything."),
                ui.download_button("download_data", "Download Cleaned Data"),
            ),
        )
    )
)

# Server logic
def server(input, output, session):
    clean_df = reactive.Value(_init_df)
    guest_summary_rv = reactive.Value(_init_gs)
    upload_msg = reactive.Value("")
    @reactive.Calc
    def cohort_retention():
        return pipeline.build_cohort_retention(clean_df())
    @reactive.Effect
    @reactive.event(input.new_data)
    def handle_upload():
        file_info = input.new_data()
        if file_info is None:
            return
        path = file_info[0]["datapath"]
        try:
            updated_df = pipeline.merge_incremental(clean_df(), path)
            updated_gs = pipeline.run_pipeline(updated_df)
            updated_df.to_csv("appointments_clean.csv", index=False)
            updated_gs.to_csv("guest_summary.csv", index=False)
            clean_df.set(updated_df)
            guest_summary_rv.set(updated_gs)
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
    @reactive.Calc
    def monthly_customers():
        df = clean_df()
        mc = df.groupby(df["Appointment Date"].dt.to_period("W"))["Guest Name"].nunique()
        mc = mc[mc.index < pipeline.today().to_period("W")]
        return mc
    @reactive.Calc
    def stylist_retention():
        df = clean_df()
        guest_summary = guest_summary_rv()
        last_active = df.groupby("Stylist")["Appointment Date"].max()
        recent_cutoff = pipeline.today() - pd.Timedelta(days=90)
        current_stylists = last_active[last_active >= recent_cutoff].index
        stylist_counts = guest_summary["PreferredStylist"].value_counts()
        main_stylists = stylist_counts[(stylist_counts >= 30) & (stylist_counts.index.isin(current_stylists))].index
        retention_data = guest_summary[guest_summary["PreferredStylist"].isin(main_stylists)]
        stylist_retention = pd.crosstab(
            retention_data["PreferredStylist"], retention_data["Status"], normalize="index"
        ) * 100
        stylist_retention = stylist_retention.reindex(columns=["Active", "At Risk", "Churned"], fill_value=0)
        stylist_retention = stylist_retention.sort_values("Active")
        return stylist_retention
    @render.text
    def active_count():
        count = (guest_summary_rv()["Status"] == "Active").sum()
        return f"{count:,}"
    @render.text
    def at_risk_count():
        count = (guest_summary_rv()["Status"] == "At Risk").sum()
        return f"{count:,}"
    @render.text
    def churned_count():
        count = (guest_summary_rv()["Status"] == "Churned").sum()
        return f"{count:,}"
    @render.text
    def active_pct():
        total = len(guest_summary_rv())
        if total == 0:
            return "—"
        pct = (guest_summary_rv()["Status"] == "Active").sum() / total * 100
        return f"{pct:.0f}% of all customers"
    @render.text
    def at_risk_pct():
        total = len(guest_summary_rv())
        if total == 0:
            return "—"
        pct = (guest_summary_rv()["Status"] == "At Risk").sum() / total * 100
        return f"{pct:.0f}% of all customers"
    @render.text
    def churned_pct():
        total = len(guest_summary_rv())
        if total == 0:
            return "—"
        pct = (guest_summary_rv()["Status"] == "Churned").sum() / total * 100
        return f"{pct:.0f}% of all customers"
    @render.text
    def new_guest_stat():
        total_guest = len(guest_summary_rv())
        if total_guest == 0:
            return "No data yet"
        single_visit = (guest_summary_rv()["VisitCount"] == 1).sum()
        no_return_pct = (single_visit / total_guest) * 100
        return f"{no_return_pct:.0f}% of new guests never return"
    @render.text
    def new_guest_detail():
        total = len(guest_summary_rv())
        if total == 0:
            return "Upload data to populate this dashboard."
        single = (guest_summary_rv()["VisitCount"] == 1).sum()
        returned = total - single
        return f"Out of {total:,} first-time visitors, only {returned:,} came back."

    @render.plot
    def cohort_chart():
        matrix = cohort_retention()
        fig, ax = plt.subplots(figsize=(16, 16), tight_layout=True)
        if matrix.empty:
            ax.text(0.5, 0.5, "Upload data on the Refresh Data tab to see this chart.",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='#999')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            return fig
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
                    text_color = "white" if val > 25 else "black"
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            fontsize=8, color=text_color)
        ax.set_xlabel("Months since first visit")
        ax.set_ylabel("Cohort (first visit month)")
        return fig
    
    @render.plot
    def trend_chart():
        fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
        if clean_df().empty:
            ax.text(0.5, 0.5, "Upload data on the Refresh Data tab to see this chart.",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='#999')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            return fig
        x_data = monthly_customers().index.to_timestamp()
        y_data = monthly_customers().values
        ax.plot(x_data, y_data, color="#2b5c8f", linewidth=2.5, label="Active Customers")
        ax.fill_between(x_data, y_data, color="#2b5c8f", alpha=0.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5, color='#e0e0e0')
        ax.grid(True, which='major', axis='x', linestyle=':', alpha=0.4, color='#e0e0e0')
        ax.tick_params(axis='both', which='major', labelsize=9, colors='#444444')
        ax.set_ylabel("Unique Customers", fontsize=10, color='#444444', labelpad=10)
        cutoff = pipeline.get_data_cutoff(clean_df())
        ax.axvline(pd.Timestamp(cutoff), color='#999', linestyle='--', linewidth=1)
        ax.annotate(str(cutoff), xy=(pd.Timestamp(cutoff), ax.get_ylim()[1]),
            xytext=(-5, -5), textcoords='offset points',
            ha='right', va='top', fontsize=9, color='#666')
        return fig
    @render.data_frame
    def customer_table():
        data = guest_summary_rv().copy()
        if input.status_filter() != "All":
            data = data[data["Status"] == input.status_filter()]
        if input.stylist_filter() != "All":
            data = data[data["PreferredStylist"] == input.stylist_filter()]
        if input.name_search():
            data = data[data["Guest Name"].str.contains(
                input.name_search(), case=False, na=False, regex=False
            )]
        if data.empty:
            return render.DataGrid(pd.DataFrame(
                {"": ["No customers to display. Upload data on the Refresh Data tab to get started."]}
            ))
        data = data.sort_values("DaysSinceLastVisit", ascending=False)
        display = data[[
            "Guest Name", "Status", "LastVisit", "DaysSinceLastVisit",
            "LastService", "PreferredStylist", "AverageGap", "VisitCount"
        ]].copy()
        display["LastVisit"] = pd.to_datetime(display["LastVisit"]).dt.date
        display["AverageGap"] = display["AverageGap"].round(0)
        return render.DataGrid(display)
    @render.plot
    def stylist_chart():
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        sr = stylist_retention()
        if sr.empty:
            ax.text(0.5, 0.5, "Upload data on the Refresh Data tab to see this chart.",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='#999')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            return fig
        names = sr.index
        active = sr["Active"]
        at_risk = sr["At Risk"]
        churned = sr["Churned"]
        ax.barh(names, active, color="#2e7d32", label="Active")
        ax.barh(names, at_risk, left=active, color="#f5a623", label="At Risk")
        ax.barh(names, churned, left=active + at_risk, color="#c0392b", label="Churned")
        ax.set_xlabel("Percentage of Customers")
        ax.legend(loc="lower right")
        return fig
app = App(app_ui, server)