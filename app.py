from shiny import App, ui, render
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Load data once at the top
guest_summary = pd.read_csv("guest_summary.csv")
df = pd.read_csv("appointments_clean.csv")
df["Appointment Date"] = pd.to_datetime(df["Appointment Date"])
monthly_customers = df.groupby(df["Appointment Date"].dt.to_period("M"))["Guest Name"].nunique()
monthly_customers = monthly_customers[monthly_customers.index < pd.Timestamp("today").to_period("M")]
stylists = sorted(guest_summary["PreferredStylist"].dropna().unique().tolist())
# Keep only stylists with enough customers to be meaningful

# Find each stylist's most recent appointment
last_active = df.groupby("Stylist")["Appointment Date"].max()
recent_cutoff = pd.Timestamp("today") - pd.Timedelta(days=90)
current_stylists = last_active[last_active >= recent_cutoff].index

stylist_counts = guest_summary["PreferredStylist"].value_counts()
main_stylists = stylist_counts[(stylist_counts >= 30) & (stylist_counts.index.isin(current_stylists))].index
retention_data = guest_summary[guest_summary["PreferredStylist"].isin(main_stylists)]

stylist_retention = pd.crosstab(
    retention_data["PreferredStylist"], retention_data["Status"], normalize="index"
) * 100
stylist_retention = stylist_retention.reindex(columns=["Active", "At Risk", "Churned"], fill_value=0)
stylist_retention = stylist_retention.sort_values("Active")

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
                choices=["All"] + stylists,
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
    )
)

# Server logic
def server(input, output, session):
    @render.text
    def active_count():
        count = (guest_summary["Status"] == "Active").sum()
        return f"{count:,}"  # we'll fill this in
    @render.text
    def at_risk_count():
        count = (guest_summary["Status"] == "At Risk").sum()
        return f"{count:,}"
    @render.text
    def churned_count():
        count = (guest_summary["Status"] == "Churned").sum()
        return f"{count:,}"
    @render.text
    def active_pct():
        total = len(guest_summary)
        pct = (guest_summary["Status"] == "Active").sum() / total * 100
        return f"{pct:.0f}% of all customers"
    @render.text
    def at_risk_pct():
        total = len(guest_summary)
        pct = (guest_summary["Status"] == "At Risk").sum() / total * 100
        return f"{pct:.0f}% of all customers"
    @render.text
    def churned_pct():
        total = len(guest_summary)
        pct = (guest_summary["Status"] == "Churned").sum() / total * 100
        return f"{pct:.0f}% of all customers"
    @render.text
    def new_guest_stat():
        single_visit = (guest_summary["VisitCount"] == 1).sum()
        total_guest = len(guest_summary)
        no_return_pct = (single_visit/total_guest) * 100
        return f"{no_return_pct:.0f}% of new guests never return"
    @render.text
    def new_guest_detail():
        single = (guest_summary["VisitCount"] == 1).sum()
        total = len(guest_summary)
        returned = total - single
        return f"Out of {total:,} first-time visitors, only {returned:,} came back."
    import matplotlib.dates as mdates

    @render.plot
    def trend_chart():
        fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
        x_data = monthly_customers.index.to_timestamp()
        y_data = monthly_customers.values
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
        return fig
    @render.data_frame
    def customer_table():
        data = guest_summary.copy()
        if input.status_filter() != "All":
            data = data[data["Status"] == input.status_filter()]
        if input.stylist_filter() != "All":
            data = data[data["PreferredStylist"] == input.stylist_filter()]
        if input.name_search():
            data = data[data["Guest Name"].str.contains(
                input.name_search(), case=False, na=False
            )]
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

        names = stylist_retention.index
        active = stylist_retention["Active"]
        at_risk = stylist_retention["At Risk"]
        churned = stylist_retention["Churned"]

        ax.barh(names, active, color="#2e7d32", label="Active")
        ax.barh(names, at_risk, left=active, color="#f5a623", label="At Risk")
        ax.barh(names, churned, left=active + at_risk, color="#c0392b", label="Churned")

        ax.set_xlabel("Percentage of Customers")
        ax.legend(loc="lower right")
        return fig
app = App(app_ui, server)