"""KPI metric card components."""

import streamlit as st


def render_kpi_row(stats: dict, change: dict | None = None, rate: float = 0.154):
    """Render a row of KPI metric cards."""
    periods = list(stats.keys())

    if len(periods) >= 2:
        # Baseline always comes first
        before_key = "Baseline" if "Baseline" in periods else periods[0]
        after_key = [p for p in periods if p != before_key][-1]
        before = stats[before_key]
        after = stats[after_key]

        cols = st.columns(5)

        cols[0].metric(
            "Days of Data",
            f"{sum(s['count'] for s in stats.values())}",
        )

        cols[1].metric(
            f"Avg Daily Usage ({before_key})",
            f"{before['mean_daily_kwh']:.1f} kWh",
        )

        cols[2].metric(
            f"Avg Daily Usage ({after_key})",
            f"{after['mean_daily_kwh']:.1f} kWh",
            delta=f"{after['mean_daily_kwh'] - before['mean_daily_kwh']:.1f} kWh",
            delta_color="inverse",
        )

        if change:
            cols[3].metric(
                "Est. Annual Savings",
                f"${change['projected_annual_savings_cost']:.0f}",
            )

            # Payback period if we have equipment cost info
            cols[4].metric(
                "Daily Savings",
                f"${change['daily_savings_cost']:.2f}",
            )
    else:
        period = periods[0]
        s = stats[period]
        cols = st.columns(4)
        cols[0].metric("Days of Data", f"{s['count']}")
        cols[1].metric("Avg Daily Usage", f"{s['mean_daily_kwh']:.1f} kWh")
        cols[2].metric("Avg Daily Cost", f"${s['mean_daily_cost']:.2f}")
        cols[3].metric("Total Cost", f"${s['total_cost']:.0f}")
