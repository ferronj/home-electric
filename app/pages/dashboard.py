"""Dashboard tab: KPIs, time series, distributions, cost analysis."""

import numpy as np
import pandas as pd
import streamlit as st

from app.components.kpi_cards import render_kpi_row
from src.analysis.cost import (
    compute_cumulative_savings,
    compute_roi_payback,
    estimate_annual_savings,
)
from src.analysis.daily_usage import compute_period_statistics, compute_usage_change
from src.viz.plots import (
    plot_cumulative_savings,
    plot_daily_usage_timeseries,
    plot_monthly_cost_comparison,
    plot_temperature_vs_usage,
    plot_usage_distribution,
)


def render_dashboard(app_state: dict):
    """Render the dashboard tab."""
    daily_df = app_state.get("daily_df")
    merged_df = app_state.get("merged_df")
    events = app_state.get("events", [])
    config = app_state.get("config")

    if daily_df is None:
        st.info("Upload electricity usage data or place CSV files in data/raw/ to get started.")
        return

    # Use merged if available (has temperature), otherwise daily
    display_df = merged_df if merged_df is not None else daily_df

    # --- KPI ROW ---
    stats = compute_period_statistics(display_df)
    change = None
    periods = list(stats.keys())
    # Ensure Baseline is first
    if "Baseline" in periods:
        periods = ["Baseline"] + [p for p in periods if p != "Baseline"]
    if len(periods) >= 2:
        change = compute_usage_change(stats, periods[0], periods[-1])

    render_kpi_row(stats, change, rate=config.electricity_rate)

    st.divider()

    # --- TIME SERIES ---
    col_temp, col_avg = st.columns(2)
    show_temp = col_temp.checkbox("Show temperature", value=False)
    show_avg = col_avg.checkbox("Show 7-day average", value=True)

    fig_ts = plot_daily_usage_timeseries(
        display_df, events,
        show_temperature=show_temp and "temp_f" in display_df.columns,
        show_rolling_avg=show_avg,
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # --- DISTRIBUTION + SCATTER ---
    col_dist, col_scatter = st.columns(2)

    with col_dist:
        if "period" in display_df.columns:
            fig_dist = plot_usage_distribution(display_df)
            st.plotly_chart(fig_dist, use_container_width=True)

    with col_scatter:
        if merged_df is not None and "temp_f" in merged_df.columns:
            fig_scatter = plot_temperature_vs_usage(merged_df)
            st.plotly_chart(fig_scatter, use_container_width=True)

    # --- COST ANALYSIS ---
    st.divider()
    st.subheader("Cost Analysis")

    if "period" in display_df.columns:
        fig_cost = plot_monthly_cost_comparison(display_df)
        st.plotly_chart(fig_cost, use_container_width=True)

    # Per-event savings and ROI
    if len(periods) >= 2 and change:
        rate = config.electricity_rate
        savings = estimate_annual_savings(stats[periods[0]], stats[periods[-1]], rate)

        col_sav, col_roi = st.columns(2)

        with col_sav:
            st.markdown("**Projected Annual Savings (simple comparison)**")
            st.markdown(f"- Energy: **{savings['annual_kwh_saved']:.0f} kWh/year**")
            st.markdown(f"- Cost: **${savings['annual_cost_saved']:.0f}/year**")
            st.markdown(f"- Monthly: **${savings['monthly_cost_saved']:.0f}/month**")
            after_days = stats[periods[-1]]["count"]
            if after_days < 90:
                st.caption(
                    f"Note: Only {after_days} days of post-install data. "
                    "Seasonal differences may skew this estimate. "
                    "Use the Bayesian Modeling tab for temperature-normalized analysis."
                )

        with col_roi:
            # Find the most recent event with a cost
            cost_events = [e for e in events if e.equipment_cost and e.equipment_cost > 0]
            if cost_events:
                latest = cost_events[-1]
                roi = compute_roi_payback(latest.net_cost, savings["annual_cost_saved"])
                st.markdown(f"**ROI for {latest.label}**")
                st.markdown(f"- Net equipment cost: **${latest.net_cost:,.0f}**")
                if roi["simple_payback_years"] < 100:
                    st.markdown(
                        f"- Payback period: **{roi['simple_payback_years']:.1f} years**"
                    )
                else:
                    st.markdown("- Payback period: **N/A** (savings too low)")
                st.markdown(f"- 10-year net savings: **${roi['total_savings_10yr']:,.0f}**")
            else:
                st.caption("Add equipment cost to events for ROI analysis.")

        # Cumulative savings chart
        if events:
            latest_event = sorted(events, key=lambda e: e.date)[-1]
            baseline_cost = stats[periods[0]]["mean_daily_cost"]
            savings_df = compute_cumulative_savings(
                display_df, pd.Timestamp(latest_event.date), baseline_cost
            )
            if not savings_df.empty:
                fig_cum = plot_cumulative_savings(
                    savings_df,
                    latest_event.label,
                    equipment_cost=latest_event.net_cost if latest_event.equipment_cost else None,
                )
                st.plotly_chart(fig_cum, use_container_width=True)

    # --- RAW DATA ---
    with st.expander("Raw Daily Data"):
        st.dataframe(display_df, use_container_width=True)
        csv = display_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "daily_usage.csv", "text/csv")
