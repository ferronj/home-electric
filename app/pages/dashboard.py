"""Dashboard tab: KPIs, time series, distributions, cost analysis."""

import numpy as np
import pandas as pd
import streamlit as st

from app.components.kpi_cards import render_kpi_row
from src.analysis.cost import (
    compute_cumulative_savings,
    compute_roi_payback,
    compute_savings_credible_interval,
    compute_savings_from_full_model_result,
    estimate_annual_savings,
)
from src.analysis.daily_usage import (
    compute_heating_cooling_degree_days,
    compute_period_statistics,
    compute_usage_change,
)
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

    # Degree-day efficiency comparison (kWh per HDD/CDD)
    if (
        "temp_f" in display_df.columns
        and "period" in display_df.columns
        and len(periods) >= 2
    ):
        dd_df = compute_heating_cooling_degree_days(display_df, base_temp_f=65.0)
        rows = []
        for p in periods:
            sub = dd_df[dd_df["period"] == p]
            total_hdd = sub["hdd"].sum()
            total_cdd = sub["cdd"].sum()
            total_kwh = sub["usage_kwh"].sum()
            kwh_per_hdd = total_kwh / total_hdd if total_hdd > 0 else float("nan")
            kwh_per_cdd = total_kwh / total_cdd if total_cdd > 0 else float("nan")
            rows.append({
                "Period": p,
                "Days": len(sub),
                "Total HDD": round(total_hdd, 0),
                "Total CDD": round(total_cdd, 0),
                "kWh / HDD": round(kwh_per_hdd, 2),
                "kWh / CDD": round(kwh_per_cdd, 2),
            })
        dd_summary = pd.DataFrame(rows).set_index("Period")
        st.markdown("**Degree-Day Efficiency** _(base 65°F)_")
        st.dataframe(dd_summary, use_container_width=True)
        st.caption(
            "Lower kWh/HDD after install = more efficient heating. "
            "kWh/CDD covers cooling-side efficiency (less meaningful when cooling load is small)."
        )

    # Per-event savings and ROI
    if len(periods) >= 2 and change:
        rate = config.electricity_rate
        savings = estimate_annual_savings(stats[periods[0]], stats[periods[-1]], rate)

        # If a Full Temperature model has been run this session, compute the
        # temperature-normalized savings — this is the corrected headline number
        # the project's CLAUDE.md flags the naive comparison as misleading for.
        norm = None
        ci = None
        latest_cost_event = next(
            (e for e in reversed(events) if e.equipment_cost and e.equipment_cost > 0),
            None,
        )
        if "temp_f" in display_df.columns:
            model_result = st.session_state.get("model_result")
            norm = compute_savings_from_full_model_result(
                model_result, display_df, rate_per_kwh=rate
            )
            if norm:
                ci = compute_savings_credible_interval(
                    model_result,
                    display_df,
                    rate_per_kwh=rate,
                    equipment_cost=(
                        latest_cost_event.net_cost if latest_cost_event else None
                    ),
                )

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

            if norm:
                st.markdown("**Temperature-Normalized Annual Savings (Bayesian)**")
                if ci:
                    kwh = ci["annual_kwh_saved"]
                    cost = ci["annual_cost_saved"]
                    hdi_pct = int(round(ci["hdi_prob"] * 100))
                    st.markdown(
                        f"- Energy: **{kwh['mean']:.0f} kWh/year** "
                        f"_({hdi_pct}% HDI: {kwh['hdi_low']:.0f} – {kwh['hdi_high']:.0f})_"
                    )
                    st.markdown(
                        f"- Cost: **${cost['mean']:.0f}/year** "
                        f"_({hdi_pct}% HDI: ${cost['hdi_low']:.0f} – ${cost['hdi_high']:.0f})_"
                    )
                    st.markdown(f"- Monthly: **${cost['mean'] / 12:.0f}/month**")
                    prob_pos = ci.get("prob_savings_positive")
                    if prob_pos is not None and prob_pos < 1.0:
                        st.caption(
                            f"Posterior probability that savings are positive: "
                            f"{prob_pos * 100:.1f}%."
                        )
                else:
                    st.markdown(f"- Energy: **{norm['annual_kwh_saved']:.0f} kWh/year**")
                    st.markdown(f"- Cost: **${norm['annual_cost_saved']:.0f}/year**")
                    st.markdown(f"- Monthly: **${norm['annual_cost_saved'] / 12:.0f}/month**")
                st.caption(
                    f"From Full Temperature model posterior "
                    f"({norm['before_period']} → {norm['after_period']}), "
                    f"applied to {norm['n_temp_days_used']} days of historical temps."
                )

        with col_roi:
            latest = latest_cost_event
            if latest:
                # Prefer the temp-normalized number for ROI when it's available
                annual_cost_for_roi = (
                    norm["annual_cost_saved"] if norm else savings["annual_cost_saved"]
                )
                roi = compute_roi_payback(latest.net_cost, annual_cost_for_roi)
                source = "Bayesian" if norm else "simple"
                st.markdown(f"**ROI for {latest.label}** _({source} savings)_")
                st.markdown(f"- Net equipment cost: **${latest.net_cost:,.0f}**")
                if roi["simple_payback_years"] < 100:
                    payback_ci = ci.get("simple_payback_years") if ci else None
                    if payback_ci:
                        hdi_pct = int(round(ci["hdi_prob"] * 100))
                        st.markdown(
                            f"- Payback period: **{roi['simple_payback_years']:.1f} years** "
                            f"_({hdi_pct}% HDI: "
                            f"{payback_ci['hdi_low']:.1f} – {payback_ci['hdi_high']:.1f})_"
                        )
                    else:
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
