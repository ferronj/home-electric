"""Sidebar: data upload, event config, cost settings."""

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src.analysis.events import load_events, save_events, validate_events
from src.data.cleaning import (
    add_event_periods,
    aggregate_to_daily,
    interpolate_temperature_gaps,
    merge_usage_temperature,
)
from src.data.loader import (
    load_multiple_pgn_files,
    load_multiple_temperature_files,
    load_nasa_power_temperature,
    load_pgn_interval_data,
)
from src.data.schemas import AppConfig, EquipmentEvent
from src.data.temperature_api import fetch_temperature_range
from src.analysis.cost import compute_electricity_rate


def render_sidebar() -> dict:
    """Render sidebar and return app state dict."""
    st.sidebar.title("Settings")
    state = {
        "interval_df": None,
        "daily_df": None,
        "merged_df": None,
        "events": [],
        "config": AppConfig(),
    }

    # --- DATA MANAGEMENT ---
    st.sidebar.header("Data")

    usage_files = st.sidebar.file_uploader(
        "Upload PGN CSV(s)",
        type=["csv"],
        accept_multiple_files=True,
        key="pgn_upload",
    )

    temp_files = st.sidebar.file_uploader(
        "Upload Temperature CSV(s)",
        type=["csv"],
        accept_multiple_files=True,
        key="temp_upload",
    )

    # Try loading from data/raw/ if no uploads
    raw_dir = Path("data/raw")
    if not usage_files:
        pgn_files = sorted(raw_dir.glob("pgn_*.csv"))
        if pgn_files:
            usage_files = [str(f) for f in pgn_files]

    if not temp_files:
        temp_csv_files = sorted(raw_dir.glob("POWER_*.csv"))
        if temp_csv_files:
            temp_files = [str(f) for f in temp_csv_files]

    # Load usage data
    if usage_files:
        interval_df = load_multiple_pgn_files(usage_files)
        state["interval_df"] = interval_df
        daily_df = aggregate_to_daily(interval_df)
        state["daily_df"] = daily_df

        date_range = (daily_df["date"].min(), daily_df["date"].max())
        st.sidebar.caption(
            f"Usage: {date_range[0].date()} to {date_range[1].date()} "
            f"({len(daily_df)} days)"
        )

    # Load temperature data
    temp_df = None
    if temp_files:
        temp_df = load_multiple_temperature_files(temp_files)
        st.sidebar.caption(f"Temperature: {len(temp_df)} days loaded")

    # Open-Meteo fetch
    st.sidebar.subheader("Temperature API")
    col1, col2 = st.sidebar.columns(2)
    lat = col1.number_input("Lat", value=45.47, format="%.2f", key="lat")
    lon = col2.number_input("Lon", value=-122.72, format="%.2f", key="lon")
    state["config"].latitude = lat
    state["config"].longitude = lon

    if st.sidebar.button("Fetch from Open-Meteo") and state["daily_df"] is not None:
        daily = state["daily_df"]
        start = daily["date"].min().date()
        end = daily["date"].max().date()
        with st.sidebar.spinner("Fetching temperature data..."):
            try:
                api_temp = fetch_temperature_range(lat, lon, start, end)
                if temp_df is not None:
                    temp_df = pd.concat([temp_df, api_temp], ignore_index=True)
                    temp_df = temp_df.dropna(subset=["temp_c"]).drop_duplicates(
                        subset=["date"], keep="last"
                    )
                    temp_df = temp_df.sort_values("date").reset_index(drop=True)
                else:
                    temp_df = api_temp
                st.sidebar.success(f"Fetched {len(api_temp)} days of temperature data")
            except Exception as e:
                st.sidebar.error(f"API error: {e}")

    # Merge data
    if state["daily_df"] is not None and temp_df is not None:
        merged = merge_usage_temperature(state["daily_df"], temp_df)
        merged = interpolate_temperature_gaps(merged)
        state["merged_df"] = merged

    # --- EQUIPMENT EVENTS ---
    st.sidebar.header("Equipment Events")
    events = load_events()

    # Display existing events
    for i, event in enumerate(events):
        with st.sidebar.expander(f"{event.label} ({event.date})"):
            new_label = st.text_input("Label", value=event.label, key=f"evt_label_{i}")
            new_date = st.date_input("Date", value=event.date, key=f"evt_date_{i}")
            new_cost = st.number_input(
                "Equipment cost ($)",
                value=event.equipment_cost or 0.0,
                min_value=0.0,
                key=f"evt_cost_{i}",
            )
            new_rebates = st.number_input(
                "Rebates ($)",
                value=event.rebates or 0.0,
                min_value=0.0,
                key=f"evt_rebates_{i}",
            )

            if st.button("Update", key=f"evt_update_{i}"):
                events[i] = EquipmentEvent(
                    name=event.name,
                    label=new_label,
                    date=new_date,
                    equipment_cost=new_cost if new_cost > 0 else None,
                    rebates=new_rebates if new_rebates > 0 else None,
                )
                save_events(events)
                st.rerun()

            if st.button("Delete", key=f"evt_delete_{i}"):
                events.pop(i)
                save_events(events)
                st.rerun()

    # Add new event
    with st.sidebar.expander("Add New Event"):
        new_name = st.text_input("Name (slug)", key="new_evt_name")
        new_label = st.text_input("Display Label", key="new_evt_label")
        new_date = st.date_input("Install Date", key="new_evt_date")
        new_cost = st.number_input("Cost ($)", value=0.0, min_value=0.0, key="new_evt_cost")
        new_rebates = st.number_input("Rebates ($)", value=0.0, min_value=0.0, key="new_evt_reb")

        if st.button("Add Event") and new_name and new_label:
            events.append(
                EquipmentEvent(
                    name=new_name,
                    label=new_label,
                    date=new_date,
                    equipment_cost=new_cost if new_cost > 0 else None,
                    rebates=new_rebates if new_rebates > 0 else None,
                )
            )
            save_events(events)
            st.rerun()

    state["events"] = events

    # Validate events
    if state["daily_df"] is not None and events:
        daily = state["daily_df"]
        warnings = validate_events(
            events, daily["date"].min().date(), daily["date"].max().date()
        )
        for w in warnings:
            st.sidebar.warning(w)

    # Add period labels to merged data
    if state["merged_df"] is not None:
        state["merged_df"] = add_event_periods(state["merged_df"], events)
    if state["daily_df"] is not None:
        state["daily_df"] = add_event_periods(state["daily_df"], events)

    # --- COST SETTINGS ---
    st.sidebar.header("Cost Settings")
    auto_rate = None
    if state["interval_df"] is not None:
        auto_rate = compute_electricity_rate(state["interval_df"])

    rate = st.sidebar.number_input(
        "Electricity rate ($/kWh)",
        value=auto_rate or 0.154,
        format="%.4f",
        key="elec_rate",
    )
    state["config"].electricity_rate = rate

    if auto_rate:
        st.sidebar.caption(f"Auto-derived rate: ${auto_rate:.4f}/kWh")

    return state
