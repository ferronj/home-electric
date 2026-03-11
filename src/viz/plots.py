"""Plotly chart builders for the dashboard tab."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.schemas import EquipmentEvent


def plot_daily_usage_timeseries(
    df: pd.DataFrame,
    events: list[EquipmentEvent],
    show_temperature: bool = False,
    show_rolling_avg: bool = True,
    rolling_window: int = 7,
) -> go.Figure:
    """Time series of daily usage with event markers and optional overlays."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["usage_kwh"],
            mode="lines",
            name="Daily Usage (kWh)",
            line=dict(color="steelblue", width=1),
            opacity=0.6,
        ),
        secondary_y=False,
    )

    if show_rolling_avg and len(df) > rolling_window:
        rolling = df["usage_kwh"].rolling(window=rolling_window, center=True).mean()
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=rolling,
                mode="lines",
                name=f"{rolling_window}-day avg",
                line=dict(color="navy", width=2),
            ),
            secondary_y=False,
        )

    if show_temperature and "temp_f" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["temp_f"],
                mode="lines",
                name="Temperature (F)",
                line=dict(color="orangered", width=1, dash="dot"),
                opacity=0.5,
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Temperature (F)", secondary_y=True)

    # Event markers
    colors = ["red", "green", "purple", "orange"]
    for i, event in enumerate(events):
        color = colors[i % len(colors)]
        event_str = str(event.date)
        fig.add_shape(
            type="line",
            x0=event_str, x1=event_str,
            y0=0, y1=1,
            yref="paper",
            line=dict(color=color, dash="dash", width=2),
        )
        fig.add_annotation(
            x=event_str, y=1, yref="paper",
            text=event.label,
            showarrow=False,
            font=dict(color=color, size=11),
            yshift=10,
        )

    fig.update_layout(
        title="Daily Electricity Usage",
        xaxis_title="Date",
        yaxis_title="Usage (kWh)",
        hovermode="x unified",
        height=400,
    )
    return fig


def plot_usage_distribution(
    df: pd.DataFrame,
    period_column: str = "period",
) -> go.Figure:
    """Histogram/KDE per period."""
    fig = go.Figure()
    periods = df[period_column].unique()

    for period in periods:
        subset = df[df[period_column] == period]
        fig.add_trace(
            go.Histogram(
                x=subset["usage_kwh"],
                name=period,
                opacity=0.6,
                nbinsx=30,
            )
        )

    fig.update_layout(
        title="Usage Distribution by Period",
        xaxis_title="Usage (kWh)",
        yaxis_title="Count",
        barmode="overlay",
        height=400,
    )
    return fig


def plot_temperature_vs_usage(
    df: pd.DataFrame,
    period_column: str = "period",
    model_fits: dict | None = None,
    temp_range: np.ndarray | None = None,
) -> go.Figure:
    """Scatter plot with optional model curve overlays."""
    fig = go.Figure()
    periods = df[period_column].unique()

    for period in periods:
        subset = df[df[period_column] == period].dropna(subset=["temp_f"])
        fig.add_trace(
            go.Scatter(
                x=subset["temp_f"],
                y=subset["usage_kwh"],
                mode="markers",
                name=period,
                opacity=0.6,
            )
        )

    # Add model fit curves if provided
    if model_fits and temp_range is not None:
        for name, fit in model_fits.items():
            fig.add_trace(
                go.Scatter(
                    x=temp_range,
                    y=fit["mean"],
                    mode="lines",
                    name=f"{name} fit",
                    line=dict(dash="dash", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([temp_range, temp_range[::-1]]),
                    y=np.concatenate([fit["hdi_high"], fit["hdi_low"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(128,128,128,0.2)",
                    line=dict(color="rgba(128,128,128,0)"),
                    name=f"{name} 95% CI",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title="Temperature vs Usage",
        xaxis_title="Temperature (F)",
        yaxis_title="Usage (kWh)",
        height=400,
    )
    return fig


def plot_monthly_cost_comparison(
    df: pd.DataFrame,
    period_column: str = "period",
) -> go.Figure:
    """Grouped bar chart of monthly costs by period."""
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    monthly = df.groupby(["month", period_column])["daily_cost"].sum().reset_index()

    fig = go.Figure()
    for period in monthly[period_column].unique():
        subset = monthly[monthly[period_column] == period]
        fig.add_trace(
            go.Bar(x=subset["month"], y=subset["daily_cost"], name=period)
        )

    fig.update_layout(
        title="Monthly Electricity Cost",
        xaxis_title="Month",
        yaxis_title="Cost ($)",
        barmode="group",
        height=400,
    )
    return fig


def plot_cumulative_savings(
    savings_df: pd.DataFrame,
    event_label: str,
    equipment_cost: float | None = None,
) -> go.Figure:
    """Line chart of cumulative savings since event."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=savings_df["date"],
            y=savings_df["cumulative_saving"],
            mode="lines",
            name="Cumulative Savings",
            fill="tozeroy",
            line=dict(color="green"),
        )
    )

    if equipment_cost is not None:
        fig.add_hline(
            y=equipment_cost,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Equipment Cost: ${equipment_cost:,.0f}",
        )

    fig.update_layout(
        title=f"Cumulative Savings Since {event_label}",
        xaxis_title="Date",
        yaxis_title="Savings ($)",
        height=400,
    )
    return fig
