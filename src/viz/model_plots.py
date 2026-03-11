"""Plotly visualization for Bayesian model results."""

import arviz as az
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_trace(idata: az.InferenceData, var_names: list[str]) -> go.Figure:
    """Plotly trace plots (posterior density + trace) for given variables."""
    n_vars = len(var_names)
    fig = make_subplots(
        rows=n_vars, cols=2,
        subplot_titles=[f"{v} posterior" for v in var_names]
        + [f"{v} trace" for v in var_names],
    )

    colors = ["steelblue", "coral", "seagreen", "mediumpurple"]

    for i, var in enumerate(var_names):
        row = i + 1
        vals = idata.posterior[var]

        # Handle multi-dimensional variables
        if vals.ndim > 2:
            # Flatten to just chain x draw
            flat_vals = vals.values.reshape(vals.shape[0], vals.shape[1], -1)
            for dim_idx in range(flat_vals.shape[2]):
                for chain in range(flat_vals.shape[0]):
                    chain_data = flat_vals[chain, :, dim_idx]
                    color = colors[chain % len(colors)]

                    # KDE-like histogram for posterior
                    fig.add_trace(
                        go.Histogram(
                            x=chain_data, opacity=0.5,
                            marker_color=color, showlegend=False,
                            nbinsx=30,
                        ),
                        row=row, col=1,
                    )
                    # Trace plot
                    fig.add_trace(
                        go.Scatter(
                            y=chain_data, mode="lines", opacity=0.5,
                            line=dict(color=color, width=0.5),
                            showlegend=False,
                        ),
                        row=row, col=2,
                    )
        else:
            for chain in range(vals.shape[0]):
                chain_data = vals[chain].values
                color = colors[chain % len(colors)]

                fig.add_trace(
                    go.Histogram(
                        x=chain_data, opacity=0.5,
                        marker_color=color, showlegend=False,
                        nbinsx=30,
                    ),
                    row=row, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        y=chain_data, mode="lines", opacity=0.5,
                        line=dict(color=color, width=0.5),
                        showlegend=False,
                    ),
                    row=row, col=2,
                )

    fig.update_layout(
        height=250 * n_vars,
        title="Trace Plots",
        showlegend=False,
    )
    return fig


def plot_posterior(idata: az.InferenceData, var_names: list[str]) -> go.Figure:
    """Posterior distributions with HDI markers."""
    fig = make_subplots(rows=1, cols=len(var_names), subplot_titles=var_names)

    for i, var in enumerate(var_names):
        vals = idata.posterior[var].values.flatten()
        hdi = az.hdi(idata, var_names=[var], hdi_prob=0.94)[var].values
        mean_val = float(np.mean(vals))

        fig.add_trace(
            go.Histogram(
                x=vals, nbinsx=50, opacity=0.7,
                marker_color="steelblue", showlegend=False,
            ),
            row=1, col=i + 1,
        )

        # HDI lines
        if hdi.ndim == 1:
            fig.add_vline(x=float(hdi[0]), line_dash="dash", line_color="red",
                          row=1, col=i + 1)
            fig.add_vline(x=float(hdi[1]), line_dash="dash", line_color="red",
                          row=1, col=i + 1)
        fig.add_vline(x=mean_val, line_color="black", row=1, col=i + 1)

    fig.update_layout(height=300, title="Posterior Distributions (94% HDI)")
    return fig


def plot_piecewise_fit(
    temp_range: np.ndarray,
    predictions: dict,
    observed_df=None,
    period_column: str = "period",
) -> go.Figure:
    """Scatter data + piecewise linear fit + credible bands."""
    fig = go.Figure()

    # Model fit
    fig.add_trace(
        go.Scatter(
            x=temp_range, y=predictions["mean"],
            mode="lines", name="Model Fit",
            line=dict(color="navy", dash="dash", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([temp_range, temp_range[::-1]]),
            y=np.concatenate([predictions["hdi_high"], predictions["hdi_low"][::-1]]),
            fill="toself",
            fillcolor="rgba(100,100,200,0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI",
        )
    )

    # Observed data
    if observed_df is not None and "temp_f" in observed_df.columns:
        clean = observed_df.dropna(subset=["temp_f"])
        if period_column in clean.columns:
            for period in clean[period_column].unique():
                subset = clean[clean[period_column] == period]
                fig.add_trace(
                    go.Scatter(
                        x=subset["temp_f"], y=subset["usage_kwh"],
                        mode="markers", name=period, opacity=0.5,
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=clean["temp_f"], y=clean["usage_kwh"],
                    mode="markers", name="Observed", opacity=0.5,
                )
            )

    fig.update_layout(
        title="Piecewise Linear Fit",
        xaxis_title="Temperature (F)",
        yaxis_title="Usage (kWh)",
        height=450,
    )
    return fig
