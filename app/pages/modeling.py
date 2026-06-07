"""Modeling tab: Bayesian model configuration, execution, and results."""

import numpy as np
import pandas as pd
import streamlit as st

from src.models.simple_normal import SimpleNormalModel
from src.models.piecewise_linear import PiecewiseLinearModel
from src.models.full_model import FullTemperatureModel
from src.models.diagnostics import (
    check_ess,
    check_rhat,
    count_divergences,
    generate_diagnostics_report,
    model_comparison,
)
from src.viz.model_plots import plot_trace, plot_posterior, plot_piecewise_fit


def render_modeling(app_state: dict):
    """Render the Bayesian modeling tab."""
    merged_df = app_state.get("merged_df")
    events = app_state.get("events", [])
    config = app_state.get("config")

    if merged_df is None:
        st.info("Load both usage and temperature data to use Bayesian modeling.")
        return

    if "temp_f" not in merged_df.columns:
        st.warning("Temperature data required for modeling. Upload or fetch temperature data.")
        return

    st.subheader("Model Configuration")

    # Model selection
    model_type = st.selectbox(
        "Model type",
        [
            "Simple Normal (before/after comparison)",
            "Piecewise Linear (single period)",
            "Full Temperature Model (before/after)",
        ],
    )

    col1, col2, col3 = st.columns(3)
    setpoint = col1.slider("Setpoint temperature (F)", 50, 75, 60)
    draws = col2.number_input("MCMC draws", value=1000, min_value=100, step=100)
    chains = col3.number_input("Chains", value=4, min_value=1, max_value=8)

    # Prepare data
    clean_df = merged_df.dropna(subset=["temp_f"]).copy()

    if clean_df.empty:
        st.error("No data with valid temperatures available.")
        return

    # Run model
    if st.button("Run Model", type="primary"):
        with st.spinner("Running MCMC sampling... This may take a minute."):
            try:
                if model_type == "Simple Normal (before/after comparison)":
                    result = _run_simple_normal(clean_df, draws, chains)
                elif model_type == "Piecewise Linear (single period)":
                    result = _run_piecewise(clean_df, setpoint, draws, chains)
                else:
                    result = _run_full_model(clean_df, setpoint, draws, chains)

                st.session_state["model_result"] = result
                runs = st.session_state.setdefault("model_runs", {})
                runs[result["type"]] = result
                st.success("Model sampling complete!")
            except Exception as e:
                st.error(f"Model error: {e}")
                return

    # Display results
    if "model_result" in st.session_state:
        result = st.session_state["model_result"]
        _display_results(result, merged_df, config)

    # Side-by-side LOO comparison once 2+ different models have been run
    runs = st.session_state.get("model_runs", {})
    if len(runs) >= 2:
        _render_model_comparison(runs)


def _run_simple_normal(df, draws, chains):
    """Run simple normal model."""
    # Pivot to before/after columns
    model_data = df.groupby(["temp_f", "period"])["usage_kwh"].mean().reset_index()
    periods = sorted(model_data["period"].unique())

    if len(periods) < 2:
        raise ValueError("Need at least 2 periods (before/after an event) for this model.")

    pivoted = model_data.pivot(index="temp_f", columns="period", values="usage_kwh")
    pivoted = pivoted[periods]
    usage_obs = pivoted.to_numpy()

    model = SimpleNormalModel()
    model.build(usage_obs=usage_obs, period_names=periods)
    model.sample(draws=draws, tune=draws, chains=chains)

    return {
        "type": "simple_normal",
        "model": model,
        "var_names": ["mu_usage", "sigma_usage"],
        "params": model.get_parameter_estimates(),
    }


def _run_piecewise(df, setpoint, draws, chains):
    """Run piecewise linear model on baseline period only."""
    baseline = df[df["period"] == df["period"].unique()[0]]
    if baseline.empty:
        baseline = df

    # Average usage per temperature
    model_data = baseline.groupby("temp_f")["usage_kwh"].mean().reset_index()
    temp_obs = model_data["temp_f"].to_numpy()
    usage_obs = model_data["usage_kwh"].to_numpy()

    model = PiecewiseLinearModel()
    model.build(temp_obs=temp_obs, usage_obs=usage_obs, setpoint=setpoint)
    model.sample(draws=draws, tune=draws, chains=chains)

    return {
        "type": "piecewise_linear",
        "model": model,
        "var_names": ["k_heat", "k_cool", "usage_min"],
        "params": model.get_parameter_estimates(),
        "temp_obs": temp_obs,
    }


def _run_full_model(df, setpoint, draws, chains):
    """Run full temperature model with separate params per period."""
    model_data = df.groupby(["temp_f", "period"])["usage_kwh"].mean().reset_index()
    periods = sorted(model_data["period"].unique())

    if len(periods) < 2:
        raise ValueError("Need at least 2 periods for the full model.")

    pivoted = model_data.pivot(index="temp_f", columns="period", values="usage_kwh")
    pivoted = pivoted[periods]

    temp_obs = pivoted.index.to_numpy()
    usage_obs = pivoted.to_numpy().T  # shape: (n_periods, n_temps)

    model = FullTemperatureModel()
    model.build(
        temp_obs=temp_obs,
        usage_obs=usage_obs,
        setpoint=setpoint,
        period_names=periods,
    )
    model.sample(draws=draws, tune=draws, chains=chains)

    return {
        "type": "full_temperature",
        "model": model,
        "var_names": ["k_heat", "k_cool", "usage_min"],
        "params": model.get_parameter_estimates(),
        "temp_obs": temp_obs,
        "periods": periods,
    }


def _display_results(result, merged_df, config):
    """Display model results."""
    model = result["model"]
    var_names = result["var_names"]

    # Diagnostics
    report = generate_diagnostics_report(model.idata, model.name)
    if report["convergence_ok"]:
        st.success("Convergence: All diagnostics passed")
    else:
        st.warning("Convergence issues detected")

    rhat_results = check_rhat(model.idata)
    ess_results = check_ess(model.idata)
    n_div = count_divergences(model.idata)

    diag_rows = []
    for param in rhat_results:
        rhat = rhat_results[param]["rhat"]
        ess = ess_results.get(param, {}).get("ess_bulk", float("nan"))
        ok = rhat_results[param]["ok"] and ess_results.get(param, {}).get("ok", False)
        diag_rows.append({
            "param": param,
            "rhat": round(rhat, 3),
            "ess_bulk": round(ess, 0),
            "ok": "✓" if ok else "✗",
        })
    diag_df = pd.DataFrame(diag_rows).set_index("param")
    st.markdown(f"**Diagnostics** _(divergences: {n_div})_")
    st.dataframe(diag_df, use_container_width=True)
    if report["recommendations"]:
        for r in report["recommendations"]:
            st.info(r)

    # Parameter summary
    st.subheader("Parameter Summary")
    summary_df = model.summary(var_names=var_names)
    st.dataframe(summary_df, use_container_width=True)

    # Trace plots
    st.subheader("Trace Plots")
    fig_trace = plot_trace(model.idata, var_names)
    st.plotly_chart(fig_trace, use_container_width=True)

    # Posterior distributions
    st.subheader("Posterior Distributions")
    # Only plot scalar parameters for posterior
    scalar_vars = [v for v in var_names if model.idata.posterior[v].ndim <= 2]
    if scalar_vars:
        fig_post = plot_posterior(model.idata, scalar_vars)
        st.plotly_chart(fig_post, use_container_width=True)

    # Model-specific visualizations
    if result["type"] == "piecewise_linear":
        st.subheader("Model Fit")
        temp_range = np.linspace(10, 100, 100)
        preds = model.predict(temp_range)
        fig_fit = plot_piecewise_fit(temp_range, preds, merged_df)
        st.plotly_chart(fig_fit, use_container_width=True)

        # Interpretation
        params = result["params"]
        st.subheader("Interpretation")
        st.markdown(
            f"- **Heating rate (k_heat)**: {params['k_heat']['mean']:.2f} kWh/F "
            f"(95% HDI: {params['k_heat']['hdi_low']:.2f} - {params['k_heat']['hdi_high']:.2f})"
        )
        st.markdown(
            f"- **Cooling rate (k_cool)**: {params['k_cool']['mean']:.2f} kWh/F "
            f"(95% HDI: {params['k_cool']['hdi_low']:.2f} - {params['k_cool']['hdi_high']:.2f})"
        )
        st.markdown(
            f"- **Baseload**: {params['baseload']['mean']:.1f} kWh/day "
            f"(95% HDI: {params['baseload']['hdi_low']:.1f} - {params['baseload']['hdi_high']:.1f})"
        )

    elif result["type"] == "full_temperature":
        st.subheader("Model Fit")
        temp_range = np.linspace(10, 100, 100)
        per_period_preds = model.predict_per_period(temp_range)

        from src.viz.plots import plot_temperature_vs_usage
        fig = plot_temperature_vs_usage(
            merged_df, model_fits=per_period_preds, temp_range=temp_range
        )
        st.plotly_chart(fig, use_container_width=True)

        # Interpretation
        params = result["params"]
        st.subheader("Interpretation")
        for period_name, pvals in params.get("periods", {}).items():
            st.markdown(
                f"**{period_name}**: k_heat = {pvals['k_heat_mean']:.2f} kWh/F, "
                f"k_cool = {pvals['k_cool_mean']:.2f} kWh/F"
            )

        if len(params.get("periods", {})) >= 2:
            period_keys = list(params["periods"].keys())
            before_kh = params["periods"][period_keys[0]]["k_heat_mean"]
            after_kh = params["periods"][period_keys[-1]]["k_heat_mean"]
            pct_change = ((after_kh - before_kh) / before_kh) * 100
            st.markdown(
                f"Heating efficiency changed by **{pct_change:.0f}%** "
                f"({before_kh:.2f} -> {after_kh:.2f} kWh/F)"
            )

    elif result["type"] == "simple_normal":
        params = result["params"]
        st.subheader("Interpretation")
        for name, p in params.items():
            st.markdown(f"**{name}**: mean usage = {p['mu_mean']:.1f} kWh/day, "
                        f"std = {p['sigma_mean']:.1f} kWh")


def _render_model_comparison(runs: dict):
    """Render LOO model comparison once 2+ different models have run."""
    st.divider()
    st.subheader("Model Comparison (LOO)")
    idatas = {name: run["model"].idata for name, run in runs.items()}
    table = model_comparison(idatas)
    if table is None:
        st.info("LOO comparison unavailable for the current set of models.")
        return
    st.dataframe(table, use_container_width=True)
    st.caption(
        "Higher elpd_loo is better. `weight` is the LOO model-stacking weight; "
        "`p_loo` is the effective number of parameters."
    )
