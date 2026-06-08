"""Cost analysis, ROI, and payback calculations."""

import numpy as np
import pandas as pd


def compute_electricity_rate(interval_df: pd.DataFrame) -> float:
    """
    Derive blended $/kWh rate from PGN interval data.
    Returns weighted average: sum(cost) / sum(usage_kwh).
    """
    total_cost = interval_df["cost"].sum()
    total_kwh = interval_df["usage_kwh"].sum()
    if total_kwh == 0:
        return 0.0
    return total_cost / total_kwh


def estimate_annual_savings(
    before_stats: dict,
    after_stats: dict,
    rate_per_kwh: float,
) -> dict:
    """
    Project annual savings from simple mean comparison.

    Returns annual_kwh_saved, annual_cost_saved, monthly_cost_saved.
    """
    daily_kwh_saved = before_stats["mean_daily_kwh"] - after_stats["mean_daily_kwh"]
    annual_kwh = daily_kwh_saved * 365
    annual_cost = annual_kwh * rate_per_kwh
    return {
        "daily_kwh_saved": daily_kwh_saved,
        "annual_kwh_saved": annual_kwh,
        "annual_cost_saved": annual_cost,
        "monthly_cost_saved": annual_cost / 12,
    }


def compute_roi_payback(
    equipment_cost: float,
    annual_savings: float,
) -> dict:
    """
    Simple payback period and cumulative savings projections.
    """
    if annual_savings <= 0:
        return {
            "simple_payback_years": float("inf"),
            "monthly_savings": 0.0,
            "total_savings_5yr": 0.0,
            "total_savings_10yr": 0.0,
        }

    return {
        "simple_payback_years": equipment_cost / annual_savings,
        "monthly_savings": annual_savings / 12,
        "total_savings_5yr": annual_savings * 5 - equipment_cost,
        "total_savings_10yr": annual_savings * 10 - equipment_cost,
    }


def compute_temperature_normalized_savings(
    model_before_params: dict,
    model_after_params: dict,
    typical_year_temperatures: pd.Series,
    setpoint: float = 60.0,
    rate_per_kwh: float = 0.154,
) -> dict:
    """
    Using piecewise linear model parameters, compute what usage WOULD have been
    before vs after for a typical temperature year.

    Each params dict should have: k_heat, k_cool, baseload

    Returns annual_kwh_saved and annual_cost_saved.
    """

    def _piecewise_usage(temp_f: float, params: dict) -> float:
        if temp_f < setpoint:
            return params["k_heat"] * (setpoint - temp_f) + params["baseload"]
        else:
            return params["k_cool"] * (temp_f - setpoint) + params["baseload"]

    before_usage = typical_year_temperatures.apply(
        lambda t: _piecewise_usage(t, model_before_params)
    )
    after_usage = typical_year_temperatures.apply(
        lambda t: _piecewise_usage(t, model_after_params)
    )

    annual_kwh_saved = (before_usage - after_usage).sum()
    return {
        "annual_kwh_saved": annual_kwh_saved,
        "annual_cost_saved": annual_kwh_saved * rate_per_kwh,
    }


def compute_savings_from_full_model_result(
    model_result: dict,
    daily_df: pd.DataFrame,
    rate_per_kwh: float,
    setpoint: float = 60.0,
) -> dict | None:
    """
    Translate a Full Temperature model result into temperature-normalized savings.

    Returns None when the result isn't a full_temperature run, has fewer than two
    periods, or the dataframe lacks usable temp_f data. First period is treated as
    "before", last as "after".
    """
    if not model_result or model_result.get("type") != "full_temperature":
        return None

    params = model_result.get("params") or {}
    periods = params.get("periods") or {}
    if len(periods) < 2:
        return None

    if "temp_f" not in daily_df.columns:
        return None
    temps = daily_df["temp_f"].dropna()
    if temps.empty:
        return None

    period_names = list(periods.keys())
    before_name, after_name = period_names[0], period_names[-1]
    baseload_mean = float(params.get("baseload", {}).get("mean", 0.0))

    before_params = {
        "k_heat": float(periods[before_name]["k_heat_mean"]),
        "k_cool": float(periods[before_name]["k_cool_mean"]),
        "baseload": baseload_mean,
    }
    after_params = {
        "k_heat": float(periods[after_name]["k_heat_mean"]),
        "k_cool": float(periods[after_name]["k_cool_mean"]),
        "baseload": baseload_mean,
    }

    typical_year_days = 365
    if len(temps) >= typical_year_days:
        typical_temps = temps.iloc[-typical_year_days:].reset_index(drop=True)
    else:
        scale = typical_year_days / len(temps)
        typical_temps = temps.reset_index(drop=True)
        result = compute_temperature_normalized_savings(
            before_params,
            after_params,
            typical_temps,
            setpoint=setpoint,
            rate_per_kwh=rate_per_kwh,
        )
        return {
            "annual_kwh_saved": result["annual_kwh_saved"] * scale,
            "annual_cost_saved": result["annual_cost_saved"] * scale,
            "before_period": before_name,
            "after_period": after_name,
            "n_temp_days_used": len(temps),
        }

    result = compute_temperature_normalized_savings(
        before_params,
        after_params,
        typical_temps,
        setpoint=setpoint,
        rate_per_kwh=rate_per_kwh,
    )
    return {
        "annual_kwh_saved": result["annual_kwh_saved"],
        "annual_cost_saved": result["annual_cost_saved"],
        "before_period": before_name,
        "after_period": after_name,
        "n_temp_days_used": len(temps),
    }


def compute_savings_credible_interval(
    model_result: dict,
    daily_df: pd.DataFrame,
    rate_per_kwh: float,
    setpoint: float = 60.0,
    n_samples: int = 1000,
    hdi_prob: float = 0.94,
    equipment_cost: float | None = None,
    seed: int | None = 0,
) -> dict | None:
    """
    Propagate posterior uncertainty from a Full Temperature model fit into the
    annual savings (and payback) distribution.

    Returns a dict with mean / hdi_low / hdi_high for annual_kwh_saved,
    annual_cost_saved, and (if equipment_cost given) simple_payback_years.
    Returns None when the result isn't a usable full_temperature run or when
    the underlying idata isn't available.
    """
    if not model_result or model_result.get("type") != "full_temperature":
        return None

    model = model_result.get("model")
    idata = getattr(model, "idata", None) if model is not None else None
    if idata is None or not hasattr(idata, "posterior"):
        return None

    params = model_result.get("params") or {}
    periods = list((params.get("periods") or {}).keys())
    if len(periods) < 2:
        return None

    if "temp_f" not in daily_df.columns:
        return None
    temps = daily_df["temp_f"].dropna().to_numpy(dtype=float)
    if temps.size == 0:
        return None

    typical_year_days = 365
    if temps.size >= typical_year_days:
        typical_temps = temps[-typical_year_days:]
        scale = 1.0
    else:
        typical_temps = temps
        scale = typical_year_days / temps.size

    try:
        import arviz as az  # local import: cost.py is imported on PyMC-less deploys
    except ImportError:
        return None

    posterior = idata.posterior
    stacked = posterior.stack(sample=("chain", "draw"))
    total_samples = stacked.sizes["sample"]
    if total_samples == 0:
        return None

    rng = np.random.default_rng(seed)
    n_draws = min(int(n_samples), total_samples)
    idx = rng.choice(total_samples, size=n_draws, replace=False)

    before_i = 0
    after_i = len(periods) - 1

    usage_min = stacked["usage_min"].values[idx]
    k_heat = stacked["k_heat"].values  # shape: (n_periods, 1, total_samples)
    k_cool = stacked["k_cool"].values
    k_heat_before = k_heat[before_i, 0, idx]
    k_heat_after = k_heat[after_i, 0, idx]
    k_cool_before = k_cool[before_i, 0, idx]
    k_cool_after = k_cool[after_i, 0, idx]

    # Vectorized piecewise usage over (n_draws, n_days).
    temps_row = typical_temps[None, :]
    heating_mask = temps_row < setpoint
    heat_delta = np.where(heating_mask, setpoint - temps_row, 0.0)
    cool_delta = np.where(heating_mask, 0.0, temps_row - setpoint)
    baseload = usage_min[:, None]

    before_usage = (
        k_heat_before[:, None] * heat_delta
        + k_cool_before[:, None] * cool_delta
        + baseload
    )
    after_usage = (
        k_heat_after[:, None] * heat_delta
        + k_cool_after[:, None] * cool_delta
        + baseload
    )

    annual_kwh_draws = (before_usage - after_usage).sum(axis=1) * scale
    annual_cost_draws = annual_kwh_draws * rate_per_kwh

    out = {
        "annual_kwh_saved": _summarize(annual_kwh_draws, hdi_prob, az),
        "annual_cost_saved": _summarize(annual_cost_draws, hdi_prob, az),
        "before_period": periods[before_i],
        "after_period": periods[after_i],
        "n_temp_days_used": int(temps.size),
        "n_samples_used": int(n_draws),
        "hdi_prob": float(hdi_prob),
    }

    if equipment_cost is not None and equipment_cost > 0:
        positive = annual_cost_draws > 0
        if positive.any():
            payback_draws = np.where(
                positive, equipment_cost / np.where(positive, annual_cost_draws, 1.0),
                float("inf"),
            )
            finite = payback_draws[np.isfinite(payback_draws)]
            if finite.size:
                out["simple_payback_years"] = _summarize(finite, hdi_prob, az)
        out["prob_savings_positive"] = float(positive.mean())

    return out


def _summarize(draws: np.ndarray, hdi_prob: float, az) -> dict:
    """Mean + HDI summary of a 1-D draws array."""
    mean = float(np.mean(draws))
    hdi = az.hdi(np.asarray(draws), hdi_prob=hdi_prob)
    low, high = float(np.min(hdi)), float(np.max(hdi))
    return {"mean": mean, "hdi_low": low, "hdi_high": high}


def compute_cumulative_savings(
    df: pd.DataFrame,
    event_date: pd.Timestamp,
    baseline_daily_cost: float,
) -> pd.DataFrame:
    """
    Compute cumulative dollar savings since an event date.

    Returns DataFrame with: date, daily_saving, cumulative_saving
    """
    after = df[df["date"] >= event_date].copy()
    after["daily_saving"] = baseline_daily_cost - after["daily_cost"]
    after["cumulative_saving"] = after["daily_saving"].cumsum()
    return after[["date", "daily_saving", "cumulative_saving"]].reset_index(drop=True)
