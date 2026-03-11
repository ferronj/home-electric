"""Model diagnostics: convergence checks and model comparison."""

import arviz as az
import numpy as np
import pandas as pd


def check_rhat(idata: az.InferenceData, threshold: float = 1.01) -> dict:
    """Check rhat for all parameters. Returns {param: {rhat, ok}}."""
    summary = az.summary(idata)
    result = {}
    for param in summary.index:
        rhat = summary.loc[param, "r_hat"]
        result[param] = {"rhat": float(rhat), "ok": rhat <= threshold}
    return result


def check_ess(
    idata: az.InferenceData, min_ess_per_chain: int = 100
) -> dict:
    """Check effective sample size. Returns {param: {ess_bulk, ok}}."""
    summary = az.summary(idata)
    result = {}
    for param in summary.index:
        ess = summary.loc[param, "ess_bulk"]
        result[param] = {"ess_bulk": float(ess), "ok": ess >= min_ess_per_chain}
    return result


def count_divergences(idata: az.InferenceData) -> int:
    """Count divergent transitions across all chains."""
    if "sample_stats" not in idata.groups():
        return 0
    diverging = idata.sample_stats.get("diverging")
    if diverging is None:
        return 0
    return int(diverging.sum().values)


def generate_diagnostics_report(idata: az.InferenceData, model_name: str) -> dict:
    """
    Generate a structured diagnostics report.

    Returns:
        convergence_ok: bool
        warnings: list of warning strings
        recommendations: list of recommendation strings
    """
    rhat_results = check_rhat(idata)
    ess_results = check_ess(idata)
    n_divergences = count_divergences(idata)

    warnings = []
    recommendations = []

    # Check rhat
    bad_rhat = [k for k, v in rhat_results.items() if not v["ok"]]
    if bad_rhat:
        warnings.append(f"r_hat > 1.01 for: {', '.join(bad_rhat)}")
        recommendations.append("Consider increasing tune/draws or reparameterizing")

    # Check ESS
    bad_ess = [k for k, v in ess_results.items() if not v["ok"]]
    if bad_ess:
        warnings.append(f"Low ESS for: {', '.join(bad_ess)}")
        recommendations.append("Consider increasing draws or adding more data")

    # Check divergences
    if n_divergences > 0:
        warnings.append(f"{n_divergences} divergent transitions detected")
        recommendations.append("Consider increasing target_accept or reparameterizing")

    return {
        "model_name": model_name,
        "convergence_ok": len(warnings) == 0,
        "n_divergences": n_divergences,
        "warnings": warnings,
        "recommendations": recommendations,
    }


def model_comparison(
    idatas: dict[str, az.InferenceData],
) -> pd.DataFrame | None:
    """Compare models using LOO. Returns comparison table or None if unavailable."""
    try:
        compare_dict = {name: idata for name, idata in idatas.items()}
        return az.compare(compare_dict)
    except Exception:
        return None
