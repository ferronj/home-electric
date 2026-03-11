"""Simple before/after Normal model — no temperature dependence."""

import arviz as az
import numpy as np
import pymc as pm

from src.models.base import BayesianModel


class SimpleNormalModel(BayesianModel):
    """
    Estimates mean and sigma of daily usage for each period.
    Handles missing data (NaN) via PyMC imputation.

    From notebook: before mu ~ 34.9 kWh, after mu ~ 28.5 kWh
    """

    def __init__(self):
        super().__init__("simple_normal")
        self.period_names: list[str] = []

    def build(
        self,
        usage_obs: np.ndarray,
        period_names: list[str] | None = None,
    ) -> None:
        """
        Build the model.

        Args:
            usage_obs: array of shape (n_obs, n_periods) with NaN for missing.
            period_names: labels for each period column.
        """
        n_periods = usage_obs.shape[1] if usage_obs.ndim > 1 else 1
        self.period_names = period_names or [f"period_{i}" for i in range(n_periods)]

        coords = {"period": self.period_names}

        with pm.Model(coords=coords) as self.model:
            sigma_usage = pm.Exponential("sigma_usage", lam=1, dims="period")
            mu_usage = pm.Normal("mu_usage", mu=25, sigma=10, dims="period")
            pm.Normal("usage", mu=mu_usage, sigma=sigma_usage, observed=usage_obs)

    def sample(self, draws=1000, tune=1000, chains=4, **kwargs):
        idata = super().sample(draws=draws, tune=tune, chains=chains, **kwargs)
        # Add posterior predictive
        with self.model:
            self.idata.extend(pm.sample_posterior_predictive(trace=self.idata))
        return self.idata

    def get_parameter_estimates(self) -> dict:
        """Return {period: {mu, sigma}} with means and HDI."""
        result = {}
        for i, name in enumerate(self.period_names):
            posterior = self.idata.posterior
            mu_vals = posterior["mu_usage"].sel(period=name)
            sigma_vals = posterior["sigma_usage"].sel(period=name)
            result[name] = {
                "mu_mean": float(mu_vals.mean().values),
                "sigma_mean": float(sigma_vals.mean().values),
            }
        return result
