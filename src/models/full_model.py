"""Temperature-dependent piecewise linear model with separate params per period."""

import arviz as az
import numpy as np
import pymc as pm

from src.models.base import BayesianModel


class FullTemperatureModel(BayesianModel):
    """
    Piecewise linear model with separate k_heat, k_cool per period.
    Shared baseload and setpoint.

    From notebook: k_heat dropped 2.23 -> 0.71 after heat pump install.
    """

    def __init__(self):
        super().__init__("full_temperature")
        self.setpoint: float = 60.0
        self.n_periods: int = 2
        self.period_names: list[str] = []
        self.temp_obs: np.ndarray | None = None

    def build(
        self,
        temp_obs: np.ndarray,
        usage_obs: np.ndarray,
        setpoint: float = 60.0,
        period_names: list[str] | None = None,
    ) -> None:
        """
        Build multi-period model.

        Args:
            temp_obs: 1D array of temperatures (F), length N.
            usage_obs: 2D array (n_periods, N) of usage. NaN for missing.
            setpoint: temperature setpoint (F).
            period_names: labels for each period.
        """
        self.setpoint = setpoint
        self.temp_obs = temp_obs
        self.n_periods = usage_obs.shape[0]
        self.period_names = period_names or [f"period_{i}" for i in range(self.n_periods)]

        coords = {"period": self.period_names}

        with pm.Model(coords=coords) as self.model:
            temp_set = pm.Data("temp_set", setpoint)
            sigma_usage = pm.Exponential("sigma_usage", lam=1)
            usage_min = pm.Uniform("usage_min", lower=0, upper=30)
            k_heat = pm.Uniform("k_heat", lower=0, upper=5, shape=(self.n_periods, 1))
            k_cool = pm.Uniform("k_cool", lower=0, upper=5, shape=(self.n_periods, 1))

            mu_usage = pm.math.switch(
                temp_obs < temp_set,
                k_heat * (temp_set - temp_obs) + usage_min,
                k_cool * (temp_obs - temp_set) + usage_min,
            )

            pm.Normal("usage", mu=mu_usage, sigma=sigma_usage, observed=usage_obs)

    def get_parameter_estimates(self) -> dict:
        """Return per-period k_heat, k_cool, plus shared baseload."""
        posterior = self.idata.posterior
        result = {"setpoint": self.setpoint, "periods": {}}

        # Shared baseload
        result["baseload"] = self._extract_param("usage_min")

        for i, name in enumerate(self.period_names):
            k_heat_vals = posterior["k_heat"].sel({"k_heat_dim_0": i, "k_heat_dim_1": 0})
            k_cool_vals = posterior["k_cool"].sel({"k_cool_dim_0": i, "k_cool_dim_1": 0})

            result["periods"][name] = {
                "k_heat_mean": float(k_heat_vals.mean().values),
                "k_cool_mean": float(k_cool_vals.mean().values),
            }

        return result

    def predict_per_period(self, temp_range: np.ndarray) -> dict[str, dict]:
        """
        Compute posterior predictive for each period.

        Returns dict keyed by period name, each with 'mean', 'hdi_low', 'hdi_high'.
        """
        posterior = self.idata.posterior
        usage_min = posterior["usage_min"].values.flatten()
        results = {}

        for i, name in enumerate(self.period_names):
            k_heat = posterior["k_heat"].sel(
                {"k_heat_dim_0": i, "k_heat_dim_1": 0}
            ).values.flatten()
            k_cool = posterior["k_cool"].sel(
                {"k_cool_dim_0": i, "k_cool_dim_1": 0}
            ).values.flatten()

            predictions = np.zeros((len(k_heat), len(temp_range)))
            for s in range(len(k_heat)):
                for j, temp in enumerate(temp_range):
                    if temp < self.setpoint:
                        predictions[s, j] = (
                            k_heat[s] * (self.setpoint - temp) + usage_min[s]
                        )
                    else:
                        predictions[s, j] = (
                            k_cool[s] * (temp - self.setpoint) + usage_min[s]
                        )

            results[name] = {
                "mean": predictions.mean(axis=0),
                "hdi_low": np.percentile(predictions, 2.5, axis=0),
                "hdi_high": np.percentile(predictions, 97.5, axis=0),
            }

        return results
