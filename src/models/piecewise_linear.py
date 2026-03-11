"""Temperature-dependent piecewise linear model for a single period."""

import arviz as az
import numpy as np
import pymc as pm

from src.models.base import BayesianModel


class PiecewiseLinearModel(BayesianModel):
    """
    Piecewise linear model: usage as a function of temperature.

    usage = k_heat * (setpoint - temp) + baseload   when temp < setpoint
    usage = k_cool * (temp - setpoint) + baseload   when temp >= setpoint

    From notebook (before period): k_heat~2.25, k_cool~0.61, baseload~14.3
    """

    def __init__(self):
        super().__init__("piecewise_linear")
        self.setpoint: float = 60.0
        self.temp_obs: np.ndarray | None = None

    def build(
        self,
        temp_obs: np.ndarray,
        usage_obs: np.ndarray,
        setpoint: float = 60.0,
    ) -> None:
        """
        Build the model for a single period.

        Args:
            temp_obs: 1D array of daily temperatures (F).
            usage_obs: 1D array of daily usage (kWh). NaN for missing.
            setpoint: temperature setpoint (F).
        """
        self.setpoint = setpoint
        self.temp_obs = temp_obs

        with pm.Model() as self.model:
            temp_set = pm.Data("temp_set", setpoint)
            sigma_usage = pm.Exponential("sigma_usage", lam=1)
            usage_min = pm.Uniform("usage_min", lower=0, upper=30)
            k_heat = pm.Uniform("k_heat", lower=0, upper=5)
            k_cool = pm.Uniform("k_cool", lower=0, upper=5)

            mu_usage = pm.math.switch(
                temp_obs < temp_set,
                k_heat * (temp_set - temp_obs) + usage_min,
                k_cool * (temp_obs - temp_set) + usage_min,
            )

            pm.Normal("usage", mu=mu_usage, sigma=sigma_usage, observed=usage_obs)

    def get_parameter_estimates(self) -> dict:
        """Return k_heat, k_cool, baseload with means and HDI."""
        return {
            "k_heat": self._extract_param("k_heat"),
            "k_cool": self._extract_param("k_cool"),
            "baseload": self._extract_param("usage_min"),
            "setpoint": self.setpoint,
        }

    def predict(self, temp_range: np.ndarray) -> dict:
        """
        Compute posterior predictive for a range of temperatures.

        Returns dict with 'mean', 'hdi_low', 'hdi_high' arrays.
        """
        posterior = self.idata.posterior
        k_heat = posterior["k_heat"].values.flatten()
        k_cool = posterior["k_cool"].values.flatten()
        usage_min = posterior["usage_min"].values.flatten()

        predictions = np.zeros((len(k_heat), len(temp_range)))
        for i in range(len(k_heat)):
            for j, temp in enumerate(temp_range):
                if temp < self.setpoint:
                    predictions[i, j] = k_heat[i] * (self.setpoint - temp) + usage_min[i]
                else:
                    predictions[i, j] = k_cool[i] * (temp - self.setpoint) + usage_min[i]

        return {
            "mean": predictions.mean(axis=0),
            "hdi_low": np.percentile(predictions, 2.5, axis=0),
            "hdi_high": np.percentile(predictions, 97.5, axis=0),
        }
