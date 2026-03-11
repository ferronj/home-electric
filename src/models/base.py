"""Abstract base class for Bayesian models."""

from abc import ABC, abstractmethod

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


class BayesianModel(ABC):
    """Abstract base for all PyMC models in this project."""

    def __init__(self, name: str):
        self.name = name
        self.model: pm.Model | None = None
        self.idata: az.InferenceData | None = None

    @abstractmethod
    def build(self, **kwargs) -> None:
        """Build the PyMC model. Sets self.model."""

    def sample(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        **kwargs,
    ) -> az.InferenceData:
        """Run MCMC sampling. Sets and returns self.idata."""
        if self.model is None:
            raise RuntimeError("Model not built. Call build() first.")

        with self.model:
            self.idata = pm.sample(
                draws=draws, tune=tune, chains=chains, **kwargs
            )
        return self.idata

    def summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """Return ArviZ summary for key parameters."""
        if self.idata is None:
            raise RuntimeError("No trace. Call sample() first.")
        return az.summary(self.idata, var_names=var_names)

    @abstractmethod
    def get_parameter_estimates(self) -> dict:
        """Return posterior means and 95% HDI for key parameters."""

    def _extract_param(self, var_name: str) -> dict:
        """Helper: extract mean and HDI for a variable from posterior."""
        posterior = self.idata.posterior[var_name]
        mean = float(posterior.mean(dim=["chain", "draw"]).values)
        hdi = az.hdi(self.idata, var_names=[var_name], hdi_prob=0.95)
        hdi_vals = hdi[var_name].values
        return {
            "mean": mean,
            "hdi_low": float(hdi_vals[0]) if hdi_vals.ndim == 1 else float(hdi_vals),
            "hdi_high": float(hdi_vals[1]) if hdi_vals.ndim == 1 else float(hdi_vals),
        }
