"""Smoke tests for Bayesian models with synthetic data."""

import numpy as np
import pytest

from src.models.simple_normal import SimpleNormalModel
from src.models.piecewise_linear import PiecewiseLinearModel
from src.models.full_model import FullTemperatureModel
from src.models.diagnostics import generate_diagnostics_report


@pytest.fixture
def synthetic_usage():
    """Generate synthetic usage data for two periods."""
    rng = np.random.default_rng(42)
    before = rng.normal(35, 20, size=50)
    after = rng.normal(28, 8, size=20)
    # Pad after with NaN to match length
    after_padded = np.full(50, np.nan)
    after_padded[:20] = after
    return np.column_stack([before, after_padded])


@pytest.fixture
def synthetic_temp_usage():
    """Generate synthetic temperature-dependent usage."""
    rng = np.random.default_rng(42)
    temps = np.linspace(20, 90, 50)
    setpoint = 60.0
    k_heat, k_cool, baseload = 2.2, 0.6, 14.0
    usage = np.array([
        k_heat * (setpoint - t) + baseload if t < setpoint
        else k_cool * (t - setpoint) + baseload
        for t in temps
    ]) + rng.normal(0, 5, size=50)
    return temps, usage


class TestSimpleNormalModel:
    def test_build(self, synthetic_usage):
        model = SimpleNormalModel()
        model.build(usage_obs=synthetic_usage, period_names=["before", "after"])
        assert model.model is not None

    def test_sample_and_params(self, synthetic_usage):
        model = SimpleNormalModel()
        model.build(usage_obs=synthetic_usage, period_names=["before", "after"])
        model.sample(draws=100, tune=100, chains=1)
        params = model.get_parameter_estimates()
        assert "before" in params
        assert "after" in params
        assert "mu_mean" in params["before"]


class TestPiecewiseLinearModel:
    def test_build(self, synthetic_temp_usage):
        temps, usage = synthetic_temp_usage
        model = PiecewiseLinearModel()
        model.build(temp_obs=temps, usage_obs=usage)
        assert model.model is not None

    def test_sample_and_params(self, synthetic_temp_usage):
        temps, usage = synthetic_temp_usage
        model = PiecewiseLinearModel()
        model.build(temp_obs=temps, usage_obs=usage)
        model.sample(draws=100, tune=100, chains=1)
        params = model.get_parameter_estimates()
        assert "k_heat" in params
        assert "k_cool" in params
        assert "baseload" in params
        # k_heat should be roughly around 2.2
        assert 0.5 < params["k_heat"]["mean"] < 5.0

    def test_predict(self, synthetic_temp_usage):
        temps, usage = synthetic_temp_usage
        model = PiecewiseLinearModel()
        model.build(temp_obs=temps, usage_obs=usage)
        model.sample(draws=100, tune=100, chains=1)
        preds = model.predict(np.linspace(20, 90, 10))
        assert "mean" in preds
        assert len(preds["mean"]) == 10


class TestFullTemperatureModel:
    def test_build(self, synthetic_temp_usage):
        temps, usage = synthetic_temp_usage
        # Create 2-period data
        usage_2d = np.stack([usage, np.full_like(usage, np.nan)])
        usage_2d[1, :10] = usage[:10] * 0.5  # reduced usage for after

        model = FullTemperatureModel()
        model.build(
            temp_obs=temps, usage_obs=usage_2d,
            period_names=["before", "after"],
        )
        assert model.model is not None


class TestDiagnostics:
    def test_generate_report(self, synthetic_temp_usage):
        temps, usage = synthetic_temp_usage
        model = PiecewiseLinearModel()
        model.build(temp_obs=temps, usage_obs=usage)
        model.sample(draws=100, tune=100, chains=1)
        report = generate_diagnostics_report(model.idata, "test_model")
        assert "model_name" in report
        assert "convergence_ok" in report
        assert isinstance(report["warnings"], list)
