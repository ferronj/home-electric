# Hand-off: Electricity Dashboard Improvements

This document describes the remaining tracks. Track A — surfacing already-built code to the UI — was completed in the commit that added this file. A new session can pick up from here.

**Recommended order: D → B → C**
- **Track D (Deploy)** — get the app live on Streamlit Community Cloud. Do this first — it's quick (mostly config files) and lets you share the dashboard immediately.
- **Track B (Bayesian uncertainty)** — propagate posterior HDI into the cost story. High analytical value.
- **Track C (Modeling polish)** — PPC plots, idata persistence, named-run registry. Quality-of-life for repeated modeling sessions.

## What Just Shipped (Track A)

- **`src/analysis/cost.py`**: New `compute_savings_from_full_model_result(model_result, daily_df, rate_per_kwh, setpoint=60.0)` adapter — translates a `FullTemperatureModel` result dict into temperature-normalized annual savings using `compute_temperature_normalized_savings()` under the hood. Treats first period as "before", last as "after". Scales when fewer than 365 days of temperature data exist.
- **`app/pages/dashboard.py`**: Cost Analysis section now also shows the Bayesian temperature-normalized savings whenever a Full Temperature model has been run in the session. ROI/payback prefers that number when available. New "Degree-Day Efficiency" table (kWh/HDD, kWh/CDD) per period, base 65°F.
- **`app/pages/modeling.py`**: Diagnostics now render as a dataframe (param, rhat, ess_bulk, ok) rather than toast warnings. A new "Model Comparison (LOO)" section appears once 2+ different model types have been run in the session. Each run is also stored in `st.session_state["model_runs"][model_type]`.
- **`pyproject.toml`**: Migrated `[tool.uv] dev-dependencies` → `[dependency-groups] dev`.
- **Tests**: 5 new unit tests in `tests/test_cost.py` covering the adapter (basic, wrong type, single period, short series, missing temp column). All 27 fast tests green.

## Track D — Deploy to Streamlit Community Cloud (do this first)

**Why**: The app only runs locally right now. The hook-f1 project (at `C:\Users\jferr\hook-f1`) is already deployed to Streamlit Community Cloud via GitHub — replicate that pattern here.

**Reference deployment (hook-f1):**
- Repo: `https://github.com/ferronj/hook-f1.git`, branch `main`
- Entry point: `dashboard.py` at repo root
- Deps: `pyproject.toml` + a fallback `requirements.txt`
- `.python-version`: `3.12`
- No `.streamlit/` dir in the deployed repo (uses Cloud defaults)
- Auto-deploys on `git push origin main`

### D1. Create GitHub repo
Create a new GitHub repo (e.g. `ferronj/electricity-use` or whatever you prefer). Set the remote:
```bash
git remote add origin https://github.com/ferronj/electricity-use.git
```

### D2. Add `requirements.txt` for Streamlit Cloud compatibility
Streamlit Cloud uses `requirements.txt` (or `pyproject.toml` — but `requirements.txt` is more reliable for the Cloud builder). Create `requirements.txt` at repo root:
```
streamlit>=1.40.0
pandas>=2.1.0
numpy>=1.26.0
plotly>=5.18.0
pymc>=5.26.0
arviz>=0.18.0,<1.0.0
scipy>=1.12.0
xarray>=2024.1.0
requests>=2.31.0
pydantic>=2.5.0
```

**PyMC on Cloud caveat**: PyMC + its C/Fortran compilation dependencies (pytensor, libgfortran) are heavy. Streamlit Community Cloud gives ~1GB RAM on the free tier. PyMC models typically need more than that for MCMC. Options:
- **Option A (recommended)**: Deploy dashboard-only — the dashboard works fully without running MCMC live. Models are run locally, and if Track C2 (idata persistence) ships first, pre-computed results can be committed to the repo as `.nc` files for the Cloud app to load.
- **Option B**: Deploy everything, accept that MCMC may OOM on Cloud. Users would see the dashboard + cost analysis but the "Run Model" button might fail. The app would still be useful for the data exploration and cost analysis features.
- **Option C**: Use `packages.txt` to install system deps (libgfortran, etc.) and increase Cloud resources (requires paid tier).

**Recommendation**: Go with Option A for now — it covers 80% of the value (the whole dashboard, cost analysis, degree-day efficiency). Add a note on the modeling tab when deployed: "Bayesian modeling requires running the app locally."

### D3. Add `.python-version` file
```
3.12
```
Streamlit Cloud reads this to pick the Python version.

### D4. Update `.gitignore` for deploy
Add these lines to `.gitignore`:
```
# Claude Code session artifacts
.claude/worktrees/
.claude/scheduled_tasks.lock

# Sampling artifacts (keep .nc in data/processed/ for pre-computed results)
*.pkl
```
Note: `data/raw/` and `data/processed/` are already gitignored. For deploy, we'll need **sample data** committed — see D5.

### D5. Commit sample data for the deployed app
The app currently requires the user to upload CSVs or have files in `data/raw/`. For the Cloud deploy, either:
- **Add a demo dataset**: commit a small representative CSV (e.g. a month of data) to `data/demo/` and update `app/components/sidebar.py` to auto-load it when `data/raw/` is empty. This way the Cloud app shows a working dashboard out of the box.
- **Or**: accept that Cloud visitors must upload their own PGN CSV. The upload widget already works.

Recommended: add demo data so the deployed app isn't empty on first visit.

### D6. Push and connect to Streamlit Cloud
```bash
git push -u origin main
```
Then:
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select repo `ferronj/electricity-use`, branch `main`, main file `app/main.py`
4. Deploy

### D7. Verify deployed app
- Confirm the dashboard loads with demo data (or upload prompt)
- Confirm temperature API fetch works from Cloud
- If deployed with PyMC: test "Run Model" button (expect possible OOM on free tier)
- If deployed without PyMC: confirm graceful messaging on the modeling tab

### Deploy checklist summary
| Step | What | Files touched |
|------|------|---------------|
| D1 | Create GitHub repo, add remote | git config |
| D2 | Add `requirements.txt` | `requirements.txt` (new) |
| D3 | Add `.python-version` | `.python-version` (new) |
| D4 | Update `.gitignore` | `.gitignore` |
| D5 | Add demo dataset (optional but recommended) | `data/demo/`, `app/components/sidebar.py` |
| D6 | Push to GitHub, connect Streamlit Cloud | — |
| D7 | Verify deployed app | — |

---

## Track B — Propagate Bayesian Uncertainty Into the Cost Story

**Why**: Track A surfaces the *correct* annual savings number, but it's still a point estimate. Payback and 5/10yr ROI on the dashboard show no uncertainty, even though the Bayesian model fits perfectly capture it. With ~23 days of post-install data, the after-period k_heat HDI is wide; users deserve to see that.

### B1. New function: `compute_savings_credible_interval`
**File**: `src/analysis/cost.py` (append after the existing adapter)

Signature:
```python
def compute_savings_credible_interval(
    model_result: dict,           # full_temperature result with .model.idata
    daily_df: pd.DataFrame,
    rate_per_kwh: float,
    setpoint: float = 60.0,
    n_samples: int = 1000,
    hdi_prob: float = 0.94,
) -> dict | None:
    # Returns:
    # {
    #   "annual_kwh_saved": {"mean": ..., "hdi_low": ..., "hdi_high": ...},
    #   "annual_cost_saved": {"mean": ..., "hdi_low": ..., "hdi_high": ...},
    #   ...
    # }
```

Implementation sketch:
1. Pull `idata = model_result["model"].idata` (note: this is the full PyMC `InferenceData`, not just point estimates — need to update the adapter or have callers pass it explicitly).
2. Stack chains: `posterior = idata.posterior.stack(sample=("chain", "draw"))`.
3. Draw `n_samples` indices from the stacked dimension.
4. For each sample, extract `(usage_min, k_heat[before], k_cool[before], k_heat[after], k_cool[after])` and run `compute_temperature_normalized_savings` on the typical-year temp series → one annual_kwh_saved value.
5. Compute mean and `arviz.hdi` over the resulting array.
6. Same for payback when equipment cost is provided.

**Tests** to add in `tests/test_cost.py`:
- A synthetic `idata` (use `arviz.from_dict` with deterministic posterior arrays) — verify that when `k_heat[after]` posterior has spread, the savings HDI is non-degenerate.
- Verify HDI low ≤ mean ≤ HDI high.
- Verify mean ≈ point estimate from `compute_savings_from_full_model_result` to within ~5%.

### B2. Dashboard wiring for credible intervals
**File**: `app/pages/dashboard.py`

In the Cost Analysis "Temperature-Normalized" block (added in Track A around line ~115):
- When the model has an `idata`, also call `compute_savings_credible_interval()`.
- Render as `mean (HDI low – HDI high)` instead of just `mean`.
- Same for the ROI block — show payback range, not just point.

### B3. 12-month forecast plot
**File**: `src/viz/plots.py` (new function)

```python
def plot_savings_forecast(
    daily_df: pd.DataFrame,             # historical merged_df
    model_result: dict,                  # full_temperature
    rate_per_kwh: float,
    months_ahead: int = 12,
) -> go.Figure:
    # Use the typical-year temp series as the "next year" predictor.
    # Draw posterior samples → per-day usage trajectories under "after" params.
    # Aggregate to monthly cumulative savings vs the "before" trajectory.
    # Plot mean line + 80% / 95% credible bands.
```

Show this on the dashboard below the existing cumulative-savings chart, only when a Full Temperature model has run.

## Track C — Modeling Tab Polish

**Why**: Once a user runs a 1-5 minute MCMC, hitting refresh today wipes everything. The modeling tab has no way to compare runs side-by-side beyond LOO. PPC plots — table stakes for Bayesian workflows — are missing.

### C1. Posterior predictive check
**File**: `src/viz/model_plots.py` (new), `app/pages/modeling.py` (call site)

```python
def plot_ppc(idata, observed_var: str = "usage") -> go.Figure:
    # Overlay 50 posterior predictive draws as faint lines + observed as markers.
```

Add to `_display_results` in `app/pages/modeling.py` for the piecewise and full-temperature models. Need to call `pm.sample_posterior_predictive` if `posterior_predictive` group not yet in idata; the simple_normal model already does this in `sample()`.

### C2. Persist idata to disk
**File**: `src/models/base.py` (add `save`/`load` methods)

```python
def save(self, path: Path) -> None:
    self.idata.to_netcdf(str(path))

@classmethod
def load(cls, path: Path) -> "BayesianModel":
    instance = cls()
    instance.idata = az.from_netcdf(str(path))
    return instance
```

Wire into `app/pages/modeling.py`:
- After successful sampling: write to `data/processed/model_results/<model_type>__<timestamp>.nc`.
- On page load: if `data/processed/model_results/` has files, hydrate `st.session_state["model_runs"]` from them.
- Add a "Clear runs" button.

Update `.gitignore` to ensure `data/processed/` stays gitignored (already is).

### C3. Named-run registry
**File**: `app/pages/modeling.py`

Replace `model_runs: dict[str, dict]` (keyed by model type) with a named registry:
```python
runs: dict[str, dict]  # keyed by user-supplied run name, e.g. "full_v1", "full_higher_setpoint"
```

UI: text input for run name before "Run Model" button (default `<model_type>_<n>`). Multi-select to choose which runs to include in LOO comparison.

This is the natural extension of Track A's `model_runs` dict — when this lands, the LOO block in `_render_model_comparison` should accept a multi-select rather than auto-including everything.

## Open Items / Known Limitations of Track A

- The temperature-normalized savings adapter uses the historical temp series as the "typical year." If `daily_df["temp_f"]` covers less than a year, the result is scaled linearly — fine for ~322 days of Portland data but worth flagging if the dataset shrinks. A more rigorous version (Track B candidate) would build a typical-year temp series by taking median temp per day-of-year across multiple years of data.
- The HDD/CDD efficiency table uses base 65°F (industry standard) but the model setpoint is configurable down to 50°F. For very different setpoints, `kWh / HDD` interpretation diverges from heating-system-output efficiency — it's still useful as a relative period-over-period comparison.
- LOO comparison can fail (`model_comparison` returns `None`) when posterior log-likelihood isn't computed. The simple_normal and full_temperature models use `pm.Normal` observed nodes which produce log-likelihood automatically; piecewise_linear should as well. If a user sees "LOO comparison unavailable," it likely means a model class needs an explicit `idata_kwargs={"log_likelihood": True}` on `pm.sample`.

## Verification That Track A Works End-to-End

```bash
uv sync                       # confirms the dependency-groups migration
uv run pytest                 # 27 fast tests, all green
uv run streamlit run app/main.py
```

Then in the browser:
1. Sidebar: load CSV, fetch temperatures.
2. Dashboard: confirm naive savings + degree-day table render.
3. Bayesian Modeling: run Simple Normal → run Full Temperature.
4. Back to Dashboard: confirm "Temperature-Normalized Annual Savings (Bayesian)" block now appears with positive numbers; ROI block now reads "_(Bayesian savings)_".
5. Back to Bayesian Modeling: confirm "Model Comparison (LOO)" section appears.
