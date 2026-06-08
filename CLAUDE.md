# Electricity Usage Analysis

## Project Overview
Streamlit app analyzing home electricity usage before/after heat pump and heat pump water heater installations. Uses Bayesian modeling (PyMC) to quantify efficiency gains as a function of temperature.

## Tech Stack
- Python 3.12, managed with `uv`
- Streamlit for the web app
- Plotly for interactive charts (not matplotlib in the app)
- PyMC 5.x / ArviZ <1.0.0 for Bayesian modeling (ArviZ 1.0 breaks PyMC compatibility)
- Pandas for data manipulation
- Open-Meteo API for temperature data (free, no key)

## Project Structure
- `src/` — pure logic, no Streamlit dependency (testable independently)
  - `src/data/` — loader.py, cleaning.py, temperature_api.py, schemas.py
  - `src/analysis/` — daily_usage.py, cost.py, events.py
  - `src/models/` — base.py, simple_normal.py, piecewise_linear.py, full_model.py, diagnostics.py
  - `src/viz/` — plots.py (dashboard), model_plots.py (Bayesian results)
- `app/` — Streamlit UI code only
  - `app/main.py` — entry point with tabs (Dashboard, Bayesian Modeling)
  - `app/pages/` — dashboard.py, modeling.py (tab content, NOT Streamlit multipage)
  - `app/components/` — sidebar.py, kpi_cards.py
- `data/raw/` — user-uploaded CSVs (gitignored)
- `data/processed/` — cached parquet files (gitignored)
- `data/events.json` — equipment install events (tracked in git)
- `tests/` — pytest tests (22 fast + 7 model smoke tests)
- `notebooks/` — original Jupyter notebook (reference only)
- `.streamlit/config.toml` — Streamlit config (headless, no sidebar nav)

## Data Formats
- **PGN CSV**: columns TYPE, DATE, START TIME, END TIME, USAGE (kWh), COST ($-prefixed string), NOTES. 15-min intervals.
- **NASA POWER CSV**: columns YEAR, MO, DY, T2M (Celsius). Sentinel -999.0 = missing.
- **Open-Meteo API**: returns daily temperature_2m_mean in Celsius as JSON.

## Key Constants
- Location: lat=45.47, lon=-122.72 (Portland, OR area)
- Default setpoint: 60F
- Electricity rate: ~$0.157/kWh (derived from data, user-overridable)

## Key Model Results (from notebook, verified in app)
- Baseline daily usage: ~34.5 kWh/day (notebook: 34.9 before temp averaging)
- After Heat Pump: ~28.7 kWh/day (notebook: 28.5)
- Piecewise linear (before): k_heat=2.25, k_cool=0.61, baseload=14.3 kWh
- Full model: k_heat dropped 2.23 -> 0.71 after install

## Conventions
- Use `uv run` to execute commands
- Venv at `.venv/` (shared with worktrees)
- Tests: `uv run pytest` (fast) or `uv run pytest tests/test_models.py` (slow, ~70s)
- All viz in app uses Plotly, not matplotlib
- Period ordering: "Baseline" always first, then "After {event}" in date order
- Plotly vlines on subplots: use add_shape + add_annotation, NOT add_vline (broken with make_subplots)

## Gotchas
- ArviZ 1.0.0 removed InferenceData from top-level namespace — pin `arviz<1.0.0`
- Plotly `add_vline` with `annotation_text` on `make_subplots` causes TypeError — use shapes instead
- Streamlit auto-detects `app/pages/` as multipage — disable with `showSidebarNavigation = false`
- numpy bool (np.True_) is not Python bool — use `==` not `is` in assertions
- The simple before/after savings comparison is misleading with limited post-install data during winter — the dashboard now also surfaces a temperature-normalized number from the Full Temperature model when one has been run on the Bayesian Modeling tab

## Running
```bash
uv run streamlit run app/main.py
uv run pytest
uv run pytest tests/test_models.py  # slow: ~70s
```

## Deployment (Streamlit Community Cloud)
- Entry point: `app/main.py`
- Python version: `.python-version` (3.12)
- Dependencies: `requirements.txt` (mirrors `pyproject.toml` `[project.dependencies]` — keep in sync)
- PyMC builds on Cloud free tier but MCMC sampling can OOM (~1GB limit). The dashboard, cost analysis, and degree-day blocks work fully without sampling; modeling tab degrades gracefully if PyMC is missing or sampling fails.
- No persistent storage on Cloud: `data/raw/` is empty in the deployed app, so visitors must upload their own PGN CSVs via the sidebar widget (or we commit demo data under `data/demo/` and have the sidebar fall back to it).
- `STREAMLIT_DISABLE_MODELING=1` env var forces the modeling tab into "disabled on Cloud" mode without needing PyMC to fail; set this in the Cloud app's secrets/env if you want to skip the sampling path entirely.
