# Market Intelligence Terminal

A personal, Bloomberg-style global market-intelligence project. Today it does
**leakage-free LSTM stock-price forecasting**; the roadmap grows it into a
continuously-ingesting dashboard for news, economics, and markets worldwide.

> **Full vision & phased plan:** [docs/VISION_AND_ROADMAP.md](docs/VISION_AND_ROADMAP.md)

## Current features (Phase 0)

- **Format-agnostic loader** for single- and multi-ticker yfinance CSVs.
- **Leakage-free LSTM**: scaler fit on the train slice only, windows never cross
  split boundaries, plus walk-forward (expanding-window) cross-validation.
- Typed config (`pydantic-settings`), tests (`pytest`), linting (`ruff`/`black`).
- Self-hosted data layer via `docker-compose` (Postgres + pgvector + Redis).

## Project layout

```
src/market_intel/
  config.py             # typed settings (.env)
  data/loaders.py       # format-agnostic price CSV loader
  data/fetch.py         # yfinance downloader
  models/windowing.py   # leakage-free split + sequence construction
  models/lstm.py        # model, train/evaluate, walk-forward CV
src/multi_feature_model.py  # thin training CLI
src/preprocess.py           # thin fetch CLI
tests/                      # incl. the leakage regression test
docker-compose.yml          # Postgres+pgvector + Redis
docs/VISION_AND_ROADMAP.md  # the plan
```

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # add -r requirements-dev.txt for tooling
cp .env.example .env                      # then edit
```

## Usage

```bash
# Fetch price CSVs into data/
python src/preprocess.py

# Train a leakage-free LSTM and print honest RMSE/MAE
python src/multi_feature_model.py AAPL

# A ticker out of a multi-ticker file, with a saved chart
python src/multi_feature_model.py MSFT --dataset tech --plot --plot-path out.png

# Walk-forward cross-validation (the realistic score)
python src/multi_feature_model.py AAPL --walk-forward 5
```

### Data layer (optional in Phase 0, required from Phase 1)

```bash
docker compose up -d        # Postgres+pgvector on 127.0.0.1:5432, Redis on 6379
```

Keep the DB bound to localhost / your Tailscale interface — never expose port 5432.

## Development

```bash
pytest                      # run tests (incl. leakage regression)
ruff check src tests
black src tests
```

## A note on correctness

The original model fit its scaler on the **entire** dataset and built sliding
windows **before** splitting — both leak future information and make reported
accuracy untrustworthy. This is fixed in `models/windowing.py` and locked in by
`tests/test_windowing.py`. See the roadmap for why this was the highest-priority
fix.
