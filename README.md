# Gold Sentinel

Institutional-grade, screenshot-first market intelligence for **XAUUSD (Gold vs USD)**.

## What this repo delivers (current MVP)

- **Web UI**: Landing page + Application screen (upload chart screenshot → structured output).
- **API**: `POST /api/analyze` accepts a single screenshot and returns:
  - directional bias (bullish / bearish / neutral)
  - confidence score (0–100)
  - deterministic safety-gated status (`TRADE` / `NO_TRADE` / `INSUFFICIENT_DATA`)
  - auditable reasoning chain
- **No price feed dependency**: analysis is computed **purely from pixels**.

## Architecture (replaceable & testable)

### Vision layer (image → features)

- `app/vision/plot_region.py`: isolates the chart pane (best-effort).
- `app/vision/structure.py`: extracts conservative structure proxies:
  - trend proxy (edge-trace slope + fit quality)
  - phase (expansion / retracement / consolidation)
  - key horizontal levels (Hough line clusters)
  - liquidity clusters (repeated y-level clustering, conservative)

### Decision layer (features → decision)

- `app/decision/engine.py`: deterministic rules + confidence scoring + **NO TRADE gates**.

### Web/API layer

- `app/main.py`: FastAPI app serving UI + JSON API.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open:
- UI: `http://localhost:8000/`
- App: `http://localhost:8000/app`
- API docs: `http://localhost:8000/docs`

## Safety posture

Gold Sentinel is a decision-support system. If the screenshot does not provide sufficient evidence, the system returns
**INSUFFICIENT DATA** or **NO TRADE** by design.
