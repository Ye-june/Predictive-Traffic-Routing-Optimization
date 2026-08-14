# TrafficFlow

[![Python 3.10](https://img.shields.io/badge/python-3.10-3776AB.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B.svg)](https://streamlit.io)

TrafficFlow predicts traffic speeds on a freeway sensor network, then uses those forecasts to pick routes. The main question: does knowing what traffic will look like in 15–60 minutes actually save travel time?

**Try the app:** https://predictive-traffic-routing-optimization.streamlit.app/

This is a historical replay demo — not live navigation. Pick a past departure time, compare routes, and see how they would have performed on traffic that really happened.

## What it does

1. Forecast speed at each sensor (XGBoost, with and without neighbor sensors)
2. Turn those speeds into travel times on a sensor graph
3. Run Dijkstra to find static, current-traffic, and forecast-based routes
4. Score each route on the traffic that actually occurred

Routes are judged on real future speeds, not on what the model guessed its own trip time would be.

## Data

[METR-LA](https://arxiv.org/abs/1707.01926) — 207 loop detectors in Los Angeles, 5-minute speed readings from March–June 2012. About 8% of values are missing (stored as zeros).

More detail in [docs/dataset.md](docs/dataset.md).

## Results

### Forecasting (test set, MAE in mph)

Models trained on data through May 23; tested on June 10–27. Every model is scored on the same forecast origin → target pairs.

| Model | 15 min | 30 min | 60 min |
| --- | ---: | ---: | ---: |
| Persistence (last value) | 3.40 | 4.10 | 5.29 |
| Historical weekday average | 4.28 | 4.33 | **4.33** |
| XGBoost (temporal) | 3.25 | 3.88 | 4.96 |
| XGBoost (+ neighbors) | **3.24** | **3.87** | 4.89 |

Adding neighbor speeds helps a little (about **0.4–1.3%** better MAE). Persistence is still hard to beat at short horizons. The weekday average does best at 60 minutes.

### Routing (72 trips, demo week)

| vs. | Mean difference | Trips improved |
| --- | ---: | ---: |
| Static free-flow route | **+1.4 min saved** | 61% |
| Current-traffic route | −0.08 min | 25% |

Forecast-based routing usually beats a naive static plan. It does not clearly beat routing on current traffic — which is worth knowing.

## How routing works

The app builds a graph from the 207 sensors (8 nearest neighbors within 8 km, with reverse edges added). Edge cost is `miles / max(speed, 5 mph) × 60`. Paths follow sensor links, not real street geometry.

The full DCRNN distance table has too many shortcuts for routing, so a sparser k-NN graph is used instead.

## Run it locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,app]"

python scripts/download_data.py
python scripts/prepare_data.py
python scripts/build_deployment_assets.py
streamlit run app/streamlit_app.py
```

The hosted app loads pre-built files from `artifacts/` and does not retrain.

To deploy on [Streamlit Community Cloud](https://share.streamlit.io), point it at `app/streamlit_app.py` on `main`. See [docs/deployment.md](docs/deployment.md).

## Repo layout

```text
app/           Streamlit UI
artifacts/     Models, demo data, metrics
scripts/       Download, train, export
src/trafficflow/   Core library
tests/
configs/       YAML settings
docs/
```

## Caveats

- Freeway sensors only — not a city street map
- Historical data, not live traffic
- Small forecasting gains from spatial features
- Predictive routing uses one forecast snapshot for all edges (realized scoring still advances time along the path)
- No model of how routed cars would affect congestion
- `configs/model.yaml` hyperparameters are not what the compact Cloud models use; training settings live in `scripts/build_deployment_assets.py`
