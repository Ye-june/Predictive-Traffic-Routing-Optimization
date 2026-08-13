# TrafficFlow

[![Python 3.10](https://img.shields.io/badge/python-3.10-3776AB.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-forecasting-2e8b57.svg)](https://xgboost.readthedocs.io)
[![NetworkX](https://img.shields.io/badge/NetworkX-routing-1f4e79.svg)](https://networkx.org)

TrafficFlow is an end-to-end spatiotemporal machine learning and network optimization system that forecasts future traffic conditions and uses those predictions to make routing decisions. The project compares traditional routing methods with prediction-aware routing to test whether better traffic forecasts produce measurable travel-time savings.

Try it here: https://predictive-traffic-routing-optimization.streamlit.app/

```text
Traffic Data → Forecasting → Predicted Edge Costs → Routing → Realized Travel Time
```

## Live demo

**Historical Replay Mode** (not live navigation).

Run locally:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
streamlit run app/streamlit_app.py
```

Deploy from GitHub with [Streamlit Community Cloud](https://share.streamlit.io): select `app/streamlit_app.py` on branch `main`. Step-by-step notes are in [docs/deployment.md](docs/deployment.md). After the first Cloud deploy, this section should be updated with the public URL.

The hosted app loads compact artifacts under `artifacts/`. It does **not** retrain models.

## Research question

Can spatial relationships between roadway sensors improve future traffic predictions, and can those predictions improve routing decisions compared with static or non-predictive routing?

Routes are scored on **realized future traffic**, not on the travel time a model predicted for itself.

## Dataset

METR-LA (Li et al., [DCRNN, ICLR 2018](https://arxiv.org/abs/1707.01926)), inspected from the downloaded files:

| Property | Observed |
| --- | --- |
| Sensors | 207 |
| Timestamps | 34,272 |
| Sampling | 5 minutes |
| Period | 2012-03-01 00:00 → 2012-06-27 23:55 |
| Target | Speed (mph) |
| Missing | 8.1094%, stored as zeros |

Details: [docs/dataset.md](docs/dataset.md).

## Forecasting results (chronological test subsample)

Train 2012-03-01 → 2012-05-23; test 2012-06-10 → 2012-06-27. Evaluation uses every 6th test timestamp. Units: mph.

| Model | 15 min MAE | 30 min MAE | 60 min MAE |
| --- | ---: | ---: | ---: |
| Persistence | 3.349 | 4.050 | 5.227 |
| Historical weekday-hour mean | 4.339 | 4.339 | **4.339** |
| Temporal XGBoost | 3.254 | 3.894 | 4.920 |
| Spatiotemporal XGBoost | **3.216** | **3.845** | 4.848 |

Neighbor speeds improve MAE by about **1.2% at 15 min**, **1.2% at 30 min**, and **1.5% at 60 min** versus the same XGBoost without spatial features. The gain is real but modest. Persistence remains a strong short-horizon baseline; the seasonal mean still wins at 60 minutes among the non-boosted models.

Source: `artifacts/demo/model_metrics.json`.

## Routing results (historical replay)

72 origin–destination trips on the demo week (2012-06-21 → 2012-06-27), scored with actual future speeds:

| Comparison | Mean savings | Median | Improved | Worsened |
| --- | ---: | ---: | ---: | ---: |
| Predictive vs static (free-flow) | **+1.53 min** | +0.18 min | 58.3% | 11.1% |
| Predictive vs current-state | −0.05 min | 0.00 min | 26.4% | 12.5% |

Mean regret versus a hindsight oracle: **0.23 min**. Predictive routing beats a static free-flow plan more often than not. It does **not** reliably beat a current-traffic shortest path on this sensor graph — that is a result, not a claim to hide.

Source: `artifacts/demo/routing_summary.json`.

## Methods

1. Persistence and train-only historical-mean baselines
2. Compact temporal XGBoost (lags, rolling means, calendar, horizon)
3. Same model plus k-nearest neighbor speeds
4. Predicted mph → minutes with `distance_miles / max(speed, 5 mph) × 60`
5. Dijkstra on a 207-node k-NN sensor graph (2,838 edges)
6. Static / current-state / predictive / oracle routes, replayed on realized traffic

The dense DCRNN distance table (11,546 edges) is too shortcut-heavy for routing. The app uses 8-nearest neighbors within 8 km, then adds reverse edges so the graph is strongly connected. Paths are **sensor relationships, not turn-by-turn roads**.

## Installation

Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,app]"
```

Training the local HDF5 pipeline also needs `requirements-dev.txt` (PyTables/h5py). The Streamlit Cloud install file is `requirements.txt`.

## Usage

```bash
python scripts/download_data.py
python scripts/inspect_data.py
python scripts/prepare_data.py
python scripts/build_graph.py
python scripts/generate_eda_figures.py
python scripts/train_baseline.py
python scripts/build_deployment_assets.py
pytest
streamlit run app/streamlit_app.py
```

## Project structure

```text
app/                     Streamlit product (home + pages)
artifacts/               Compact models, demo week, metrics (committed)
configs/                 YAML for data, models, routing
docs/                    Dataset and deployment notes
scripts/                 Download, clean, train, export
src/trafficflow/         Installable package
tests/
```

## Limitations

- Not a navigation product and not live traffic.
- Sensor graph ≠ complete street map.
- METR-LA is freeway loop detectors only.
- 5.28% of cells remain missing after short-gap interpolation.
- Routing does not model congestion feedback from the routed vehicles.
- Spatial MAE gains are small; current-state routing is already competitive.

## License

MIT. Cite Li et al. (ICLR 2018) if you use METR-LA.
