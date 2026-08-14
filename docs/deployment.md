# Deploy TrafficFlow on Streamlit Community Cloud

The app is inference-only. It loads compact artifacts from `artifacts/` and does not retrain models.

## Local

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
streamlit run app/streamlit_app.py
```

If artifacts are missing:

```bash
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/build_deployment_assets.py
```

No secrets or API keys are required.

## Streamlit Community Cloud

1. Push this repository to GitHub (already the source of truth).
2. Sign in at [share.streamlit.io](https://share.streamlit.io).
3. Create an app.
4. Select the `Ye-june/Predictive-Traffic-Routing-Optimization` repository.
5. Branch: `main`.
6. Main file: `app/streamlit_app.py`.
7. Python version: 3.10 (see `runtime.txt`).
8. Deploy.

`requirements.txt` is the Cloud install file. `packages.txt` installs `libgomp1` for XGBoost on Linux.

After the first successful deploy, paste the public URL into the README if it is not already there.

## What is hosted

| Item | Notes |
| --- | --- |
| Demo traffic | 7 days, 2012-06-21 to 2012-06-27 |
| Models | Compact temporal + spatiotemporal XGBoost (~0.3 MB each) |
| Graph | 207 nodes, 2,838 routing edges (k-NN, not a street map) |
| Mode | Historical replay only — not live navigation |

Raw METR-LA HDF5 files are not deployed.
