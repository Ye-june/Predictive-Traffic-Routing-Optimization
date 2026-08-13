# METR-LA dataset notes

This project uses **METR-LA**, the loop-detector traffic speed corpus popularized by Li et al., *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting* (ICLR 2018).

Numbers below come from `scripts/inspect_data.py` and `scripts/prepare_data.py` unless labeled as literature reference.

## Why METR-LA

The dataset is a standard benchmark for spatiotemporal traffic forecasting. It provides:

- repeated observations at a regular interval
- a large set of spatially distributed sensors
- a speed variable that maps directly to travel time
- publicly documented sensor coordinates

## Files downloaded by `scripts/download_data.py`

| File | Role |
| --- | --- |
| `data/raw/metr-la.h5` | Timestamp × sensor speed matrix (mph) |
| `data/raw/graph_sensor_ids.txt` | Ordered sensor identifiers |
| `data/raw/graph_sensor_locations.csv` | Latitude / longitude for each sensor |
| `data/raw/distances_la_2012.csv` | Pairwise distances from the DCRNN release |

Raw files are never overwritten. SHA-256 of `metr-la.h5` is checked against the Hugging Face mirror checksum.

## Inspected properties

| Property | Observed |
| --- | --- |
| Sensors | 207 |
| Timestamps | 34,272 |
| Sampling | 5 minutes, regular grid, no missing timestamps |
| Date range | 2012-03-01 00:00:00 to **2012-06-27 23:55:00** |
| Pandas NA | 0 |
| Sentinel zeros | 575,302 (8.1094%) |
| Speed min / max | 0.0 / 70.0 mph (zeros are missing, not stopped traffic) |
| Median (including zeros) | 62.44 mph |
| Mean of valid speeds | 58.46 mph |
| P01 / P99 valid speeds | 13.13 / 69.75 mph |
| Duplicate timestamps | 0 |
| Negative speeds | 0 |

Literature often describes the window as 1 March–30 June 2012. The downloaded file **ends 27 June 2012 23:55**. Sensor count, timestep count, 5-minute frequency, and ~8.11% missingness match the DCRNN references.

There are **2,148 timestamps** where every sensor is missing. Those full outages are why long-gap imputation is refused.

Sensor IDs, HDF5 columns, and location rows match exactly (207/207).

## Cleaning decisions

Documented in `outputs/metrics/cleaning_summary.json`:

1. Treat values `<= 0` as missing (confirmed: all 575,302 missing cells are zeros, not NaN).
2. Treat speeds `> 90 mph` as missing. No values above 70 mph were present, so this rule did not change the matrix.
3. Interpolate along time within each sensor, limit **12 steps (1 hour)**.
4. Leave longer gaps as NaN and store masks.

| Quantity | Count | Share of cells |
| --- | --- | --- |
| Originally missing | 575,302 | 8.11% |
| Imputed (short gaps) | 200,380 | 2.82% |
| Remaining missing | 374,922 | 5.28% |

Limitations: interpolation cannot recover multi-hour or multi-day outages; it uses only a sensor's own history.

## Graph

`scripts/build_graph.py` keeps DCRNN distance rows whose `from` and `to` IDs are both in the 207-sensor set.

| Property | Observed |
| --- | --- |
| Distance-file IDs | 4,106 (broader PeMS set) |
| Overlap with METR-LA | 207 / 207 |
| Directed METR-LA edges | 11,546 |
| Distance units | meters (`cost`), also stored as miles |
| Distance range (METR-LA pairs) | 33.5 m to 11.90 km (median 7.43 km) |
| Weakly connected | yes (207 nodes) |
| Strongly connected | no (206 + singleton `717804`) |
| Isolated nodes | 0 |

Routing origin-destination pairs should be sampled inside the 206-node strongly connected component. Edges are **sensor relationships / road-network distances between detectors**, not a complete street map.

## Units used in this project

| Quantity | Unit |
| --- | --- |
| Speed | miles per hour (mph) |
| Geographic coordinates | WGS84 degrees |
| Pairwise DCRNN `cost` | meters |
| Routing travel time | minutes |

## Sources

- Traffic HDF5 mirrors (Hugging Face): `jimmygao3218/METRLA`, `MintBruce/SkyTraffic`
- Original DCRNN release: https://github.com/liyaguang/DCRNN
- Paper: https://arxiv.org/abs/1707.01926
