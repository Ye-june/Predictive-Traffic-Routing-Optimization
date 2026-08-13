# METR-LA Data Quality Report

Generated from the downloaded files. Figures are computed, not assumed.

## Summary

- Observations: **7094304**
- Timestamps: **34272**
- Sensors: **207**
- Date range: **2012-03-01 00:00:00** to **2012-06-27 23:55:00**
- Sampling interval: **5.0** minutes

## Missingness

- Null values: 0 (0.0%)
- Sentinel values (0.0): 575302 (8.1094%)
- Combined missing: 575302 (8.1094%)

## Value range (including sentinels unless noted)

- Min: 0.0
- Max: 70.0
- Median: 62.44444444444444
- Mean of valid (non-missing) speeds: 58.45972532036902
- P01 / P99 of valid speeds: 13.125 / 69.75
- Negative values: 0
- Non-positive values: 575302

## Literature comparison

- `n_sensors_reference`: 207
- `n_sensors_observed`: 207
- `n_timesteps_reference`: 34272
- `n_timesteps_observed`: 34272
- `frequency_minutes_reference`: 5
- `frequency_minutes_observed`: 5.0
- `start_date_reference`: 2012-03-01
- `end_date_reference`: 2012-06-30
- `reported_missing_pct`: 8.11
- `observed_combined_missing_pct`: 8.1094

## Issues

- None

## Notes

- 575302 non-positive values present. METR-LA commonly encodes missing readings as 0; confirm before imputation.
