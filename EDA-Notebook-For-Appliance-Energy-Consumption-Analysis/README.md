# Appliance Energy Consumption - Sensor-Based EDA

Explanatory data analysis of household appliance energy use, combining indoor wireless sensor readings with outdoor weather data. Framed around the needs of a smart-home energy management company: when energy is used, and which sensor signals are actually informative.

Based on [Candanedo, Feldheim & Deramaix (2017)](https://doi.org/10.1016/j.enbuild.2017.01.083) and the [UCI Appliances Energy Prediction dataset](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) — 19,735 records at 10-minute intervals, Jan–May 2016, with 9 indoor temperature sensors, 9 humidity sensors, and 6 weather variables.

## Key findings

- **Consumption is time-driven.** Usage bottoms out around 03:00 (~48 Wh) and peaks at 18:00 (~190 Wh). `hour` is the strongest single correlate in the dataset (r = 0.22).
- **Weekends differ modestly.** 100.6 Wh average vs 96.6 Wh on weekdays; Monday highest (111 Wh), Tuesday lowest (87 Wh).
- **No single sensor dominates.** Strongest signals are `T2` (0.12) and `RH_out` (−0.15) — appliance use is behaviour-driven, so sensors capture it only indirectly.
- **But sensors add value in aggregate.** A linear model on time + weather reaches R² = 0.097 (MAE 54.3 Wh); adding indoor sensors raises it to R² = 0.172 (MAE 52.5 Wh).

Both R² values are low in absolute terms, as expected for a linear model here — the comparison isolates the sensor network's contribution rather than aiming for forecast accuracy.

## Running it

```bash
pip install pandas numpy matplotlib scikit-learn
jupyter notebook appliance_energy_eda.ipynb
```

Download `energydata_complete.csv` from UCI and update the path in the import cell.

## Notes and limitations

Correlations are associations, not causal effects. The model uses a random 80/20 split, which leaks future information for time series data - a chronological split would be needed for any real forecasting claim. Natural extensions: non-linear models as in the source paper, lag features, and occupancy proxies.
