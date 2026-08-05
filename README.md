# Sensor-Based Analysis of Appliance Energy Consumption in a Low-Energy House

Explanatory data analysis (EDA) of household appliance energy consumption using indoor wireless sensor data and outdoor weather variables. The analysis is framed around the needs of a smart-home energy management company: when energy is used, where it is used, and which sensor signals are actually informative.

Based on the dataset and findings of Candanedo, Feldheim & Deramaix (2017).

## Dataset

| | |
|---|---|
| Source | [UCI ML Repository — Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) |
| Records | 19,735 |
| Features | 29 (before feature engineering) |
| Period | January – May 2016 |
| Granularity | 10-minute intervals |
| Target | `Appliances` — energy consumption in Wh |

The data combines a target energy variable, `lights`, nine indoor temperature sensors (`T1`–`T9`), nine indoor humidity sensors (`RH_1`–`RH_9`), six outdoor weather variables from a nearby airport station, and two random variables (`rv1`, `rv2`) included in the original study as a sanity check.

There are no missing values and no duplicate rows.

## Business questions

The notebook answers eight questions, each with code, a chart, and an interpretation:

1. At what times of day is appliance energy consumption highest?
2. Do weekends and weekdays show different usage patterns?
3. Which indoor temperature sensors are most associated with consumption?
4. Which indoor humidity sensors are most associated with consumption?
5. How does consumption vary across days of the week?
6. What weather conditions are most related to appliance energy use?
7. What characterises high-consumption periods?
8. Which variables overall are most strongly related to consumption?

## Key findings

**Consumption is strongly time-driven.** Usage bottoms out around 03:00 (≈48 Wh average) and peaks at 18:00 (≈190 Wh), with a secondary bump around late morning. `hour` is the single strongest correlate of consumption in the whole dataset (r = 0.22).

**Weekends differ, but modestly.** Weekend mean consumption is 100.6 Wh against 96.6 Wh on weekdays. The gap in medians is larger (70 vs 60 Wh), suggesting a flatter, more spread-out weekend profile rather than a sharply higher one. By day, Monday is highest (111 Wh) and Tuesday lowest (87 Wh).

**No single sensor dominates.** Individual correlations are weak: the strongest temperature signals are `T2` (0.12) and `T6` (0.12), and the strongest humidity signal is outdoor humidity `RH_out` (−0.15). This mirrors the source paper — appliance use is driven by occupant behaviour, and sensors capture it only indirectly.

**Sensors still add value in aggregate.** A linear regression using only time and weather features reaches R² = 0.097 (MAE 54.3 Wh). Adding the indoor temperature and humidity sensors raises it to R² = 0.172 (MAE 52.5 Wh) — a ~77% improvement in explained variance. Weak individual correlations do not mean weak collective signal.

| Model | Features | MAE (Wh) | R² |
|---|---|---|---|
| Baseline | time + weather | 54.33 | 0.097 |
| Extended | time + weather + indoor sensors | 52.51 | 0.172 |

Both R² values are low in absolute terms. This is expected for a linear model on this dataset; the paper itself reports that non-linear models are needed for competitive accuracy. The comparison here is intended to isolate the contribution of the sensor network, not to produce a production forecaster.

## Repository contents

```
├── notebooks/
│   └── appliance_energy_eda.ipynb    # full analysis
├── data/
│   └── energydata_complete.csv       # download from UCI (see below)
├── requirements.txt
└── README.md
```

## Running the notebook

```bash
git clone https://github.com/AkshaMakarani/Python-Learning.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook
```

Download `energydata_complete.csv` from the [UCI page](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) and place it in `data/`.

The notebook currently reads from a Colab path:

```python
df = pd.read_csv('/content/energydata_complete.csv')
```

Change this to `data/energydata_complete.csv` to run locally, or upload the CSV to your Colab session before executing.

**Requirements:** `pandas`, `numpy`, `matplotlib`, `scikit-learn`


## Method notes

- Time features (`hour`, `day_name`, `is_weekend`, `month`) are derived from the timestamp, since smart-home recommendations are time-dependent.
- `high_consumption` is defined as the top quartile of the `Appliances` distribution.
- Relationships are measured with Pearson correlation and are **associations, not causal effects**.
- The model uses a plain 80/20 random split. For a time series this leaks future information into training; a chronological split would be more appropriate for any forecasting claim.


## Future work

- Non-linear models (gradient boosting, random forest) as used in the source paper
- Lag features and rolling windows to capture usage momentum
- Chronological train/test split for honest forecasting evaluation
- Occupancy proxies and routine detection


## References

Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings*, 140, 81–97. https://doi.org/10.1016/j.enbuild.2017.01.083

Appliances Energy Prediction dataset. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
