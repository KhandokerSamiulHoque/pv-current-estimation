# Estimating PV Short-Circuit Current From Weather: Clean vs. Dusty Modules

A 2026 reanalysis of the field data from my BSc thesis at BRAC University (2021). Two identical PV modules, one cleaned and one left to collect dust, were logged side by side for five months (November to March) together with module temperature, wind speed, humidity and air pressure. The question is how well short-circuit current can be estimated from those same-time readings, and how much dust costs.

The original thesis notebook is in [pv-module-weather-analysis](https://github.com/KhandokerSamiulHoque/pv-module-weather-analysis). This version fixes the evaluation: each month's last recorded day is held out, models only train on earlier data, and the results come with month-level uncertainty.

## Setup
- **Inputs:** clean and dusty module temperature, wind speed, humidity, air pressure, clock hour.
- **Targets:** short-circuit current (mA) of the clean and the dusty module.
- **Models:** a time-of-day median baseline, linear regression, and a random forest (80 trees, depth 24, seed 42).
- **Evaluation:** five rolling monthly origins. Non-numeric entries and physically impossible humidity and pressure values (counts in `results/published/data_quality_diagnostics.json`).

## Results

| Model | Mean R² | 95% CI (month bootstrap) | Mean MAE (mA) |
|---|---:|---|---:|
| Time-of-day median | 0.740 | 0.469 to 0.910 | 102.2 |
| Linear regression | 0.754 | 0.626 to 0.843 | 128.2 |
| Random forest | 0.847 | 0.804 to 0.890 | 86.8 |

The random forest has the highest and most stable R², but with only five months the gain over linear regression is not statistically significant (Wilcoxon on monthly MAE, p = 0.19). Between 10:00 and 14:00 the dusty module produced a median 78% to 96% of the clean module's current depending on the month, with the largest loss in February.

## Reproduce
```bash
pip install -r requirements.txt
cd src && python reanalyze_pv.py
```
Outputs go to `results/run/`. `results/published/` holds the reference results and figures. A rerun on 2026-10-20 matched them (largest difference 1.4e-14, floating-point rounding).

## Limits
- One site, two modules, five months. This is not a multi-site or multi-year result.
- These are same-time estimates from measured weather, not day-ahead forecasts.
- The confidence intervals rest on five monthly blocks and should be read as descriptive.

## Data
`data/` holds the monthly training and held-out CSVs from the thesis measurement campaign. The thesis was a team project. The data are shared here with my thesis co-authors' consent.

Author: Khandoker Samiul Hoque (ORCID [0009-0008-0756-2395](https://orcid.org/0009-0008-0756-2395)). Code license: MIT. Manuscript in preparation.
