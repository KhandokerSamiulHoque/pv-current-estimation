from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WORK = ROOT / "results" / "run"
FIG = WORK / "figures"
MONTHS = ["November", "December", "January", "February", "March"]
FEATURES = [
    "Clean Module Temperature (c)",
    "Dusty Module Temperature (c)",
    "Wind Speed (m/s)",
    "Humidity (%)",
    "Air Pressure (hPa)",
    "Tr (Hour)",
]
TARGETS = [
    "Clean Module Short Circuit Current (mA)",
    "Dusty Module Short Circuit Current (mA)",
]
PANEL_LABELS = {TARGETS[0]: "Clean", TARGETS[1]: "Dusty"}
MODEL_ORDER = ["Time-bin median", "Linear regression", "Random forest"]
COLORS = {
    "Time-bin median": "#777777",
    "Linear regression": "#D97706",
    "Random forest": "#1D4ED8",
}


def clean_frame(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, low_memory=False)
    out = raw.copy()
    coercions = {}
    for col in FEATURES + TARGETS:
        before = int(out[col].notna().sum())
        values = out[col].astype(str).str.replace("%", "", regex=False).str.strip()
        out[col] = pd.to_numeric(values, errors="coerce")
        coercions[col] = before - int(out[col].notna().sum())
    invalid_physical = {
        "humidity_outside_(0,1]": int(((out["Humidity (%)"] <= 0) | (out["Humidity (%)"] > 1)).sum()),
        "pressure_outside_[800,1100]_hPa": int(((out["Air Pressure (hPa)"] < 800) |
                                                  (out["Air Pressure (hPa)"] > 1100)).sum()),
    }
    out.loc[(out["Humidity (%)"] <= 0) | (out["Humidity (%)"] > 1), "Humidity (%)"] = np.nan
    out.loc[(out["Air Pressure (hPa)"] < 800) | (out["Air Pressure (hPa)"] > 1100),
            "Air Pressure (hPa)"] = np.nan
    used = out[FEATURES + TARGETS].copy()
    duplicate_without_index = int(used.duplicated().sum())
    rows_before = len(used)
    used = used.dropna().reset_index(drop=True)
    return used, {
        "file": path.name,
        "rows_raw": int(rows_before),
        "rows_usable": int(len(used)),
        "rows_dropped": int(rows_before - len(used)),
        "coercions_to_missing": coercions,
        "invalid_physical_values_set_missing": invalid_physical,
        "duplicate_rows_on_model_fields": duplicate_without_index,
        "humidity_min": float(used["Humidity (%)"].min()),
        "humidity_max": float(used["Humidity (%)"].max()),
        "pressure_min_hpa": float(used["Air Pressure (hPa)"].min()),
        "pressure_max_hpa": float(used["Air Pressure (hPa)"].max()),
        "recorded_days_from_clock_resets": int(1 + (np.diff(used["Tr (Hour)"].to_numpy()) < -1).sum()),
    }


def time_bin_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    train_bins = np.round(train["Tr (Hour)"].to_numpy() * 12) / 12
    test_bins = np.round(test["Tr (Hour)"].to_numpy() * 12) / 12
    temp = train[TARGETS].copy()
    temp["bin"] = train_bins
    med = temp.groupby("bin")[TARGETS].median()
    fallback = train[TARGETS].median().to_numpy()
    pred = np.empty((len(test), 2), dtype=float)
    for i, b in enumerate(test_bins):
        if b in med.index:
            pred[i] = med.loc[b].to_numpy()
        else:
            nearest = med.index[np.argmin(np.abs(med.index.to_numpy() - b))]
            pred[i] = med.loc[nearest].to_numpy() if len(med) else fallback
    return pred


def metric_rows(month: str, panel: str, model: str, actual: np.ndarray, pred: np.ndarray) -> dict:
    rmse = float(mean_squared_error(actual, pred) ** 0.5)
    span = float(np.max(actual) - np.min(actual))
    return {
        "month": month,
        "panel": panel,
        "model": model,
        "n": int(len(actual)),
        "r2": float(r2_score(actual, pred)),
        "mae_mA": float(mean_absolute_error(actual, pred)),
        "rmse_mA": rmse,
        "nrmse_range_pct": float(100 * rmse / span) if span > 0 else np.nan,
    }


def bootstrap_summary(metrics: pd.DataFrame, metric: str, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        d = metrics[metrics.model == model]
        month_values = d.groupby("month", sort=False)[metric].mean().reindex(MONTHS).to_numpy()
        boot = np.array([
            month_values[rng.integers(0, len(month_values), len(month_values))].mean()
            for _ in range(10000)
        ])
        rows.append({
            "model": model,
            "metric": metric,
            "mean": float(d[metric].mean()),
            "median": float(d[metric].median()),
            "ci95_low_month_block": float(np.quantile(boot, 0.025)),
            "ci95_high_month_block": float(np.quantile(boot, 0.975)),
        })
    return pd.DataFrame(rows)


def style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8.2,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.2,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })


def make_figures(metrics: pd.DataFrame, predictions: pd.DataFrame, ratio_rows: pd.DataFrame,
                 final_train: pd.DataFrame) -> None:
    style()
    FIG.mkdir(parents=True, exist_ok=True)

    heat = metrics.copy()
    heat["row"] = heat["model"].str.replace(" regression", " reg.") + " - " + heat["panel"]
    row_order = [
        f"{m.replace(' regression', ' reg.')} - {p}"
        for m in MODEL_ORDER for p in ["Clean", "Dusty"]
    ]
    matrix = heat.pivot(index="row", columns="month", values="r2").reindex(index=row_order, columns=MONTHS)
    fig, ax = plt.subplots(figsize=(4.78, 3.15), constrained_layout=True)
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor="white", cbar_kws={"label": "$R^2$"}, ax=ax)
    ax.set_title("Held-out-day $R^2$ by model and module", loc="center", pad=7)
    ax.set_xlabel("Evaluation month")
    ax.set_ylabel("")
    fig.savefig(FIG / "fig1_r2_heatmap.png")
    plt.close(fig)

    march = predictions[predictions.month == "March"].copy()
    march["time_bin"] = np.floor(march["Tr (Hour)"] * 12) / 12
    agg = march.groupby(["panel", "time_bin"], as_index=False).agg(
        actual=("actual", "mean"),
        rf=("Random forest", "mean"),
        linear=("Linear regression", "mean"),
        baseline=("Time-bin median", "mean"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(4.78, 4.05), sharex=True, constrained_layout=True)
    for ax, panel in zip(axes, ["Clean", "Dusty"]):
        d = agg[agg.panel == panel]
        ax.plot(d.time_bin, d.actual, color="#111111", lw=1.25, label="Observed")
        ax.plot(d.time_bin, d.rf, color=COLORS["Random forest"], lw=1.0, label="Random forest")
        ax.plot(d.time_bin, d.linear, color=COLORS["Linear regression"], lw=0.9, label="Linear regression")
        ax.plot(d.time_bin, d.baseline, color=COLORS["Time-bin median"], lw=0.8, ls="--", label="Time-bin median")
        ax.set_ylabel(f"{panel} $I_{{sc}}$ (mA)")
        ax.grid(True, color="#D1D5DB", linewidth=0.5)
    axes[0].set_title("March held-out day: observed and estimated current", loc="center", pad=28)
    axes[-1].set_xlabel("Recorded clock hour")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.965), columnspacing=1.4, handlelength=2.2)
    fig.savefig(FIG / "fig2_march_predictions.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(4.78, 2.80), sharey=True, constrained_layout=True)
    x = np.arange(len(MODEL_ORDER))
    for ax, panel in zip(axes, ["Clean", "Dusty"]):
        d = metrics[metrics.panel == panel]
        for i, model in enumerate(MODEL_ORDER):
            vals = d[d.model == model].set_index("month").reindex(MONTHS)["mae_mA"].to_numpy()
            jitter = np.linspace(-0.08, 0.08, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=16, color=COLORS[model], alpha=0.75, zorder=3)
            ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1), fmt="D", ms=4,
                        color="#111111", mfc=COLORS[model], capsize=3, lw=0.8, zorder=4)
        ax.set_xticks(x, ["Time-bin\nmedian", "Linear\nregression", "Random\nforest"])
        ax.set_title(f"{panel} module", loc="center", pad=8)
        ax.set_xlabel("Model")
    axes[0].set_ylabel("MAE (mA); dots are held-out months")
    fig.savefig(FIG / "fig3_mae_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.78, 2.75), constrained_layout=True)
    sns.boxplot(data=ratio_rows, x="month", y="dusty_to_clean_ratio", order=MONTHS,
                color="#BFDBFE", width=0.55, fliersize=1.5, linewidth=0.8, ax=ax)
    ax.axhline(1.0, color="#111111", ls="--", lw=0.8, label="Equal current")
    ax.set_title("Paired current ratio during the recorded 10:00-14:00 window", loc="center", pad=7)
    ax.set_xlabel("Evaluation month")
    ax.set_ylabel("Dusty $I_{sc}$ / clean $I_{sc}$")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(FIG / "fig4_paired_current_ratio.png")
    plt.close(fig)

    corr_cols = FEATURES + TARGETS
    sample = final_train[corr_cols].sample(n=min(50000, len(final_train)), random_state=42)
    corr = sample.corr(method="spearman")
    short = {
        FEATURES[0]: "Clean temp.", FEATURES[1]: "Dusty temp.", FEATURES[2]: "Wind",
        FEATURES[3]: "Humidity", FEATURES[4]: "Pressure", FEATURES[5]: "Clock hour",
        TARGETS[0]: "Clean Isc", TARGETS[1]: "Dusty Isc",
    }
    corr = corr.rename(index=short, columns=short)
    fig, ax = plt.subplots(figsize=(4.78, 4.15), constrained_layout=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.4, cbar_kws={"label": "Spearman $\\rho$"}, ax=ax)
    ax.set_title("Rank correlations in the rolling training record", loc="center", pad=7)
    ax.tick_params(axis="x", rotation=38)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.savefig(FIG / "fig5_spearman_heatmap.png")
    plt.close(fig)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    trains, tests, quality = {}, {}, []
    for month in MONTHS:
        trains[month], q = clean_frame(DATA / f"only_{month}_train.csv")
        q["partition"] = "train"
        q["month"] = month
        quality.append(q)
        tests[month], q = clean_frame(DATA / f"{month}_test.csv")
        q["partition"] = "test"
        q["month"] = month
        quality.append(q)

    metric_records = []
    pred_records = []
    ratio_records = []
    rolling_parts: list[pd.DataFrame] = []
    prior_tests: list[pd.DataFrame] = []
    for month in MONTHS:
        rolling_parts.append(trains[month])
        train = pd.concat(rolling_parts + prior_tests, ignore_index=True)
        test = tests[month]
        x_train, y_train = train[FEATURES], train[TARGETS]
        x_test, y_test = test[FEATURES], test[TARGETS]

        linear = LinearRegression().fit(x_train, y_train)
        rf = RandomForestRegressor(
            n_estimators=80,
            criterion="squared_error",
            max_features=1.0,
            min_samples_leaf=2,
            max_depth=24,
            random_state=42,
            n_jobs=-1,
        ).fit(x_train, y_train)
        preds = {
            "Time-bin median": time_bin_predict(train, test),
            "Linear regression": linear.predict(x_test),
            "Random forest": rf.predict(x_test),
        }

        for j, target in enumerate(TARGETS):
            actual = y_test[target].to_numpy()
            panel = PANEL_LABELS[target]
            for model, pred in preds.items():
                metric_records.append(metric_rows(month, panel, model, actual, pred[:, j]))
            for i in range(len(test)):
                pred_records.append({
                    "month": month,
                    "panel": panel,
                    "Tr (Hour)": float(test.iloc[i]["Tr (Hour)"]),
                    "actual": float(actual[i]),
                    **{model: float(arr[i, j]) for model, arr in preds.items()},
                })

        max_clean = float(test[TARGETS[0]].max())
        active = test[(test["Tr (Hour)"] >= 10) & (test["Tr (Hour)"] <= 14) &
                      (test[TARGETS[0]] > 0.10 * max_clean) & (test[TARGETS[0]] > 0)]
        for _, row in active.iterrows():
            ratio_records.append({
                "month": month,
                "clock_hour": float(row["Tr (Hour)"]),
                "clean_mA": float(row[TARGETS[0]]),
                "dusty_mA": float(row[TARGETS[1]]),
                "dusty_to_clean_ratio": float(row[TARGETS[1]] / row[TARGETS[0]]),
            })
        prior_tests.append(test)

    metrics = pd.DataFrame(metric_records)
    predictions = pd.DataFrame(pred_records)
    ratios = pd.DataFrame(ratio_records)
    metrics.to_csv(WORK / "model_metrics_by_month_panel.csv", index=False)
    predictions.to_csv(WORK / "predictions_all_models.csv", index=False)
    ratios.to_csv(WORK / "paired_current_ratios.csv", index=False)
    pd.DataFrame(quality).to_json(WORK / "data_quality_diagnostics.json", orient="records", indent=2)

    rng = np.random.default_rng(20260810)
    summaries = pd.concat([
        bootstrap_summary(metrics, "r2", rng),
        bootstrap_summary(metrics, "mae_mA", rng),
        bootstrap_summary(metrics, "rmse_mA", rng),
    ], ignore_index=True)
    summaries.to_csv(WORK / "aggregate_model_summary.csv", index=False)

    paired = metrics.pivot_table(index=["month", "panel"], columns="model", values="mae_mA")
    per_month_diff = (paired["Random forest"] - paired["Linear regression"]).groupby("month").mean().reindex(MONTHS)
    try:
        w = wilcoxon(per_month_diff.to_numpy(), alternative="two-sided", method="exact")
        wilcoxon_result = {"statistic": float(w.statistic), "p_value": float(w.pvalue)}
    except ValueError:
        wilcoxon_result = {"statistic": None, "p_value": None}

    ratio_summary = ratios.groupby("month")["dusty_to_clean_ratio"].agg(
        n="size", median="median", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75)
    ).reindex(MONTHS)
    ratio_summary.to_csv(WORK / "paired_current_ratio_summary.csv")

    final_train = pd.concat(rolling_parts + prior_tests[:-1], ignore_index=True)
    make_figures(metrics, predictions, ratios, final_train)

    result = {
        "analysis_definition": {
            "task": "conditional short-circuit-current estimation from same-time recorded meteorological and module-temperature inputs",
            "not_claimed": "day-ahead weather forecasting",
            "features": FEATURES,
            "targets": TARGETS,
            "evaluation": "five rolling monthly origins; each current-month final-day file held out; prior held-out days become historical for later origins",
            "rf": {"n_estimators": 80, "criterion": "squared_error", "max_features": 1.0,
                   "min_samples_leaf": 2, "max_depth": 24, "random_state": 42},
            "uncertainty": "95% percentile bootstrap over five month blocks; descriptive because n=5",
        },
        "quality": quality,
        "aggregate_summary": summaries.to_dict("records"),
        "rf_minus_linear_mae_by_month_mA": {k: float(v) for k, v in per_month_diff.items()},
        "wilcoxon_rf_vs_linear_month_average_mae": wilcoxon_result,
        "paired_ratio_summary": ratio_summary.reset_index().to_dict("records"),
    }
    (WORK / "validated_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "metrics_rows": len(metrics),
        "prediction_rows": len(predictions),
        "ratio_rows": len(ratios),
        "summary": summaries.to_dict("records"),
        "wilcoxon": wilcoxon_result,
        "figures": [str(p) for p in sorted(FIG.glob("*.png"))],
    }, indent=2))


if __name__ == "__main__":
    main()
