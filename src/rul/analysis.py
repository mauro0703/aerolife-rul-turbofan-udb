"""Development-only EDA, engine-level inference and publication figures."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from .data import SENSORS, SYMBOLS

NAVY, BLUE, TEAL, ORANGE = "#12314b", "#246ba0", "#158c83", "#d88736"


def style():
    sns.set_theme(style="whitegrid", font_scale=0.92)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": NAVY,
            "text.color": NAVY,
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
        }
    )


def holm_adjust(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p)
    adjusted = np.minimum(
        1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p))))
    )
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def analyze_development(dev, features, folder="reports"):
    """Paired early/late means use ONE difference per independent engine."""
    out = Path(folder)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    rows = []
    for sensor in ("s2", "s3", "s4", "s7", "s11", "s12", "s20", "s21"):
        diffs = []
        for _, g in dev.groupby("engine_id"):
            diffs.append(float(g[sensor].tail(30).mean() - g[sensor].head(30).mean()))
        d = np.array(diffs)
        test = stats.wilcoxon(d, alternative="two-sided", method="auto")
        nonzero = d[d != 0]
        ranks = stats.rankdata(np.abs(nonzero))
        effect = float(np.sum(np.sign(nonzero) * ranks) / ranks.sum())
        rng = np.random.default_rng(42)
        boot = rng.choice(d, size=(2000, len(d)), replace=True).mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rows.append(
            {
                "sensor": sensor,
                "symbol": SYMBOLS[int(sensor[1:]) - 1],
                "n_engines": len(d),
                "mean_late_minus_early": float(d.mean()),
                "median_difference": float(np.median(d)),
                "mean_ci95_lower": float(lo),
                "mean_ci95_upper": float(hi),
                "wilcoxon_W": float(test.statistic),
                "p_raw": float(test.pvalue),
                "rank_biserial": effect,
            }
        )
    tests = pd.DataFrame(rows)
    tests["p_holm"] = holm_adjust(tests.p_raw)
    tests.to_csv(out / "hypothesis_tests.csv", index=False)
    lifetime = dev.groupby("engine_id").cycle.max()
    summary = {
        "development_rows": len(dev),
        "development_engines": len(lifetime),
        "lifetime_min": int(lifetime.min()),
        "lifetime_median": float(lifetime.median()),
        "lifetime_max": int(lifetime.max()),
        "constant_raw_channels": [
            c
            for c in [*SENSORS, "setting_1", "setting_2", "setting_3"]
            if dev[c].nunique() <= 1
        ],
    }
    # Flags are descriptive and never used to remove legitimate degradation.
    q1, q3 = dev[SENSORS].quantile(0.25), dev[SENSORS].quantile(0.75)
    iqr = q3 - q1
    flags = ((dev[SENSORS] < q1 - 3 * iqr) | (dev[SENSORS] > q3 + 3 * iqr)) & (iqr > 0)
    summary["statistical_outlier_cells_3iqr"] = int(flags.sum().sum())
    summary["outliers_removed"] = 0
    pd.DataFrame(
        {
            "sensor": SENSORS,
            "q1_dev": q1.values,
            "q3_dev": q3.values,
            "iqr_dev": iqr.values,
            "flagged_cells": flags.sum().values,
        }
    ).to_csv(out / "outlier_audit.csv", index=False)
    style()
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.2))
    ax[0].hist(lifetime, bins=14, color=BLUE, edgecolor="white")
    ax[0].set(
        xlabel="Duración observada hasta fallo [ciclos]",
        ylabel="Motores de desarrollo",
        title="Variación entre unidades",
    )
    for engine in sorted(dev.engine_id.unique())[:4]:
        g = dev[dev.engine_id == engine]
        ax[1].plot(
            g.cycle,
            g.s3.rolling(15, min_periods=1).mean(),
            label=f"Motor {engine}",
            linewidth=1.5,
        )
    ax[1].set(
        xlabel="Edad [ciclos]",
        ylabel="T30, media causal de 15 ciclos [K]",
        title="Temperatura de salida del HPC",
    )
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figs / "eda_lifetime_trajectories.png", bbox_inches="tight")
    plt.close(fig)
    # The matrix is descriptive; p-values are NOT computed treating cycles as independent.
    sensors = ["s2", "s3", "s4", "s7", "s11", "s12", "s20", "s21"]
    corr = dev[sensors + ["rul_true"]].corr(method="spearman")
    corr.to_csv(out / "spearman_descriptive.csv")
    fig, ax = plt.subplots(figsize=(6.1, 4.5))
    sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        cmap="vlag",
        annot=True,
        fmt=".2f",
        ax=ax,
        annot_kws={"size": 7},
        cbar_kws={"label": "Spearman rho (descriptivo)"},
    )
    ax.set_title("Dependencia multivariable | desarrollo", fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / "eda_correlation.png", bbox_inches="tight")
    plt.close(fig)
    return summary, tests


def result_figures(cv, learning, predictions, importance, folder="reports/figures"):
    style()
    out = Path(folder)
    means = (
        cv.groupby("candidate", sort=False)
        .agg(mean=("rmse", "mean"), sd=("rmse", "std"))
        .sort_values("mean")
    )
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.4))
    ax[0].barh(
        means.index[::-1],
        means["mean"][::-1],
        xerr=means.sd[::-1],
        color=BLUE,
        capsize=3,
    )
    ax[0].set(
        xlabel="RMSE CV [ciclos]; barra = DE entre folds",
        title="Comparación sobre motores no vistos",
    )
    ax[0].tick_params(axis="y", labelsize=8)
    ax[1].plot(
        learning.n_engines,
        learning.train_rmse,
        "o-",
        label="Ajuste: snapshots train",
        color=TEAL,
    )
    ax[1].plot(
        learning.n_engines,
        learning.validation_rmse,
        "o-",
        label="Validación: mismos motores",
        color=ORANGE,
    )
    ax[1].set(
        xlabel="Motores utilizados para ajustar",
        ylabel="RMSE [ciclos]",
        title="Curva de aprendizaje (un fold)",
    )
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "model_selection.png", bbox_inches="tight")
    plt.close(fig)
    p = predictions
    fig, ax = plt.subplots(1, 3, figsize=(10.6, 3.1))
    ax[0].scatter(p.rul_true, p.rul_pred, s=18, color=BLUE, alpha=0.75)
    lim = max(p.rul_true.max(), p.rul_pred.max()) + 5
    ax[0].plot([0, lim], [0, lim], "--", color="gray", linewidth=1)
    ax[0].set(
        xlabel="RUL NASA [ciclos]",
        ylabel="RUL estimada [ciclos]",
        title="Prueba oficial sin recortar",
    )
    ax[1].scatter(p.rul_true, p.error, s=18, color=TEAL, alpha=0.75)
    ax[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    ax[1].set(
        xlabel="RUL NASA [ciclos]",
        ylabel="Estimación - real [ciclos]",
        title="Sesgo según horizonte",
    )
    ax[2].hist(p.error, bins=15, color=ORANGE, edgecolor="white")
    ax[2].set(
        xlabel="Error [ciclos]", ylabel="Motores", title="Distribución de residuos"
    )
    fig.tight_layout()
    fig.savefig(out / "test_residuals.png", bbox_inches="tight")
    plt.close(fig)
    sort = p.sort_values("rul_true").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.6, 2.8))
    ax.fill_between(
        sort.index,
        sort.lower90,
        sort.upper90,
        color=BLUE,
        alpha=0.18,
        label="Intervalo nominal 90%",
    )
    ax.plot(sort.index, sort.rul_pred, color=BLUE, linewidth=1.3, label="Predicción")
    ax.scatter(sort.index, sort.rul_true, color=ORANGE, s=13, label="RUL real")
    ax.set(
        xlabel="Motores de prueba ordenados por RUL real (solo presentación)",
        ylabel="RUL [ciclos]",
        title="Cobertura empírica en prueba independiente",
    )
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "test_intervals.png", bbox_inches="tight")
    plt.close(fig)
    top = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.barh(top.feature, top.importance, color=TEAL)
    ax.set(
        xlabel="Importancia interna del modelo (no causal)",
        title="Diez características con mayor peso",
    )
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)
