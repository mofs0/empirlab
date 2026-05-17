"""Publication-quality plotting utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def plot_regression_table(
    results: list[dict],
    dep_var: str = "y",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Coefficient plot from a list of regression result dicts.

    Each dict: {"label": str, "coef": float, "ci_lo": float, "ci_hi": float}
    """
    fig, ax = plt.subplots(figsize=figsize)
    ys     = range(len(results))
    coefs  = [r["coef"] for r in results]
    lo     = [r["coef"] - r["ci_lo"] for r in results]
    hi     = [r["ci_hi"] - r["coef"] for r in results]
    labels = [r["label"] for r in results]
    ax.errorbar(coefs, list(ys), xerr=[lo, hi], fmt="o", color="steelblue",
                ecolor="steelblue", elinewidth=1.5, capsize=4)
    ax.axvline(0, color="gray", linestyle="--", lw=1)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Coefficient")
    ax.set_title(f"Regression coefficients — {dep_var}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_event_study(
    event_df: pd.DataFrame,
    figsize: tuple = (9, 4),
    title: str = "Event Study — Cumulative Abnormal Returns",
) -> plt.Figure:
    """Standard event-study CAR plot.

    event_df must have columns: event_day, car, and optionally ci_lo / ci_hi.
    """
    fig, ax = plt.subplots(figsize=figsize)
    days = event_df["event_day"].values
    car  = event_df["car"].values
    ax.plot(days, car, "b-o", markersize=4, label="CAR")
    if "ci_lo" in event_df.columns and "ci_hi" in event_df.columns:
        ax.fill_between(days, event_df["ci_lo"], event_df["ci_hi"],
                        alpha=0.2, color="steelblue", label="95% CI")
    ax.axhline(0, color="gray", lw=1, linestyle="--")
    ax.axvline(0, color="red", lw=1, linestyle="--", label="Event day")
    ax.set_xlabel("Days relative to event")
    ax.set_ylabel("CAR")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    plt.tight_layout()
    return fig


def plot_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
    max_display: int = 15,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Bar plot of mean |SHAP| values (no shap library required)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[-max_display:]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh([feature_names[i] for i in order], mean_abs[order], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig
