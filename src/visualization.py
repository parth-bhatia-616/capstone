
import os, sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(__file__))
from config import PLOTS_DIR

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})
CMAP_CHEM = {"NMC": "#2196F3", "LFP": "#4CAF50", "NCA": "#FF9800", "LCO": "#9C27B0"}
CMAP_GRADE = {"A": "#4CAF50", "B": "#FF9800", "C": "#F44336"}


def _save(fig, name: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  ✓ Saved → {path}")
    return path


# ── 01: Degradation Curves ────────────────────────────────────────────────────

def plot_degradation_curves(df: pd.DataFrame,
                             max_bat: int = 4,
                             save: bool = True) -> plt.Figure:
    chemistries = df["chemistry"].unique() if "chemistry" in df.columns else [None]
    n = min(len(chemistries), 4)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    cap_col = "measured_capacity_Ah" if "measured_capacity_Ah" in df.columns else "Capacity"

    for ax, chem in zip(axes, list(chemistries)[:n]):
        sub = df[df["chemistry"] == chem] if chem else df
        for j, bat in enumerate(sub["battery_id"].unique()[:max_bat]):
            d = sub[sub["battery_id"] == bat].sort_values("cycle")
            c = CMAP_CHEM.get(chem, "#607D8B")
            ax.plot(d["cycle"], d[cap_col], color=c, alpha=0.85 - j*0.15, lw=1.5)
        ax.set(title=f"{chem or 'Battery'}", xlabel="Cycle", ylabel="Capacity (Ah)")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Capacity Degradation by Chemistry", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save:
        _save(fig, "01_degradation_curves.png")
    return fig


# ── 02: SoH Distribution ─────────────────────────────────────────────────────

def plot_soh_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    chemistries = df["chemistry"].unique() if "chemistry" in df.columns else [None]

    for chem in chemistries:
        d = df[df["chemistry"] == chem]["soh"] if chem else df["soh"]
        ax1.hist(d.dropna(), bins=30, alpha=0.55, density=True,
                 color=CMAP_CHEM.get(chem, "#607D8B"), label=chem or "All")
    ax1.axvline(80, color="green",  ls="--", lw=1.5, label="Grade A (80%)")
    ax1.axvline(60, color="orange", ls="--", lw=1.5, label="Grade B (60%)")
    ax1.set(xlabel="SoH (%)", ylabel="Density",
            title="SoH Distribution by Chemistry")
    ax1.legend(fontsize=9)

    if "chemistry" in df.columns and len(chemistries) > 1:
        sns.violinplot(data=df, x="chemistry", y="soh",
                       palette=CMAP_CHEM, ax=ax2, inner="quartile",
                       order=sorted(chemistries))
    else:
        ax2.violinplot(df["soh"].dropna(), showmedians=True)
    ax2.axhline(80, color="green",  ls="--", alpha=0.6)
    ax2.axhline(60, color="orange", ls="--", alpha=0.6)
    ax2.set(xlabel="Chemistry", ylabel="SoH (%)", title="SoH Violin Plot")

    fig.suptitle("State-of-Health Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save:
        _save(fig, "02_soh_distribution.png")
    return fig


# ── 03: IC / DV Feature Evolution ────────────────────────────────────────────

def plot_ic_dv_features(df: pd.DataFrame,
                         battery_id: str = None,
                         save: bool = True) -> plt.Figure:
    if battery_id is None and "battery_id" in df.columns:
        battery_id = df["battery_id"].iloc[0]
    d = df[df["battery_id"] == battery_id].sort_values("cycle") if battery_id else df

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    plots = [
        ("ic_peak_height",  "#2196F3", "IC Peak Height (dQ/dV)",     "Phase capacity indicator"),
        ("ic_peak_voltage", "#9C27B0", "IC Peak Voltage (V)",         "Peak shift = lithiation loss"),
        ("dv_peak_height",  "#FF9800", "DV Peak Height (dV/dQ)",      "Two-phase region strength"),
        ("eis_R0",          "#F44336", "EIS R₀ (Ω)",                  "Series resistance growth"),
    ]
    for ax, (col, color, ylabel, subtitle) in zip(axes.flatten(), plots):
        if col in d.columns:
            ax.plot(d["cycle"], d[col], color=color, lw=2, marker="o",
                    markersize=3, alpha=0.85)
            ax.set(xlabel="Cycle", ylabel=ylabel,
                   title=f"{ylabel}\n({subtitle})")
            ax.grid(True, alpha=0.25)
        else:
            ax.text(0.5, 0.5, f"{col}\nnot available", ha="center",
                    va="center", transform=ax.transAxes, color="gray")

    fig.suptitle(f"IC/DV/EIS Feature Evolution — {battery_id}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save:
        _save(fig, "03_ic_dv_features.png")
    return fig


# ── 04: Correlation Matrix ────────────────────────────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame,
                              features: List[str] = None,
                              save: bool = True) -> plt.Figure:
    default = ["cycle", "internal_resistance", "ic_peak_height",
               "dv_peak_height", "eis_R0", "eis_Rct", "v_mean",
               "energy_Wh", "dod_mean", "crate_mean", "soh"]
    cols = [c for c in (features or default) if c in df.columns]
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={"size": 8}, linewidths=0.4)
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save:
        _save(fig, "04_correlation_matrix.png")
    return fig


# ── 05: SoH Predictions ──────────────────────────────────────────────────────

def plot_soh_predictions(y_true: np.ndarray,
                          preds: Dict[str, np.ndarray],
                          save: bool = True) -> plt.Figure:
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, preds.items()):
        mae = np.mean(np.abs(y_true - y_pred))
        r2  = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
        ax.scatter(y_true, y_pred, alpha=0.35, s=18, color="#2196F3", edgecolors="none")
        lim = [min(y_true.min(), y_pred.min()) - 2,
               max(y_true.max(), y_pred.max()) + 2]
        ax.plot(lim, lim, "r--", lw=2)
        ax.set(xlabel="True SoH (%)", ylabel="Predicted SoH (%)",
               title=f"{name}\nMAE={mae:.2f}%  R²={r2:.4f}")
        ax.grid(True, alpha=0.25)

    fig.suptitle("SoH Prediction Accuracy", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save:
        _save(fig, "05_soh_predictions.png")
    return fig


# ── 06: Feature Importance ────────────────────────────────────────────────────

def plot_feature_importance(importances: Dict[str, float],
                             top_n: int = 20,
                             title: str = "Random Forest Feature Importance",
                             save: bool = True) -> plt.Figure:
    sorted_fi = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, vals = zip(*sorted_fi)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(names)))
    ax.barh(names[::-1], vals[::-1], color=colors)
    for i, (name, val) in enumerate(zip(reversed(names), reversed(vals))):
        ax.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=8)
    ax.set(xlabel="Importance Score", title=title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    if save:
        _save(fig, "06_feature_importance.png")
    return fig


# ── 07: Ensemble Comparison ───────────────────────────────────────────────────

def plot_ensemble_comparison(y_true: np.ndarray,
                              preds: Dict[str, np.ndarray],
                              n_samples: int = 100,
                              save: bool = True) -> plt.Figure:
    n = min(n_samples, len(y_true))
    idx = np.arange(n)
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

    ax1.plot(idx, y_true[:n], "k-", lw=2.5, label="True SoH", zorder=5)
    for (name, yp), col in zip(preds.items(), colors):
        lw = 2.5 if "ensemble" in name.lower() else 1.3
        ls = "-" if "ensemble" in name.lower() else "--"
        ax1.plot(idx, yp[:n], ls, color=col, lw=lw, label=name, alpha=0.88)
    ax1.set(ylabel="SoH (%)", title="Model Predictions vs True SoH")
    ax1.legend(fontsize=9, loc="upper right"); ax1.grid(True, alpha=0.25)

    for (name, yp), col in zip(preds.items(), colors):
        ax2.plot(idx, np.abs(y_true[:n] - yp[:n]), lw=1.5, color=col, label=name, alpha=0.85)
    ax2.set(xlabel="Sample", ylabel="Absolute Error (%)",
            title="Absolute Error by Model")
    ax2.legend(fontsize=9, loc="upper right"); ax2.grid(True, alpha=0.25)

    fig.suptitle("Ensemble Fusion — Full Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save:
        _save(fig, "07_ensemble_comparison.png")
    return fig


# ── 08: SoC Trajectory ────────────────────────────────────────────────────────

def plot_soc_trajectory(times: np.ndarray,
                         soc_true: np.ndarray,
                         soc_ekf: np.ndarray,
                         soc_ukf: np.ndarray = None,
                         soc_fused: np.ndarray = None,
                         voltages: np.ndarray = None,
                         save: bool = True) -> plt.Figure:
    n_rows = 2 if voltages is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(times, soc_true * 100, "k-",  lw=2.5, label="True SoC",   zorder=5)
    ax.plot(times, soc_ekf  * 100, "b--", lw=1.8, label="AEKF",       alpha=0.9)
    if soc_ukf   is not None:
        ax.plot(times, soc_ukf  * 100, "g:",  lw=1.8, label="UKF",   alpha=0.85)
    if soc_fused is not None:
        ax.plot(times, soc_fused* 100, "r-",  lw=2.2, label="Fused", zorder=4)
    ax.set(ylabel="SoC (%)", title="SoC Estimation — Kalman + ML Fusion",
           ylim=[-2, 102])
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25)

    if voltages is not None:
        axes[1].plot(times, voltages, "#E91E63", lw=1.8)
        axes[1].set(xlabel="Time (s)", ylabel="Voltage (V)",
                    title="Terminal Voltage Profile")
        axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    if save:
        _save(fig, "08_soc_trajectory.png")
    return fig


# ── 09: Cell Balance + SoH Gauge ─────────────────────────────────────────────

def plot_cell_balance_gauge(cell_voltages: List[float],
                             mode: str, soh: float, grade: str,
                             save: bool = True) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    v = np.array(cell_voltages)
    mean_v = v.mean()
    dv = v.max() - v.min()

    # Voltage bars
    colors = ["#F44336" if vi == v.max() else
              "#FF9800" if vi == v.min() else "#2196F3"
              for vi in v]
    ax1.bar([f"Cell {i+1}" for i in range(len(v))], v,
            color=colors, edgecolor="black", lw=1.5)
    ax1.axhline(mean_v, color="navy", ls="--", lw=1.5,
                label=f"Mean: {mean_v:.3f} V")
    ax1.set(ylabel="Voltage (V)", title="4S Cell Voltage Balance",
            ylim=[v.min() - 0.06, v.max() + 0.08])
    ax1.legend()
    ax1.text(0.5, 0.97, f"ΔV={dv*1000:.0f} mV  |  {mode}",
             ha="center", va="top", transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    # Gauge arc
    theta = np.linspace(0, np.pi, 300)
    ax2.plot(np.cos(theta), np.sin(theta), color="#E0E0E0", lw=16, solid_capstyle="butt")
    tf = np.linspace(0, np.pi * np.clip(soh / 100, 0, 1), 300)
    gc = CMAP_GRADE.get(grade, "#2196F3")
    ax2.plot(np.cos(tf), np.sin(tf), color=gc, lw=16, solid_capstyle="butt")
    ax2.text(0, -0.15, f"{soh:.1f}%", ha="center", fontsize=22, fontweight="bold")
    ax2.text(0, -0.42, f"Grade {grade}", ha="center", fontsize=16,
             color=gc, fontweight="bold")
    ax2.set(xlim=(-1.3, 1.3), ylim=(-0.6, 1.1), aspect="equal",
            title="Battery Health Gauge")
    ax2.axis("off")

    fig.tight_layout()
    if save:
        _save(fig, "09_cell_balance_gauge.png")
    return fig


# ── 10: Training Curves ───────────────────────────────────────────────────────

def plot_training_curves(history: Dict, title: str = "Training Loss",
                          save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    tr = history.get("train_loss", [])
    va = history.get("val_loss",   [])
    ep = list(range(1, len(tr) + 1))
    ax.plot(ep, tr, "#2196F3", lw=2, label="Train Loss")
    if va:
        ax.plot(range(1, len(va) + 1), va, "#F44336", lw=2,
                ls="--", label="Val Loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title=title)
    ax.legend(); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if save:
        _save(fig, "10_training_curves.png")
    return fig


if __name__ == "__main__":
    # Smoke-test with random data
    rng = np.random.default_rng(42)
    y_true = rng.uniform(60, 100, 200)
    preds = {
        "RF":       y_true + rng.normal(0, 2, 200),
        "XGBoost":  y_true + rng.normal(0, 1.5, 200),
        "Ensemble": y_true + rng.normal(0, 0.9, 200),
    }
    plot_soh_predictions(y_true, preds, save=True)
    plot_ensemble_comparison(y_true, preds, save=True)
    plot_cell_balance_gauge([3.65, 3.72, 3.68, 3.70],
                             "ACTIVE_BALANCING", 82.3, "A", save=True)
    print("✓ All test plots saved")