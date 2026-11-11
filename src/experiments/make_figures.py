"""
Thesis figures from the Random Forest and MLP runs.

Writes to results/figures/, as PDF (for LaTeX) and PNG (for slides):

  fig1_cross_vehicle_heatmap   train-on / test-on MCC matrix (RF, single
                               vehicle), zero-shot and benign-calibrated
  fig2_threshold_rules         MCC per held-out vehicle under each threshold
                               rule, RF and MLP
  fig3_training_arms           single vs pooled vs FedAvg vs FedProx per
                               held-out vehicle, every seed shown
  fig4_per_attack              recall per attack type: same vehicle vs
                               unseen vehicle (calibrated)
  fig5_scores_rf / _mlp / _mlp_fedavg
                               score distributions, benign vs attack, on the
                               sources and on the unseen vehicle, with the
                               three thresholds marked
  fig6_feature_sets            the four feature representations compared

Only source-selected configurations are plotted (selected=True), except in
fig6, which shows every feature set on purpose.

Run from the project root, after the experiment scripts:
    python src/experiments/make_figures.py
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURE_SETS, RESULTS, SCORES_DIR, VEHICLES  # noqa: E402

FIGURES = RESULTS / "figures"
LABELS = {"impala": "Impala", "traverse": "Traverse",
          "silverado": "Silverado", "forester": "Forester"}
FEATURE_LABELS = {"raw": "raw", "stat": "stat", "stat_sf": "stat-sf", "all": "all"}

INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8985"
GRID, SURFACE = "#e3e2de", "#ffffff"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed order
REFERENCE = "#b8b7b2"                                   # oracle = a bound
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("thesis_blue", BLUE_RAMP)

RULES = [("mcc_zs", "zero-shot: source-selected", SERIES[0], None),
         ("mcc_zsb", "zero-shot: budget on source benign", SERIES[1], None),
         ("mcc_cal", "calibrated: budget on target benign", SERIES[2], None),
         ("mcc_oracle", "oracle (upper bound)", REFERENCE, "///")]
ARMS = [("single", "single vehicle", SERIES[0]),
        ("pooled", "pooled (centralised)", SERIES[1]),
        ("fedavg", "FedAvg", SERIES[2]),
        ("fedprox", "FedProx (selected mu)", SERIES[3])]


def style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.size": 9, "text.color": INK,
        "axes.labelcolor": INK_2, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False, "pdf.fonttype": 42,
    })


def recede(ax, grid_axis="y"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)
    ax.yaxis.grid(grid_axis == "y")
    ax.xaxis.grid(grid_axis == "x")


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIGURES}/{name}.pdf and .png")


def flag(df, col="selected"):
    df[col] = df[col].astype(str).str.lower() == "true"
    return df


def seed_mean(df, metric):
    """Seed-averaged value per held-out vehicle (rows in a seed averaged first)."""
    return (df.groupby(["held_out", "seed"])[metric].mean()
              .groupby("held_out").mean().reindex(VEHICLES))


def bar_labels(ax, pos, vals, size=6.8):
    for p, v in zip(pos, vals):
        if np.isfinite(v):
            ax.text(p, max(v, 0) + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=size, color=INK_2)


# ----------------------------------------------------------------------

def fig_heatmap(rf):
    single = rf[(rf["mode"] == "single") & rf["selected"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9),
                             gridspec_kw={"wspace": 0.35})
    for ax, (rule, title) in zip(axes, [("mcc_zs", "Zero-shot (source-selected)"),
                                        ("mcc_cal", "Benign target-calibrated")]):
        m = np.full((4, 4), np.nan)
        for i, tr in enumerate(VEHICLES):
            for j, te in enumerate(VEHICLES):
                s = single[(single["train_vehicles"] == tr) &
                           (single["test_vehicle"] == te)]
                if len(s):
                    m[i, j] = s["mcc_zs" if tr == te else rule].iloc[0]
        ax.imshow(np.clip(m, 0, 1), cmap=SEQUENTIAL, vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                if np.isfinite(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                            fontsize=8.5,
                            color="#ffffff" if m[i, j] > 0.55 else INK)
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=INK, linewidth=1.4))
        off = ~np.eye(4, dtype=bool)
        ax.set_title(f"{title}\nunseen-vehicle mean {np.nanmean(m[off]):.2f}",
                     fontsize=9.5, color=INK, pad=8)
        ax.set_xticks(range(4), [LABELS[v] for v in VEHICLES], fontsize=8)
        ax.set_yticks(range(4), [LABELS[v] for v in VEHICLES], fontsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.text(0.5, -0.02, "tested on", ha="center", color=INK_2)
    fig.text(0.03, 0.5, "trained on", va="center", rotation="vertical", color=INK_2)
    fig.text(0.5, -0.10,
             "Random Forest, one source vehicle, source-selected feature set. "
             "Boxed diagonal: same vehicle (source-selected threshold) in both panels.",
             ha="center", color=INK_MUTED, fontsize=7.5)
    fig.suptitle("MCC across vehicles", fontsize=11, color=INK, y=1.04)
    save(fig, "fig1_cross_vehicle_heatmap")


def fig_thresholds(rf, mlp):
    panels = [("Random Forest, pooled", rf[(rf["mode"] == "pooled_3")]),
              ("MLP, pooled", mlp[mlp["arm"] == "pooled"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    x = np.arange(4)
    w = 0.2
    for ax, (title, df) in zip(axes, panels):
        cross = df[df["selected"] & (df["setting"] == "cross")]
        for k, (col, label, colour, hatch) in enumerate(RULES):
            vals = seed_mean(cross, col).to_numpy()
            pos = x + (k - 1.5) * w
            ax.bar(pos, np.clip(vals, 0, None), w * 0.88, color=colour,
                   edgecolor=SURFACE, linewidth=1.0, hatch=hatch,
                   label=label if ax is axes[0] else None)
            bar_labels(ax, pos, vals, 5.8)
        ax.set_xticks(x, [LABELS[v] for v in VEHICLES], fontsize=8)
        ax.set_title(title, fontsize=9.5, color=INK_2)
        ax.set_ylim(0, 1.05)
        recede(ax)
    axes[0].set_ylabel("MCC on the held-out vehicle")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.95), fontsize=8)
    fig.suptitle("MCC under each threshold rule, by held-out vehicle",
                 fontsize=11, color=INK, y=1.02, x=0.02, ha="left")
    save(fig, "fig2_threshold_rules")


def fig_arms(mlp):
    cross = mlp[mlp["selected"] & (mlp["setting"] == "cross")]
    metrics = [("auprc", "AUPRC (no threshold)"),
               ("mcc_zs", "MCC, zero-shot"),
               ("mcc_cal", "MCC, benign-calibrated")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), sharey=True)
    x = np.arange(4)
    w = 0.19
    rng = np.random.default_rng(0)
    for ax, (metric, title) in zip(axes, metrics):
        for k, (arm, label, colour) in enumerate(ARMS):
            s = cross[cross["arm"] == arm]
            ps = s.groupby(["held_out", "seed"])[metric].mean()
            pos = x + (k - 1.5) * w
            for i, v in enumerate(VEHICLES):
                if v not in ps.index.get_level_values(0):
                    continue
                vals = ps.xs(v, level="held_out").to_numpy()
                jitter = rng.uniform(-w * 0.25, w * 0.25, len(vals))
                ax.scatter(pos[i] + jitter, vals, s=9, color=colour, alpha=0.35,
                           linewidths=0)
                ax.scatter([pos[i]], [vals.mean()], s=46, color=colour,
                           edgecolors=SURFACE, linewidths=1.5, zorder=3,
                           label=label if (i == 0 and ax is axes[0]) else None)
        ax.set_xticks(x, [LABELS[v] for v in VEHICLES], fontsize=8)
        ax.set_title(title, fontsize=9.5, color=INK_2)
        ax.set_ylim(-0.05, 1.05)
        recede(ax)
        for b in x[:-1] + 0.5:
            ax.axvline(b, color=GRID, linewidth=0.8)
    axes[0].set_ylabel("value on the held-out vehicle")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.95), fontsize=8)
    fig.suptitle("Training strategies on the held-out vehicle "
                 "(large dot = mean over seeds, small dots = single seeds)",
                 fontsize=10.5, color=INK, y=1.02, x=0.02, ha="left")
    save(fig, "fig3_training_arms")


def fig_per_attack():
    """
    Recall per attack type. The unseen-vehicle bars use the calibrated
    threshold only: at the zero-shot threshold some models flag almost
    every window, so recall there is high only because false alarms are
    too (see the false-alarm rates printed by analyse_results.py).
    """
    pa = flag(pd.read_csv(RESULTS / "rf_per_attack.csv"))
    pa = pa[pa["selected"] & (pa["mode"] == "single")]
    within = pa[pa["setting"] == "within"].groupby("attack_type")["recall_zs"].mean()
    cross = pa[pa["setting"] == "cross"]
    t = pd.DataFrame({"within": within,
                      "cal": cross.groupby("attack_type")["recall_cal"].mean()})
    t = t.sort_values("cal")
    y = np.arange(len(t))
    h = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 0.36 * len(t) + 1.4))
    bars = [("within", "same vehicle (source-selected threshold)", SERIES[0], h / 2),
            ("cal", "unseen vehicle (benign-calibrated threshold)", SERIES[2], -h / 2)]
    for col, label, colour, off in bars:
        vals = t[col].to_numpy()
        ax.barh(y + off, np.nan_to_num(vals), h * 0.9, color=colour,
                edgecolor=SURFACE, linewidth=0.8, label=label)
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(v + 0.01, yi + off, f"{v:.2f}", va="center",
                        fontsize=6.5, color=INK_2)
    ax.set_yticks(y, t.index, fontsize=8.5)
    ax.set_xlabel("recall")
    ax.set_xlim(0, 1.12)
    recede(ax, "x")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=7.5)
    fig.suptitle("Recall by attack type (Random Forest, single source vehicle)",
                 fontsize=11, color=INK, x=0.02, ha="left", y=0.99)
    save(fig, "fig4_per_attack")


def fig_scores(prefix, name, title, logit):
    files = {v: SCORES_DIR / f"{prefix}_{v}.npz" for v in VEHICLES}
    if not all(f.exists() for f in files.values()):
        print(f"  skipped {name}: score files {prefix}_*.npz not found")
        return

    def tx(p):
        p = np.asarray(p, dtype=float)
        if not logit:
            return p
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    fig, axes = plt.subplots(4, 2, figsize=(8.0, 9.0), sharex=True)
    for i, v in enumerate(VEHICLES):
        z = np.load(files[v])
        panels = [
            (axes[i, 0], z["source_score"], z["source_label"],
             [("zs", float(z["thr_zs"]), "-"),
              ("oracle", float(z["thr_oracle_source"]), ":")]),
            (axes[i, 1], z["target_score"], z["target_label"],
             [("zs", float(z["thr_zs"]), "-"),
              ("cal", float(z["thr_cal"]), "--"),
              ("oracle", float(z["thr_oracle"]), ":")]),
        ]
        lo, hi = (-12, 12) if logit else (0, 1)
        bins = np.linspace(lo, hi, 61)
        for ax, s, y, thrs in panels:
            for cls, label, colour in [(0, "benign", SERIES[0]),
                                       (1, "attack", SERIES[1])]:
                vals = np.clip(tx(s[y == cls]), lo, hi)
                if len(vals) == 0:
                    continue
                hist, _ = np.histogram(vals, bins=bins)
                frac = hist / hist.sum()
                ax.stairs(frac, bins, fill=True, color=colour, alpha=0.35)
                ax.stairs(frac, bins, color=colour, linewidth=1.2)
            for tag, thr, ls in thrs:
                if np.isfinite(thr):
                    ax.axvline(float(np.clip(tx(thr), lo, hi)), color=INK,
                               linestyle=ls, linewidth=1.1)
            ax.set_yscale("log")
            ax.set_ylim(1e-5, 1.5)
            recede(ax)
        axes[i, 0].set_ylabel(f"{LABELS[v]} held out\nshare of windows",
                              fontsize=8)
    axes[0, 0].set_title("source vehicles (test recordings)", fontsize=9.5,
                         color=INK_2)
    axes[0, 1].set_title("unseen vehicle", fontsize=9.5, color=INK_2)
    xlabel = "model score, log-odds" if logit else "model score"
    axes[3, 0].set_xlabel(xlabel)
    axes[3, 1].set_xlabel(xlabel)
    handles = [Patch(facecolor=SERIES[0], alpha=0.5, label="benign windows"),
               Patch(facecolor=SERIES[1], alpha=0.5, label="attack windows"),
               Line2D([], [], color=INK, ls="-", label="source-selected threshold"),
               Line2D([], [], color=INK, ls="--", label="target-calibrated threshold"),
               Line2D([], [], color=INK, ls=":", label="oracle threshold")]
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(handles=handles, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.985), fontsize=8)
    fig.suptitle(title, fontsize=11, color=INK, y=1.01, x=0.02, ha="left")
    save(fig, name)


def fig_features(rf):
    cross = rf[(rf["setting"] == "cross") & (rf["mode"] == "pooled_3")]
    metrics = [("auroc", "AUROC"), ("auprc", "AUPRC"),
               ("mcc_zs", "MCC zero-shot"), ("mcc_cal", "MCC calibrated")]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    x = np.arange(len(metrics))
    w = 0.2
    for k, g in enumerate(FEATURE_SETS):
        s = cross[cross["feature_set"] == g]
        vals = [s.groupby("held_out")[m].mean().reindex(VEHICLES).mean()
                for m, _ in metrics]
        pos = x + (k - 1.5) * w
        ax.bar(pos, np.clip(vals, 0, None), w * 0.88, color=SERIES[k],
               edgecolor=SURFACE, linewidth=1.0, label=FEATURE_LABELS[g])
        bar_labels(ax, pos, vals, 6.5)
    ax.set_xticks(x, [m[1] for m in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("mean over held-out vehicles")
    recede(ax)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=8.5,
              title="feature set", title_fontsize=8)
    ax.set_title("Feature representations on the unseen vehicle "
                 "(Random Forest, pooled)", fontsize=11, color=INK,
                 loc="left", pad=36)
    save(fig, "fig6_feature_sets")


def main():
    style()
    rf = flag(pd.read_csv(RESULTS / "rf_results.csv"))
    mlp = flag(pd.read_csv(RESULTS / "mlp_results.csv"))
    if "degenerate" in mlp:
        mlp = mlp[~flag(mlp, "degenerate")["degenerate"]]
    print("building figures ...")
    fig_heatmap(rf)
    fig_thresholds(rf, mlp)
    fig_arms(mlp)
    fig_per_attack()
    fig_scores("rf", "fig5_scores_rf",
               "Score distributions, Random Forest (pooled, selected features)",
               logit=False)
    fig_scores("mlp_pooled", "fig5_scores_mlp",
               "Score distributions, MLP (pooled, first seed)", logit=True)
    fig_scores("mlp_fedavg", "fig5_scores_mlp_fedavg",
               "Score distributions, MLP (FedAvg, first seed)", logit=True)
    fig_features(rf)


if __name__ == "__main__":
    main()
