"""
Final comparison tables and figures: CAN-Train-and-Test against ROAD.

Reads the CAN-Train-and-Test results (results/*.csv, the controlled
leave-one-vehicle-out experiments) and the ROAD results (results/road/),
and writes, in results/road/:

  tables/  comparison_regimes.csv   zero-shot, calibrated, oracle; both datasets
           road_arms.csv            training strategies on ROAD
           road_features.csv        feature sets on ROAD (Random Forest)
           road_calibration.csv     promised vs reached false-alarm rate on ROAD
           road_per_attack.csv      recall per ROAD attack type, zs and cal
           road_false_alarms.csv    false-alarm rates on the attack captures
           road_ambient_by_recording.csv  ambient recordings above each
                                    threshold (descriptive only)
  figures/ figR1_regimes            MCC per threshold rule, both datasets
           figR2_scores_*           score distributions on ROAD
           figR3_arms               training strategies on ROAD
           figR4_per_attack         recall per ROAD attack type

Nothing here is tuned; it only summarises what was run. CAN-Train-and-Test
numbers are means over the four held-out vehicles; ROAD is one vehicle, so
its MLP numbers are means over seeds (seed spread shown as +-).

Run from the project root:
    python src/experiments/analyse_road.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUDGETS, FEATURE_SETS, QUICK, RESULTS, VEHICLES  # noqa: E402
import make_figures as mf  # noqa: E402

ROAD = RESULTS / ("road_quick" if QUICK else "road")
TABLES = ROAD / "tables"
FIGS = ROAD / "figures"
ARMS = ["single", "pooled", "fedavg", "fedprox"]
ARM_LABEL = {"single": "single", "pooled": "pooled", "fedavg": "FedAvg",
             "fedprox": "FedProx"}
METRICS = ["auroc", "auprc", "mcc_zs", "mcc_zsb", "mcc_cal", "mcc_oracle"]


def flag(df, col):
    df[col] = df[col].astype(str).str.lower() == "true"
    return df


def load():
    ctt_rf = flag(pd.read_csv(RESULTS / "rf_results.csv"), "selected")
    ctt_mlp = flag(pd.read_csv(RESULTS / "mlp_results.csv"), "selected")
    road_rf = flag(pd.read_csv(ROAD / "rf_results.csv"), "selected")
    road_mlp = flag(pd.read_csv(ROAD / "mlp_results.csv"), "selected")
    return ctt_rf, ctt_mlp, road_rf, road_mlp


def ctt_value(df, metric):
    """Seeds averaged within each held-out vehicle, then over vehicles."""
    per = df.groupby(["held_out", "seed"])[metric].mean().groupby("held_out").mean()
    return float(per.mean())


def road_value(df, metric):
    """Mean over seeds (rows within a seed averaged first) and seed sd."""
    per = df.groupby("seed")[metric].mean()
    return float(per.mean()), float(per.std(ddof=1)) if len(per) > 1 else 0.0


def configs(ctt_rf, ctt_mlp, road_rf, road_mlp):
    """(name, CTT rows, ROAD rows) for each headline configuration."""
    out = [("RF pooled",
            ctt_rf[(ctt_rf["mode"] == "pooled_3") & ctt_rf["selected"]
                   & (ctt_rf["setting"] == "cross")],
            road_rf[(road_rf["mode"] == "pooled") & road_rf["selected"]
                    & (road_rf["setting"] == "external")])]
    for arm in ARMS:
        out.append((f"MLP {ARM_LABEL[arm]}",
                    ctt_mlp[(ctt_mlp["arm"] == arm) & ctt_mlp["selected"]
                            & (ctt_mlp["setting"] == "cross")],
                    road_mlp[(road_mlp["arm"] == arm) & road_mlp["selected"]
                             & (road_mlp["setting"] == "external")]))
    return out


# ----------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------

def table_selection():
    print("=" * 96 + "\nSETTINGS FOR ROAD, chosen on the four CAN-Train-and-Test "
          "vehicles only\n" + "=" * 96)
    for fam in ("rf", "mlp"):
        sel = pd.read_csv(ROAD / f"{fam}_selected.csv")
        for _, r in sel.iterrows():
            print(f"  {fam.upper():<4}{r['parameter']:<13}{str(r['chosen']):<10}"
                  f"({r['criterion']}: {r['scores']})")


def table_regimes(cfgs):
    print("\n" + "=" * 96 + "\nZERO-SHOT vs BENIGN-CALIBRATED: CAN-Train-and-Test "
          "(mean of 4 held-out vehicles) and ROAD\n" + "=" * 96)
    rows = []
    for name, ctt, road in cfgs:
        entry = {"model": name}
        for m in METRICS:
            entry[f"ctt_{m}"] = ctt_value(ctt, m) if len(ctt) else np.nan
            for s in sorted(road["road_set"].unique()):
                mean, sd = road_value(road[road["road_set"] == s], m)
                entry[f"road_{s}_{m}"] = mean
                entry[f"road_{s}_{m}_seed_sd"] = sd
        rows.append(entry)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "comparison_regimes.csv", index=False)

    sets = [c.split("_")[1] for c in t.columns
            if c.startswith("road_") and c.endswith("_auroc")]
    for label, prefix in [("CAN-Train-and-Test", "ctt")] + \
            [(f"ROAD, {s}", f"road_{s}") for s in sets]:
        print(f"\n  {label}")
        print(f"    {'model':<14}{'AUROC':>7}{'AUPRC':>7}{'zs':>7}{'zsb':>7}"
              f"{'cal':>7}{'oracle':>8}")
        for _, r in t.iterrows():
            print(f"    {r['model']:<14}" + "".join(
                f"{r[f'{prefix}_{m}']:>7.3f}" for m in METRICS[:-1])
                + f"{r[f'{prefix}_mcc_oracle']:>8.3f}")
    return t


def table_arms(road_mlp):
    print("\n" + "=" * 96 + "\nTRAINING STRATEGIES ON ROAD (fabrication), "
          "mean +- seed sd\n" + "=" * 96)
    ext = road_mlp[road_mlp["selected"] & (road_mlp["setting"] == "external")
                   & (road_mlp["road_set"] == "fabrication")]
    rows = []
    print(f"  {'arm':<10}" + "".join(f"{m:>16}" for m in
                                      ["auroc", "auprc", "mcc_zs", "mcc_cal"]))
    for arm in ARMS:
        s = ext[ext["arm"] == arm]
        entry = {"arm": arm}
        cells = ""
        for m in ["auroc", "auprc", "mcc_zs", "mcc_cal"]:
            mean, sd = road_value(s, m)
            entry[m], entry[f"{m}_seed_sd"] = mean, sd
            cells += f"{mean:>10.3f}+-{sd:<4.2f}"
        rows.append(entry)
        print(f"  {ARM_LABEL[arm]:<10}{cells}")
    fp = road_mlp[(road_mlp["arm"] == "fedprox") & (road_mlp["road_set"] == "fabrication")]
    print("\n  FedProx at every mu (MCC cal / AUPRC; * = selected)")
    for mu in sorted(fp["mu"].dropna().unique()):
        s = fp[fp["mu"] == mu]
        mark = "*" if s["selected"].any() else " "
        print(f"    mu {mu:<6}{mark} {road_value(s, 'mcc_cal')[0]:.3f} / "
              f"{road_value(s, 'auprc')[0]:.3f}")
    pd.DataFrame(rows).to_csv(TABLES / "road_arms.csv", index=False)


def table_features(road_rf):
    print("\n" + "=" * 96 + "\nFEATURE SETS ON ROAD (Random Forest pooled, "
          "fabrication)\n" + "=" * 96)
    ext = road_rf[(road_rf["mode"] == "pooled") & (road_rf["setting"] == "external")
                  & (road_rf["road_set"] == "fabrication")]
    rows = []
    print(f"  {'features':<10}{'AUROC':>8}{'AUPRC':>8}{'zs':>8}{'cal':>8}{'oracle':>8}")
    for g in FEATURE_SETS:
        r = ext[ext["feature_set"] == g].iloc[0]
        rows.append({"feature_set": g, **{m: r[m] for m in METRICS}})
        mark = "   <- selected" if r["selected"] else ""
        print(f"  {g:<10}{r['auroc']:>8.3f}{r['auprc']:>8.3f}{r['mcc_zs']:>8.3f}"
              f"{r['mcc_cal']:>8.3f}{r['mcc_oracle']:>8.3f}{mark}")
    pd.DataFrame(rows).to_csv(TABLES / "road_features.csv", index=False)


def table_calibration():
    """Promised vs reached false alarms on ROAD, and the rule on held-out
    ambient recordings (leave one ambient recording out)."""
    print("\n" + "=" * 96 + "\nBENIGN CALIBRATION ON ROAD\n" + "=" * 96)
    rows = []
    print(f"  {'model':<22}{'budget':>8}{'zs flags':>10}{'held-out ambient':>18}"
          f"{'benign in attack':>18}")
    print(f"  {'':<22}{'':>8}{'ambient':>10}{'recording (mean)':>18}"
          f"{'captures':>18}")
    for f in sorted((ROAD / "scores").glob("*_fabrication.npz")):
        z = np.load(f)
        budget = BUDGETS[str(z["budget"])]
        clean, recs = z["target_benign_score"], z["target_benign_recording"]
        s, y = z["target_score"], z["target_label"]
        zs_fpr = float(np.mean(clean >= float(z["thr_zs"])))
        held = []
        for r in np.unique(recs):
            fit = clean[recs != r]
            thr = float(np.quantile(fit, 1 - budget))
            held.append(float(np.mean(clean[recs == r] >= thr)))
        reached = float(np.mean(s[y == 0] >= float(z["thr_cal"])))
        name = f.stem.replace("_fabrication", "")
        rows.append({"model": name, "budget": budget, "zs_fpr_ambient": zs_fpr,
                     "heldout_ambient_fpr_mean": float(np.mean(held)),
                     "heldout_ambient_fpr_max": float(np.max(held)),
                     "reached_fpr_benign_in_attack": reached})
        print(f"  {name:<22}{budget:>8.2%}{zs_fpr:>10.2%}{np.mean(held):>18.3%}"
              f"{reached:>18.2%}")
    pd.DataFrame(rows).to_csv(TABLES / "road_calibration.csv", index=False)


def table_attacks(road_rf, road_mlp):
    """Recall per attack type at the zero-shot and the calibrated threshold,
    with the false-alarm rate on the benign windows of the same captures
    (recall means little without it)."""
    print("\n" + "=" * 96 + "\nRECALL PER ROAD ATTACK TYPE (zero-shot and calibrated "
          "threshold)\n" + "=" * 96)
    rf = flag(pd.read_csv(ROAD / "rf_per_attack.csv"), "selected")
    rf = rf[rf["selected"] & (rf["mode"] == "pooled")]
    mlp = flag(pd.read_csv(ROAD / "mlp_per_attack.csv"), "selected")
    mlp = mlp[mlp["selected"]]
    models = [("RF pooled", rf, road_rf[(road_rf["mode"] == "pooled")
                                        & road_rf["selected"]])]
    for arm in ARMS:
        models.append((f"MLP {ARM_LABEL[arm]}", mlp[mlp["arm"] == arm],
                       road_mlp[(road_mlp["arm"] == arm) & road_mlp["selected"]]))

    t = pd.DataFrame()
    for name, pa, res in models:
        for rule in ("zs", "cal"):
            t[f"{name} {rule}"] = pa.groupby("attack_type")[f"recall_{rule}"].mean()
    t["windows"] = rf.groupby("attack_type")["n_attack_windows"].max()
    t = t.sort_values("MLP FedAvg zs", ascending=False)
    t.to_csv(TABLES / "road_per_attack.csv")

    print("\n  false-alarm rate on the benign windows of the attack captures")
    print(f"    {'model':<14}" + "".join(f"{s + ' ' + r:>18}" for s in
                                         ("fabrication", "masquerade") for r in ("zs", "cal")))
    fpr_rows = []
    for name, _, res in models:
        cells, entry = "", {"model": name}
        for sset in ("fabrication", "masquerade"):
            sub = res[(res["setting"] == "external") & (res["road_set"] == sset)]
            for r in ("zs", "cal"):
                v = sub.groupby("seed")[f"fpr_{r}"].mean().mean() if len(sub) else np.nan
                entry[f"{sset}_fpr_{r}"] = v
                cells += f"{v:>18.2%}"
        fpr_rows.append(entry)
        print(f"    {name:<14}{cells}")
    pd.DataFrame(fpr_rows).to_csv(TABLES / "road_false_alarms.csv", index=False)

    show = ["RF pooled zs", "RF pooled cal", "MLP pooled zs", "MLP pooled cal",
            "MLP FedAvg zs", "MLP FedAvg cal", "windows"]
    print("\n  recall per attack type")
    print("  " + t[show].round(3).to_string().replace("\n", "\n  "))
    return t


def table_ambient():
    """Which ambient recordings lie above each threshold (descriptive only)."""
    print("\n" + "=" * 96 + "\nAMBIENT RECORDINGS ABOVE EACH THRESHOLD "
          "(share of the recording's windows; descriptive)\n" + "=" * 96)
    rows = []
    for prefix, name in [("rf_pooled", "RF pooled"), ("mlp_pooled", "MLP pooled"),
                         ("mlp_fedavg", "MLP FedAvg")]:
        f = ROAD / "scores" / f"{prefix}_fabrication.npz"
        if not f.exists():
            continue
        z = np.load(f)
        clean, recs = z["target_benign_score"], z["target_benign_recording"]
        att = z["target_score"][z["target_label"] == 1]
        ben = z["target_score"][z["target_label"] == 0]
        thr_zs, thr_cal = float(z["thr_zs"]), float(z["thr_cal"])
        print(f"\n  {name}: attack windows above cal {np.mean(att >= thr_cal):.2%}, "
              f"above zs {np.mean(att >= thr_zs):.2%}; benign windows of the attack "
              f"captures above zs {np.mean(ben >= thr_zs):.2%}")
        print(f"    {'ambient recording':<46}{'windows':>9}{'above zs':>10}"
              f"{'above cal':>11}{'top 0.01% here':>16}")
        n_top = max(1, int(round(len(clean) * BUDGETS[str(z['budget'])])))
        top = recs[np.argsort(clean)[-n_top:]]
        for r in np.unique(recs):
            m = recs == r
            entry = {"model": name, "recording": r.replace("__ambient", ""),
                     "windows": int(m.sum()),
                     "share_above_zs": float(np.mean(clean[m] >= thr_zs)),
                     "share_above_cal": float(np.mean(clean[m] >= thr_cal)),
                     "windows_in_top_budget": int(np.sum(top == r))}
            rows.append(entry)
            print(f"    {entry['recording']:<46}{entry['windows']:>9,}"
                  f"{entry['share_above_zs']:>10.1%}{entry['share_above_cal']:>11.3%}"
                  f"{entry['windows_in_top_budget']:>16}")
    pd.DataFrame(rows).to_csv(TABLES / "road_ambient_by_recording.csv", index=False)


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------

def fig_regimes(t):
    models = ["RF pooled", "MLP pooled", "MLP FedAvg"]
    rules = mf.RULES
    fig, axes = plt.subplots(1, len(models), figsize=(10.0, 3.5), sharey=True)
    x = np.arange(2)
    w = 0.2
    for ax, model in zip(axes, models):
        r = t[t["model"] == model].iloc[0]
        for k, (col, label, colour, hatch) in enumerate(rules):
            vals = [r[f"ctt_{col}"], r[f"road_fabrication_{col}"]]
            pos = x + (k - 1.5) * w
            ax.bar(pos, np.clip(vals, 0, None), w * 0.88, color=colour,
                   edgecolor=mf.SURFACE, linewidth=1.0, hatch=hatch,
                   label=label if ax is axes[0] else None)
            mf.bar_labels(ax, pos, vals, 6.0)
        ax.set_xticks(x, ["CAN-Train-and-Test\n(mean of 4 held out)", "ROAD"],
                      fontsize=8)
        ax.set_title(model, fontsize=9.5, color=mf.INK_2)
        ax.set_ylim(0, 1.05)
        mf.recede(ax)
    axes[0].set_ylabel("MCC on the unseen vehicle")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.95), fontsize=8)
    fig.suptitle("Threshold rules on CAN-Train-and-Test and on ROAD",
                 fontsize=11, color=mf.INK, y=1.02, x=0.02, ha="left")
    mf.save(fig, "figR1_regimes")


def fig_scores(prefix, name, title, logit):
    f = ROAD / "scores" / f"{prefix}_fabrication.npz"
    if not f.exists():
        return
    z = np.load(f)

    def tx(p):
        p = np.asarray(p, dtype=float)
        if not logit:
            return p
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    lo, hi = (-12, 12) if logit else (0, 1)
    bins = np.linspace(lo, hi, 61)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.8), sharex=True)
    panels = [(axes[0], "CAN-Train-and-Test (test recordings)", z["source_score"],
               z["source_label"], [(float(z["thr_zs"]), "-"),
                                   (float(z["thr_oracle_source"]), ":")]),
              (axes[1], "ROAD (attack captures)", z["target_score"],
               z["target_label"], [(float(z["thr_zs"]), "-"),
                                   (float(z["thr_cal"]), "--"),
                                   (float(z["thr_oracle"]), ":")])]
    for ax, sub, s, y, thrs in panels:
        for cls, colour in [(0, mf.SERIES[0]), (1, mf.SERIES[1])]:
            vals = np.clip(tx(s[y == cls]), lo, hi)
            if len(vals) == 0:
                continue
            hist, _ = np.histogram(vals, bins=bins)
            frac = hist / hist.sum()
            ax.stairs(frac, bins, fill=True, color=colour, alpha=0.35)
            ax.stairs(frac, bins, color=colour, linewidth=1.2)
        for thr, ls in thrs:
            if np.isfinite(thr):
                ax.axvline(float(np.clip(tx(thr), lo, hi)), color=mf.INK,
                           linestyle=ls, linewidth=1.1)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1.5)
        ax.set_title(sub, fontsize=9, color=mf.INK_2)
        ax.set_xlabel("model score, log-odds" if logit else "model score")
        mf.recede(ax)
    axes[0].set_ylabel("share of windows")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=mf.SERIES[0], alpha=0.5, label="benign windows"),
               Patch(facecolor=mf.SERIES[1], alpha=0.5, label="attack windows"),
               Line2D([], [], color=mf.INK, ls="-", label="source-selected"),
               Line2D([], [], color=mf.INK, ls="--", label="target-calibrated"),
               Line2D([], [], color=mf.INK, ls=":", label="oracle")]
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    fig.legend(handles=handles, ncol=5, loc="upper center",
               bbox_to_anchor=(0.5, 0.93), fontsize=7.5)
    fig.suptitle(title, fontsize=10.5, color=mf.INK, y=1.02, x=0.02, ha="left")
    mf.save(fig, name)


def fig_arms(road_mlp):
    ext = road_mlp[road_mlp["selected"] & (road_mlp["setting"] == "external")
                   & (road_mlp["road_set"] == "fabrication")]
    metrics = [("auprc", "AUPRC"), ("mcc_zs", "MCC zero-shot"),
               ("mcc_cal", "MCC calibrated")]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(metrics))
    w = 0.19
    rng = np.random.default_rng(0)
    for k, (arm, label, colour) in enumerate(mf.ARMS):
        s = ext[ext["arm"] == arm]
        for i, (m, _) in enumerate(metrics):
            vals = s.groupby("seed")[m].mean().to_numpy()
            pos = x[i] + (k - 1.5) * w
            ax.scatter(pos + rng.uniform(-w * 0.25, w * 0.25, len(vals)), vals,
                       s=9, color=colour, alpha=0.35, linewidths=0)
            ax.scatter([pos], [vals.mean()], s=46, color=colour,
                       edgecolors=mf.SURFACE, linewidths=1.5, zorder=3,
                       label=label if i == 0 else None)
    ax.set_xticks(x, [m[1] for m in metrics])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("value on ROAD")
    mf.recede(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.95), fontsize=8)
    fig.suptitle("Training strategies on ROAD (large dot = mean over seeds)",
                 fontsize=10.5, color=mf.INK, y=1.02, x=0.02, ha="left")
    mf.save(fig, "figR3_arms")


def fig_attacks(t):
    """Fabrication attack types, zero-shot and calibrated threshold."""
    t = t[~t.index.str.endswith("_masquerade")]
    models = ["RF pooled", "MLP pooled", "MLP FedAvg"]
    t = t.sort_values("MLP FedAvg zs")
    y = np.arange(len(t))
    h = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 0.45 * len(t) + 1.6), sharey=True)
    for ax, rule, title in [(axes[0], "zs", "zero-shot threshold"),
                            (axes[1], "cal", "benign-calibrated threshold")]:
        for k, (m, colour) in enumerate(zip(models, mf.SERIES)):
            off = (1 - k) * h
            vals = t[f"{m} {rule}"].to_numpy()
            ax.barh(y + off, np.nan_to_num(vals), h * 0.9, color=colour,
                    edgecolor=mf.SURFACE, linewidth=0.8,
                    label=m if ax is axes[0] else None)
            for yi, v in zip(y, vals):
                if np.isfinite(v):
                    ax.text(v + 0.01, yi + off, f"{v:.2f}", va="center",
                            fontsize=6.0, color=mf.INK_2)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("recall")
        ax.set_title(title, fontsize=9.5, color=mf.INK_2)
        mf.recede(ax, "x")
    axes[0].set_yticks(y, t.index, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.95), fontsize=8)
    fig.suptitle("Recall per ROAD attack type (fabrication); read with the "
                 "false-alarm rates in the table", fontsize=10.5, color=mf.INK,
                 x=0.02, ha="left", y=1.0)
    mf.save(fig, "figR4_per_attack")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    mf.FIGURES = FIGS
    mf.style()
    ctt_rf, ctt_mlp, road_rf, road_mlp = load()
    road_mlp = road_mlp[~flag(road_mlp, "degenerate")["degenerate"]] \
        if "degenerate" in road_mlp else road_mlp

    table_selection()
    t = table_regimes(configs(ctt_rf, ctt_mlp, road_rf, road_mlp))
    table_arms(road_mlp)
    table_features(road_rf)
    table_calibration()
    ta = table_attacks(road_rf, road_mlp)
    table_ambient()

    print("\nbuilding figures ...")
    fig_regimes(t)
    fig_scores("rf_pooled", "figR2_scores_rf", "Scores on ROAD, Random Forest (pooled)",
               logit=False)
    fig_scores("mlp_pooled", "figR2_scores_mlp_pooled",
               "Scores on ROAD, MLP (pooled, first seed)", logit=True)
    fig_scores("mlp_fedavg", "figR2_scores_mlp_fedavg",
               "Scores on ROAD, MLP (FedAvg, first seed)", logit=True)
    fig_arms(road_mlp)
    fig_attacks(ta)
    print(f"\n  tables in {TABLES}/, figures in {FIGS}/")


if __name__ == "__main__":
    main()
