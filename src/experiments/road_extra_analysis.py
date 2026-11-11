"""
Extra ROAD analyses asked for by the supervisor (email of 30 September).
Nothing is trained or tuned here: the script only reads the ROAD results and
per-window scores that run_road.py already saved, and describes them.

1. Why benign calibration fails on ROAD
   For each saved model (Random Forest pooled, MLP pooled, FedAvg, FedProx;
   the MLP scores are from the first seed), the scores of three groups of
   windows are compared:
     ambient            ROAD's attack-free recordings, the data the
                        calibrated threshold is set on
     normal in captures the benign windows inside the attack captures
     attack             the attack windows
   Figure: distribution (left) and tail view (right: share of windows at or
   above each score, on a log scale), with the zero-shot and calibrated
   thresholds and the budget B.
   Table:  road_score_separation.csv

2. ROAD results per attack type, marked by whether the attack type was
   represented in the CAN-Train-and-Test source data (ATTACK_MAP below).
   Recall at the zero-shot and calibrated thresholds for all five models
   (mean over seeds; MLP single also over its four single-vehicle models),
   and a threshold-free AUROC per attack type for the four models whose
   per-window scores were saved (first seed for the MLPs).
   Tables: road_per_attack_full.csv, road_per_attack_by_group.csv

Outputs go to results/road/tables/ and results/road/figures/ (road_quick/
with QUICK=1).

Run from the project root, after run_road.py and analyse_road.py:
    python src/experiments/road_extra_analysis.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUDGETS, QUICK, RESULTS  # noqa: E402
import make_figures as mf  # noqa: E402

ROAD = RESULTS / ("road_quick" if QUICK else "road")
TABLES = ROAD / "tables"
FIGS = ROAD / "figures"

# models with saved per-window scores (first seed for the MLPs)
SCORED_MODELS = [("rf_pooled", "RF pooled"), ("mlp_pooled", "MLP pooled"),
                 ("mlp_fedavg", "MLP FedAvg"), ("mlp_fedprox", "MLP FedProx")]
SETS = ["fabrication", "masquerade"]

# ----------------------------------------------------------------------
# Was each ROAD attack type represented in the CAN-Train-and-Test sources?
#
# Decided from the published descriptions of the two datasets, not from
# any result:
#   CAN-Train-and-Test (Lampe & Meng): every attack injects extra frames
#     (DoS, fuzzing, RPM/speed/gear spoofing, double and triple spoofing,
#     interval, standstill, systematic). There are no masquerade attacks.
#   ROAD (Verma et al.): fuzzing injects frames with random IDs and
#     payloads; the other fabrication attacks inject one forged frame per
#     legitimate frame of the target ID; the masquerade versions are the
#     same captures with the legitimate target frames removed.
#
#   yes     same attack type in CAN-Train-and-Test
#   partly  same kind of attack on the same kind of signal, different value
#   no      target signal or attack method not in CAN-Train-and-Test
# ----------------------------------------------------------------------
ATTACK_MAP = {
    # revised on 5 October: mechanism AND targeted signal (Table 5.6)
    "fuzzing": ("represented", "fuzzing",
                "injection; random IDs and payloads, as in CAN-Train-and-Test"),
    "max_speedometer": ("represented", "speed spoofing",
                        "injection; same signal (vehicle speed)"),
    "correlated_signal": ("partly", "speed spoofing",
                          "injection; different signal (four wheel speeds)"),
    "max_engine_coolant_temp": ("partly", "none",
                                "injection; different signal (coolant temperature)"),
    "reverse_light_on": ("partly", "gear spoofing",
                         "injection; different signal (reverse light)"),
    "reverse_light_off": ("partly", "gear spoofing",
                          "injection; different signal (reverse light)"),
}
MASQ_NOTE = ("masquerade: legitimate frames replaced, not added; "
             "CAN-Train-and-Test has no masquerade attacks")


def represented(attack_type):
    """(represented, closest CTT attack, note) for one ROAD attack type."""
    if attack_type.endswith("_masquerade"):
        return "not", "none (no masquerade attacks)", MASQ_NOTE
    if attack_type in ATTACK_MAP:
        return ATTACK_MAP[attack_type]
    return "UNMAPPED", "?", "attack type not in ATTACK_MAP: add it before use"


def flag(df, col):
    df[col] = df[col].astype(str).str.lower() == "true"
    return df


def load_npz(prefix, sset):
    f = ROAD / "scores" / f"{prefix}_{sset}.npz"
    return np.load(f) if f.exists() else None


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ----------------------------------------------------------------------
# 1. score separation and the calibration figure
# ----------------------------------------------------------------------

def groups(z):
    s, y = z["target_score"].astype(float), z["target_label"]
    return {"ambient": z["target_benign_score"].astype(float),
            "normal in captures": s[y == 0],
            "attack": s[y == 1]}


def table_separation():
    print("\n" + "=" * 96 + "\n1. SCORE SEPARATION ON ROAD (why calibration fails)\n"
          "   first seed for the MLPs; 'ambient' = the recordings the calibrated "
          "threshold is set on\n" + "=" * 96)
    rows = []
    for sset in SETS:
        for prefix, name in SCORED_MODELS:
            z = load_npz(prefix, sset)
            if z is None:
                print(f"  (no scores for {prefix} {sset})")
                continue
            g = groups(z)
            budget = BUDGETS[str(z["budget"])]
            thr_zs, thr_cal = float(z["thr_zs"]), float(z["thr_cal"])
            att, amb, cap = g["attack"], g["ambient"], g["normal in captures"]
            med_att = float(np.median(att))
            row = {"road_set": sset, "model": name, "budget": budget,
                   "n_ambient": len(amb), "n_normal_in_captures": len(cap),
                   "n_attack": len(att), "thr_zs": thr_zs, "thr_cal": thr_cal}
            for key, v in g.items():
                k = key.replace(" ", "_")
                row[f"{k}_above_zs"] = float(np.mean(v >= thr_zs))
                row[f"{k}_above_cal"] = float(np.mean(v >= thr_cal))
                for q in (0.5, 0.9, 0.99, 0.999, 0.9999):
                    row[f"{k}_q{q}"] = float(np.quantile(v, q))
            both = np.concatenate([att, amb])
            lab = np.r_[np.ones(len(att)), np.zeros(len(amb))]
            row["auroc_attack_vs_ambient"] = float(roc_auc_score(lab, both))
            both = np.concatenate([att, cap])
            lab = np.r_[np.ones(len(att)), np.zeros(len(cap))]
            row["auroc_attack_vs_normal_in_captures"] = float(roc_auc_score(lab, both))
            # how unusual is a typical attack window, measured on normal driving?
            row["ambient_share_at_or_above_median_attack"] = float(np.mean(amb >= med_att))
            rows.append(row)

            print(f"\n  {sset:<12}{name}")
            print(f"    windows: ambient {len(amb):,}, normal in captures "
                  f"{len(cap):,}, attack {len(att):,}")
            print(f"    {'':<22}{'above zs':>10}{'above cal':>11}"
                  f"{'median':>10}{'99th pct':>10}{'99.99th pct':>13}")
            for key, v in g.items():
                k = key.replace(" ", "_")
                print(f"    {key:<22}{row[f'{k}_above_zs']:>10.2%}"
                      f"{row[f'{k}_above_cal']:>11.3%}{row[f'{k}_q0.5']:>10.4f}"
                      f"{row[f'{k}_q0.99']:>10.4f}{row[f'{k}_q0.9999']:>13.4f}")
            print(f"    AUROC attack vs ambient {row['auroc_attack_vs_ambient']:.3f}, "
                  f"vs normal in captures {row['auroc_attack_vs_normal_in_captures']:.3f}")
            print(f"    a typical (median) attack window scores lower than "
                  f"{row['ambient_share_at_or_above_median_attack']:.1%} of the "
                  f"ambient windows; the calibrated threshold needs it above "
                  f"all but {budget:.2%} of them")
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "road_score_separation.csv", index=False)
    return t


def fig_calibration(prefix, name, sset, fig_name):
    z = load_npz(prefix, sset)
    if z is None:
        return
    g = groups(z)
    budget = BUDGETS[str(z["budget"])]
    thr_zs, thr_cal = float(z["thr_zs"]), float(z["thr_cal"])
    # colour and line style both carry the group, so identity is not colour alone
    style = {"ambient": (mf.SERIES[0], "-", "ambient recordings (calibration data)"),
             "normal in captures": (mf.SERIES[2], "--", "normal windows in attack captures"),
             "attack": (mf.SERIES[1], "-", "attack windows")}
    lo, hi = -14.0, 14.0
    bins = np.linspace(lo, hi, 71)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.1))
    for key, v in g.items():
        colour, ls, _ = style[key]
        x = np.clip(logit(v), lo, hi)
        hist, _ = np.histogram(x, bins=bins)
        frac = hist / max(hist.sum(), 1)
        if key == "attack":
            ax1.stairs(frac, bins, fill=True, color=colour, alpha=0.25)
        ax1.stairs(frac, bins, color=colour, linestyle=ls, linewidth=1.4)
        xs = np.sort(x)
        surv = (len(xs) - np.arange(len(xs))) / len(xs)
        ax2.step(xs, surv, where="post", color=colour, linestyle=ls, linewidth=1.6)
    xs_all = np.concatenate([np.clip(logit(v), lo, hi) for v in g.values()])
    x_lo, x_hi = float(xs_all.min()) - 0.5, float(xs_all.max()) + 0.5
    for ax in (ax1, ax2):
        ax.set_xlim(x_lo, x_hi)
        ax.axvline(float(np.clip(logit(thr_zs), lo, hi)), color=mf.INK,
                   linestyle="-", linewidth=1.0)
        ax.axvline(float(np.clip(logit(thr_cal), lo, hi)), color=mf.INK,
                   linestyle=(0, (4, 2)), linewidth=1.0)
        ax.set_yscale("log")
        ax.set_xlabel("model score (log-odds)")
        mf.recede(ax)
    n_min = min(len(v) for v in g.values())
    ax1.set_ylim(1e-5, 1.5)
    ax1.set_ylabel("share of windows")
    ax1.set_title("score distribution", fontsize=9, color=mf.INK_2)
    ax2.axhline(budget, color=mf.INK_MUTED, linestyle=":", linewidth=1.0)
    ax2.text(0.02, budget * 1.4, f"budget B = {budget:.2%}", fontsize=7,
             color=mf.INK_2, transform=ax2.get_yaxis_transform())
    ax2.set_ylim(min(0.5 / n_min, budget / 3), 1.5)
    ax2.set_ylabel("share of windows at or above score")
    ax2.set_title("tail: where the thresholds sit", fontsize=9, color=mf.INK_2)
    above = "\n".join(f"{style[k][2].split(' (')[0]}: {np.mean(v >= thr_cal):.3%}"
                      for k, v in g.items())
    ax2.text(0.02, 0.30, "above the calibrated threshold\n" + above,
             transform=ax2.transAxes, ha="left", va="bottom", fontsize=6.8,
             color=mf.INK, bbox=dict(facecolor=mf.SURFACE, edgecolor=mf.GRID,
                                     boxstyle="round,pad=0.3"))
    handles = [Line2D([], [], color=style[k][0], linestyle=style[k][1], lw=1.6,
                      label=style[k][2]) for k in style]
    handles += [Line2D([], [], color=mf.INK, ls="-", lw=1.0,
                       label="zero-shot threshold"),
                Line2D([], [], color=mf.INK, ls=(0, (4, 2)), lw=1.0,
                       label="calibrated threshold")]
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    fig.legend(handles=handles, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.92), fontsize=7.2)
    seed_note = "" if prefix.startswith("rf") else ", first seed"
    fig.suptitle(f"ROAD {sset}: {name}{seed_note}", fontsize=10.5, color=mf.INK,
                 y=1.0, x=0.02, ha="left")
    mf.save(fig, fig_name)


# ----------------------------------------------------------------------
# 2. per attack type, with representation in CAN-Train-and-Test
# ----------------------------------------------------------------------

def per_attack_recall():
    rf = flag(pd.read_csv(ROAD / "rf_per_attack.csv"), "selected")
    rf = rf[rf["selected"] & (rf["mode"] == "pooled")]
    mlp = flag(pd.read_csv(ROAD / "mlp_per_attack.csv"), "selected")
    mlp = mlp[mlp["selected"]]
    models = [("RF pooled", rf)] + [
        (f"MLP {lab}", mlp[mlp["arm"] == arm])
        for arm, lab in [("single", "single"), ("pooled", "pooled"),
                         ("fedavg", "FedAvg"), ("fedprox", "FedProx")]]
    windows = rf.groupby("attack_type")["n_attack_windows"].max()
    sets = rf.groupby("attack_type")["road_set"].first()
    return models, windows, sets


def false_alarm_context():
    """False-alarm rate on the benign windows of the captures, per model."""
    rf = flag(pd.read_csv(ROAD / "rf_results.csv"), "selected")
    rf = rf[rf["selected"] & (rf["mode"] == "pooled")]
    mlp = flag(pd.read_csv(ROAD / "mlp_results.csv"), "selected")
    mlp = mlp[mlp["selected"]]
    out = {}
    for name, res in [("RF pooled", rf)] + [
            (f"MLP {lab}", mlp[mlp["arm"] == arm])
            for arm, lab in [("single", "single"), ("pooled", "pooled"),
                             ("fedavg", "FedAvg"), ("fedprox", "FedProx")]]:
        for sset in SETS:
            sub = res[(res["setting"] == "external") & (res["road_set"] == sset)]
            for r in ("zs", "cal"):
                out[(name, sset, r)] = (sub.groupby("seed")[f"fpr_{r}"].mean().mean()
                                        if len(sub) else np.nan)
    return out


def table_per_attack():
    print("\n" + "=" * 96 + "\n2. ROAD RESULTS PER ATTACK TYPE, with representation in "
          "CAN-Train-and-Test\n" + "=" * 96)
    models, windows, sets = per_attack_recall()
    rows = []
    for atk in windows.index:
        rep, closest, note = represented(atk)
        row = {"road_set": sets[atk], "attack_type": atk,
               "attack_windows": int(windows[atk]), "represented_in_ctt": rep,
               "closest_ctt_attack": closest, "note": note}
        for name, pa in models:
            sub = pa[pa["attack_type"] == atk]
            for r in ("zs", "cal"):
                row[f"{name} recall_{r}"] = (float(sub[f"recall_{r}"].mean())
                                             if len(sub) else np.nan)
        for prefix, name in SCORED_MODELS:
            z = load_npz(prefix, sets[atk])
            if z is None:
                row[f"{name} auroc"] = np.nan
                continue
            y, a, s = z["target_label"], z["target_attack"], z["target_score"]
            pos = (y == 1) & (a == atk)
            neg = y == 0
            lab = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
            row[f"{name} auroc"] = (float(roc_auc_score(lab, np.r_[s[pos], s[neg]]))
                                    if pos.any() and neg.any() else np.nan)
        rows.append(row)
    t = pd.DataFrame(rows)
    order = {"represented": 0, "partly": 1, "not": 2, "UNMAPPED": 3}
    t = t.sort_values(["road_set", "represented_in_ctt", "attack_type"],
                      key=lambda c: c.map(order) if c.name == "represented_in_ctt"
                      else c).reset_index(drop=True)
    t.to_csv(TABLES / "road_per_attack_full.csv", index=False)

    if (t["represented_in_ctt"] == "UNMAPPED").any():
        print("  WARNING: some attack types are not in ATTACK_MAP: "
              + ", ".join(t.loc[t["represented_in_ctt"] == "UNMAPPED", "attack_type"]))

    fpr = false_alarm_context()
    names = [m for m, _ in models]
    for sset in SETS:
        sub = t[t["road_set"] == sset]
        if sub.empty:
            continue
        print(f"\n  {sset.upper()}")
        print("  false-alarm rate on the benign windows of these captures:  "
              + ";  ".join(f"{n} zs {fpr[(n, sset, 'zs')]:.1%} / cal "
                           f"{fpr[(n, sset, 'cal')]:.3%}" for n in names))
        print(f"\n  {'attack type':<36}{'in CTT':<8}{'windows':>8}  "
              + "".join(f"{n.replace('MLP ', ''):>12}" for n in names))
        for label, suffix, fmt in [("recall at zero-shot", "recall_zs", "{:>12.3f}"),
                                   ("recall at calibrated", "recall_cal", "{:>12.3f}"),
                                   ("AUROC (threshold-free)", "auroc", "{:>12.3f}")]:
            print(f"  -- {label}")
            for _, r in sub.iterrows():
                cells = ""
                for n in names:
                    v = r.get(f"{n} {suffix}", np.nan)
                    cells += fmt.format(v) if pd.notna(v) else f"{'-':>12}"
                print(f"  {r['attack_type']:<36}{r['represented_in_ctt']:<8}"
                      f"{r['attack_windows']:>8,}  {cells}")
    print("\n  closest CAN-Train-and-Test attack and reason:")
    for _, r in t.iterrows():
        print(f"    {r['attack_type']:<36}{r['represented_in_ctt']:<8}"
              f"{r['closest_ctt_attack']}: {r['note']}")

    # descriptive summary by group; few attack types per group, so no claim
    auroc_cols = [f"{n} auroc" for _, n in SCORED_MODELS]
    g = (t.groupby(["road_set", "represented_in_ctt"])
          .agg(attack_types=("attack_type", "count"),
               attack_windows=("attack_windows", "sum"),
               **{c.replace(" ", "_"): (c, "mean") for c in auroc_cols})
          .reset_index())
    g.to_csv(TABLES / "road_per_attack_by_group.csv", index=False)
    print("\n  mean AUROC per group (descriptive only: 1-5 attack types per group)")
    print("  " + g.round(3).to_string(index=False).replace("\n", "\n  "))
    return t


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    mf.FIGURES = FIGS
    mf.style()
    table_separation()
    table_per_attack()
    print("\nbuilding figures ...")
    for prefix, name in SCORED_MODELS:
        fig_calibration(prefix, name, "fabrication", f"figR5_calibration_{prefix}")
        fig_calibration(prefix, name, "masquerade", f"figR6_masquerade_{prefix}")
    print(f"\n  tables in {TABLES}/, figures in {FIGS}/")


if __name__ == "__main__":
    main()
