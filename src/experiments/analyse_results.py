"""
Tables and vehicle-level comparisons from the Random Forest and MLP runs.

Reads results/rf_*.csv and results/mlp_*.csv and writes results/tables/.

How differences are judged
--------------------------
There are four vehicles, so there are four independent observations. The
seeds of the MLP runs are repeated measurements on the same four vehicles,
not extra vehicles, so they are averaged within each vehicle first. Every
comparison then reports:

  - the difference for each held-out vehicle (seed-averaged)
  - how many of the four vehicles move in the same direction
  - an exact sign-flip test on the four differences. With four vehicles the
    smallest possible two-sided p-value is 2/16 = 0.125, so no comparison
    here can reach p < 0.05. That is a property of the design, and it is
    stated rather than worked around.
  - for the MLP, whether each vehicle's difference is stable across seeds
    (larger than twice its standard error over seeds). This says the
    difference is not a training accident on that vehicle; it does not say
    it generalises to other vehicles.

Run from the project root, after run_baselines.py and run_federated.py:
    python src/experiments/analyse_results.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURE_SETS, RESULTS, VEHICLES  # noqa: E402

TABLES = RESULTS / "tables"
ARMS = ["single", "pooled", "fedavg", "fedprox"]
RF_MODES = ["single", "pooled_3", "pooled_3_balanced", "pooled_3_matched"]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def sign_flip_p(d):
    """Exact two-sided sign-flip test on the mean of d."""
    d = np.asarray([x for x in d if np.isfinite(x)])
    if len(d) == 0:
        return np.nan
    obs = abs(d.mean())
    flips = [abs((d * np.array(s)).mean())
             for s in itertools.product([1, -1], repeat=len(d))]
    return float(np.mean([f >= obs - 1e-12 for f in flips]))


def per_seed(df, metric):
    """One value per (held_out, seed): rows within a seed are averaged first
    (this is where the three single-vehicle models become one number)."""
    return df.groupby(["held_out", "seed"])[metric].mean()


def per_vehicle(df, metric):
    return per_seed(df, metric).groupby("held_out").mean().reindex(VEHICLES)


def compare(a_df, b_df, metric, name_a, name_b):
    a, b = per_seed(a_df, metric), per_seed(b_df, metric)
    diff = (a - b).dropna()
    by_v = diff.groupby("held_out")
    mean_v = by_v.mean().reindex(VEHICLES)
    n_seeds = by_v.size().reindex(VEHICLES)
    sd_v = by_v.std(ddof=1).reindex(VEHICLES)
    se_v = sd_v / np.sqrt(n_seeds)
    stable = (mean_v.abs() > 2 * se_v) if n_seeds.max() > 1 else \
        pd.Series(np.nan, index=VEHICLES)

    overall = float(np.nanmean(mean_v))
    same = (int(np.sum(np.sign(mean_v.dropna()) == np.sign(overall)))
            if overall != 0 else 0)
    row = {"comparison": f"{name_a} - {name_b}", "metric": metric,
           "mean_diff": overall,
           "vehicles_same_sign": f"{same}/{mean_v.notna().sum()}",
           "sign_flip_p": sign_flip_p(mean_v.to_numpy())}
    for v in VEHICLES:
        row[f"diff_{v}"] = mean_v[v]
        row[f"seed_sd_{v}"] = sd_v[v]
        row[f"stable_{v}"] = stable[v]
    return row


def print_comparisons(rows, title):
    print(f"\n{title}")
    print(f"  {'comparison':<30}{'metric':<9}" + "".join(f"{v[:9]:>10}" for v in VEHICLES)
          + f"{'mean':>8}{'same sign':>11}{'p':>7}")
    for r in rows:
        cells = ""
        for v in VEHICLES:
            d = r[f"diff_{v}"]
            mark = "*" if r[f"stable_{v}"] is True or r[f"stable_{v}"] == 1.0 else " "
            cells += f"{d:>+9.3f}{mark}" if np.isfinite(d) else f"{'--':>10}"
        print(f"  {r['comparison']:<30}{r['metric']:<9}{cells}"
              f"{r['mean_diff']:>+8.3f}{r['vehicles_same_sign']:>11}"
              f"{r['sign_flip_p']:>7.3f}")


def load():
    rf = pd.read_csv(RESULTS / "rf_results.csv")
    mlp = pd.read_csv(RESULTS / "mlp_results.csv")
    for df in (rf, mlp):
        df["selected"] = df["selected"].astype(str).str.lower() == "true"
    if "degenerate" in mlp:
        mlp["degenerate"] = mlp["degenerate"].astype(str).str.lower() == "true"
        mlp = mlp[~mlp["degenerate"]]
    return rf, mlp


# ----------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------

def selections():
    frames = [pd.read_csv(RESULTS / f) for f in
              ("rf_selected.csv", "mlp_selected.csv") if (RESULTS / f).exists()]
    sel = pd.concat(frames, ignore_index=True)
    sel.to_csv(TABLES / "selections.csv", index=False)
    print("\n" + "=" * 90)
    print("A. WHAT WAS SELECTED, ON SOURCE VEHICLES ONLY")
    print("=" * 90)
    piv = sel.pivot_table(index=["family", "parameter"], columns="held_out",
                          values="chosen", aggfunc="first").reindex(columns=VEHICLES)
    print(piv.to_string())


def regimes(rf, mlp):
    """Zero-shot vs calibrated, with the source-side numbers next to them."""
    print("\n" + "=" * 90)
    print("B. ZERO-SHOT vs BENIGN-CALIBRATED  (headline configurations)")
    print("=" * 90)

    configs = [("RF pooled_3", rf, rf["mode"] == "pooled_3")]
    for arm in ARMS:
        configs.append((f"MLP {arm}", mlp, mlp["arm"] == arm))

    rows = []
    for name, df, mask in configs:
        cross = df[mask & df["selected"] & (df["setting"] == "cross")]
        src = df[mask & df["selected"] & (df["setting"] == "within_sources")]
        for v in VEHICLES:
            c = cross[cross["held_out"] == v]
            s = src[src["held_out"] == v]
            if not len(c):
                continue
            row = {"model": name, "held_out": v}
            for m in ["auroc", "auprc", "mcc_05", "mcc_zs", "mcc_zsb",
                      "mcc_cal", "mcc_oracle", "fpr_zs", "fpr_cal",
                      "recall_zs", "recall_cal"]:
                row[m] = per_seed(c, m).mean()
            for m in ["auroc", "auprc", "mcc_zs", "mcc_oracle"]:
                row[f"source_{m}"] = per_seed(s, m).mean() if len(s) else np.nan
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "regimes.csv", index=False)

    for name in t["model"].unique():
        s = t[t["model"] == name]
        print(f"\n  {name}")
        print(f"    {'held out':<11}{'src MCC':>8}{'src AUPRC':>10}  |"
              f"{'AUROC':>7}{'AUPRC':>7}{'zs':>7}{'zsb':>7}{'cal':>7}{'oracle':>8}")
        for _, r in s.iterrows():
            print(f"    {r['held_out']:<11}{r['source_mcc_zs']:>8.3f}"
                  f"{r['source_auprc']:>10.3f}  |{r['auroc']:>7.3f}{r['auprc']:>7.3f}"
                  f"{r['mcc_zs']:>7.3f}{r['mcc_zsb']:>7.3f}{r['mcc_cal']:>7.3f}"
                  f"{r['mcc_oracle']:>8.3f}")
        m = s.mean(numeric_only=True)
        print(f"    {'MEAN':<11}{m['source_mcc_zs']:>8.3f}{m['source_auprc']:>10.3f}  |"
              f"{m['auroc']:>7.3f}{m['auprc']:>7.3f}{m['mcc_zs']:>7.3f}"
              f"{m['mcc_zsb']:>7.3f}{m['mcc_cal']:>7.3f}{m['mcc_oracle']:>8.3f}")

    # the gap, split into parts that can each be measured
    print("\n  Parts of the cross-vehicle gap (mean over held-out vehicles, MCC"
          "\n  unless stated). Each is one measurable difference, not a share of"
          "\n  a total.")
    print(f"    {'model':<14}{'total gap':>10}{'threshold':>11}{'ranking':>9}"
          f"{'ranking':>9}{'residual':>11}")
    print(f"    {'':<14}{'src - zs':>10}{'cal - zsb':>11}{'dAUROC':>9}"
          f"{'dAUPRC':>9}{'oracle-cal':>11}")
    parts = []
    for name in t["model"].unique():
        s = t[t["model"] == name].mean(numeric_only=True)
        p = {"model": name,
             "total_gap": s["source_mcc_zs"] - s["mcc_zs"],
             "threshold_shift": s["mcc_cal"] - s["mcc_zsb"],
             "ranking_loss_auroc": s["source_auroc"] - s["auroc"],
             "ranking_loss_auprc": s["source_auprc"] - s["auprc"],
             "residual_to_oracle": s["mcc_oracle"] - s["mcc_cal"]}
        parts.append(p)
        print(f"    {name:<14}{p['total_gap']:>10.3f}{p['threshold_shift']:>11.3f}"
              f"{p['ranking_loss_auroc']:>9.3f}{p['ranking_loss_auprc']:>9.3f}"
              f"{p['residual_to_oracle']:>11.3f}")
    pd.DataFrame(parts).to_csv(TABLES / "gap_parts.csv", index=False)
    print("    total gap   : source-side MCC minus zero-shot MCC on the new vehicle"
          "\n    threshold   : what moving the SAME budget rule from source to target"
          "\n                  benign traffic gains (threshold shift)"
          "\n    ranking     : AUROC / AUPRC lost on the new vehicle; no threshold"
          "\n                  can fix this part (representation shift)"
          "\n    residual    : what even a calibrated threshold leaves below the oracle")

    cmp_rows = []
    for name, df, mask in configs:
        cross = df[mask & df["selected"] & (df["setting"] == "cross")]
        a = cross.assign(v=cross["mcc_cal"])
        b = cross.assign(v=cross["mcc_zs"])
        cmp_rows.append(compare(a, b, "v", f"{name}: cal", "zs"))
        cmp_rows[-1]["metric"] = "MCC"
    print_comparisons(cmp_rows, "  Calibrated minus zero-shot, per held-out vehicle "
                                "(* = stable across seeds)")
    return cmp_rows


def arms(rf, mlp):
    print("\n" + "=" * 90)
    print("C. TRAINING STRATEGIES  (selected feature set, budget and mu)")
    print("=" * 90)
    cross = mlp[mlp["selected"] & (mlp["setting"] == "cross")]
    rows = []

    print("\n  MLP, seed-averaged value per held-out vehicle (+- seed sd)")
    table = []
    for metric in ["auroc", "auprc", "mcc_zs", "mcc_cal"]:
        print(f"\n    {metric}")
        for arm in ARMS:
            s = cross[cross["arm"] == arm]
            ps = per_seed(s, metric)
            cells = ""
            entry = {"family": "mlp", "arm": arm, "metric": metric}
            for v in VEHICLES:
                vals = ps.xs(v, level="held_out") if v in ps.index.get_level_values(0) else pd.Series(dtype=float)
                entry[v] = vals.mean()
                entry[f"{v}_seed_sd"] = vals.std(ddof=1)
                cells += f"{vals.mean():>8.3f}+-{vals.std(ddof=1):<5.3f}"
            entry["mean"] = np.nanmean([entry[v] for v in VEHICLES])
            table.append(entry)
            print(f"      {arm:<9}{cells}{entry['mean']:>8.3f}")

    for metric in ["mcc_zs", "mcc_cal", "auprc"]:
        a = {arm: cross[cross["arm"] == arm] for arm in ARMS}
        rows.append(compare(a["pooled"], a["single"], metric, "MLP pooled", "single"))
        rows.append(compare(a["fedavg"], a["single"], metric, "MLP fedavg", "single"))
        rows.append(compare(a["fedavg"], a["pooled"], metric, "MLP fedavg", "pooled"))
        rows.append(compare(a["fedprox"], a["fedavg"], metric, "MLP fedprox", "fedavg"))

    rcross = rf[rf["selected"] & (rf["setting"] == "cross")]
    for metric in ["mcc_zs", "mcc_cal", "auprc"]:
        r = {m: rcross[rcross["mode"] == m] for m in RF_MODES}
        rows.append(compare(r["pooled_3"], r["single"], metric, "RF pooled_3", "single"))
        rows.append(compare(r["pooled_3"], r["pooled_3_balanced"], metric,
                            "RF pooled_3", "balanced"))
        rows.append(compare(r["pooled_3_balanced"], r["pooled_3_matched"], metric,
                            "RF balanced", "matched"))
        for mode in RF_MODES:
            s = r[mode]
            entry = {"family": "rf", "arm": mode, "metric": metric}
            for v in VEHICLES:
                entry[v] = s[s["held_out"] == v][metric].mean()
            entry["mean"] = np.nanmean([entry[v] for v in VEHICLES])
            table.append(entry)

    pd.DataFrame(table).to_csv(TABLES / "arms.csv", index=False)
    print_comparisons(rows, "  Vehicle-level differences (* = stable across seeds; "
                            "RF has one seed, so no *)")

    # mu sensitivity
    fp = mlp[(mlp["arm"] == "fedprox") & (mlp["setting"] == "cross")]
    print("\n  FedProx at every mu (MCC at cal, seed-averaged; [ ] = selected)")
    print(f"    {'mu':<8}" + "".join(f"{v[:9]:>12}" for v in VEHICLES))
    mu_rows = []
    for mu in sorted(fp["mu"].dropna().unique()):
        cells = ""
        for v in VEHICLES:
            s = fp[(fp["mu"] == mu) & (fp["held_out"] == v)]
            val = s["mcc_cal"].mean()
            chosen = bool(s["selected"].any())
            cells += f"{('[' if chosen else ' ')}{val:.3f}{(']' if chosen else ' ')}".rjust(12)
            mu_rows.append({"mu": mu, "held_out": v, "mcc_cal": val,
                            "auprc": s["auprc"].mean(),
                            "client_drift": s["client_drift"].mean(),
                            "selected": chosen})
        print(f"    {mu:<8}{cells}")
    pd.DataFrame(mu_rows).to_csv(TABLES / "fedprox_mu.csv", index=False)
    return rows


def features(rf):
    print("\n" + "=" * 90)
    print("D. FEATURE REPRESENTATION  (RF, every feature set)")
    print("=" * 90)
    cross = rf[rf["setting"] == "cross"]
    rows = []
    for mode in ["pooled_3", "single"]:
        for metric in ["auprc", "mcc_zs", "mcc_cal"]:
            print(f"\n  {mode}, {metric}")
            print(f"    {'features':<10}" + "".join(f"{v[:9]:>10}" for v in VEHICLES)
                  + f"{'mean':>8}")
            for g in FEATURE_SETS:
                s = cross[(cross["mode"] == mode) & (cross["feature_set"] == g)]
                vals = [s[s["held_out"] == v][metric].mean() for v in VEHICLES]
                rows.append({"mode": mode, "metric": metric, "feature_set": g,
                             **dict(zip(VEHICLES, vals)), "mean": np.nanmean(vals)})
                print(f"    {g:<10}" + "".join(f"{x:>10.3f}" for x in vals)
                      + f"{np.nanmean(vals):>8.3f}")
    pd.DataFrame(rows).to_csv(TABLES / "feature_sets.csv", index=False)

    imp_path = RESULTS / "rf_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        top = (imp[(imp["feature_set"] == "all") & (imp["mode"] == "pooled_3")]
               .groupby("feature")["importance"].mean()
               .sort_values(ascending=False))
        top.to_csv(TABLES / "importance_all_pooled.csv")
        print("\n  most important features (all, pooled_3, mean over folds)")
        for f, i in top.head(8).items():
            print(f"    {f:<28}{i:.3f}")


def attacks():
    print("\n" + "=" * 90)
    print("E. ATTACK-WISE RECALL  (same thresholds as the headline results)")
    print("=" * 90)
    rows = []
    rf = pd.read_csv(RESULTS / "rf_per_attack.csv")
    rf["selected"] = rf["selected"].astype(str).str.lower() == "true"
    rf = rf[rf["selected"] & (rf["mode"] == "single")]
    within = rf[rf["setting"] == "within"].groupby("attack_type")["recall_zs"].mean()
    cross = rf[rf["setting"] == "cross"]
    zs = cross.groupby("attack_type")["recall_zs"].mean()
    cal = cross.groupby("attack_type")["recall_cal"].mean()
    ora = cross.groupby("attack_type")["recall_oracle"].mean()
    n = cross.groupby("attack_type")["n_attack_windows"].max()
    n_targets = cross.groupby("attack_type")["test_vehicle"].nunique()
    t = pd.DataFrame({"same_vehicle_zs": within, "unseen_zs": zs,
                      "unseen_cal": cal, "unseen_oracle": ora,
                      "max_windows": n, "target_vehicles": n_targets})
    t = t.sort_values("unseen_cal", ascending=False)
    t.to_csv(TABLES / "per_attack_rf_single.csv")
    print("\n  RF single-vehicle models (selected feature set), mean recall")
    print(f"    {'attack':<18}{'same veh':>9}{'unseen zs':>11}{'unseen cal':>12}"
          f"{'oracle':>8}{'targets':>9}")
    for a, r in t.iterrows():
        w = "   --" if pd.isna(r["same_vehicle_zs"]) else f"{r['same_vehicle_zs']:.3f}"
        print(f"    {a:<18}{w:>9}{r['unseen_zs']:>11.3f}{r['unseen_cal']:>12.3f}"
              f"{r['unseen_oracle']:>8.3f}{int(r['target_vehicles']):>9}")

    res = pd.read_csv(RESULTS / "rf_results.csv")
    res["selected"] = res["selected"].astype(str).str.lower() == "true"
    rc = res[res["selected"] & (res["mode"] == "single") &
             (res["setting"] == "cross")]
    print(f"\n    false-alarm rate on the same unseen-vehicle benign windows:"
          f"\n      zs {rc['fpr_zs'].mean():.1%}   cal {rc['fpr_cal'].mean():.1%}"
          f"\n    Recall at zs is high only because almost everything is flagged;"
          f"\n    compare attack types at cal.")

    seen = cross.groupby("seen_in_training")[["recall_zs", "recall_cal"]].mean()
    print("\n    attack type seen in training?   zs     cal")
    for k, r in seen.iterrows():
        print(f"      {str(bool(k)):<30}{r['recall_zs']:.3f}  {r['recall_cal']:.3f}")

    mp = RESULTS / "mlp_per_attack.csv"
    if mp.exists():
        m = pd.read_csv(mp)
        m["selected"] = m["selected"].astype(str).str.lower() == "true"
        m = m[m["selected"] & (m["setting"] == "cross")]
        piv = (m.groupby(["attack_type", "arm"])["recall_cal"].mean()
                .unstack("arm").reindex(columns=ARMS))
        piv.to_csv(TABLES / "per_attack_mlp_arms.csv")
        print("\n  MLP arms, recall at cal (mean over vehicles and seeds)")
        print("    " + piv.round(3).to_string().replace("\n", "\n    "))


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    rf, mlp = load()
    selections()
    r1 = regimes(rf, mlp)
    r2 = arms(rf, mlp)
    features(rf)
    attacks()
    pd.DataFrame(r1 + r2).to_csv(TABLES / "comparisons.csv", index=False)
    print(f"\n  With 4 held-out vehicles the smallest possible p is 0.125."
          f"\n  wrote tables to {TABLES}/")


if __name__ == "__main__":
    main()
