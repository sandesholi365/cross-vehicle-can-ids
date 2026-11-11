"""
Sensitivity check: recording-level source calibration split
(supervisor's email of 5 October). CAN-Train-and-Test only; ROAD is not
touched.

In the main experiments, the windows used to set the zero-shot threshold
(the "calibration windows", 15% of the training windows) are drawn at random
from the training recordings. They therefore sit next to windows the model
was trained on. Everything else in the thesis is split by whole recording.

This script repeats the main CAN-Train-and-Test comparison with a
calibration split that respects recordings, and compares it with the saved
results. Two variants are available, because some attack types have only one
training recording per vehicle and cannot give up a whole recording:

  original   15% of the training windows, drawn at random (as saved)
  recording  for each source vehicle and attack type with at least two
             training recordings, the last training recording (sorted by
             name) becomes a calibration recording; the other training
             recordings are used for fitting. Attack types with a single
             training recording keep it for fitting (listed by "coverage").
  block      the last 15% of every training recording (a contiguous block in
             time) is used for calibration, the first 85% for fitting. Every
             attack type is covered; calibration and fitting windows only
             meet at one boundary per recording.

In both variants the test recordings are the same as before, and so are the
features, preprocessing, models, seeds and threshold rules. The settings
already chosen by the inner loop are READ from results/rf_selected.csv and
results/mlp_selected.csv (not selected again).

Models: Random Forest pooled, and the MLP trained pooled, with FedAvg and with
FedProx at the selected mu (the headline configurations of Table 6.2).

Run from the project root, in this order:
    python src/experiments/calib_sensitivity.py check      (seed 0, original
                                     split: must reproduce the saved results)
    python src/experiments/calib_sensitivity.py coverage   (how the two
                                     variants look per vehicle; seconds)
    python src/experiments/calib_sensitivity.py run recording   (all seeds)
    python src/experiments/calib_sensitivity.py run block       (all seeds)
    python src/experiments/calib_sensitivity.py compare    (tables)

Outputs in results/sensitivity_calib/:
  coverage.csv     how the recording-level split looks per vehicle
  check.csv        re-run of the original split against the saved results
  runs_<variant>.csv  one row per model, held-out vehicle and seed
  summary.csv      original against recording-level, per model
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import (BUDGETS, QUICK, RESULTS, VEHICLES, best_threshold,  # noqa: E402
                    evaluate, feature_columns, heldout_benign, load_all, scored,
                    threshold_rules)
import run_baselines as rb  # noqa: E402
import run_federated as rfd  # noqa: E402

OUT = RESULTS / ("sensitivity_calib_quick" if QUICK else "sensitivity_calib")
MLP_ARMS = ["pooled", "fedavg", "fedprox"]
METRICS = ["auroc", "auprc", "mcc_zs", "mcc_zsb", "mcc_cal", "mcc_oracle",
           "fpr_zs", "fpr_cal"]


def flag(df, col="selected"):
    df[col] = df[col].astype(str).str.lower() == "true"
    return df


# ----------------------------------------------------------------------
# the recording-level split
# ----------------------------------------------------------------------

def split_by_recording(df):
    """(fit, calib, test): calibration made of whole training recordings."""
    train_recs, test_recs = common.split_recordings(df)
    train = df[df["recording"].isin(train_recs)]
    calib_recs = []
    for _, grp in train.groupby("attack_type"):
        recs = sorted(grp["recording"].unique())
        if len(recs) >= 2:
            calib_recs.append(recs[-1])
    is_cal = train["recording"].isin(calib_recs)
    return train[~is_cal], train[is_cal], df[df["recording"].isin(test_recs)]


def split_by_block(df, share=common.CALIB_FRACTION):
    """(fit, calib, test): the last `share` of every training recording."""
    train_recs, test_recs = common.split_recordings(df)
    train = df[df["recording"].isin(train_recs)]
    pos = train.groupby("recording")["window_index"].rank(method="first", pct=True)
    is_cal = pos > (1.0 - share)
    return train[~is_cal], train[is_cal], df[df["recording"].isin(test_recs)]


SPLITS = {"recording": split_by_recording, "block": split_by_block}


def coverage_report(data, variant):
    parts_v = {v: SPLITS[variant](data[v]) for v in VEHICLES}
    print(f"\n  variant: {variant}")
    return coverage(data, parts_v, variant), parts_v


def coverage(data, parts_rec, variant="recording"):
    rows = []
    for v in VEHICLES:
        fit, cal, test = parts_rec[v]
        types_train = set(pd.concat([fit, cal])["attack_type"].unique())
        types_cal = set(cal["attack_type"].unique())
        rows.append({
            "vehicle": v, "fit_windows": len(fit), "calib_windows": len(cal),
            "calib_share_of_training": len(cal) / (len(fit) + len(cal)),
            "variant": variant,
            "calib_recordings": cal["recording"].nunique(),
            "calib_attack_windows": int(cal["label"].sum()),
            "calib_attack_share": float(cal["label"].mean()) if len(cal) else np.nan,
            "attack_types_without_calib_recording":
                ", ".join(sorted(types_train - types_cal))})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"coverage_{variant}.csv", index=False)
    for _, r in t.iterrows():
        print(f"  {r['vehicle']:<10} fit {r['fit_windows']:>8,}  calib "
              f"{r['calib_windows']:>7,} ({r['calib_share_of_training']:.1%} of training, "
              f"{r['calib_recordings']} recordings, {r['calib_attack_share']:.1%} attack)")
        if r["attack_types_without_calib_recording"]:
            print(f"             no calibration data for: "
                  f"{r['attack_types_without_calib_recording']}")
        if r["calib_windows"] == 0 or r["calib_attack_windows"] == 0:
            print("             WARNING: no calibration windows with attacks for this "
                  "vehicle; the zero-shot threshold cannot be set with this variant")
    return t


# ----------------------------------------------------------------------
# one evaluation, written the same way as run_baselines / run_federated
# ----------------------------------------------------------------------

def selections():
    rf = pd.read_csv(RESULTS / "rf_selected.csv")
    mlp = pd.read_csv(RESULTS / "mlp_selected.csv")
    out = {}
    for v in VEHICLES:
        r = dict(zip(rf.loc[rf["held_out"] == v, "parameter"],
                     rf.loc[rf["held_out"] == v, "chosen"]))
        m = dict(zip(mlp.loc[mlp["held_out"] == v, "parameter"],
                     mlp.loc[mlp["held_out"] == v, "chosen"]))
        out[v] = {"rf_features": str(r["feature_set"]), "rf_budget": str(r["budget"]),
                  "mlp_features": str(m["feature_set"]), "mlp_budget": str(m["budget"]),
                  "mu": float(m["fedprox_mu"])}
    return out


def evaluate_run(prob, p_clean, p_bsrc, thr_zs, budget, y_target, p_src, y_src):
    rules = threshold_rules(thr_zs, p_bsrc, p_clean, BUDGETS[budget])
    row = evaluate(y_target, prob, rules)
    out = {m: row[m] for m in METRICS}
    out["zs_flags_target_attack_free"] = float(np.mean(p_clean >= thr_zs))
    src = evaluate(y_src, p_src, {"zs": thr_zs, "zsb": rules["zsb"]})
    out["source_mcc_zs"] = src["mcc_zs"]
    out["source_auroc"] = src["auroc"]
    return out


def run_split(split, data, P, clean, evalset, seeds, sel):
    rows = []
    everything = rfd.all_columns(data)
    for held_out in VEHICLES:
        s = sel[held_out]
        sources = [v for v in VEHICLES if v != held_out]
        src_calib = pd.concat([P[v][1] for v in sources], ignore_index=True)
        src_test = scored(pd.concat([P[v][2] for v in sources], ignore_index=True))
        benign_src = heldout_benign(P, sources)
        y_t = evalset[held_out]["label"].to_numpy()

        # Random Forest, pooled (proportional), selected feature set
        t0 = time.time()
        cols = feature_columns(data[VEHICLES[0]], s["rf_features"])
        train = rb.pooled_variants(P, sources)["pooled_3"]
        model = rb.fit_model(train[cols], train["label"])
        thr_zs, _ = best_threshold(src_calib["label"].to_numpy(),
                                   rb.score(model, src_calib, cols))
        r = evaluate_run(rb.score(model, evalset[held_out], cols),
                         rb.score(model, clean[held_out], cols),
                         rb.score(model, benign_src, cols), thr_zs, s["rf_budget"],
                         y_t, rb.score(model, src_test, cols),
                         src_test["label"].to_numpy())
        rows.append({"split": split, "model": "RF pooled", "held_out": held_out,
                     "seed": rb.RF_SEED, **r})
        print(f"  {split:<9} RF pooled    hold out {held_out:<10}"
              f"{time.time() - t0:>5.0f}s  zs {r['mcc_zs']:.3f}  cal {r['mcc_cal']:.3f}")

        # MLP arms
        cols = feature_columns(data[VEHICLES[0]], s["mlp_features"])
        sc = rfd.Scaler(pd.concat([P[v][0] for v in sources]), everything)
        clients = [(sc.X(P[v][0], cols), rfd.labels(P[v][0])) for v in sources]
        X = {"calib": sc.X(src_calib, cols), "bsrc": sc.X(benign_src, cols),
             "src": sc.X(src_test, cols), "t": sc.X(evalset[held_out], cols),
             "clean": sc.X(clean[held_out], cols)}
        for seed in seeds:
            for arm in MLP_ARMS:
                t0 = time.time()
                mu = s["mu"] if arm == "fedprox" else 0.0
                model, _ = rfd.train_arm(arm, clients, len(cols), seed, mu)
                thr_zs, _ = best_threshold(src_calib["label"].to_numpy(),
                                           rfd.predict(model, X["calib"]))
                r = evaluate_run(rfd.predict(model, X["t"]),
                                 rfd.predict(model, X["clean"]),
                                 rfd.predict(model, X["bsrc"]), thr_zs,
                                 s["mlp_budget"], y_t, rfd.predict(model, X["src"]),
                                 src_test["label"].to_numpy())
                name = {"pooled": "MLP pooled", "fedavg": "MLP FedAvg",
                        "fedprox": "MLP FedProx"}[arm]
                rows.append({"split": split, "model": name, "held_out": held_out,
                             "seed": seed, **r})
                print(f"  {split:<9} {name:<12} hold out {held_out:<10} seed {seed}"
                      f"{time.time() - t0:>5.0f}s  zs {r['mcc_zs']:.3f}  "
                      f"cal {r['mcc_cal']:.3f}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# saved results of the main experiments, in the same form
# ----------------------------------------------------------------------

def saved_original():
    rf = flag(pd.read_csv(RESULTS / "rf_results.csv"))
    mlp = flag(pd.read_csv(RESULTS / "mlp_results.csv"))
    rows = []
    configs = [("RF pooled", rf, rf["mode"] == "pooled_3")]
    for arm, name in [("pooled", "MLP pooled"), ("fedavg", "MLP FedAvg"),
                      ("fedprox", "MLP FedProx")]:
        configs.append((name, mlp, mlp["arm"] == arm))
    for name, df, mask in configs:
        cross = df[mask & df["selected"] & (df["setting"] == "cross")]
        src = df[mask & df["selected"] & (df["setting"] == "within_sources")]
        for (v, seed), c in cross.groupby(["held_out", "seed"]):
            row = {"split": "original", "model": name, "held_out": v, "seed": seed}
            row.update({m: float(c[m].mean()) for m in METRICS if m in c})
            sr = src[(src["held_out"] == v) & (src["seed"] == seed)]
            row["source_mcc_zs"] = float(sr["mcc_zs"].mean()) if len(sr) else np.nan
            row["source_auroc"] = float(sr["auroc"].mean()) if len(sr) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# the three commands
# ----------------------------------------------------------------------

def check():
    """Original split, first seed only: must reproduce the saved results."""
    OUT.mkdir(parents=True, exist_ok=True)
    data, parts, clean, evalset = load_all()
    new = run_split("original", data, parts, clean, evalset, [rfd.SEEDS[0]],
                    selections())
    old = saved_original()
    m = new.merge(old, on=["model", "held_out", "seed"], suffixes=("", "_saved"))
    rows = []
    for metric in ("auroc", "mcc_zs", "mcc_zsb", "mcc_cal"):
        d = (m[metric] - m[f"{metric}_saved"]).abs()
        rows.append({"metric": metric, "runs_compared": len(m),
                     "max_abs_difference": float(d.max())})
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "check.csv", index=False)
    print("\nCHECK: original split re-run against the saved results")
    print("  " + rep.to_string(index=False).replace("\n", "\n  "))
    if len(m) == 0 or rep["max_abs_difference"].max() > 1e-3:
        print("  WARNING: the re-run does not reproduce the saved results. "
              "Do not start 'run'; send this log.")
    else:
        print("  OK: the script reproduces the main experiments. "
              "You can start 'run'.")


def show_coverage():
    OUT.mkdir(parents=True, exist_ok=True)
    data, _, _, _ = load_all()
    print("\nCALIBRATION SPLITS THAT RESPECT RECORDINGS")
    for variant in SPLITS:
        coverage_report(data, variant)


def run(variant):
    if variant not in SPLITS:
        raise SystemExit(f"variant must be one of {list(SPLITS)}")
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    data, _, clean, evalset = load_all()
    t, parts_v = coverage_report(data, variant)
    if ((t["calib_windows"] == 0) | (t["calib_attack_windows"] == 0)).any():
        raise SystemExit("A vehicle has no calibration windows with attacks in this "
                         "variant. Use the other variant.")
    res = run_split(variant, data, parts_v, clean, evalset, rfd.SEEDS, selections())
    res.to_csv(OUT / f"runs_{variant}.csv", index=False)
    print(f"\n  total {(time.time() - t_start) / 60:.0f} min")
    compare()


def compare():
    runs = [pd.read_csv(f) for f in sorted(OUT.glob("runs_*.csv"))]
    if not runs:
        raise SystemExit("no runs yet: use 'run recording' or 'run block' first")
    old = saved_original()
    both = pd.concat([old] + runs, ignore_index=True)
    splits = ["original"] + [s for s in SPLITS if (both["split"] == s).any()]
    cols = ["source_mcc_zs", "auroc", "mcc_zs", "mcc_zsb", "mcc_cal", "mcc_oracle",
            "fpr_zs"]
    # seeds averaged within each vehicle first, then the mean over vehicles
    per_v = both.groupby(["split", "model", "held_out"])[
        cols + ["zs_flags_target_attack_free"]].mean().reset_index()
    per_v["threshold_shift"] = per_v["mcc_cal"] - per_v["mcc_zsb"]
    per_v["total_gap"] = per_v["source_mcc_zs"] - per_v["mcc_zs"]
    summ = per_v.groupby(["model", "split"])[
        cols + ["threshold_shift", "total_gap"]].mean().reset_index()
    summ.to_csv(OUT / "summary.csv", index=False)
    per_v.to_csv(OUT / "per_vehicle.csv", index=False)

    order = ["RF pooled", "MLP pooled", "MLP FedAvg", "MLP FedProx"]
    print("\n" + "=" * 100)
    print("CALIBRATION SPLIT: original (random 15% of training windows) against "
          "variants that respect recordings\nmean over the four held-out vehicles "
          "(MLP: seeds averaged within each vehicle first)")
    print("=" * 100)
    head = ["src MCC", "AUROC", "zs", "zsb", "cal", "oracle", "FPR zs",
            "thr shift", "gap"]
    print(f"  {'model':<13}{'split':<11}" + "".join(f"{h:>10}" for h in head))
    for model in order:
        for split in splits:
            r = summ[(summ["model"] == model) & (summ["split"] == split)]
            if r.empty:
                continue
            r = r.iloc[0]
            vals = [r["source_mcc_zs"], r["auroc"], r["mcc_zs"], r["mcc_zsb"],
                    r["mcc_cal"], r["mcc_oracle"], r["fpr_zs"],
                    r["threshold_shift"], r["total_gap"]]
            print(f"  {model:<13}{split:<11}" + "".join(f"{x:>10.3f}" for x in vals))

    for split in splits[1:]:
        print(f"\n  zero-shot threshold: share of the new vehicle's attack-free "
              f"windows flagged ({split})")
        rec = per_v[per_v["split"] == split]
        for model in order:
            r = rec[rec["model"] == model].set_index("held_out")
            if r.empty:
                continue
            print(f"    {model:<13}" + "  ".join(
                f"{v} {r.loc[v, 'zs_flags_target_attack_free']:.2%}" for v in VEHICLES
                if v in r.index))

    print("\n  key comparison: FedAvg minus pooled MLP, zero-shot MCC, per vehicle")
    for split in splits:
        p = per_v[(per_v["split"] == split)].pivot(index="held_out", columns="model",
                                                   values="mcc_zs")
        if not {"MLP FedAvg", "MLP pooled"} <= set(p.columns):
            continue
        d = (p["MLP FedAvg"] - p["MLP pooled"]).reindex(VEHICLES)
        print(f"    {split:<10}" + "  ".join(f"{v} {d[v]:+.2f}" for v in VEHICLES)
              + f"   mean {d.mean():+.2f}, positive on {int((d > 0).sum())} of 4")
    print(f"\n  tables in {OUT}/")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "run":
        run(sys.argv[2] if len(sys.argv) > 2 else "")
    else:
        {"check": check, "coverage": show_coverage, "compare": compare}.get(
            what, lambda: print(__doc__))()
