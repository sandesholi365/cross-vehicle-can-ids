"""
Direction 1 (supervisor's email of 20 October): what does model averaging do
to the scores, and to the transfer of a source-only threshold to a new
vehicle?

Question, in neutral words: when models trained on different vehicles are
combined (federated averaging, one weight average, or combining their
outputs), how do the benign score distribution and the transfer of the
zero-shot threshold change? Nothing is concluded in advance; the script only
measures.

CAN-Train-and-Test only. ROAD and CrySyS are NOT touched. The method of the
thesis is not changed: the feature set, the budget B and mu are READ from
results/mlp_selected.csv; network, scaling, class weight, threshold rules and
seeds are those of run_federated.py.

Settings (fixed in the diary on 20 October, nothing else)
---------------------------------------------------------
  pooled         all source windows together, 20 epochs (as in the thesis)
  fedavg         10 rounds x 2 local epochs (as in the thesis)
  avg_weights    one-shot weight average: three single-vehicle models trained
                 for 20 epochs from the SAME starting weights, averaged once,
                 weighted by number of training windows
  ens_logit      the same three models; weighted mean of their logits
  ens_prob       the same three models; weighted mean of their probabilities
  interp_a       weights (1 - a) * pooled + a * avg_weights, on the grid
                 a = 0, 0.1, ..., 1 (a = 0 is pooled, a = 1 is avg_weights;
                 both ends are taken from those rows)
  interp_sel     the interpolation at the a chosen on the SOURCE vehicles

How a is chosen without the held-out vehicle (step "select")
------------------------------------------------------------
The same inner leave-one-source-out loop as the thesis: for each held-out
vehicle H, train on two of the three sources, treat the third as new, rotate,
and average over the selection seeds 100-102. Criterion, fixed before any
run: the a with the highest mean inner zero-shot MCC; if several a are within
0.005 of the best, the SMALLEST a (closest to pooled) is taken. H is not
used in this step. The whole inner curve is saved.

Measured for every setting, held-out vehicle and evaluation seed (0-9)
---------------------------------------------------------------------
  threshold-independent  AUROC and AUPRC on the source test data and on H
  threshold-dependent    MCC at zs, zsb, cal and oracle; false-alarm rate and
                         recall at zs and cal; threshold shift (cal - zsb)
  benign score distribution, on the logit scale, raw and in units of the
                         source class gap G (median source attack logit -
                         median source benign logit, source test windows),
                         for three groups:
                           srccal  benign source calibration windows
                           tclean  H's attack-free recordings
                           tbenign benign windows inside H's attack recordings
                         median, IQR, sd, q90, q99, q99.9, q99.99. Location
                         values in units of G are measured from the median
                         source benign logit (as headroom in the thesis).
                         headroom (zero-shot threshold above the source
                         benign median); where the zero-shot threshold falls
                         among H's benign scores (share of tclean and tbenign
                         windows at or above it)
  probability calibration  Brier score and expected calibration error
                         (15 equal-width bins) on source test data and on H
  member disagreement    for avg_weights and the ensembles: mean over windows
                         of the spread (sd) of the three members' logits, on
                         source benign and on H's attack-free windows; in logit
                         units and in units of the one-shot average's G

The new vehicle's data is used only to DESCRIBE the outcome.

For Direction 2, the per-window scores of pooled and fedavg are saved for all
ten seeds (results/direction1/scores/).

Run from the project root, in this order:
    QUICK=1 python -u src/experiments/direction1_averaging.py select  (pipeline
    QUICK=1 python -u src/experiments/direction1_averaging.py run      check,
                                                         a few minutes)
    python -u src/experiments/direction1_averaging.py select   (about 15 min)
    python -u src/experiments/direction1_averaging.py run      (about 1-2 h)
    python -u src/experiments/direction1_averaging.py report   (seconds)
"run" continues where it stopped if it is interrupted.

Outputs in results/direction1/:
    alpha_selection_log.csv   every inner score behind the choice of a
    alpha_selected.csv        the a chosen for each held-out vehicle
    runs.csv                  one row per setting, held-out vehicle and seed
    check.csv                 pooled and fedavg against results/mlp_results.csv
                              (and avg_weights against explain_arms, if there)
    per_vehicle.csv           seed means per setting and vehicle
    summary.csv               means over the four vehicles
    sign_consistency.csv      per-vehicle differences against pooled
    scores/                   per-window scores of pooled and fedavg
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BUDGETS, QUICK, RESULTS, VEHICLES, best_threshold,  # noqa: E402
                    budget_threshold, evaluate, feature_columns, load_all,
                    threshold_rules)
import run_federated as rfd  # noqa: E402
import explain_arms as ea  # noqa: E402
import explain_avg as eav  # noqa: E402

OUT = RESULTS / ("direction1_quick" if QUICK else "direction1")
SCORES = OUT / "scores"
SEEDS = [0] if QUICK else list(range(10))
SELECTION_SEEDS = rfd.SELECTION_SEEDS
ALPHAS = [round(0.1 * i, 1) for i in range(11)]
ALPHA_TIE = 0.005
ECE_BINS = 15
QUANTILES = {"q90": 0.90, "q99": 0.99, "q999": 0.999, "q9999": 0.9999}
GROUPS = ["srccal", "tclean", "tbenign"]

TRAINED = ["pooled", "fedavg", "avg_weights", "ens_logit", "ens_prob"]
INTERIOR = [a for a in ALPHAS if 0 < a < 1]
ORDER = (["pooled"] + [f"interp_{a}" for a in INTERIOR] +
         ["avg_weights", "ens_logit", "ens_prob", "fedavg", "interp_sel"])


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------

def make_models(clients, nf, seed):
    """pooled, the three members, and their weight average (same start)."""
    pooled = rfd.train_arm("pooled", clients, nf, seed)[0]
    members = eav.train_members(clients, nf, seed)
    sizes = [len(X) for X, _ in clients]
    avg_state = rfd.average_weights([m.state_dict() for m in members], sizes)
    return pooled, members, sizes, avg_state


def interp(nf, pooled, avg_state, alpha):
    return eav.from_state(nf, eav.mix(pooled.state_dict(), avg_state, alpha))


def brier(p, y):
    return float(np.mean((p.astype(np.float64) - y) ** 2)) if len(y) else np.nan


def ece(p, y, bins=ECE_BINS):
    """Expected calibration error with equal-width bins on [0, 1]."""
    if len(y) == 0:
        return np.nan
    p = p.astype(np.float64)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / len(y))


def dist_stats(z, prefix, m0, G):
    """Benign logit distribution: raw and in units of G."""
    out = {}
    if len(z) == 0:
        return out
    q25, q50, q75 = np.quantile(z, [0.25, 0.5, 0.75])
    vals = {"median": q50, "iqr": q75 - q25, "sd": float(np.std(z))}
    vals.update({k: float(np.quantile(z, q)) for k, q in QUANTILES.items()})
    for k, v in vals.items():
        out[f"{prefix}_{k}"] = float(v)
        # spreads are divided by G; locations are measured from the source
        # benign median first
        rel = v / G if k in ("iqr", "sd") else (v - m0) / G
        out[f"{prefix}_{k}_G"] = float(rel)
    return out


def member_disagreement(members, X):
    """Mean over windows of the sd of the members' logits (logit units)."""
    z = np.stack([ea.logits(m, X) for m in members])
    return float(np.mean(np.std(z, axis=0)))


# ----------------------------------------------------------------------
# step 1: choose a on the source vehicles only
# ----------------------------------------------------------------------

def select():
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    data, parts, clean, evalset = load_all()
    sel = ea.selections()
    everything = rfd.all_columns(data)
    log, chosen = [], []
    for held_out in VEHICLES:
        sources = [v for v in VEHICLES if v != held_out]
        cols = feature_columns(data[VEHICLES[0]], sel[held_out]["features"])
        nf = len(cols)
        print(f"\n--- held out {held_out} (not used here); features "
              f"{sel[held_out]['features']} ---")
        for pretend in sources:
            inner = [v for v in sources if v != pretend]
            # exactly as run_federated.select: scaler on the inner sources only
            sc = rfd.Scaler(pd.concat([parts[v][0] for v in inner]), everything)
            calib = pd.concat([parts[v][1] for v in inner], ignore_index=True)
            clients = [(sc.X(parts[v][0], cols), rfd.labels(parts[v][0]))
                       for v in inner]
            for sd in SELECTION_SEEDS:
                t0 = time.time()
                pooled, _, _, avg_state = make_models(clients, nf, sd)
                for a in ALPHAS:
                    model = interp(nf, pooled, avg_state, a)
                    row = rfd.inner_eval(model, sc, cols, calib, clean[pretend],
                                         evalset[pretend])
                    log.append({"held_out": held_out, "pretend_target": pretend,
                                "inner_sources": "+".join(inner), "seed": sd,
                                "alpha": a, **row})
                cur = pd.DataFrame(log)
                cur = cur[(cur["held_out"] == held_out) &
                          (cur["pretend_target"] == pretend) & (cur["seed"] == sd)]
                print(f"  {'+'.join(inner):<20} -> {pretend:<10} seed {sd} "
                      f"{time.time() - t0:>4.0f}s  inner zs MCC by a: "
                      + " ".join(f"{r.mcc_zs:.2f}" for r in cur.itertuples()))
        lg = pd.DataFrame(log)
        lg = lg[lg["held_out"] == held_out]
        means = lg.groupby("alpha")["mcc_zs"].mean()
        best = float(means.max())
        a_sel = float(min(a for a in ALPHAS if means[a] >= best - ALPHA_TIE))
        chosen.append({"held_out": held_out, "alpha": a_sel,
                       "criterion": "mean inner zero-shot MCC; ties within "
                                    f"{ALPHA_TIE} to the smallest alpha",
                       "best_mean_inner_mcc_zs": best,
                       "chosen_mean_inner_mcc_zs": float(means[a_sel]),
                       "scores": "; ".join(f"{a} {means[a]:.4f}" for a in ALPHAS),
                       "data_used": f"inner leave-one-source-out on "
                                    f"{', '.join(sources)}"})
        print(f"  selected for {held_out}: alpha {a_sel} "
              f"(mean inner zs MCC {means[a_sel]:.3f}; best {best:.3f})")
        pd.DataFrame(log).to_csv(OUT / "alpha_selection_log.csv", index=False)
        pd.DataFrame(chosen).to_csv(OUT / "alpha_selected.csv", index=False)
    print(f"\n  total {(time.time() - t_start) / 60:.0f} min; "
          f"saved alpha_selection_log.csv and alpha_selected.csv in {OUT}/")


def alphas_selected():
    f = OUT / "alpha_selected.csv"
    if not f.exists():
        raise SystemExit(f"{f} not found: run the 'select' step first.")
    a = pd.read_csv(f)
    out = dict(zip(a["held_out"], a["alpha"].astype(float)))
    missing = [v for v in VEHICLES if v not in out]
    if missing:
        raise SystemExit(f"alpha not selected for {missing}: run 'select' again.")
    return out


# ----------------------------------------------------------------------
# step 2: measure every setting on the held-out vehicle
# ----------------------------------------------------------------------

def measure(model, d, budget):
    """Everything for one model. d comes from explain_arms.prepare."""
    out = {}
    # outcome, on probabilities, exactly as in run_federated
    p = {k: rfd.predict(model, d[k]) for k in ("calib", "bsrc", "src", "t", "clean")}
    thr_zs, _ = best_threshold(d["y_calib"], p["calib"])
    rules = threshold_rules(thr_zs, p["bsrc"], p["clean"], budget)
    ev = evaluate(d["y_t"], p["t"], rules)
    for k in ("auroc", "auprc", "mcc_zs", "mcc_zsb", "mcc_cal", "mcc_oracle",
              "fpr_zs", "fpr_cal", "recall_zs", "recall_cal",
              "thr_zs", "thr_zsb", "thr_cal"):
        out[k] = ev.get(k, np.nan)
    out["threshold_shift_mcc"] = out["mcc_cal"] - out["mcc_zsb"]
    sev = evaluate(d["y_src"], p["src"], {"zs": thr_zs, "zsb": rules["zsb"]})
    out["src_auroc"], out["src_auprc"] = sev["auroc"], sev["auprc"]
    out["src_mcc_zs"] = sev["mcc_zs"]
    out["zs_flags_tclean"] = float(np.mean(p["clean"] >= thr_zs))
    tben = d["y_t"] == 0
    out["zs_flags_tbenign"] = float(np.mean(p["t"][tben] >= thr_zs))
    out["cal_flags_tbenign"] = float(np.mean(p["t"][tben] >= rules["cal"]))

    # probability calibration
    out["brier_src"] = brier(p["src"], d["y_src"])
    out["ece_src"] = ece(p["src"], d["y_src"])
    out["brier_t"] = brier(p["t"], d["y_t"])
    out["ece_t"] = ece(p["t"], d["y_t"])

    # benign score distributions, logit scale
    z = {k: ea.logits(model, d[k]) for k in ("calib", "src", "t", "clean")}
    src_b = z["src"][d["y_src"] == 0]
    src_a = z["src"][d["y_src"] == 1]
    m0 = float(np.median(src_b))
    G = float(np.median(src_a) - m0)
    out["class_gap_logit"] = G
    out["spread"] = float(np.std(src_b)) / G           # as in the thesis table
    zs_logit, _ = best_threshold(d["y_calib"], z["calib"])
    out["thr_zs_logit"] = zs_logit
    out["headroom"] = (zs_logit - m0) / G
    groups = {"srccal": z["calib"][d["y_calib"] == 0], "tclean": z["clean"],
              "tbenign": z["t"][tben]}
    for g, zz in groups.items():
        out.update(dist_stats(zz, g, m0, G))
    # zero-shot threshold relative to the new vehicle's benign tail (in G)
    for g in ("tclean", "tbenign"):
        out[f"thr_minus_{g}_q99_G"] = (zs_logit - out[f"{g}_q99"]) / G
        out[f"thr_minus_{g}_q9999_G"] = (zs_logit - out[f"{g}_q9999"]) / G
    return out, p, z, rules


def save_scores(arm, held_out, seed, p, z, rules, d, G, zs_logit, budget):
    SCORES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SCORES / f"{arm}_{held_out}_s{seed}.npz",
        target_score=p["t"].astype(np.float32),
        target_logit=z["t"].astype(np.float32),
        target_label=d["y_t"].astype(np.int8),
        thr_zs=rules["zs"], thr_cal=rules["cal"],
        thr_zs_logit=zs_logit,
        thr_cal_logit=budget_threshold(z["clean"], budget),
        class_gap_logit=G)


def done_keys():
    f = OUT / "runs.csv"
    if not f.exists():
        return set(), []
    old = pd.read_csv(f)
    rows = old.to_dict("records")
    keys = {(r["held_out"], int(r["seed"])) for r in rows
            if r["setting"] == "fedavg"}   # fedavg is written last per seed
    rows = [r for r in rows if (r["held_out"], int(r["seed"])) in keys]
    return keys, rows


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    a_sel = alphas_selected()
    data, parts, clean, evalset = load_all()
    sel = ea.selections()
    keys, rows = done_keys()
    if keys:
        print(f"  continuing: {len(keys)} (vehicle, seed) pairs already done")
    for held_out in VEHICLES:
        s = sel[held_out]
        d = ea.prepare(held_out, data, parts, clean, evalset, s)
        nf = len(d["cols"])
        budget = BUDGETS[s["budget"]]
        print(f"\n--- hold out {held_out}: features {s['features']}, budget "
              f"{s['budget']}, alpha selected {a_sel[held_out]} ---")
        for seed in SEEDS:
            if (held_out, seed) in keys:
                continue
            t0 = time.time()
            pooled, members, sizes, avg_state = make_models(d["clients"], nf, seed)
            fedavg = rfd.train_arm("fedavg", d["clients"], nf, seed)[0]
            models = {"pooled": pooled,
                      "avg_weights": eav.from_state(nf, avg_state),
                      "ens_logit": eav.Ensemble(members, sizes, "logit"),
                      "ens_prob": eav.Ensemble(members, sizes, "prob")}
            for a in INTERIOR:
                models[f"interp_{a}"] = interp(nf, pooled, avg_state, a)
            models["fedavg"] = fedavg
            new = []
            # member disagreement is a property of the three members, so it is
            # computed once and given in logit units and in units of the
            # one-shot average's class gap G (the same value on all three rows)
            dis = {"disagree_src_benign": member_disagreement(members, d["bsrc"]),
                   "disagree_tclean": member_disagreement(members, d["clean"])}
            G_avg = None
            for name in [o for o in ORDER if o in models]:
                r, p, z, rules = measure(models[name], d, budget)
                if name == "avg_weights":
                    G_avg = r["class_gap_logit"]
                if name in ("avg_weights", "ens_logit", "ens_prob"):
                    for k, v in dis.items():
                        r[k] = v
                        r[f"{k}_G"] = v / G_avg
                alpha = (0.0 if name == "pooled" else 1.0 if name == "avg_weights"
                         else float(name.split("_")[1]) if name.startswith("interp_")
                         else np.nan)
                new.append({"setting": name, "alpha": alpha, "held_out": held_out,
                            "seed": seed, "sources": "+".join(d["sources"]),
                            "budget": s["budget"], **r})
                if name in ("pooled", "fedavg"):
                    save_scores(name, held_out, seed, p, z, rules, d,
                                r["class_gap_logit"], r["thr_zs_logit"], budget)
            # interp_sel: the row of the selected alpha (no new training)
            a = a_sel[held_out]
            src = ("pooled" if a == 0 else "avg_weights" if a == 1
                   else f"interp_{a}")
            base = next(x for x in new if x["setting"] == src)
            new.append({**{k: v for k, v in base.items() if not k.startswith("disagree_")},
                        "setting": "interp_sel", "alpha": a})
            # write fedavg last, so an interrupted seed is redone in full
            new.sort(key=lambda x: x["setting"] == "fedavg")
            rows += new
            pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)
            got = {x["setting"]: x for x in new}
            print(f"  seed {seed} {time.time() - t0:>4.0f}s   zs MCC / FPR zs / spread: "
                  + "  ".join(f"{k} {got[k]['mcc_zs']:.2f}/{got[k]['fpr_zs']:.3f}/"
                              f"{got[k]['spread']:.2f}"
                              for k in ("pooled", "avg_weights", "ens_logit",
                                        "ens_prob", "fedavg", "interp_sel")))
    check()
    print(f"\n  total {(time.time() - t_start) / 60:.0f} min")
    report()


# ----------------------------------------------------------------------
# checks and report
# ----------------------------------------------------------------------

def check():
    """pooled and fedavg must reproduce mlp_results.csv for every seed."""
    if QUICK:
        print("\nCHECK skipped in QUICK mode: the saved thesis results use all "
              "windows,\n  QUICK uses every 25th. The full run does the check.")
        return True
    runs = pd.read_csv(OUT / "runs.csv")
    saved = pd.read_csv(RESULTS / "mlp_results.csv")
    saved["selected"] = saved["selected"].astype(str).str.lower() == "true"
    saved = saved[saved["selected"] & (saved["setting"] == "cross") &
                  saved["arm"].isin(["pooled", "fedavg"])]
    rows = []
    for arm in ("pooled", "fedavg"):
        m = runs[runs["setting"] == arm].merge(
            saved[saved["arm"] == arm], on=["held_out", "seed"],
            suffixes=("", "_saved"))
        for k in ("auroc", "auprc", "mcc_zs", "mcc_zsb", "mcc_cal", "mcc_oracle"):
            rows.append({"compared": f"{arm} vs results/mlp_results.csv",
                         "metric": k, "runs": len(m),
                         "max_abs_difference": float((m[k] - m[f"{k}_saved"]).abs().max())
                         if len(m) else np.nan})
    prev = ea.OUT / "runs.csv"
    if prev.exists():
        old = pd.read_csv(prev)
        m = runs[runs["setting"] == "avg_weights"].merge(
            old[old["variant"] == "fedavg_20x1"], on=["held_out", "seed"],
            suffixes=("", "_old"))
        for k in ("auroc", "mcc_zs", "fpr_zs"):
            rows.append({"compared": "avg_weights vs explain_arms fedavg_20x1",
                         "metric": k, "runs": len(m),
                         "max_abs_difference": float((m[k] - m[f"{k}_old"]).abs().max())
                         if len(m) else np.nan})
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "check.csv", index=False)
    print("\nCHECK: re-trained models against the saved results")
    print("  " + rep.to_string(index=False).replace("\n", "\n  "))
    main = rep[rep["compared"].str.contains("mlp_results")]
    ok = (len(main) > 0 and (main["runs"] > 0).all()
          and main["max_abs_difference"].max() <= 1e-3)
    print("  OK: pooled and FedAvg reproduce the thesis results." if ok else
          "  WARNING: not reproduced. Send this log before using any table.")
    return ok


def report():
    runs = pd.read_csv(OUT / "runs.csv")
    # interp_sel copies the row of the chosen alpha; member disagreement belongs
    # to avg_weights and the ensembles only, so it is blanked here
    dis = [c for c in runs.columns if c.startswith("disagree_")]
    runs.loc[runs["setting"] == "interp_sel", dis] = np.nan
    num = [c for c in runs.columns if c not in
           ("setting", "held_out", "seed", "sources", "budget")]
    per_v = runs.groupby(["setting", "held_out"])[num].mean()
    per_v.reset_index().to_csv(OUT / "per_vehicle.csv", index=False)
    summ = per_v.groupby(level=0).mean()
    summ = summ.reindex([o for o in ORDER if o in summ.index])
    summ.to_csv(OUT / "summary.csv")
    n_seeds = runs["seed"].nunique()

    def table(title, cols, fmt=".3f"):
        print(f"\n{title}")
        print(f"  {'setting':<14}" + "".join(f"{h:>13}" for h, _ in cols))
        for name, r in summ.iterrows():
            print(f"  {name:<14}" + "".join(
                f"{format(r[c], fmt) if c in r and np.isfinite(r[c]) else '-':>13}"
                for _, c in cols))

    print("\n" + "=" * 104)
    print(f"DIRECTION 1   mean over 4 held-out vehicles ({n_seeds} seeds averaged "
          f"within each vehicle first)")
    print("=" * 104)
    table("1. Threshold-independent", [("src AUROC", "src_auroc"),
          ("src AUPRC", "src_auprc"), ("AUROC", "auroc"), ("AUPRC", "auprc")])
    table("2. Threshold-dependent", [("src MCC", "src_mcc_zs"), ("zs", "mcc_zs"),
          ("zsb", "mcc_zsb"), ("cal", "mcc_cal"), ("oracle", "mcc_oracle"),
          ("cal-zsb", "threshold_shift_mcc"), ("FPR zs", "fpr_zs"),
          ("recall zs", "recall_zs"), ("FPR cal", "fpr_cal"),
          ("recall cal", "recall_cal")])
    table("3. Benign logits in units of G (from the source benign median)",
          [("spread", "spread"), ("headroom", "headroom"),
           ("src q99", "srccal_q99_G"), ("src q9999", "srccal_q9999_G"),
           ("clean med", "tclean_median_G"), ("clean q99", "tclean_q99_G"),
           ("clean q9999", "tclean_q9999_G"), ("ben q99", "tbenign_q99_G"),
           ("ben q9999", "tbenign_q9999_G")], ".2f")
    table("4. Where the zero-shot threshold falls on the new vehicle",
          [("flags clean", "zs_flags_tclean"), ("flags benign", "zs_flags_tbenign"),
           ("thr-clean99", "thr_minus_tclean_q99_G"),
           ("thr-ben99", "thr_minus_tbenign_q99_G")])
    table("5. Probability calibration and member disagreement",
          [("Brier src", "brier_src"), ("ECE src", "ece_src"),
           ("Brier tgt", "brier_t"), ("ECE tgt", "ece_t"),
           ("disagr src", "disagree_src_benign_G"),
           ("disagr tgt", "disagree_tclean_G")])

    # per-vehicle differences against pooled, with sign consistency
    rows = []
    pv = per_v.reset_index()
    for name in [o for o in ORDER if o != "pooled" and (pv["setting"] == o).any()]:
        for metric in ("auroc", "auprc", "mcc_zs", "mcc_cal", "fpr_zs", "spread",
                       "headroom", "tbenign_q9999_G"):
            diffs = []
            for v in VEHICLES:
                a = pv[(pv["setting"] == name) & (pv["held_out"] == v)]
                b = pv[(pv["setting"] == "pooled") & (pv["held_out"] == v)]
                diffs.append(float(a[metric].iloc[0] - b[metric].iloc[0])
                             if len(a) and len(b) else np.nan)
            d = np.array(diffs)
            pos, neg = int(np.sum(d > 0)), int(np.sum(d < 0))
            rows.append({"setting": name, "metric": metric,
                         **{v: x for v, x in zip(VEHICLES, diffs)},
                         "mean": float(np.nanmean(d)),
                         "same_sign": f"{max(pos, neg)}/{int(np.isfinite(d).sum())}"})
    sc = pd.DataFrame(rows)
    sc.to_csv(OUT / "sign_consistency.csv", index=False)
    print("\n6. Per vehicle, setting minus pooled (seed means)")
    for metric in ("mcc_zs", "fpr_zs", "spread"):
        print(f"  [{metric}]")
        sub = sc[sc["metric"] == metric]
        print(f"    {'setting':<14}" + "".join(f"{v[:9]:>10}" for v in VEHICLES)
              + f"{'mean':>9}{'sign':>7}")
        for r in sub.itertuples():
            print(f"    {r.setting:<14}" + "".join(
                f"{getattr(r, v):>+10.3f}" for v in VEHICLES)
                + f"{r.mean:>+9.3f}{r.same_sign:>7}")

    a = OUT / "alpha_selected.csv"
    if a.exists():
        print("\n7. alpha chosen on the source vehicles")
        for r in pd.read_csv(a).itertuples():
            print(f"    {r.held_out:<10} alpha {r.alpha}   (inner zs MCC by alpha: "
                  f"{r.scores})")
    print(f"\n  Tables in {OUT}/. No conclusion is drawn here; read them per "
          f"vehicle and by sign consistency, as in the thesis.")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    {"select": select, "run": run, "report": report,
     "check": check}.get(what, lambda: print(__doc__))()
