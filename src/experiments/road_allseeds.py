"""
ROAD results per attack type and score separation over all ten MLP seeds
(supervisor's email of 5 October).

run_road.py saved per-window scores for the first seed only, so the
attack-level AUROC could only be computed for that seed. This script
re-runs the ROAD part of the MLP with exactly the frozen settings:

  - the settings chosen earlier on CAN-Train-and-Test are READ from
    results/road/mlp_selected.csv (feature set, budget, FedProx mu);
    nothing is selected or tuned again
  - the same data, scaling, network, training code and seeds as run_road.py
  - only the selected arms: single (each vehicle), pooled, FedAvg and
    FedProx at the selected mu

and saves the per-window scores of every seed. It then checks that the
re-run reproduces the saved results in results/road/mlp_results.csv
(AUROC, MCC at zs and cal). If it does not, stop and report: the analysis
should not be used until the difference is understood.

Two parts, run in this order from the project root:
    python src/experiments/road_allseeds.py run       (trains, a few hours)
    python src/experiments/road_allseeds.py analyse   (tables, minutes)

Outputs in results/road_allseeds/:
  scores/<model>_s<seed>_<set>.npz   per-window scores of every run
  reproduction_check.csv             re-run against saved results
  tables/per_attack_auroc.csv        AUROC per attack type, mean and seed sd
  tables/separation.csv              score separation per model, mean and sd
  tables/per_attack_by_class.csv     mean AUROC per representation class
The Random Forest has no seeds; its single saved model in results/road/scores
is used as it is.
"""

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BUDGETS, QUICK, RESULTS, VEHICLES, best_threshold,  # noqa: E402
                    evaluate, feature_columns, heldout_benign, load_all,
                    scored, threshold_rules)
import run_federated as rfd  # noqa: E402
import run_road as rr  # noqa: E402

ROAD = RESULTS / ("road_quick" if QUICK else "road")
OUT = RESULTS / ("road_allseeds_quick" if QUICK else "road_allseeds")
SCORES = OUT / "scores"
TABLES = OUT / "tables"
SETS = ["fabrication", "masquerade"]

# Table 5.6, revised: mechanism and targeted signal (see the Methodology)
#   represented  mechanism and targeted signal both occur in CAN-Train-and-Test
#   partly       mechanism occurs, targeted signal does not
#   not          mechanism does not occur (masquerade)
ATTACK_CLASS = {
    "fuzzing": "represented",
    "max_speedometer": "represented",
    "correlated_signal": "partly",
    "max_engine_coolant_temp": "partly",
    "reverse_light_on": "partly",
    "reverse_light_off": "partly",
}


def attack_class(attack_type):
    if attack_type.endswith("_masquerade"):
        return "not"
    return ATTACK_CLASS.get(attack_type, "UNMAPPED")


def selected():
    s = pd.read_csv(ROAD / "mlp_selected.csv")
    d = dict(zip(s["parameter"], s["chosen"]))
    return str(d["feature_set"]), str(d["budget"]), float(d["fedprox_mu"])


# ----------------------------------------------------------------------
# part 1: re-run and save every seed
# ----------------------------------------------------------------------

def run():
    SCORES.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    data, parts, _, _ = load_all()
    road_clean, road_sets = rr.load_road()
    group, b, mu_sel = selected()
    cols = feature_columns(data[VEHICLES[0]], group)
    print(f"\nfrozen settings read from {ROAD}/mlp_selected.csv: features {group}, "
          f"budget {b}, FedProx mu {mu_sel}")

    sc_all = rfd.Scaler(pd.concat([parts[v][0] for v in VEHICLES]),
                        rfd.all_columns(data))
    clients = [(sc_all.X(parts[v][0], cols), rfd.labels(parts[v][0]))
               for v in VEHICLES]
    src_calib = pd.concat([parts[v][1] for v in VEHICLES], ignore_index=True)
    X_calib = sc_all.X(src_calib, cols)
    X_bsrc = sc_all.X(heldout_benign(parts, VEHICLES), cols)
    X_clean = sc_all.X(road_clean, cols)
    X_sets = {k: sc_all.X(v, cols) for k, v in road_sets.items()}

    # shared per-window information, saved once
    for name, df in road_sets.items():
        np.savez_compressed(
            SCORES / f"windows_{name}.npz",
            target_label=df["label"].to_numpy(np.int8),
            target_attack=df["attack_type"].to_numpy(str),
            target_recording=df["recording"].to_numpy(str),
            benign_recording=road_clean["recording"].to_numpy(str))

    runs = [("single", v, 0.0) for v in VEHICLES]
    runs += [("pooled", "all", 0.0), ("fedavg", "all", 0.0),
             ("fedprox", "all", mu_sel)]
    rows = []
    for seed in rfd.SEEDS:
        print(f"  seed {seed}")
        for arm, label, mu in runs:
            t0 = time.time()
            if arm == "single":
                i = VEHICLES.index(label)
                model = rfd.train_local(rfd.new_model(len(cols), seed), *clients[i],
                                        rfd.CENTRAL_EPOCHS, seed=seed)
                calib_v = parts[label][1]
                thr_zs, _ = best_threshold(calib_v["label"].to_numpy(),
                                           rfd.predict(model, sc_all.X(calib_v, cols)))
                p_bsrc = rfd.predict(model, sc_all.X(heldout_benign(parts, [label]),
                                                     cols))
                name = f"mlp_single_{label}"
            else:
                model, _ = rfd.train_arm(arm, clients, len(cols), seed, mu)
                thr_zs, _ = best_threshold(src_calib["label"].to_numpy(),
                                           rfd.predict(model, X_calib))
                p_bsrc = rfd.predict(model, X_bsrc)
                name = f"mlp_{arm}"
            p_clean = rfd.predict(model, X_clean)
            rules = threshold_rules(thr_zs, p_bsrc, p_clean, BUDGETS[b])
            for sset, x in X_sets.items():
                prob = rfd.predict(model, x)
                y = road_sets[sset]["label"].to_numpy()
                m = evaluate(y, prob, rules)
                np.savez_compressed(
                    SCORES / f"{name}_s{seed}_{sset}.npz",
                    target_score=prob.astype(np.float32),
                    benign_score=p_clean.astype(np.float32),
                    thr_zs=thr_zs, thr_cal=rules["cal"], budget=b)
                rows.append({"model": name, "arm": arm, "train_vehicles": label,
                             "seed": seed, "road_set": sset, "auroc": m["auroc"],
                             "mcc_zs": m["mcc_zs"], "mcc_cal": m["mcc_cal"]})
            print(f"    {name:<22}{time.time() - t0:>5.0f}s")
    pd.DataFrame(rows).to_csv(OUT / "rerun_results.csv", index=False)
    print(f"\n  total {(time.time() - t_start) / 60:.0f} min")
    check_reproduction()


def check_reproduction():
    """The re-run must reproduce the results saved by run_road.py."""
    new = pd.read_csv(OUT / "rerun_results.csv")
    old = pd.read_csv(ROAD / "mlp_results.csv")
    old["selected"] = old["selected"].astype(str).str.lower() == "true"
    old = old[(old["setting"] == "external") & old["selected"]]
    old_key = old["arm"].astype(str) + "|" + old["train_vehicles"].astype(str)
    old = old.assign(key=np.where(old["arm"] == "single", old_key,
                                  old["arm"].astype(str) + "|all"))
    new = new.assign(key=new["arm"] + "|" + new["train_vehicles"])
    m = new.merge(old[["key", "seed", "road_set", "auroc", "mcc_zs", "mcc_cal"]],
                  on=["key", "seed", "road_set"], suffixes=("", "_saved"))
    rows = []
    for metric in ("auroc", "mcc_zs", "mcc_cal"):
        diff = (m[metric] - m[f"{metric}_saved"]).abs()
        rows.append({"metric": metric, "runs_compared": len(m),
                     "max_abs_difference": float(diff.max()),
                     "mean_abs_difference": float(diff.mean())})
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "reproduction_check.csv", index=False)
    print("\nREPRODUCTION CHECK against results/road/mlp_results.csv")
    print("  " + rep.to_string(index=False).replace("\n", "\n  "))
    worst = rep["max_abs_difference"].max()
    if len(m) == 0:
        print("  WARNING: no runs could be matched. Stop and report.")
    elif worst < 1e-6:
        print("  OK: the re-run reproduces the saved results exactly.")
    elif worst < 1e-3:
        print("  OK: tiny numerical differences only (below 0.001).")
    else:
        print("  WARNING: the re-run does NOT reproduce the saved results. "
              "Do not use the analysis; send this log.")


# ----------------------------------------------------------------------
# part 2: analysis over all seeds
# ----------------------------------------------------------------------

def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(lab, np.r_[pos, neg]))


def run_stats(score, benign, thr_cal, y, attack, budget):
    """Per-run statistics: separation and AUROC per attack type."""
    att, cap = score[y == 1], score[y == 0]
    out = {"amb_above_cal": float(np.mean(benign >= thr_cal)),
           "cap_above_cal": float(np.mean(cap >= thr_cal)),
           "att_above_cal": float(np.mean(att >= thr_cal)),
           "auroc_vs_ambient": auroc(att, benign),
           "auroc_vs_captures": auroc(att, cap),
           "amb_ge_median_attack": float(np.mean(benign >= np.median(att)))}
    per_type = {t: auroc(score[(y == 1) & (attack == t)], cap)
                for t in np.unique(attack[y == 1])}
    return out, per_type


def models_available():
    pattern = re.compile(r"^(mlp_.+)_s\d+_(fabrication|masquerade)\.npz$")
    names = sorted({m.group(1) for p in SCORES.glob("mlp_*.npz")
                    if (m := pattern.match(p.name))})
    groups = {"MLP pooled": ["mlp_pooled"], "MLP FedAvg": ["mlp_fedavg"],
              "MLP FedProx": ["mlp_fedprox"],
              "MLP single": [n for n in names if n.startswith("mlp_single_")]}
    return {k: v for k, v in groups.items() if v}


def analyse():
    TABLES.mkdir(parents=True, exist_ok=True)
    sep_rows, type_rows = [], []
    for sset in SETS:
        w = SCORES / f"windows_{sset}.npz"
        if not w.exists():
            continue
        w = np.load(w)
        y, attack = w["target_label"], w["target_attack"]

        # Random Forest: one deterministic model, saved by run_road.py
        f = ROAD / "scores" / f"rf_pooled_{sset}.npz"
        if f.exists():
            z = np.load(f)
            st, pt = run_stats(z["target_score"], z["target_benign_score"],
                               float(z["thr_cal"]), z["target_label"],
                               z["target_attack"], None)
            sep_rows.append({"road_set": sset, "model": "RF pooled", "seed": 0, **st})
            for t, v in pt.items():
                type_rows.append({"road_set": sset, "model": "RF pooled", "seed": 0,
                                  "attack_type": t, "auroc": v})

        for model, names in models_available().items():
            for seed in rfd.SEEDS:
                per_seed_sep, per_seed_type = [], []
                for n in names:            # MLP single: one model per vehicle
                    f = SCORES / f"{n}_s{seed}_{sset}.npz"
                    if not f.exists():
                        continue
                    z = np.load(f)
                    st, pt = run_stats(z["target_score"], z["benign_score"],
                                       float(z["thr_cal"]), y, attack, None)
                    per_seed_sep.append(st)
                    per_seed_type.append(pt)
                if not per_seed_sep:
                    continue
                # average the single-vehicle models within a seed first
                sep_rows.append({"road_set": sset, "model": model, "seed": seed,
                                 **pd.DataFrame(per_seed_sep).mean().to_dict()})
                for t, v in pd.DataFrame(per_seed_type).mean().items():
                    type_rows.append({"road_set": sset, "model": model, "seed": seed,
                                      "attack_type": t, "auroc": v})

    sep = pd.DataFrame(sep_rows)
    typ = pd.DataFrame(type_rows)
    sep_sum = sep.groupby(["road_set", "model"]).agg(["mean", "std"])
    sep_sum = sep_sum.drop(columns="seed", level=0)
    sep_sum.to_csv(TABLES / "separation.csv")

    typ["class"] = typ["attack_type"].map(attack_class)
    t_sum = (typ.groupby(["road_set", "attack_type", "class", "model"])["auroc"]
                .agg(["mean", "std", "count"]).reset_index())
    t_sum.to_csv(TABLES / "per_attack_auroc.csv", index=False)
    by_class = (typ.groupby(["road_set", "class", "model", "seed"])["auroc"].mean()
                   .groupby(["road_set", "class", "model"]).agg(["mean", "std"])
                   .reset_index())
    by_class.to_csv(TABLES / "per_attack_by_class.csv", index=False)

    order = ["RF pooled", "MLP single", "MLP pooled", "MLP FedAvg", "MLP FedProx"]
    print("=" * 100)
    print("SCORE SEPARATION ON ROAD, mean (seed sd) over all MLP seeds; RF has one model")
    print("=" * 100)
    cols = [("amb_above_cal", "{:.3%}"), ("cap_above_cal", "{:.3%}"),
            ("att_above_cal", "{:.3%}"), ("auroc_vs_ambient", "{:.3f}"),
            ("auroc_vs_captures", "{:.3f}"), ("amb_ge_median_attack", "{:.1%}")]
    for sset in SETS:
        print(f"\n  {sset}")
        print(f"  {'model':<13}" + "".join(f"{c:>24}" for c, _ in cols))
        for model in order:
            if (sset, model) not in sep_sum.index:
                continue
            r = sep_sum.loc[(sset, model)]
            cells = ""
            for c, fmt in cols:
                mean, sd = r[(c, "mean")], r[(c, "std")]
                cell = fmt.format(mean) + ("" if pd.isna(sd) else
                                           " (" + fmt.format(sd) + ")")
                cells += f"{cell:>24}"
            print(f"  {model:<13}{cells}")

    print("\n" + "=" * 100)
    print("AUROC PER ROAD ATTACK TYPE, mean (seed sd) over all MLP seeds; "
          "against the benign windows of the same captures")
    print("=" * 100)
    for sset in SETS:
        sub = t_sum[t_sum["road_set"] == sset]
        if sub.empty:
            continue
        print(f"\n  {sset}")
        print(f"  {'attack type':<36}{'class':<13}" + "".join(f"{m:>16}" for m in order))
        for (atk, cls), g in sub.groupby(["attack_type", "class"]):
            cells = ""
            for model in order:
                r = g[g["model"] == model]
                if r.empty:
                    cells += f"{'-':>16}"
                    continue
                mean, sd = r["mean"].iloc[0], r["std"].iloc[0]
                cell = f"{mean:.3f}" + ("" if pd.isna(sd) else f" ({sd:.3f})")
                cells += f"{cell:>16}"
            print(f"  {atk:<36}{cls:<13}{cells}")
    print("\n  mean AUROC per class (descriptive; few attack types per class)")
    print("  " + by_class.round(3).to_string(index=False).replace("\n", "\n  "))
    if (typ["class"] == "UNMAPPED").any():
        print("  WARNING: unmapped attack types: "
              + ", ".join(sorted(typ.loc[typ["class"] == "UNMAPPED", "attack_type"]
                                 .unique())))
    print(f"\n  tables in {TABLES}/")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "run":
        run()
    elif what == "analyse":
        analyse()
    elif what == "check":
        check_reproduction()
    else:
        print(__doc__)
