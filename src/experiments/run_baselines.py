"""
Random Forest baselines under leave-one-vehicle-out, with source-only
selection and separate zero-shot and benign-calibrated results.

What is run
-----------
For every held-out vehicle H:

1. SELECTION, on the three source vehicles only (common.py explains the
   inner loop). A pooled Random Forest is trained for each feature set on
   two sources and scored on the third, three times over. From those
   inner scores:
     feature set  highest mean inner AUPRC
     budget B     highest mean inner MCC at the target-calibrated rule
   H is not touched.

2. EVALUATION on H, for all four feature sets (the feature-representation
   analysis) and every training mode:
     single             one source vehicle
     pooled_3           three sources, sampled in proportion to their size
     pooled_3_balanced  same volume, equal windows per source
     pooled_3_matched   equal per source, cut to one vehicle's volume
   Rows with the selected feature set are marked selected=True. Those are
   the headline results; the rest are the feature analysis.

Fixed settings, chosen before any experiment and not tuned: 300 trees,
min_samples_leaf=2, class_weight="balanced_subsample", windows of 100
messages, 15% of training windows held back for calibration, pooled
training capped at MAX_TRAIN windows.

Outputs, in results/
  rf_selection_log.csv  every inner-loop score behind every choice
  rf_selected.csv       the choice made for each held-out vehicle
  rf_results.csv        one row per train/test evaluation
  rf_per_attack.csv     recall per attack type at the zs, zsb, cal and
                        oracle thresholds
  rf_importance.csv     feature importances
  scores/rf_<vehicle>.npz  per-window scores of the selected pooled_3
                        model, for the score-distribution figure

Run from the project root:
    QUICK=1 python src/experiments/run_baselines.py    (pipeline check)
    python src/experiments/run_baselines.py            (the real run)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BUDGETS, FEATURE_SETS, QUICK, RESULTS, VEHICLES,  # noqa: E402
                    attack_types_trained_on, best_threshold, evaluate,
                    feature_columns, heldout_benign, load_all, per_attack_rows,
                    scored,
                    pick_best, pick_budget, save_scores, subsample,
                    threshold_rules)

N_TREES = 40 if QUICK else 300
MIN_SAMPLES_LEAF = 2
MAX_TRAIN = 400_000
RF_SEED = 42
POOLED_MODES = ["pooled_3", "pooled_3_balanced", "pooled_3_matched"]


def fit_model(X, y):
    model = RandomForestClassifier(
        n_estimators=N_TREES, min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced_subsample", n_jobs=-1, random_state=RF_SEED)
    model.fit(X, y)
    return model


def score(model, df, cols):
    if len(df) == 0:
        return np.array([])
    return model.predict_proba(df[cols])[:, 1]


def proportional_pool(parts, sources):
    """Sources pooled in proportion to their size, capped for runtime."""
    fits = [parts[v][0] for v in sources]
    n = min(min(len(f) for f in fits) * len(sources), MAX_TRAIN)
    return subsample(pd.concat(fits, ignore_index=True), n)


def pooled_variants(parts, sources):
    fits = [parts[v][0] for v in sources]
    smallest = min(len(f) for f in fits)
    balanced = pd.concat([subsample(f, smallest) for f in fits],
                         ignore_index=True)
    matched_total = int(np.mean([len(f) for f in fits]))
    per_source = matched_total // len(sources)
    matched = pd.concat([subsample(f, per_source) for f in fits],
                        ignore_index=True)
    return {"pooled_3": proportional_pool(parts, sources),
            "pooled_3_balanced": balanced,
            "pooled_3_matched": matched}


# ----------------------------------------------------------------------
# 1. source-only selection
# ----------------------------------------------------------------------

def select(held_out, data, parts, clean, evalset):
    sources = [v for v in VEHICLES if v != held_out]
    log = []
    for pretend in sources:
        inner = [v for v in sources if v != pretend]
        train = proportional_pool(parts, inner)
        calib = pd.concat([parts[v][1] for v in inner], ignore_index=True)
        for group in FEATURE_SETS:
            cols = feature_columns(data[VEHICLES[0]], group)
            model = fit_model(train[cols], train["label"])
            thr_zs, _ = best_threshold(calib["label"].to_numpy(),
                                       score(model, calib, cols))
            rules = {"zs": thr_zs}
            p_clean = score(model, clean[pretend], cols)
            rules.update({k: v for k, v in threshold_rules(
                thr_zs, np.array([]), p_clean, 0.0).items()
                if k.startswith("cal_")})
            row = {"held_out": held_out, "pretend_target": pretend,
                   "inner_sources": "+".join(inner), "feature_set": group}
            row.update(evaluate(evalset[pretend]["label"].to_numpy(),
                                score(model, evalset[pretend], cols), rules))
            log.append(row)
            print(f"    inner: train {'+'.join(inner):<20} -> {pretend:<10} "
                  f"{group:<8} AUPRC {row['auprc']:.3f}   "
                  + "  ".join(f"{t} {row[f'mcc_cal_{t}']:.3f}" for t in BUDGETS))
    log = pd.DataFrame(log)

    group, _, auprc_means = pick_best(log, "feature_set", FEATURE_SETS, "auprc")
    budget, mcc_means = pick_budget(log[log["feature_set"] == group])

    used = f"inner leave-one-source-out on {', '.join(sources)}"
    chosen = [
        {"family": "rf", "held_out": held_out, "parameter": "feature_set",
         "chosen": group, "criterion": "mean inner AUPRC",
         "scores": "; ".join(f"{k} {v:.4f}" for k, v in auprc_means.items()),
         "data_used": used},
        {"family": "rf", "held_out": held_out, "parameter": "budget",
         "chosen": budget, "criterion": f"mean inner MCC at cal ({group})",
         "scores": "; ".join(f"{k} {v:.4f}" for k, v in mcc_means.items()),
         "data_used": used},
    ]
    print(f"  selected for {held_out}: feature set {group}, budget {budget} "
          f"({BUDGETS[budget]:.2%} false alarms)")
    return {"feature_set": group, "budget": budget}, log, chosen


# ----------------------------------------------------------------------
# 2. evaluation
# ----------------------------------------------------------------------

def main():
    RESULTS.mkdir(exist_ok=True)
    t_start = time.time()
    print("loading features ...")
    data, parts, clean, evalset = load_all()

    print("\n" + "=" * 84)
    print("1. SOURCE-ONLY SELECTION")
    print("=" * 84)
    sel, logs, chosen = {}, [], []
    for held_out in VEHICLES:
        print(f"\n  held out {held_out} (not used here)")
        sel[held_out], log, ch = select(held_out, data, parts, clean, evalset)
        logs.append(log)
        chosen += ch
    pd.concat(logs).to_csv(RESULTS / "rf_selection_log.csv", index=False)
    pd.DataFrame(chosen).to_csv(RESULTS / "rf_selected.csv", index=False)

    print("\n" + "=" * 84)
    print("2. EVALUATION ON THE HELD-OUT VEHICLE")
    print("=" * 84)
    results, attack_rows, importance_rows = [], [], []

    def record(row, df, prob, rules, seen, info):
        results.append(row)
        keep = {t: rules[t] for t in ("zs", "zsb", "cal") if t in rules}
        keep["oracle"] = row["thr_oracle"]
        attack_rows.extend(per_attack_rows(df, prob, keep, seen, info))

    for group in FEATURE_SETS:
        cols = feature_columns(data[VEHICLES[0]], group)
        print(f"\nfeature set: {group}  ({len(cols)} features)")

        # ---- single vehicle -----------------------------------------
        for src in VEHICLES:
            fit, calib, test = parts[src]
            test = scored(test)
            model = fit_model(fit[cols], fit["label"])
            thr_zs, _ = best_threshold(calib["label"].to_numpy(),
                                       score(model, calib, cols))
            p_bsrc = score(model, heldout_benign(parts, [src]), cols)
            seen = attack_types_trained_on(fit)
            importance_rows += [{"feature_set": group, "mode": "single",
                                 "train_vehicles": src, "feature": c,
                                 "importance": float(i)}
                                for c, i in zip(cols, model.feature_importances_)]

            prob = score(model, test, cols)
            info = {"family": "rf", "feature_set": group, "mode": "single",
                    "train_vehicles": src, "test_vehicle": src,
                    "held_out": src, "setting": "within", "seed": RF_SEED,
                    "selected": group == sel[src]["feature_set"]}
            row = {**info, "budget": ""}
            row.update(evaluate(test["label"].to_numpy(), prob,
                                {"05": 0.5, "zs": thr_zs}))
            record(row, test, prob, {"zs": thr_zs}, seen, info)

            for tgt in [v for v in VEHICLES if v != src]:
                b = sel[tgt]["budget"]
                prob = score(model, evalset[tgt], cols)
                rules = threshold_rules(thr_zs, p_bsrc,
                                        score(model, clean[tgt], cols), BUDGETS[b])
                info = {**info, "test_vehicle": tgt, "held_out": tgt,
                        "setting": "cross",
                        "selected": group == sel[tgt]["feature_set"]}
                row = {**info, "budget": b}
                row.update(evaluate(evalset[tgt]["label"].to_numpy(), prob, rules))
                record(row, evalset[tgt], prob, rules, seen, info)
            print(f"  single {src:<10} done")

        # ---- leave one vehicle out ----------------------------------
        for held_out in VEHICLES:
            sources = [v for v in VEHICLES if v != held_out]
            label = "+".join(sources)
            b = sel[held_out]["budget"]
            is_sel = group == sel[held_out]["feature_set"]
            src_calib = pd.concat([parts[v][1] for v in sources], ignore_index=True)
            src_test = scored(pd.concat([parts[v][2] for v in sources],
                                        ignore_index=True))
            benign_src = heldout_benign(parts, sources)

            for mode, train in pooled_variants(parts, sources).items():
                t0 = time.time()
                model = fit_model(train[cols], train["label"])
                thr_zs, _ = best_threshold(src_calib["label"].to_numpy(),
                                           score(model, src_calib, cols))
                p_bsrc = score(model, benign_src, cols)
                p_clean = score(model, clean[held_out], cols)
                prob = score(model, evalset[held_out], cols)
                rules = threshold_rules(thr_zs, p_bsrc, p_clean, BUDGETS[b])
                seen = attack_types_trained_on(train)
                importance_rows += [{"feature_set": group, "mode": mode,
                                     "train_vehicles": label, "feature": c,
                                     "importance": float(i)}
                                    for c, i in zip(cols, model.feature_importances_)]

                info = {"family": "rf", "feature_set": group, "mode": mode,
                        "train_vehicles": label, "test_vehicle": held_out,
                        "held_out": held_out, "setting": "cross",
                        "seed": RF_SEED, "selected": is_sel}
                row = {**info, "budget": b}
                row.update(evaluate(evalset[held_out]["label"].to_numpy(),
                                    prob, rules))
                record(row, evalset[held_out], prob, rules, seen, info)

                p_src = score(model, src_test, cols)
                src_rules = {"05": 0.5, "zs": thr_zs, "zsb": rules["zsb"]}
                src_info = {**info, "test_vehicle": "sources",
                            "setting": "within_sources"}
                srow = {**src_info, "budget": b}
                srow.update(evaluate(src_test["label"].to_numpy(), p_src,
                                     src_rules))
                record(srow, src_test, p_src, src_rules, seen, src_info)

                print(f"  {mode:<18} hold out {held_out:<10} "
                      f"{time.time() - t0:>4.0f}s   "
                      f"zs {row['mcc_zs']:.3f}  zsb {row['mcc_zsb']:.3f}  "
                      f"cal {row['mcc_cal']:.3f}  oracle {row['mcc_oracle']:.3f}  "
                      f"AUROC {row['auroc']:.3f}  AUPRC {row['auprc']:.3f}"
                      + ("   <- selected" if is_sel and mode == "pooled_3" else ""))

                if is_sel and mode == "pooled_3":
                    save_scores(
                        f"rf_{held_out}",
                        target_score=prob.astype(np.float32),
                        target_label=evalset[held_out]["label"].to_numpy(np.int8),
                        target_attack=evalset[held_out]["attack_type"].to_numpy(str),
                        target_benign_score=p_clean.astype(np.float32),
                        source_score=p_src.astype(np.float32),
                        source_label=src_test["label"].to_numpy(np.int8),
                        thr_zs=thr_zs, thr_zsb=rules["zsb"], thr_cal=rules["cal"],
                        thr_oracle=row["thr_oracle"],
                        thr_oracle_source=srow["thr_oracle"],
                        feature_set=group, budget=b, mode=mode)

    res = pd.DataFrame(results)
    res.to_csv(RESULTS / "rf_results.csv", index=False)
    pd.DataFrame(attack_rows).to_csv(RESULTS / "rf_per_attack.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(RESULTS / "rf_importance.csv", index=False)

    summarise(res, sel)
    print(f"\ntotal runtime {(time.time() - t_start) / 60:.0f} min")
    print(f"saved rf_results.csv, rf_per_attack.csv, rf_importance.csv, "
          f"rf_selection_log.csv, rf_selected.csv and scores/ in {RESULTS}/")


def summarise(res, sel):
    cross = res[res["setting"] == "cross"]

    print(f"\n\n{'#' * 84}\nSUMMARY (Random Forest)\n{'#' * 84}")

    print("\n--- choices made on the source vehicles ---")
    for v in VEHICLES:
        print(f"  held out {v:<10} feature set {sel[v]['feature_set']:<8} "
              f"budget {sel[v]['budget']}")

    print("\n--- headline: pooled_3, selected feature set, per held-out vehicle ---")
    print(f"  {'held out':<11}{'AUROC':>7}{'AUPRC':>7}   {'zero-shot':^15}"
          f"   {'calibrated':>10}{'oracle':>8}")
    print(f"  {'':<11}{'':>7}{'':>7}   {'zs':>7}{'zsb':>8}   {'cal':>10}{'':>8}")
    head = cross[(cross["mode"] == "pooled_3") & cross["selected"]]
    for v in VEHICLES:
        r = head[head["held_out"] == v].iloc[0]
        print(f"  {v:<11}{r['auroc']:>7.3f}{r['auprc']:>7.3f}   "
              f"{r['mcc_zs']:>7.3f}{r['mcc_zsb']:>8.3f}   "
              f"{r['mcc_cal']:>10.3f}{r['mcc_oracle']:>8.3f}")
    print(f"  {'MEAN':<11}{head['auroc'].mean():>7.3f}{head['auprc'].mean():>7.3f}"
          f"   {head['mcc_zs'].mean():>7.3f}{head['mcc_zsb'].mean():>8.3f}   "
          f"{head['mcc_cal'].mean():>10.3f}{head['mcc_oracle'].mean():>8.3f}")
    print("  (MCC columns; zsb and cal use the same budget, so cal - zsb is the"
          "\n   part of the gap that comes from where the threshold sits)")

    print("\n--- feature representation: pooled_3, mean over held-out vehicles ---")
    print(f"  {'features':<10}{'AUROC':>8}{'AUPRC':>8}{'zs':>8}{'cal':>8}{'oracle':>8}")
    for g in FEATURE_SETS:
        s = cross[(cross["mode"] == "pooled_3") & (cross["feature_set"] == g)]
        print(f"  {g:<10}{s['auroc'].mean():>8.3f}{s['auprc'].mean():>8.3f}"
              f"{s['mcc_zs'].mean():>8.3f}{s['mcc_cal'].mean():>8.3f}"
              f"{s['mcc_oracle'].mean():>8.3f}")
    print("  (the budget in 'cal' is each fold's selected budget)")

    print("\n--- single vs pooled, selected feature set, MCC at cal ---")
    print(f"  {'held out':<11}{'single':>9}{'pooled_3':>10}{'balanced':>10}"
          f"{'matched':>9}")
    sel_rows = cross[cross["selected"]]
    for v in VEHICLES:
        s = sel_rows[sel_rows["held_out"] == v]
        vals = [s[s["mode"] == m]["mcc_cal"].mean()
                for m in ["single", *POOLED_MODES]]
        print(f"  {v:<11}" + "".join(f"{x:>9.3f} " for x in vals))


if __name__ == "__main__":
    main()
