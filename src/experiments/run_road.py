"""
External validation on ROAD with the methodology frozen on CAN-Train-and-Test.

Nothing in this script is chosen on ROAD. It reuses the functions of
run_baselines.py and run_federated.py unchanged:

  - the same features, windows, preprocessing and model settings
  - the same inner selection loop, now run over all four CAN-Train-and-Test
    vehicles (each in turn the pretend unseen vehicle, trained on the other
    three), to choose the feature set, the false-alarm budget B and the
    FedProx mu for the final models
  - final models trained on all four CAN-Train-and-Test vehicles:
      Random Forest: single (each vehicle alone), pooled, pooled balanced,
                     pooled at one vehicle's data volume; all feature sets
      MLP:           single (each vehicle alone), pooled, FedAvg and
                     FedProx with four clients (one per vehicle); ten seeds
  - the same threshold rules:
      zs   MCC-optimal on the training vehicles' calibration windows
      zsb  budget B on the training vehicles' held-out attack-free windows
      cal  budget B on ROAD's ambient (attack-free) recordings
      oracle, as an upper bound only

ROAD is scored on its attack captures only; ambient recordings are used only
for the calibrated threshold. Fabrication and masquerade captures are scored
as two separate sets (road_set column), because the masquerade versions are
made from the same recordings as the fabrication ones.

Outputs, in results/road/
  rf_selected.csv, mlp_selected.csv, *_selection_log.csv
  rf_results.csv, mlp_results.csv, rf_per_attack.csv, mlp_per_attack.csv
  scores/*.npz

Run once, from the project root:
    QUICK=1 python src/experiments/run_road.py     (pipeline check only)
    python src/experiments/run_road.py             (the real run)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BUDGETS, FEATURE_SETS, QUICK, RESULTS, VEHICLES,  # noqa: E402
                    attack_types_trained_on, best_threshold, evaluate,
                    feature_columns, heldout_benign, load_all, load_vehicle,
                    per_attack_rows, scored, threshold_rules)
import run_baselines as rb  # noqa: E402
import run_federated as rfd  # noqa: E402

OUT = RESULTS / ("road_quick" if QUICK else "road")
SCORES = OUT / "scores"
MODE_NAMES = {"pooled_3": "pooled", "pooled_3_balanced": "pooled_balanced",
              "pooled_3_matched": "pooled_matched"}


def load_road():
    df = load_vehicle("road")
    clean = df[df["attack_type"] == "attack-free"]
    attacks = scored(df)
    masq = attacks["attack_type"].str.endswith("_masquerade")
    sets = {"fabrication": attacks[~masq], "masquerade": attacks[masq]}
    print(f"  road       attack-free {len(clean):,} windows; scored: "
          + "; ".join(f"{k} {len(v):,} ({int(v['label'].sum()):,} attack)"
                      for k, v in sets.items()))
    return clean, {k: v for k, v in sets.items() if len(v)}


def save(name, **arrays):
    SCORES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SCORES / f"{name}.npz", **arrays)


def score_sets(scorer, road_sets):
    return {k: scorer(v) for k, v in road_sets.items()}


def record(results, attack_rows, info, road_sets, probs, rules, seen):
    """One result row and per-attack rows per ROAD evaluation set."""
    rows = []
    for name, df in road_sets.items():
        row = {**info, "road_set": name}
        row.update(evaluate(df["label"].to_numpy(), probs[name], rules))
        results.append(row)
        keep = {t: rules[t] for t in ("zs", "zsb", "cal")}
        keep["oracle"] = row["thr_oracle"]
        attack_rows += per_attack_rows(df, probs[name], keep, seen,
                                       {**info, "road_set": name})
        rows.append(row)
    return rows


# ----------------------------------------------------------------------
# Random Forest
# ----------------------------------------------------------------------

def run_rf(data, parts, clean, evalset, road_clean, road_sets):
    print("\n" + "=" * 84 + "\nRANDOM FOREST: selection on the four CAN-Train-and-Test "
          "vehicles\n" + "=" * 84)
    sel, log, chosen = rb.select("road", data, parts, clean, evalset)
    log.to_csv(OUT / "rf_selection_log.csv", index=False)
    pd.DataFrame(chosen).to_csv(OUT / "rf_selected.csv", index=False)
    b = sel["budget"]

    results, attack_rows = [], []
    src_calib = pd.concat([parts[v][1] for v in VEHICLES], ignore_index=True)
    src_test = scored(pd.concat([parts[v][2] for v in VEHICLES], ignore_index=True))
    benign_src = heldout_benign(parts, VEHICLES)

    for group in FEATURE_SETS:
        cols = feature_columns(data[VEHICLES[0]], group)
        is_sel = group == sel["feature_set"]
        print(f"\nfeature set: {group}" + ("   <- selected" if is_sel else ""))

        runs = []
        for v in VEHICLES:
            runs.append(("single", v, parts[v][0], parts[v][1],
                         heldout_benign(parts, [v])))
        for mode, train in rb.pooled_variants(parts, VEHICLES).items():
            runs.append((MODE_NAMES[mode], "+".join(VEHICLES), train, src_calib,
                         benign_src))

        for mode, label, train, calib, benign in runs:
            t0 = time.time()
            model = rb.fit_model(train[cols], train["label"])

            def sc(df):
                return rb.score(model, df, cols)
            thr_zs, _ = best_threshold(calib["label"].to_numpy(), sc(calib))
            p_clean = sc(road_clean)
            rules = threshold_rules(thr_zs, sc(benign), p_clean, BUDGETS[b])
            probs = score_sets(sc, road_sets)
            info = {"family": "rf", "feature_set": group, "mode": mode,
                    "train_vehicles": label, "test_vehicle": "road",
                    "setting": "external", "seed": rb.RF_SEED,
                    "selected": is_sel, "budget": b}
            rows = record(results, attack_rows, info, road_sets, probs, rules,
                          attack_types_trained_on(train))
            main = rows[0]
            print(f"  {mode:<16} {label:<34}{time.time() - t0:>4.0f}s  "
                  f"zs {main['mcc_zs']:.3f}  cal {main['mcc_cal']:.3f}  "
                  f"oracle {main['mcc_oracle']:.3f}  AUROC {main['auroc']:.3f}")

            if mode == "pooled" and is_sel:
                p_src = sc(src_test)
                srow = {**info, "test_vehicle": "sources",
                        "setting": "within_sources", "road_set": ""}
                srow.update(evaluate(src_test["label"].to_numpy(), p_src,
                                     {"05": 0.5, "zs": thr_zs, "zsb": rules["zsb"]}))
                results.append(srow)
                for name, df in road_sets.items():
                    row = [r for r in rows if r["road_set"] == name][0]
                    save(f"rf_pooled_{name}",
                         target_score=probs[name].astype(np.float32),
                         target_label=df["label"].to_numpy(np.int8),
                         target_attack=df["attack_type"].to_numpy(str),
                         target_recording=df["recording"].to_numpy(str),
                         target_benign_score=p_clean.astype(np.float32),
                         target_benign_recording=road_clean["recording"].to_numpy(str),
                         source_score=p_src.astype(np.float32),
                         source_label=src_test["label"].to_numpy(np.int8),
                         thr_zs=thr_zs, thr_zsb=rules["zsb"], thr_cal=rules["cal"],
                         thr_oracle=row["thr_oracle"],
                         thr_oracle_source=srow["thr_oracle"],
                         feature_set=group, budget=b, mode=mode)

    pd.DataFrame(results).to_csv(OUT / "rf_results.csv", index=False)
    pd.DataFrame(attack_rows).to_csv(OUT / "rf_per_attack.csv", index=False)
    return sel


# ----------------------------------------------------------------------
# MLP
# ----------------------------------------------------------------------

def run_mlp(data, parts, clean, evalset, road_clean, road_sets):
    print("\n" + "=" * 84 + "\nMLP: selection on the four CAN-Train-and-Test "
          "vehicles\n" + "=" * 84)
    sel, log, chosen = rfd.select("road", data, parts, clean, evalset)
    log.to_csv(OUT / "mlp_selection_log.csv", index=False)
    pd.DataFrame(chosen).to_csv(OUT / "mlp_selected.csv", index=False)
    group, b, mu_sel = sel["feature_set"], sel["budget"], sel["mu"]
    cols = feature_columns(data[VEHICLES[0]], group)

    sc_all = rfd.Scaler(pd.concat([parts[v][0] for v in VEHICLES]),
                        rfd.all_columns(data))
    clients = [(sc_all.X(parts[v][0], cols), rfd.labels(parts[v][0]))
               for v in VEHICLES]
    src_calib = pd.concat([parts[v][1] for v in VEHICLES], ignore_index=True)
    src_test = scored(pd.concat([parts[v][2] for v in VEHICLES], ignore_index=True))
    X = {"calib": sc_all.X(src_calib, cols),
         "benign_src": sc_all.X(heldout_benign(parts, VEHICLES), cols),
         "src_test": sc_all.X(src_test, cols),
         "road_clean": sc_all.X(road_clean, cols)}
    X_sets = {k: sc_all.X(v, cols) for k, v in road_sets.items()}
    seen_all = attack_types_trained_on(pd.concat([parts[v][0] for v in VEHICLES]))
    label_all = "+".join(VEHICLES)

    runs = [("single", v, 0.0) for v in VEHICLES]
    runs += [("pooled", label_all, 0.0), ("fedavg", label_all, 0.0)]
    runs += [("fedprox", label_all, mu) for mu in rfd.MU_VALUES]

    print(f"\n  features {group}, budget {b}, mu {mu_sel}")
    results, attack_rows = [], []
    for seed in rfd.SEEDS:
        print(f"  seed {seed}")
        for arm, label, mu in runs:
            t0 = time.time()
            if arm == "single":
                i = VEHICLES.index(label)
                model = rfd.train_local(rfd.new_model(len(cols), seed), *clients[i],
                                        rfd.CENTRAL_EPOCHS, seed=seed)
                drift = np.nan
                calib_v = parts[label][1]
                thr_zs, _ = best_threshold(
                    calib_v["label"].to_numpy(),
                    rfd.predict(model, sc_all.X(calib_v, cols)))
                p_bsrc = rfd.predict(model, sc_all.X(heldout_benign(parts, [label]),
                                                     cols))
                seen = attack_types_trained_on(parts[label][0])
            else:
                model, drift = rfd.train_arm(arm, clients, len(cols), seed, mu)
                thr_zs, _ = best_threshold(src_calib["label"].to_numpy(),
                                           rfd.predict(model, X["calib"]))
                p_bsrc = rfd.predict(model, X["benign_src"])
                seen = seen_all

            p_clean = rfd.predict(model, X["road_clean"])
            rules = threshold_rules(thr_zs, p_bsrc, p_clean, BUDGETS[b])
            probs = {k: rfd.predict(model, x) for k, x in X_sets.items()}
            mode = f"fedprox_mu{mu}" if arm == "fedprox" else arm
            is_sel = arm != "fedprox" or mu == mu_sel
            info = {"family": "mlp", "feature_set": group, "mode": mode,
                    "arm": arm, "mu": mu if arm == "fedprox" else np.nan,
                    "train_vehicles": label, "test_vehicle": "road",
                    "setting": "external", "seed": seed, "selected": is_sel,
                    "budget": b, "client_drift": drift}
            rows = record(results, attack_rows, info, road_sets, probs, rules, seen)
            main = rows[0]

            if arm != "single" and is_sel and seed == rfd.SEEDS[0]:
                p_src = rfd.predict(model, X["src_test"])
                srow = {**info, "test_vehicle": "sources",
                        "setting": "within_sources", "road_set": ""}
                srow.update(evaluate(src_test["label"].to_numpy(), p_src,
                                     {"05": 0.5, "zs": thr_zs, "zsb": rules["zsb"]}))
                results.append(srow)
                for name, df in road_sets.items():
                    row = [r for r in rows if r["road_set"] == name][0]
                    save(f"mlp_{arm}_{name}",
                         target_score=probs[name].astype(np.float32),
                         target_label=df["label"].to_numpy(np.int8),
                         target_attack=df["attack_type"].to_numpy(str),
                         target_recording=df["recording"].to_numpy(str),
                         target_benign_score=p_clean.astype(np.float32),
                         target_benign_recording=road_clean["recording"].to_numpy(str),
                         source_score=p_src.astype(np.float32),
                         source_label=src_test["label"].to_numpy(np.int8),
                         thr_zs=thr_zs, thr_zsb=rules["zsb"], thr_cal=rules["cal"],
                         thr_oracle=row["thr_oracle"],
                         thr_oracle_source=srow["thr_oracle"],
                         feature_set=group, budget=b, mode=mode)

            name = f"single:{label}" if arm == "single" else mode
            print(f"    {name:<18}{time.time() - t0:>4.0f}s   zs {main['mcc_zs']:.3f}"
                  f"  cal {main['mcc_cal']:.3f}  oracle {main['mcc_oracle']:.3f}"
                  f"  AUPRC {main['auprc']:.3f}")

    pd.DataFrame(results).to_csv(OUT / "mlp_results.csv", index=False)
    pd.DataFrame(attack_rows).to_csv(OUT / "mlp_per_attack.csv", index=False)
    return sel


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print("loading features ...")
    data, parts, clean, evalset = load_all()
    road_clean, road_sets = load_road()

    rf_sel = run_rf(data, parts, clean, evalset, road_clean, road_sets)
    mlp_sel = run_mlp(data, parts, clean, evalset, road_clean, road_sets)

    print("\n" + "#" * 84)
    print(f"  RF  chosen on CAN-Train-and-Test: features {rf_sel['feature_set']}, "
          f"budget {rf_sel['budget']}")
    print(f"  MLP chosen on CAN-Train-and-Test: features {mlp_sel['feature_set']}, "
          f"budget {mlp_sel['budget']}, mu {mlp_sel['mu']}")
    print(f"  total runtime {(time.time() - t_start) / 60:.0f} min")
    print(f"  saved to {OUT}/  (summary: python src/experiments/analyse_road.py)")


if __name__ == "__main__":
    main()
