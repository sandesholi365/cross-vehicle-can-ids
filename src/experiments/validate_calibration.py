"""
Final check of the two threshold settings on CAN-Train-and-Test.

THE TWO RULES (fixed; nothing here changes them)
------------------------------------------------
Zero-shot
    The threshold is the MCC-optimal cut-off on the source vehicles'
    calibration windows. It is moved to the unseen vehicle unchanged.

Benign target calibration
    Inputs: the model's scores on the unseen vehicle's attack-free
    recordings, and the false-alarm budget B chosen on the source vehicles.
    Threshold = the (1 - B) quantile of those scores (linear
    interpolation), so a share B of the attack-free windows lies above it.
    No attack windows and no labels from the unseen vehicle are used.

WHAT IS CHECKED, PER HELD-OUT VEHICLE
-------------------------------------
1. Zero-shot on normal traffic: the share of the unseen vehicle's
   attack-free windows that the zero-shot threshold flags. This is the
   false-alarm rate a zero-shot detector would actually have.

2. The calibration rule on fresh attack-free traffic. Each vehicle has
   several attack-free recordings. The rule is applied to all of them but
   one, and the false-alarm rate is measured on the one left out, in turn
   for each. If the rule is sound, this rate should be near B. This uses
   attack-free traffic only.

3. How many attack-free windows lie above the threshold at budget B
   (B times the number of windows). When this is around one, the threshold
   is set by a single window and is sensitive to it.

4. For reference, the false-alarm rate on benign windows inside the attack
   recordings (the mismatch reported in check_calibration.py).

All budgets are shown so the sensitivity is visible; the selected one is
marked. The selected pooled Random Forest and the selected MLP arms (first
seed) are retrained with the settings already chosen. Nothing is selected
or tuned here.

Outputs, in results/tables/
  calibration_validation.csv

Run from the project root (about 10 minutes):
    python src/experiments/validate_calibration.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BUDGETS, RESULTS, VEHICLES, best_threshold,  # noqa: E402
                    budget_threshold, feature_columns, load_all)
import run_baselines as rb  # noqa: E402
import run_federated as rfd  # noqa: E402

TABLES = RESULTS / "tables"


def selected(family):
    sel = pd.read_csv(RESULTS / f"{family}_selected.csv")
    return {(r["held_out"], r["parameter"]): r["chosen"] for _, r in sel.iterrows()}


def rf_models(v, data, parts, clean, evalset):
    """Yields (name, zs threshold, scorer) for the selected pooled RF."""
    sel = selected("rf")
    cols = feature_columns(data[VEHICLES[0]], sel[(v, "feature_set")])
    sources = [s for s in VEHICLES if s != v]
    train = rb.pooled_variants(parts, sources)["pooled_3"]
    model = rb.fit_model(train[cols], train["label"])
    calib = pd.concat([parts[s][1] for s in sources], ignore_index=True)
    thr_zs, _ = best_threshold(calib["label"].to_numpy(), rb.score(model, calib, cols))
    yield "RF pooled", sel[(v, "budget")], thr_zs, lambda df: rb.score(model, df, cols)


def mlp_models(v, data, parts, clean, evalset):
    """Yields the selected MLP arms (pooled, FedAvg, FedProx), first seed."""
    sel = selected("mlp")
    cols = feature_columns(data[VEHICLES[0]], sel[(v, "feature_set")])
    mu = float(sel[(v, "fedprox_mu")])
    sources = [s for s in VEHICLES if s != v]
    sc = rfd.Scaler(pd.concat([parts[s][0] for s in sources]), rfd.all_columns(data))
    clients = [(sc.X(parts[s][0], cols), rfd.labels(parts[s][0])) for s in sources]
    calib = pd.concat([parts[s][1] for s in sources], ignore_index=True)
    for arm, name in [("pooled", "MLP pooled"), ("fedavg", "MLP FedAvg"),
                      ("fedprox", "MLP FedProx")]:
        model, _ = rfd.train_arm(arm, clients, len(cols), seed=0, mu=mu)
        thr_zs, _ = best_threshold(calib["label"].to_numpy(),
                                   rfd.predict(model, sc.X(calib, cols)))
        yield (name, sel[(v, "budget")], thr_zs,
               lambda df, m=model: rfd.predict(m, sc.X(df, cols)))


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    print("loading features ...")
    data, parts, clean, evalset = load_all()
    rows = []

    for v in VEHICLES:
        recs = sorted(clean[v]["recording"].unique())
        sizes = clean[v]["recording"].value_counts()
        print(f"\n{'=' * 92}\nheld out {v}: attack-free recordings "
              + ", ".join(f"{r.split('__')[0]} ({sizes[r]:,} windows)" for r in recs)
              + f"\n{'=' * 92}")

        for gen in (rf_models, mlp_models):
            for name, chosen, thr_zs, score in gen(v, data, parts, clean, evalset):
                s_clean = score(clean[v])
                rec_of = clean[v]["recording"].to_numpy()
                ev = evalset[v]
                s_eval = score(ev)
                benign_eval = ev["label"].to_numpy() == 0

                zs_fpr = float(np.mean(s_clean >= thr_zs))
                print(f"\n  {name}   zero-shot threshold flags {zs_fpr:.2%} of "
                      f"this vehicle's attack-free windows")
                print(f"    {'budget':<10}{'windows':>9}   {'held-out attack-free':<26}"
                      f"{'benign in attack':>17}")
                print(f"    {'':<10}{'above':>9}   {'recording, each':<26}"
                      f"{'recordings':>17}")

                for tag, b in BUDGETS.items():
                    thr = budget_threshold(s_clean, b)
                    held = []
                    for r in recs:
                        fit = s_clean[rec_of != r]
                        test = s_clean[rec_of == r]
                        held.append(float(np.mean(test >= budget_threshold(fit, b))))
                    in_attack = float(np.mean(s_eval[benign_eval] >= thr))
                    mark = " <" if tag == chosen else ""
                    held_txt = "  ".join(f"{h:.3%}" for h in held)
                    print(f"    {b:<10.2%}{b * len(s_clean):>9.1f}   {held_txt:<26}"
                          f"{in_attack:>17.2%}{mark}")
                    rows.append({
                        "held_out": v, "model": name, "budget": b,
                        "selected_budget": tag == chosen,
                        "attack_free_windows": len(s_clean),
                        "windows_above_threshold": b * len(s_clean),
                        "zero_shot_fpr_attack_free": zs_fpr,
                        "heldout_recording_fpr_mean": float(np.mean(held)),
                        "heldout_recording_fpr_max": float(np.max(held)),
                        **{f"heldout_fpr_{r.split('__')[0]}": h
                           for r, h in zip(recs, held)},
                        "fpr_benign_in_attack_recordings": in_attack,
                    })

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "calibration_validation.csv", index=False)

    sel = out[out["selected_budget"]]
    print(f"\n{'=' * 92}\nSUMMARY at the selected budget "
          f"(held-out attack-free recording, mean over recordings)\n{'=' * 92}")
    print(f"  {'model':<13}" + "".join(f"{v[:9]:>12}" for v in VEHICLES))
    for m in sel["model"].unique():
        cells = "".join(
            f"{sel[(sel['model'] == m) & (sel['held_out'] == v)]['heldout_recording_fpr_mean'].iloc[0]:>12.3%}"
            for v in VEHICLES)
        print(f"  {m:<13}{cells}")
    print(f"  promised: {sel['budget'].iloc[0]:.2%}")
    print(f"\n  wrote {TABLES}/calibration_validation.csv")


if __name__ == "__main__":
    main()
