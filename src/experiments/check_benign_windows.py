"""
Why does benign calibration miss its false-alarm budget?

check_calibration.py showed that the threshold set on a vehicle's
attack-free recordings flags far more than the promised share of the benign
windows inside its attack recordings. There are two possible reasons, and
they mean different things for the thesis:

  (a) Windows close to an attack. A window counts as benign if it contains
      no injected message, but the traffic around an injection can still be
      disturbed (other ECUs react, timing shifts). If the false alarms sit
      next to attack windows, the "excess" is largely a labelling boundary
      effect, not a calibration failure.

  (b) Different recording conditions. If the false alarms are spread over
      windows far from any attack, the benign traffic in the attack
      recordings simply differs from the attack-free recordings (a
      different session, drive or operating state). Then calibration fails
      because the attack-free recordings do not represent normal traffic
      well enough.

This script retrains the selected pooled Random Forest and the selected
pooled MLP (first seed) for each held-out vehicle, with exactly the settings
already chosen on the source vehicles, and breaks the false alarms down by
distance to the nearest attack window, by source set and by recording.
Nothing is selected or tuned here.

Outputs, in results/tables/
  benign_by_distance.csv     false-alarm rate by distance to an attack
  benign_by_recording.csv    false-alarm rate per recording

Run from the project root (about 10 minutes):
    python src/experiments/check_benign_windows.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BUDGETS, RESULTS, VEHICLES, budget_threshold,  # noqa: E402
                    feature_columns, load_all)
import run_baselines as rb  # noqa: E402
import run_federated as rfd  # noqa: E402

TABLES = RESULTS / "tables"
BUCKETS = [(1, 1, "next to an attack (1 window)"),
           (2, 5, "2-5 windows away"),
           (6, 20, "6-20 windows away"),
           (21, np.inf, "more than 20 windows away")]


def distance_to_attack(df):
    """For every window, how many windows away the nearest attack window is."""
    out = pd.Series(np.inf, index=df.index)
    order = "window_index" if "window_index" in df.columns else None
    for _, grp in df.groupby("recording"):
        g = grp.sort_values(order) if order else grp
        pos = (g[order].to_numpy() if order else np.arange(len(g)))
        att = np.sort(pos[g["label"].to_numpy() == 1])
        if len(att) == 0:
            continue
        i = np.searchsorted(att, pos)
        left = np.abs(pos - att[np.clip(i - 1, 0, len(att) - 1)])
        right = np.abs(att[np.clip(i, 0, len(att) - 1)] - pos)
        out.loc[g.index] = np.minimum(left, right)
    return out


def selected(family):
    sel = pd.read_csv(RESULTS / f"{family}_selected.csv")
    return {(r["held_out"], r["parameter"]): r["chosen"] for _, r in sel.iterrows()}


def rf_scores(held_out, data, parts, clean, evalset, sel):
    group = sel[(held_out, "feature_set")]
    cols = feature_columns(data[VEHICLES[0]], group)
    sources = [v for v in VEHICLES if v != held_out]
    train = rb.pooled_variants(parts, sources)["pooled_3"]
    model = rb.fit_model(train[cols], train["label"])
    return (rb.score(model, clean[held_out], cols),
            rb.score(model, evalset[held_out], cols))


def mlp_scores(held_out, data, parts, clean, evalset, sel):
    group = sel[(held_out, "feature_set")]
    cols = feature_columns(data[VEHICLES[0]], group)
    sources = [v for v in VEHICLES if v != held_out]
    sc = rfd.Scaler(pd.concat([parts[v][0] for v in sources]), rfd.all_columns(data))
    clients = [(sc.X(parts[v][0], cols), rfd.labels(parts[v][0])) for v in sources]
    model, _ = rfd.train_arm("pooled", clients, len(cols), seed=0)
    return (rfd.predict(model, sc.X(clean[held_out], cols)),
            rfd.predict(model, sc.X(evalset[held_out], cols)))


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    print("loading features ...")
    data, parts, clean, evalset = load_all()
    families = [("RF pooled", "rf", rf_scores), ("MLP pooled", "mlp", mlp_scores)]

    dist_rows, rec_rows = [], []
    for name, fam, scorer in families:
        sel = selected(fam)
        print(f"\n{'=' * 88}\n{name}: false alarms at the calibrated threshold\n{'=' * 88}")
        for v in VEHICLES:
            budget = BUDGETS[sel[(v, "budget")]]
            p_clean, p_eval = scorer(v, data, parts, clean, evalset, sel)
            thr = budget_threshold(p_clean, budget)
            ev = evalset[v]
            benign = ev["label"].to_numpy() == 0
            alarm = p_eval >= thr
            dist = distance_to_attack(ev).to_numpy()

            print(f"\n  held out {v}   (budget {budget:.2%}; reached "
                  f"{alarm[benign].mean():.2%} on {benign.sum():,} benign windows)")
            print(f"    {'benign windows':<30}{'windows':>10}{'false alarms':>14}"
                  f"{'rate':>9}")
            for lo, hi, label in BUCKETS:
                m = benign & (dist >= lo) & (dist <= hi)
                rate = alarm[m].mean() if m.any() else np.nan
                print(f"    {label:<30}{m.sum():>10,}{alarm[m].sum():>14,}"
                      f"{rate:>9.2%}")
                dist_rows.append({"model": name, "held_out": v, "bucket": label,
                                  "windows": int(m.sum()),
                                  "false_alarms": int(alarm[m].sum()),
                                  "rate": rate, "budget": budget})

            # by source set: the recording name ends in __set_XX_tag
            sets = (ev["recording"].str.extract(r"__(set_\d+)_")[0]
                    .fillna("unknown").to_numpy())
            print(f"    {'by source set':<30}")
            for s in sorted(set(sets)):
                m = benign & (sets == s)
                print(f"      {s:<28}{m.sum():>10,}{alarm[m].sum():>14,}"
                      f"{alarm[m].mean():>9.2%}")

            recs = []
            for rec in sorted(ev["recording"].unique()):
                m = benign & (ev["recording"].to_numpy() == rec)
                if m.any():
                    recs.append({"model": name, "held_out": v, "recording": rec,
                                 "benign_windows": int(m.sum()),
                                 "false_alarms": int(alarm[m].sum()),
                                 "rate": float(alarm[m].mean())})
            rec_rows += recs
            top = sorted(recs, key=lambda r: r["false_alarms"], reverse=True)[:5]
            print("    recordings with the most false alarms")
            for r in top:
                print(f"      {r['recording']:<36}{r['benign_windows']:>8,}"
                      f"{r['false_alarms']:>8,}{r['rate']:>9.2%}")

    pd.DataFrame(dist_rows).to_csv(TABLES / "benign_by_distance.csv", index=False)
    pd.DataFrame(rec_rows).to_csv(TABLES / "benign_by_recording.csv", index=False)
    print(f"\n  wrote {TABLES}/benign_by_distance.csv and benign_by_recording.csv")


if __name__ == "__main__":
    main()
