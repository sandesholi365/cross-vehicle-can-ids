"""
Does benign calibration deliver the false-alarm rate it promises?

The target-calibrated threshold ("cal") is placed so that a fraction B of
the new vehicle's ATTACK-FREE recordings would raise an alarm. It is then
used on the new vehicle's ATTACK recordings, which also contain many benign
windows. If those benign windows look like the attack-free recordings, the
false-alarm rate there is close to B. If they do not (for example because
the car was in a different operating state, such as accessory mode or
standstill), the real false-alarm rate can be far higher than B.

For every saved model this prints:
  - the budget B and the false-alarm rate actually reached on the benign
    windows that were scored
  - how the scores of the two kinds of benign traffic compare
  - the false-alarm rate per recording type, so the recordings that cause
    the excess can be named

Reads results/scores/*.npz (written by run_baselines.py and
run_federated.py). Writes results/tables/calibration_check.csv and
results/tables/calibration_by_recording_type.csv.

Run from the project root:
    python src/experiments/check_calibration.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUDGETS, RESULTS, SCORES_DIR, VEHICLES  # noqa: E402

TABLES = RESULTS / "tables"
MODELS = [("rf", "RF pooled"), ("mlp_pooled", "MLP pooled"),
          ("mlp_fedavg", "MLP FedAvg"), ("mlp_fedprox", "MLP FedProx")]


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    summary, by_type = [], []

    print("=" * 92)
    print("BENIGN CALIBRATION: PROMISED vs REACHED FALSE-ALARM RATE")
    print("=" * 92)
    print("  promised = budget B, measured on the attack-free recordings")
    print("  reached  = share of benign windows inside the scored (attack)")
    print("             recordings that are above the same threshold")

    for prefix, name in MODELS:
        files = {v: SCORES_DIR / f"{prefix}_{v}.npz" for v in VEHICLES}
        if not all(f.exists() for f in files.values()):
            continue
        print(f"\n  {name}")
        print(f"    {'held out':<11}{'promised':>9}{'reached':>9}{'x budget':>10}"
              f"   {'benign score, 99th pct':>24}   worst recording types")
        print(f"    {'':<11}{'':>9}{'':>9}{'':>10}   "
              f"{'attack-free':>12}{'in attacks':>12}")
        for v in VEHICLES:
            z = np.load(files[v])
            thr = float(z["thr_cal"])
            budget = BUDGETS[str(z["budget"])]
            s, y, t = z["target_score"], z["target_label"], z["target_attack"]
            clean = z["target_benign_score"]
            benign = y == 0
            reached = float(np.mean(s[benign] >= thr))
            q_clean = float(np.quantile(clean, 0.99))
            q_attack = float(np.quantile(s[benign], 0.99))

            rows = []
            for atk in sorted(set(t[benign])):
                m = benign & (t == atk)
                rows.append({"model": name, "held_out": v,
                             "recording_type": atk,
                             "benign_windows": int(m.sum()),
                             "false_alarm_rate": float(np.mean(s[m] >= thr)),
                             "false_alarms": int(np.sum(s[m] >= thr))})
            by_type += rows
            worst = sorted(rows, key=lambda r: r["false_alarms"], reverse=True)[:3]
            worst_txt = ", ".join(f"{r['recording_type']} {r['false_alarm_rate']:.0%}"
                                  for r in worst if r["false_alarms"] > 0)

            summary.append({"model": name, "held_out": v, "budget": budget,
                            "reached_fpr": reached,
                            "times_budget": reached / budget,
                            "q99_attack_free": q_clean,
                            "q99_benign_in_attack_recordings": q_attack,
                            "attack_free_windows": len(clean),
                            "benign_windows_scored": int(benign.sum())})
            print(f"    {v:<11}{budget:>9.2%}{reached:>9.2%}{reached / budget:>9.0f}x"
                  f"   {q_clean:>12.3f}{q_attack:>12.3f}   {worst_txt}")

    pd.DataFrame(summary).to_csv(TABLES / "calibration_check.csv", index=False)
    pd.DataFrame(by_type).to_csv(TABLES / "calibration_by_recording_type.csv",
                                 index=False)
    print("\n  Scores are on the model's own scale (RF: probability; MLP:")
    print("  probability, shown as a quantile, so compare within a row only).")
    print(f"  wrote {TABLES}/calibration_check.csv and "
          f"calibration_by_recording_type.csv")


if __name__ == "__main__":
    main()
