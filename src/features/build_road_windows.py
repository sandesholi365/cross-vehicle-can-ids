"""
Windows and features for ROAD, built exactly like CAN-Train-and-Test.

Uses the same window_features() function as build_windows.py, so the
window size, the features and their definitions are identical.

Input: the converted ROAD files from convert.py
  data/processed/road/ambient/*.csv   attack-free driving
  data/processed/road/attacks/*.csv   attack captures

Rules, fixed before ROAD is used:
  - Ambient recordings get attack_type "attack-free". They are used only
    to set the benign-calibrated threshold, never scored (the same rule as
    for CAN-Train-and-Test).
  - An attack capture with no labelled attack message is left out: its
    injected traffic cannot be labelled, so keeping it would put attack
    traffic in the data labelled as normal.
  - The attack type is the file name without its capture number, e.g.
    max_speedometer_attack_2_masquerade -> max_speedometer_masquerade.
  - ROAD provides masquerade versions of most captures, made from the same
    recordings as the fabrication versions. They are kept as their own
    attack types and are evaluated as a separate set, so no capture is
    counted twice in one result.

Output: data/processed/features/road_w100.csv

Run from the project root:
    python src/features/build_road_windows.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_windows import CHUNK_WINDOWS, OUT, WINDOW_SIZE, window_features  # noqa: E402

ROAD = Path("data/processed/road")


def attack_type_of(stem):
    """'max_speedometer_attack_2_masquerade' -> 'max_speedometer_masquerade'."""
    parts = [p for p in stem.split("_") if not re.fullmatch(r"\d+", p)]
    parts = [p for p in parts if p != "attack"]
    return "_".join(parts)


def window_file(path, recording, attack_type):
    frames, offset = [], 0
    for chunk in pd.read_csv(
        path,
        usecols=["timestamp", "arbitration_id", "dlc", "data_field", "label"],
        dtype={"arbitration_id": str, "data_field": str,
               "dlc": "int16", "label": "int8"},
        chunksize=CHUNK_WINDOWS * WINDOW_SIZE,
    ):
        part = window_features(chunk, "road", recording, attack_type, offset)
        if part is not None:
            frames.append(part)
            offset += len(part)
    return pd.concat(frames, ignore_index=True) if frames else None


def main():
    if not ROAD.exists():
        raise SystemExit(f"{ROAD} not found. Convert ROAD first:\n"
                         "  python -c \"import sys; sys.path.insert(0, 'src/data'); "
                         "import convert; convert.convert_road()\"")
    OUT.mkdir(parents=True, exist_ok=True)
    tables, skipped = [], []

    print("ambient (attack-free) ...")
    for path in sorted((ROAD / "ambient").glob("*.csv")):
        t = window_file(path, f"{path.stem}__ambient", "attack-free")
        if t is not None:
            tables.append(t)
            print(f"    {path.stem:<48}{len(t):>8,} windows")

    print("\nattack captures ...")
    for path in sorted((ROAD / "attacks").glob("*.csv")):
        n_attack = int(pd.read_csv(path, usecols=["label"])["label"].sum())
        if n_attack == 0:
            skipped.append(path.stem)
            continue
        t = window_file(path, f"{path.stem}__attack", attack_type_of(path.stem))
        if t is not None:
            tables.append(t)
            print(f"    {path.stem:<48}{len(t):>8,} windows, "
                  f"{int(t['label'].sum()):>6,} attack")

    table = pd.concat(tables, ignore_index=True)
    dest = OUT / f"road_w{WINDOW_SIZE}.csv"
    table.to_csv(dest, index=False)

    free = table["attack_type"] == "attack-free"
    print(f"\n  -> {dest.name}: {len(table):,} windows "
          f"({int(free.sum()):,} attack-free, {int((~free).sum()):,} in attack "
          f"captures, {int(table['label'].sum()):,} attack)")
    if skipped:
        print(f"  left out (no labelled attack messages): {', '.join(skipped)}")

    # same time unit as CAN-Train-and-Test? stat_* features depend on it
    ref = OUT / f"impala_w{WINDOW_SIZE}.csv"
    if ref.exists():
        a = pd.read_csv(ref, usecols=["stat_iat_mean"])["stat_iat_mean"].median()
        b = table["stat_iat_mean"].median()
        ratio = b / a if a else np.nan
        print(f"\n  time-unit check: median gap between messages "
              f"CAN-Train-and-Test (Impala) {a:.6f}, ROAD {b:.6f} "
              f"(ratio {ratio:.2f})")
        if not (0.05 < ratio < 20):
            print("  WARNING: the two datasets may use different time units. "
                  "Stop and check before running any experiment.")
        else:
            print("  OK: same order of magnitude.")


if __name__ == "__main__":
    main()
