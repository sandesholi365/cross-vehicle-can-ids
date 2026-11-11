"""
Windows and features for CrySyS, built exactly like CAN-Train-and-Test and
ROAD (same window_features() function, so window size, features and their
definitions are identical).

No model is involved. The labels come from crysys_labels.py, which must have
been run first (data/processed/crysys_labels/). Rules, fixed before CrySyS
is used for anything (frozen with the method, supervisor's email of
30 October):

  - Only traces that passed every label check are used (label_report.csv,
    column "passed"): 624 masquerade and 592 fabrication traces in the full
    run. Failed traces are left out with their reason (already listed in
    label_report.csv); nothing is rescued.
  - The 26 benign traces get attack_type "attack-free". As for CTT and ROAD
    they are used only to set the benign-calibrated threshold, never scored.
  - Frame labels are the ones derived by crysys_labels.py (labels/<trace>.npy,
    one 0/1 per frame in file order). The JSON markers are NOT used as labels.
  - A window is attack (1) if it holds at least one attack frame (as for CTT).
  - Windows listed in labels/<trace>_uncertain_windows.npy get uncertain = 1.
    They are left out of every window-level result. For alarm episodes they
    keep their place in time and are treated as attack windows, so they can
    never be counted as a false alarm.
  - attack_type = <kind>_<strategy>, e.g. fabrication_ADD-INCR or
    masquerade_DOUBLE-REPLAY; crysys_set = benign / fabrication / masquerade.
    Masquerade and fabrication are evaluated as separate sets (both are made
    from the same benign traces, so no trace is counted twice in one result).

Output: data/processed/features/crysys_w100.csv  (plus crysys_windows_report.csv)

Run from the project root (after crysys_labels.py):
    python src/features/build_crysys_windows.py --root data/raw/crysys
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data"))
from build_windows import OUT, WINDOW_SIZE, window_features  # noqa: E402
import crysys_labels as cl  # noqa: E402

LABELS = Path("data/processed/crysys_labels")


def read_frames(path):
    """The frames of a candump log, read with crysys_labels.read_log, the same
    reader that produced the labels, so frames and labels line up one to one."""
    log = cl.read_log(path)
    if log["bad"]:
        raise SystemExit(f"{path.name}: {log['bad']} lines could not be read. "
                         "Send this message.")
    ids = np.array([format(int(x), "03X") for x in log["id"]], dtype=object)
    data = np.array([str(x).upper() for x in log["data"]], dtype=object)
    return log["t"], ids, data


def attack_type_of(stem):
    mech, targets = cl.parse_name(stem)
    kind = {"inj": "fabrication", "mod": "masquerade"}[mech]
    types = sorted({x["type"] for x in targets})
    return kind, f"{kind}_{'DOUBLE-' if len(targets) == 2 else ''}{'+'.join(types)}"


def windows_of(path, label, recording, attack_type):
    ts, ids, data = read_frames(path)
    if len(ts) != len(label):
        raise SystemExit(f"{path.name}: {len(ts)} frames but {len(label)} labels. "
                         "Send this message.")
    df = pd.DataFrame({"timestamp": ts, "arbitration_id": ids,
                       "dlc": np.array([len(d) // 2 for d in data], dtype=np.int16),
                       "data_field": data, "label": label.astype(np.int8)})
    return window_features(df, "crysys", recording, attack_type, 0)


def main():
    ap = argparse.ArgumentParser(description="CrySyS windows")
    ap.add_argument("--root", required=True, help="extracted CrySyS dataset folder")
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    root, labels, out = Path(args.root), Path(args.labels), Path(args.out)
    rep = pd.read_csv(labels / "label_report.csv")
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"crysys_w{WINDOW_SIZE}.csv"
    tmp = dest.with_suffix(".partial.csv")
    if tmp.exists():
        tmp.unlink()

    logs = {p.stem: p for p in root.rglob("*.log") if not p.name.startswith("._")}
    report, first = [], True

    def write(t):
        nonlocal first
        t.to_csv(tmp, mode="w" if first else "a", header=first, index=False)
        first = False

    ben = rep[rep["kind"] == "benign"]
    print(f"benign traces: {len(ben)}")
    for r in ben.itertuples():
        p = logs.get(r.trace)
        if p is None:
            raise SystemExit(f"{r.trace}.log not found under {root}")
        ts, _, _ = read_frames(p)
        t = windows_of(p, np.zeros(len(ts), np.int8), r.trace, "attack-free")
        t["scenario"], t["crysys_set"], t["uncertain"] = r.scenario, "benign", 0
        write(t)
        report.append({"trace": r.trace, "set": "benign", "used": True, "windows": len(t),
                       "attack_windows": 0, "uncertain_windows": 0, "reason": ""})

    att = rep[rep["kind"].isin(["fabrication", "masquerade"])]
    n_used = 0
    for i, r in enumerate(att.itertuples(), 1):
        passed = str(r.passed).lower() == "true"
        if not passed:
            report.append({"trace": r.trace, "set": r.kind, "used": False, "windows": 0,
                           "attack_windows": 0, "uncertain_windows": 0,
                           "reason": "failed the label check (see label_report.csv)"})
            continue
        p = logs.get(r.trace)
        lab_f = labels / "labels" / f"{r.trace}.npy"
        if p is None or not lab_f.exists():
            raise SystemExit(f"{r.trace}: log or label file missing. Send this message.")
        kind, atype = attack_type_of(r.trace)
        t = windows_of(p, np.load(lab_f), r.trace, atype)
        unc_f = labels / "labels" / f"{r.trace}_uncertain_windows.npy"
        # only fabrication traces can have uncertain windows (crysys_labels.py)
        unc = (np.load(unc_f) if kind == "fabrication" and unc_f.exists()
               else np.array([], dtype=np.int64))
        t["scenario"], t["crysys_set"] = r.scenario, kind
        t["uncertain"] = np.isin(t["window_index"].to_numpy(), unc).astype(np.int8)
        write(t)
        n_used += 1
        report.append({"trace": r.trace, "set": kind, "used": True, "windows": len(t),
                       "attack_windows": int(t["label"].sum()),
                       "uncertain_windows": int(t["uncertain"].sum()), "reason": ""})
        if i % 50 == 0 or i == len(att):
            print(f"  {i}/{len(att)} attacked traces ({n_used} used)", flush=True)

    tmp.replace(dest)
    rp = pd.DataFrame(report)
    rp.to_csv(out / "crysys_windows_report.csv", index=False)
    print(f"\n  -> {dest}")
    for s, g in rp[rp["used"]].groupby("set"):
        print(f"  {s:<12} {len(g):>4} traces  {int(g['windows'].sum()):>9,} windows  "
              f"{int(g['attack_windows'].sum()):>8,} attack  "
              f"{int(g['uncertain_windows'].sum()):>3} uncertain (left out)")
    print(f"  left out: {int((~rp['used']).sum())} traces that failed the label check")

    # same time unit as CAN-Train-and-Test? stat_* features depend on it
    ref = out / f"impala_w{WINDOW_SIZE}.csv"
    if ref.exists():
        a = pd.read_csv(ref, usecols=["stat_iat_mean"])["stat_iat_mean"].median()
        b = pd.read_csv(dest, usecols=["stat_iat_mean"])["stat_iat_mean"].median()
        ratio = b / a if a else np.nan
        print(f"\n  time-unit check: median gap between messages CAN-Train-and-Test "
              f"(Impala) {a:.6f}, CrySyS {b:.6f} (ratio {ratio:.2f})")
        print("  OK: same order of magnitude." if 0.05 < ratio < 20 else
              "  WARNING: the datasets may use different time units. Stop and send this.")


if __name__ == "__main__":
    main()
