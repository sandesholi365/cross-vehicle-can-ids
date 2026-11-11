"""
Check and document that the pooled, FedAvg and FedProx MLPs are compared
fairly (supervisor's email of 30 September): same source data, same model
and features, same preprocessing, comparable training settings.

Nothing is trained or changed. The script
  1. reads the saved results (results/mlp_*.csv and results/road/mlp_*.csv)
     and checks that, within each held-out vehicle (and for ROAD), every arm
     used the same feature set, budget and seeds;
  2. rebuilds the training inputs exactly as run_federated.py / run_road.py
     do and records, per arm: training windows per vehicle, attack share,
     class weight, scaling, network size, initial weights and the amount of
     training (gradient steps, windows seen);
  3. lists the differences that exist by design, taken from the code.

Differences are documented, not fixed: the methodology is frozen.

Outputs, in results/audit/:
  arm_parity.csv          one row per check and configuration
  arm_parity_report.txt   the same, readable

Run from the project root:
    python src/experiments/check_arm_parity.py
"""

import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import QUICK, RESULTS, VEHICLES, feature_columns, load_all  # noqa: E402
import run_federated as rfd  # noqa: E402

AUDIT = RESULTS / "audit"
ARMS = ["pooled", "fedavg", "fedprox"]
LINES = []


def say(text=""):
    print(text)
    LINES.append(text)


def flag(df, col):
    df[col] = df[col].astype(str).str.lower() == "true"
    return df


def digest(arr):
    """Short fingerprint of an array, to show two things are identical."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    return hashlib.md5(a.tobytes()).hexdigest()[:10]


def selections():
    """Selected feature set, budget and mu per configuration."""
    out = {}
    for path, key in [(RESULTS / "mlp_selected.csv", None),
                      (RESULTS / "road" / "mlp_selected.csv", "road")]:
        if not path.exists():
            continue
        s = pd.read_csv(path)
        for held, grp in s.groupby("held_out"):
            d = dict(zip(grp["parameter"], grp["chosen"]))
            d["criterion_feature_set"] = grp.loc[grp["parameter"] == "feature_set",
                                                 "criterion"].iloc[0]
            d["criterion_budget"] = grp.loc[grp["parameter"] == "budget",
                                            "criterion"].iloc[0]
            out[key or held] = d
    return out


# ----------------------------------------------------------------------
# 1. from the saved results
# ----------------------------------------------------------------------

def check_results(rows):
    say("=" * 90)
    say("1. SAVED RESULTS: same feature set, budget and seeds for every arm")
    say("=" * 90)
    sources = [(RESULTS / "mlp_results.csv", "held_out", "cross"),
               (RESULTS / "road" / "mlp_results.csv", None, "external")]
    for path, held_col, setting in sources:
        if not path.exists():
            say(f"  (missing {path})")
            continue
        r = flag(pd.read_csv(path), "selected")
        r = r[r["selected"] & r["arm"].isin(ARMS) & (r["setting"] == setting)]
        configs = r.groupby(held_col) if held_col else [("road", r)]
        for cfg, grp in configs:
            per_arm = {}
            for arm in ARMS:
                a = grp[grp["arm"] == arm]
                per_arm[arm] = {
                    "feature_set": sorted(a["feature_set"].astype(str).unique()),
                    "budget": sorted(a["budget"].astype(str).unique()),
                    "seeds": sorted(int(x) for x in a["seed"].unique()),
                    "mu": sorted(float(x) for x in a["mu"].dropna().unique()),
                }
            for item in ("feature_set", "budget", "seeds"):
                vals = [per_arm[a][item] for a in ARMS]
                same = all(v == vals[0] for v in vals)
                rows.append({"config": cfg, "check": f"results: {item}",
                             **{a: str(per_arm[a][item]) for a in ARMS},
                             "verdict": "same" if same else "DIFFERS"})
                say(f"  {cfg:<10}{item:<12}"
                    + "  ".join(f"{a} {per_arm[a][item] if item != 'seeds' else len(per_arm[a][item])}"
                                for a in ARMS)
                    + f"   -> {'same' if same else 'DIFFERS'}")
            say(f"  {cfg:<10}{'fedprox mu':<12}{per_arm['fedprox']['mu']} (selected)")


# ----------------------------------------------------------------------
# 2. rebuilt training inputs
# ----------------------------------------------------------------------

def steps(n):
    return math.ceil(n / rfd.BATCH)


def check_inputs(rows, data, parts, sel):
    say("")
    say("=" * 90)
    say("2. REBUILT TRAINING INPUTS (as in run_federated.py and run_road.py)")
    say("=" * 90)
    everything = rfd.all_columns(data)
    configs = [(v, [s for s in VEHICLES if s != v]) for v in VEHICLES]
    configs.append(("road", list(VEHICLES)))
    for cfg, sources in configs:
        if cfg not in sel:
            say(f"  (no selection found for {cfg})")
            continue
        group = str(sel[cfg]["feature_set"])
        cols = feature_columns(data[VEHICLES[0]], group)
        fit = {v: parts[v][0] for v in sources}
        sc = rfd.Scaler(pd.concat([fit[v] for v in sources]), everything)
        clients = [(sc.X(fit[v], cols), rfd.labels(fit[v])) for v in sources]
        n_i = [len(x) for x, _ in clients]
        pooled_X = torch.cat([c[0] for c in clients])
        pooled_y = torch.cat([c[1] for c in clients])

        say(f"\n  held out: {cfg}   sources: {', '.join(sources)}   features: "
            f"{group} ({len(cols)})")
        # same data
        for v, (x, y) in zip(sources, clients):
            say(f"    client {v:<10}{len(x):>9,} training windows, attack share "
                f"{float(y.mean()):.2%}, class weight {float(rfd.pos_weight_for(y)):.2f}")
        say(f"    pooled            {len(pooled_X):>9,} training windows "
            f"(= sum of the clients: {len(pooled_X) == sum(n_i)}), attack share "
            f"{float(pooled_y.mean()):.2%}, class weight "
            f"{float(rfd.pos_weight_for(pooled_y)):.2f}")
        same_rows = digest(pooled_X.numpy()) == digest(
            np.concatenate([c[0].numpy() for c in clients]))
        rows.append({"config": cfg, "check": "training windows",
                     "pooled": f"{len(pooled_X)} (all sources together)",
                     "fedavg": " + ".join(str(n) for n in n_i),
                     "fedprox": " + ".join(str(n) for n in n_i),
                     "verdict": "same" if same_rows else "DIFFERS"})
        weights = [float(rfd.pos_weight_for(y)) for _, y in clients]
        wp = float(rfd.pos_weight_for(pooled_y))
        rows.append({"config": cfg, "check": "class weight (attack)",
                     "pooled": f"{wp:.2f}",
                     "fedavg": ", ".join(f"{w:.2f}" for w in weights),
                     "fedprox": ", ".join(f"{w:.2f}" for w in weights),
                     "verdict": "same" if all(abs(w - wp) < 1e-9 for w in weights)
                     else "differs (computed per client)"})
        # same features and preprocessing
        rows.append({"config": cfg, "check": "features",
                     "pooled": f"{group} ({len(cols)})", "fedavg": f"{group} ({len(cols)})",
                     "fedprox": f"{group} ({len(cols)})", "verdict": "same"})
        scale_fp = digest(np.r_[[sc.centre[c] for c in cols], [sc.scale[c] for c in cols]])
        rows.append({"config": cfg, "check": "scaling (median/IQR + arcsinh)",
                     "pooled": f"fitted on all source training windows [{scale_fp}]",
                     "fedavg": f"same statistics [{scale_fp}]",
                     "fedprox": f"same statistics [{scale_fp}]",
                     "verdict": "same (shared statistics: an assumption of the "
                                "federated setting)"})
        say(f"    scaling statistics fingerprint {scale_fp}, used by every arm")
        # same network and starting weights
        seed = rfd.SEEDS[0]
        m_pooled = rfd.new_model(len(cols), seed)
        m_fed = rfd.new_model(len(cols), seed)
        n_params = sum(p.numel() for p in m_pooled.parameters())
        same_init = all(torch.equal(a, b) for a, b in
                        zip(m_pooled.state_dict().values(), m_fed.state_dict().values()))
        arch = f"{len(cols)}-{rfd.HIDDEN[0]}-{rfd.HIDDEN[1]}-1, LayerNorm, {n_params} weights"
        rows.append({"config": cfg, "check": "network", "pooled": arch,
                     "fedavg": arch, "fedprox": arch, "verdict": "same"})
        rows.append({"config": cfg, "check": f"initial weights (seed {seed})",
                     "pooled": "new_model(seed)", "fedavg": "new_model(seed)",
                     "fedprox": "new_model(seed)",
                     "verdict": "same" if same_init else "DIFFERS"})
        say(f"    network {arch}; identical initial weights for the same seed: "
            f"{same_init}")
        # amount of training
        n = sum(n_i)
        pooled_steps = rfd.CENTRAL_EPOCHS * steps(n)
        fed_steps = rfd.ROUNDS * rfd.LOCAL_EPOCHS * sum(steps(k) for k in n_i)
        seen_pooled = rfd.CENTRAL_EPOCHS * n
        seen_fed = rfd.ROUNDS * rfd.LOCAL_EPOCHS * n
        rows.append({"config": cfg, "check": "windows seen in training",
                     "pooled": f"{seen_pooled:,} ({rfd.CENTRAL_EPOCHS} epochs)",
                     "fedavg": f"{seen_fed:,} ({rfd.ROUNDS} rounds x "
                               f"{rfd.LOCAL_EPOCHS} local epochs)",
                     "fedprox": f"{seen_fed:,} (same)",
                     "verdict": "same" if seen_pooled == seen_fed else "DIFFERS"})
        rows.append({"config": cfg, "check": "gradient steps (all clients)",
                     "pooled": f"{pooled_steps:,}", "fedavg": f"{fed_steps:,}",
                     "fedprox": f"{fed_steps:,}",
                     "verdict": "comparable (differs only by rounding of the last "
                                "batch per client)"
                     if abs(pooled_steps - fed_steps) <= rfd.ROUNDS * rfd.LOCAL_EPOCHS
                     * len(n_i) else "DIFFERS"})
        say(f"    windows seen: pooled {seen_pooled:,}, federated {seen_fed:,}; "
            f"gradient steps: pooled {pooled_steps:,}, federated {fed_steps:,} "
            f"(summed over clients)")
        weights_fedavg = [k / n for k in n_i]
        say("    FedAvg averaging weights (share of training windows): "
            + ", ".join(f"{v} {w:.1%}" for v, w in zip(sources, weights_fedavg))
            + "; pooled training weights each vehicle by the same shares")
        rows.append({"config": cfg, "check": "weight of each vehicle",
                     "pooled": "by number of windows (implicit)",
                     "fedavg": "by number of windows (averaging weights "
                               + ", ".join(f"{w:.1%}" for w in weights_fedavg) + ")",
                     "fedprox": "same as FedAvg", "verdict": "same"})
        rows.append({"config": cfg, "check": "selection of feature set and budget",
                     "pooled": str(sel[cfg]["criterion_feature_set"]),
                     "fedavg": "uses the pooled MLP's choice",
                     "fedprox": "uses the pooled MLP's choice; mu chosen with FedProx",
                     "verdict": "by design: chosen with the pooled model"})


# ----------------------------------------------------------------------
# 3. differences by design, from the code
# ----------------------------------------------------------------------

DESIGN = [
    ("optimiser", "Adam, lr {lr}, weight decay {wd}, gradient clip {clip}; one "
     "optimiser for all {ep} epochs",
     "same settings, but a new Adam optimiser in every round and client (its "
     "memory restarts each round)", "differs by design"),
    ("batches", "batches of {b} windows mixed from all source vehicles",
     "batches of {b} windows from one vehicle (the client)", "differs by design"),
    ("loss", "binary cross-entropy with class weight",
     "same; FedProx adds (mu/2)*||w - w_global||^2", "same (FedProx: by design)"),
    ("zero-shot threshold", "set on the source calibration windows of all sources",
     "same windows (clients share model scores, not data)", "same"),
    ("evaluation seeds", "{seeds}", "{seeds}", "same"),
]


def check_design(rows):
    say("")
    say("=" * 90)
    say("3. DIFFERENCES BY DESIGN (from the code of run_federated.py)")
    say("=" * 90)
    fmt = {"lr": rfd.LR, "wd": rfd.WEIGHT_DECAY, "clip": rfd.GRAD_CLIP,
           "ep": rfd.CENTRAL_EPOCHS, "b": rfd.BATCH, "seeds": rfd.SEEDS}
    for check, pooled, fed, verdict in DESIGN:
        p, f = pooled.format(**fmt), fed.format(**fmt)
        rows.append({"config": "all", "check": check, "pooled": p, "fedavg": f,
                     "fedprox": f, "verdict": verdict})
        say(f"  {check}")
        say(f"    pooled:    {p}")
        say(f"    federated: {f}")
        say(f"    -> {verdict}")


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    rows = []
    if QUICK:
        say("QUICK mode: every 25th window only. For checking the script, not "
            "for the thesis.")
    check_results(rows)
    sel = selections()
    data, parts, _, _ = load_all()
    check_inputs(rows, data, parts, sel)
    check_design(rows)
    table = pd.DataFrame(rows, columns=["config", "check", "pooled", "fedavg",
                                        "fedprox", "verdict"])
    table.to_csv(AUDIT / "arm_parity.csv", index=False)
    bad = table[table["verdict"].str.startswith("DIFFERS")]
    say("")
    say("=" * 90)
    say(f"SUMMARY: {len(table)} checks; {len(bad)} unexpected differences"
        + ("" if bad.empty else ":"))
    for _, r in bad.iterrows():
        say(f"  {r['config']}: {r['check']}")
    say("Differences marked 'by design' are documented, not fixed: the method "
        "is frozen.")
    (AUDIT / "arm_parity_report.txt").write_text("\n".join(LINES) + "\n")
    print(f"\n  wrote {AUDIT}/arm_parity.csv and arm_parity_report.txt")


if __name__ == "__main__":
    main()
