"""
Direction 1, final control (supervisor's email of 30 October): does the
effect come from weight averaging in general, or from averaging models that
are specialised to different vehicles?

Comparison, nothing else
------------------------
  vehicle averaging   (already in results/direction1/runs.csv, not retrained)
      avg_weights       one-shot weight average of three single-vehicle models
      fedavg            FedAvg, 10 rounds x 2 local epochs, one client per vehicle
  random-partition averaging   (new, this script)
      rand_avg_weights  the same one-shot weight average, but the three models
                        are trained on three RANDOM parts of the pooled source
                        windows instead of on the three vehicles
      rand_fedavg       the same FedAvg, with the three random parts as clients
  pooled                reference (from results/direction1/runs.csv)

The random parts are matched to the vehicle clients
---------------------------------------------------
For held-out vehicle H, the training windows of the three source vehicles
are put together and split at random into three parts. Part k has exactly
the same number of windows AND the same number of attack windows as source
vehicle k. So the averaging weights (window counts) and each client's class
weight (pos_weight, computed per client) are identical to the vehicle
version; the only difference is that every part is a random mix of all three
vehicles instead of one vehicle. Every source training window is used exactly
once. The split is drawn with its own fixed seed (7000 + training seed), so it
is the same every time the script runs. partitions.csv records each part's
size, attack count and vehicle mix.

Same starting weights, same everything else
-------------------------------------------
For a given seed, pooled, FedAvg, every vehicle member and every random
member start from the SAME initial weights: rfd.new_model(n_features, seed),
an MLP with two hidden layers (64, 32) and LayerNorm (no BatchNorm, so there
are no running statistics: every parameter in the state dict is a weight that
is averaged). The one-shot members are copies of that one start and are
trained with eav.train_members, the function used for the vehicle members
(20 epochs, Adam, same batch size, member c uses the training seed
seed * 1000 + c). Averaging is rfd.average_weights, weighted by the window
counts. The MD5 of the starting weights is written to runs.csv for every row.
Feature set, budget, scaling (fitted on the source training windows only),
threshold rules and evaluation seeds 0-9 are those of the thesis and of
direction1_averaging.py. Nothing is selected or tuned here: the control has
no settings of its own.

How it will be read (fixed before any run)
------------------------------------------
Per held-out vehicle (seed means) and over the four vehicles, for zero-shot
MCC, false-alarm rate at the zero-shot threshold, spread, headroom, AUROC,
AUPRC, calibrated MCC, source MCC, Brier and ECE:
  vehicle averaging minus random averaging, with sign consistency (x/4)
  "share" = (random - pooled) / (vehicle - pooled): how much of the change
            from pooled that vehicle averaging brings is also brought by
            random averaging (about 1 = averaging in general, about 0 =
            specific to averaging vehicle-specialised models). Shown only
            where vehicle and pooled differ by at least 0.01 (otherwise the
            ratio is noise), per vehicle and for the four-vehicle mean.
  member disagreement of the random members, next to that of the vehicle
            members (from direction1/runs.csv).
The new vehicle's data is used only to describe the outcome.

Check
-----
For seed 0 the vehicle avg_weights and fedavg are retrained here and must
equal results/direction1/runs.csv (so both versions come from the same code
and data). The partitions are checked to be exact (sizes, attack counts,
every window once).

Run from the project root (after direction1_averaging.py run):
    QUICK=1 python -u src/experiments/direction1_control.py run   (pipeline check)
    python -u src/experiments/direction1_control.py run           (about 20-40 min)
    python -u src/experiments/direction1_control.py report        (seconds)
"run" continues where it stopped if it is interrupted.

Outputs in results/direction1_control/:
    runs.csv          one row per setting, held-out vehicle and seed
    partitions.csv    size, attack count and vehicle mix of every random part
    check.csv         retrained vehicle averaging vs direction1/runs.csv
    per_vehicle.csv   seed means per setting and vehicle (with pooled and the
                      vehicle averaging from direction1)
    comparison.csv    per vehicle: vehicle minus random, share, sign consistency
"""

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUDGETS, QUICK, RESULTS, VEHICLES, load_all  # noqa: E402
import run_federated as rfd  # noqa: E402
import explain_arms as ea  # noqa: E402
import explain_avg as eav  # noqa: E402
import direction1_averaging as d1  # noqa: E402

OUT = RESULTS / ("direction1_control_quick" if QUICK else "direction1_control")
SEEDS = d1.SEEDS
CHECK_SEEDS = [0]
PARTITION_OFFSET = 7000
MIN_CHANGE = 0.01      # a share is shown only where |vehicle - pooled| >= this
NEW = ["rand_avg_weights", "rand_fedavg"]          # rand_fedavg is written last
COMPARE = [("avg_weights", "rand_avg_weights"), ("fedavg", "rand_fedavg")]
METRICS = ["mcc_zs", "fpr_zs", "spread", "headroom", "auroc", "auprc", "mcc_cal",
           "src_mcc_zs", "brier_src", "ece_src"]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def state_md5(model):
    h = hashlib.md5()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def random_partition(clients, seed):
    """Three random parts of the pooled windows, matched to the clients in
    size and in number of attack windows. Returns the parts and, per part,
    the source vehicle index of each of its windows."""
    X = torch.cat([c[0] for c in clients])
    y = torch.cat([c[1] for c in clients])
    owner = np.concatenate([np.full(len(c[0]), i) for i, c in enumerate(clients)])
    yy = y.numpy()
    rng = np.random.default_rng(PARTITION_OFFSET + seed)
    pos = rng.permutation(np.flatnonzero(yy == 1))
    neg = rng.permutation(np.flatnonzero(yy == 0))
    parts, owners, ip, ineg = [], [], 0, 0
    for Xc, yc in clients:
        n_pos = int(yc.sum().item())
        n_neg = len(yc) - n_pos
        idx = np.sort(np.concatenate([pos[ip:ip + n_pos], neg[ineg:ineg + n_neg]]))
        ip, ineg = ip + n_pos, ineg + n_neg
        t = torch.from_numpy(idx)
        parts.append((X[t], y[t]))
        owners.append(owner[idx])
    # exact: every window once, same sizes and attack counts as the clients
    assert ip == len(pos) and ineg == len(neg), "partition does not use every window"
    for (Xc, yc), (Xp, yp) in zip(clients, parts):
        assert len(Xp) == len(Xc) and float(yp.sum()) == float(yc.sum())
    return parts, owners


def done_keys():
    f = OUT / "runs.csv"
    if not f.exists():
        return set(), []
    rows = pd.read_csv(f).to_dict("records")
    keys = {(r["held_out"], int(r["seed"])) for r in rows if r["setting"] == NEW[-1]}
    return keys, [r for r in rows if (r["held_out"], int(r["seed"])) in keys]


def d1_runs():
    f = d1.OUT / "runs.csv"
    if not f.exists():
        raise SystemExit(f"{f} not found: run direction1_averaging.py first.")
    return pd.read_csv(f)


# ----------------------------------------------------------------------
# run
# ----------------------------------------------------------------------

def run():
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    prev = d1_runs()
    data, parts, clean, evalset = load_all()
    sel = ea.selections()
    keys, rows = done_keys()
    pf = OUT / "partitions.csv"
    prow = pd.read_csv(pf).to_dict("records") if pf.exists() and keys else []
    prow = [r for r in prow if (r["held_out"], int(r["seed"])) in keys]
    check_rows = []
    if keys:
        print(f"  continuing: {len(keys)} (vehicle, seed) pairs already done")
    for held_out in VEHICLES:
        s = sel[held_out]
        d = ea.prepare(held_out, data, parts, clean, evalset, s)
        nf = len(d["cols"])
        budget = BUDGETS[s["budget"]]
        sizes = [len(X) for X, _ in d["clients"]]
        print(f"\n--- hold out {held_out}: features {s['features']}, budget "
              f"{s['budget']}, client sizes {sizes} ---")
        for seed in SEEDS:
            if (held_out, seed) in keys:
                continue
            t0 = time.time()
            init = state_md5(rfd.new_model(nf, seed))
            assert init == state_md5(rfd.new_model(nf, seed)), "start not repeatable"
            rparts, owners = random_partition(d["clients"], seed)
            for k, (o, (Xp, yp)) in enumerate(zip(owners, rparts)):
                prow.append({"held_out": held_out, "seed": seed, "part": k,
                             "matched_to": d["sources"][k], "windows": len(Xp),
                             "attack_windows": int(yp.sum().item()),
                             "partition_seed": PARTITION_OFFSET + seed,
                             **{f"share_{v}": float(np.mean(o == i))
                                for i, v in enumerate(d["sources"])}})
            members = eav.train_members(rparts, nf, seed)
            models = {
                "rand_avg_weights": eav.from_state(nf, rfd.average_weights(
                    [m.state_dict() for m in members], sizes)),
                "rand_fedavg": rfd.train_arm("fedavg", rparts, nf, seed)[0],
            }
            dis = {"disagree_src_benign": d1.member_disagreement(members, d["bsrc"]),
                   "disagree_tclean": d1.member_disagreement(members, d["clean"])}
            new = []
            for name in NEW:
                r = d1.measure(models[name], d, budget)[0]
                if name == "rand_avg_weights":
                    for k, v in dis.items():
                        r[k] = v
                        r[f"{k}_G"] = v / r["class_gap_logit"]
                new.append({"setting": name, "held_out": held_out, "seed": seed,
                            "sources": "+".join(d["sources"]), "budget": s["budget"],
                            "init_md5": init,
                            "partition_seed": PARTITION_OFFSET + seed, **r})
            # check: the vehicle versions, retrained here, equal direction1
            if seed in CHECK_SEEDS:
                vm = eav.train_members(d["clients"], nf, seed)
                again = {"avg_weights": eav.from_state(nf, rfd.average_weights(
                             [m.state_dict() for m in vm], sizes)),
                         "fedavg": rfd.train_arm("fedavg", d["clients"], nf, seed)[0]}
                for name, model in again.items():
                    r = d1.measure(model, d, budget)[0]
                    old = prev[(prev["setting"] == name) & (prev["held_out"] == held_out)
                               & (prev["seed"] == seed)]
                    for k in ("auroc", "mcc_zs", "fpr_zs", "spread", "headroom"):
                        check_rows.append({
                            "compared": f"{name} retrained vs direction1/runs.csv",
                            "held_out": held_out, "seed": seed, "metric": k,
                            "found": len(old),
                            "abs_difference": (abs(r[k] - float(old[k].iloc[0]))
                                               if len(old) else np.nan)})
                pd.DataFrame(check_rows).to_csv(OUT / "check.csv", index=False)
            rows += new
            pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)
            pd.DataFrame(prow).to_csv(OUT / "partitions.csv", index=False)
            got = {x["setting"]: x for x in new}
            ref = prev[(prev["held_out"] == held_out) & (prev["seed"] == seed)]
            ref = {r["setting"]: r for r in ref.to_dict("records")}
            show = [("pooled", ref), ("avg_weights", ref), ("rand_avg_weights", got),
                    ("fedavg", ref), ("rand_fedavg", got)]
            print(f"  seed {seed} {time.time() - t0:>4.0f}s   zs MCC / FPR zs / spread: "
                  + "  ".join(f"{k} {src[k]['mcc_zs']:.2f}/{src[k]['fpr_zs']:.3f}/"
                              f"{src[k]['spread']:.2f}" for k, src in show if k in src))
    check()
    print(f"\n  total {(time.time() - t_start) / 60:.0f} min")
    report()


def check():
    f = OUT / "check.csv"
    if not f.exists():
        print("\nCHECK: no check rows yet (they are made with seed 0).")
        return False
    rep = pd.read_csv(f)
    summ = rep.groupby(["compared", "metric"]).agg(
        runs=("found", "sum"), max_abs_difference=("abs_difference", "max")).reset_index()
    print("\nCHECK: vehicle averaging retrained here vs results/direction1/runs.csv")
    print("  " + summ.to_string(index=False).replace("\n", "\n  "))
    ok = (len(summ) > 0 and (rep["found"] == 1).all()
          and summ["max_abs_difference"].max() <= 1e-6)
    print("  OK: same models as Direction 1." if ok else
          "  WARNING: not reproduced. Send this log before using any table.")
    p = OUT / "partitions.csv"
    if p.exists():
        pt = pd.read_csv(p)
        print(f"  partitions: {len(pt)} parts written; sizes and attack counts were "
              f"asserted equal to the vehicle clients while running.")
    return ok


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------

def report():
    new = pd.read_csv(OUT / "runs.csv")
    prev = d1_runs()
    pairs = set(zip(new["held_out"], new["seed"]))
    prev = prev[[(h, s) in pairs for h, s in zip(prev["held_out"], prev["seed"])]]
    prev = prev[prev["setting"].isin(["pooled", "avg_weights", "fedavg"])]
    allr = pd.concat([prev, new], ignore_index=True, sort=False)
    cols = [c for c in METRICS + ["disagree_src_benign_G", "disagree_tclean_G"]
            if c in allr.columns]
    per_v = allr.groupby(["setting", "held_out"])[cols].mean()
    per_v.reset_index().to_csv(OUT / "per_vehicle.csv", index=False)
    order = ["pooled", "avg_weights", "rand_avg_weights", "fedavg", "rand_fedavg"]
    summ = per_v.groupby(level=0).mean().reindex([o for o in order
                                                  if o in per_v.index.get_level_values(0)])
    summ.to_csv(OUT / "summary.csv")

    print("\n" + "=" * 100)
    print(f"DIRECTION 1 CONTROL   vehicle vs random-partition averaging, mean over "
          f"{per_v.index.get_level_values(1).nunique()} held-out vehicles "
          f"({new['seed'].nunique()} seeds averaged within each vehicle first)")
    print("=" * 100)
    heads = [("zs MCC", "mcc_zs"), ("FPR zs", "fpr_zs"), ("spread", "spread"),
             ("headroom", "headroom"), ("AUROC", "auroc"), ("AUPRC", "auprc"),
             ("cal MCC", "mcc_cal"), ("src MCC", "src_mcc_zs"),
             ("Brier src", "brier_src"), ("ECE src", "ece_src"),
             ("disagr", "disagree_src_benign_G")]
    print(f"  {'setting':<18}" + "".join(f"{h:>10}" for h, _ in heads))
    for name, r in summ.iterrows():
        print(f"  {name:<18}" + "".join(
            f"{r[c]:>10.3f}" if c in r and np.isfinite(r[c]) else f"{'-':>10}"
            for _, c in heads))

    rows = []
    for veh, rnd in COMPARE:
        for m in METRICS:
            diffs, shares = [], []
            for v in VEHICLES:
                try:
                    a = per_v.loc[(veh, v), m]
                    b = per_v.loc[(rnd, v), m]
                    p0 = per_v.loc[("pooled", v), m]
                except KeyError:
                    diffs.append(np.nan)
                    shares.append(np.nan)
                    continue
                diffs.append(float(a - b))
                shares.append(float((b - p0) / (a - p0))
                              if abs(a - p0) >= MIN_CHANGE else np.nan)
            dd = np.array(diffs)
            try:
                ma, mb, mp = (summ.loc[x, m] for x in (veh, rnd, "pooled"))
                mean_change = float(ma - mp)
                mean_share = (float((mb - mp) / (ma - mp))
                              if abs(ma - mp) >= MIN_CHANGE else np.nan)
            except KeyError:
                mean_change, mean_share = np.nan, np.nan
            pos, neg = int(np.sum(dd > 0)), int(np.sum(dd < 0))
            rows.append({"vehicle_setting": veh, "random_setting": rnd, "metric": m,
                         **{f"diff_{v}": x for v, x in zip(VEHICLES, diffs)},
                         "diff_mean": float(np.nanmean(dd)) if np.isfinite(dd).any() else np.nan,
                         "same_sign": f"{max(pos, neg)}/{int(np.isfinite(dd).sum())}",
                         **{f"share_{v}": x for v, x in zip(VEHICLES, shares)},
                         "vehicle_minus_pooled_mean": mean_change,
                         "share_of_mean": mean_share})
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "comparison.csv", index=False)
    for veh, rnd in COMPARE:
        print(f"\n  {veh} minus {rnd}, per vehicle (seed means), and share of the "
              f"change from pooled that {rnd} also brings")
        print(f"    {'metric':<11}" + "".join(f"{v[:9]:>10}" for v in VEHICLES)
              + f"{'mean':>9}{'sign':>7}   share per vehicle   "
              + "veh-pooled  share of mean")
        for r in comp[(comp["vehicle_setting"] == veh)].itertuples():
            sh = " ".join(f"{x:+.2f}" if np.isfinite(x) else "  -  "
                          for x in (getattr(r, f"share_{v}") for v in VEHICLES))
            print(f"    {r.metric:<11}" + "".join(
                f"{getattr(r, f'diff_{v}'):>+10.3f}" for v in VEHICLES)
                + f"{r.diff_mean:>+9.3f}{r.same_sign:>7}   {sh}  "
                + f"{r.vehicle_minus_pooled_mean:>+10.3f}  "
                + (f"{r.share_of_mean:>+6.2f}" if np.isfinite(r.share_of_mean) else "     -"))
    p = OUT / "partitions.csv"
    if p.exists():
        pt = pd.read_csv(p)
        sh = [c for c in pt.columns if c.startswith("share_")]
        mx = pt[sh].max(axis=1)
        print(f"\n  random parts: largest share of any one vehicle in a part: "
              f"median {mx.median():.2f}, max {mx.max():.2f} "
              f"(a vehicle client would be 1.00)")
    print(f"\n  Tables in {OUT}/. No conclusion is drawn here; read per vehicle and "
          f"by sign consistency, as in the thesis.")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    {"run": run, "report": report, "check": check}.get(what, lambda: print(__doc__))()
