"""
Why does FedAvg keep a usable zero-shot threshold on a new vehicle when the
pooled MLP does not? (supervisor's email of 5 October, point 6)

CAN-Train-and-Test only. ROAD is not touched. Nothing here changes the method
of the thesis: the settings chosen by the inner loop are READ from
results/mlp_selected.csv, and the main results are not re-selected.

What the script does
--------------------
For every held-out vehicle and seed it trains the MLP in several ways, with
the same network, data, scaling, class weight and threshold rules as
run_federated.py:

  main arms (must reproduce the saved results exactly)
    pooled                   all source windows mixed, 20 epochs
    fedavg                   10 rounds x 2 local epochs, one client per vehicle
    fedprox                  as fedavg, with the selected mu

  in-between variants (each changes ONE thing, to find the cause)
    pooled_vehicle_batches   pooled, but every mini-batch comes from one vehicle
                             (FedAvg's clients only ever see one vehicle)
    pooled_adam_reset        pooled, but Adam is restarted every 2 epochs
                             (FedAvg restarts Adam at every round)
    pooled_short             pooled, but only 7 epochs (about the number of
                             sequential steps on FedAvg's averaged path:
                             10 rounds x 2 epochs x about one third of the data)
    fedavg_1x20              FedAvg with 20 rounds x 1 local epoch
                             (averages twice as often, closer to pooled)
    fedavg_20x1              FedAvg with 1 round x 20 local epochs: three
                             single-vehicle models trained from the same start
                             and averaged once

For each trained model it measures, all on the logit scale (the score before
the sigmoid):

  outcome   AUROC, MCC at zs / zsb / cal, and the share of the new vehicle's
            attack-free windows that the zero-shot threshold flags
  where the scores sit, in units of the model's own class gap on the sources
            (G = median attack logit - median benign logit, source test):
              headroom  how far the zero-shot threshold is above typical
                        source benign windows
              shift     how far the new vehicle's benign windows move up
              margin    how far the new vehicle's attacks are above the
                        threshold
  mechanism
              weight norm, and the input-gradient norm (how strongly the
              logit reacts to a change in the input)
              exposure: input-gradient weight on features that differ
              between the SOURCE vehicles (no target data needed)
              predicted shift: first-order estimate of the benign shift from
              the gradients and the feature shift of the new vehicle
              per-source-vehicle AUROC

The new vehicle's attack-free traffic is used here only to DESCRIBE what
happens, never to choose anything.

Run from the project root, in this order:
    python -u src/experiments/explain_arms.py run      (about 30-40 min)
    python -u src/experiments/explain_arms.py report   (seconds; tables)

Outputs in results/explain_arms/:
    runs.csv        one row per variant, held-out vehicle and seed
    check.csv       main arms against the saved results
    summary.csv     means per variant (seeds, then vehicles)
    per_vehicle.csv seed means per vehicle
    false_alarms_per_vehicle.csv  false alarms, recall, spread (report, part 6)
"""

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BUDGETS, QUICK, RESULTS, VEHICLES, best_threshold,  # noqa: E402
                    budget_threshold, evaluate, feature_columns, heldout_benign,
                    load_all, scored, threshold_rules)
import run_federated as rfd  # noqa: E402

OUT = RESULTS / ("explain_arms_quick" if QUICK else "explain_arms")
SEEDS = [0] if QUICK else [0, 1, 2, 3, 4]
GRAD_SAMPLE = 20_000
POOLED_SHORT_EPOCHS = 7

VARIANTS = {
    "pooled":                 {"kind": "pooled"},
    "fedavg":                 {"kind": "fed", "rounds": 10, "local": 2, "mu": 0.0},
    "fedprox":                {"kind": "fed", "rounds": 10, "local": 2, "mu": "selected"},
    "pooled_vehicle_batches": {"kind": "pooled", "vehicle_batches": True},
    "pooled_adam_reset":      {"kind": "pooled", "reset_every": 2},
    "pooled_short":           {"kind": "pooled", "epochs": POOLED_SHORT_EPOCHS},
    "fedavg_1x20":            {"kind": "fed", "rounds": 20, "local": 1, "mu": 0.0},
    "fedavg_20x1":            {"kind": "fed", "rounds": 1, "local": 20, "mu": 0.0},
}
MAIN = ["pooled", "fedavg", "fedprox"]


# ----------------------------------------------------------------------
# training: the main arms call run_federated directly; the variants are
# copies of the same loops with one change each
# ----------------------------------------------------------------------

def train_pooled_variant(clients, n_features, seed, epochs=rfd.CENTRAL_EPOCHS,
                         vehicle_batches=False, reset_every=None):
    if not vehicle_batches and reset_every is None:
        X = torch.cat([c[0] for c in clients])
        y = torch.cat([c[1] for c in clients])
        return rfd.train_local(rfd.new_model(n_features, seed), X, y, epochs,
                               seed=seed)
    model = rfd.new_model(n_features, seed)
    X = torch.cat([c[0] for c in clients])
    y = torch.cat([c[1] for c in clients])
    owner = torch.cat([torch.full((len(c[0]),), i) for i, c in enumerate(clients)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=rfd.pos_weight_for(y))
    gen = torch.Generator().manual_seed(seed)
    opt = None
    model.train()
    for ep in range(epochs):
        if opt is None or (reset_every and ep % reset_every == 0):
            opt = torch.optim.Adam(model.parameters(), lr=rfd.LR,
                                   weight_decay=rfd.WEIGHT_DECAY)
        if vehicle_batches:
            batches = []
            for i in range(len(clients)):
                idx = torch.nonzero(owner == i).squeeze(1)
                idx = idx[torch.randperm(len(idx), generator=gen)]
                batches += [idx[s:s + rfd.BATCH] for s in range(0, len(idx), rfd.BATCH)]
            order = torch.randperm(len(batches), generator=gen).tolist()
            batches = [batches[k] for k in order]
        else:
            perm = torch.randperm(len(X), generator=gen)
            batches = [perm[s:s + rfd.BATCH] for s in range(0, len(X), rfd.BATCH)]
        for idx in batches:
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), rfd.GRAD_CLIP)
            opt.step()
    return model


def train_fed_variant(clients, n_features, seed, rounds, local, mu):
    """run_federated.train_federated with the rounds and local epochs as inputs."""
    global_model = rfd.new_model(n_features, seed)
    sizes = [len(X) for X, _ in clients]
    for r in range(rounds):
        global_params = [p.detach().clone() for p in global_model.parameters()]
        states = []
        for c, (X, y) in enumerate(clients):
            local_model = copy.deepcopy(global_model)
            rfd.train_local(local_model, X, y, local,
                            global_params=global_params if mu > 0 else None, mu=mu,
                            seed=seed * 1000 + r * 10 + c)
            states.append(local_model.state_dict())
        global_model.load_state_dict(rfd.average_weights(states, sizes))
    return global_model


def train_variant(name, clients, n_features, seed, mu_selected):
    v = VARIANTS[name]
    if name == "pooled":
        return rfd.train_arm("pooled", clients, n_features, seed)[0]
    if name == "fedavg":
        return rfd.train_arm("fedavg", clients, n_features, seed)[0]
    if name == "fedprox":
        return rfd.train_arm("fedprox", clients, n_features, seed, mu_selected)[0]
    if v["kind"] == "pooled":
        return train_pooled_variant(clients, n_features, seed,
                                    epochs=v.get("epochs", rfd.CENTRAL_EPOCHS),
                                    vehicle_batches=v.get("vehicle_batches", False),
                                    reset_every=v.get("reset_every"))
    return train_fed_variant(clients, n_features, seed, v["rounds"], v["local"],
                             v["mu"])


# ----------------------------------------------------------------------
# measurements
# ----------------------------------------------------------------------

@torch.no_grad()
def logits(model, X):
    model.eval()
    if len(X) == 0:
        return np.array([])
    return np.concatenate([model(X[s:s + 8192]).double().numpy()
                           for s in range(0, len(X), 8192)])


def sample_rows(X, n, seed=0):
    if len(X) <= n:
        return X
    idx = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
    return X[np.sort(idx)]


def input_gradients(model, X):
    """Per-window gradient of the logit with respect to the (scaled) input."""
    model.eval()
    Xs = X.clone().requires_grad_(True)
    model(Xs).sum().backward()
    return Xs.grad.detach().double().numpy()


def weight_norm(model):
    """Frobenius norm of the Linear layers' weights (LayerNorm gains left out)."""
    return float(np.sqrt(sum(float((m.weight.detach() ** 2).sum())
                             for m in model.modules() if isinstance(m, nn.Linear))))


def measure(model, d, thr_budget):
    """Everything for one trained model. d holds the prepared tensors."""
    out = {}
    # --- outcome, exactly as in run_federated (probabilities) ---
    p = {k: rfd.predict(model, d[k]) for k in ("calib", "bsrc", "src", "t", "clean")}
    thr_zs, _ = best_threshold(d["y_calib"], p["calib"])
    rules = threshold_rules(thr_zs, p["bsrc"], p["clean"], thr_budget)
    ev = evaluate(d["y_t"], p["t"], rules)
    for k in ("auroc", "auprc", "mcc_zs", "mcc_zsb", "mcc_cal", "mcc_oracle", "fpr_zs"):
        out[k] = ev[k]
    out["source_mcc_zs"] = evaluate(d["y_src"], p["src"], {"zs": thr_zs})["mcc_zs"]
    out["zs_flags_target_attack_free"] = float(np.mean(p["clean"] >= thr_zs))

    # --- where the scores sit, on the logit scale ---
    z = {k: logits(model, d[k]) for k in ("calib", "bsrc", "src", "t", "clean")}
    zs_thr, _ = best_threshold(d["y_calib"], z["calib"])
    src_b = z["src"][d["y_src"] == 0]
    src_a = z["src"][d["y_src"] == 1]
    tgt_a = z["t"][d["y_t"] == 1]
    med_b = float(np.median(src_b))
    G = float(np.median(src_a) - med_b)
    out["class_gap_logit"] = G
    out["source_benign_sd_logit"] = float(np.std(src_b))
    out["headroom"] = (zs_thr - med_b) / G
    out["shift"] = (float(np.median(z["clean"])) - med_b) / G
    out["tail_shift"] = (float(np.quantile(z["clean"], 0.99))
                         - float(np.quantile(z["bsrc"], 0.99))) / G
    out["margin"] = (float(np.median(tgt_a)) - zs_thr) / G
    out["threshold_shift_logit"] = (budget_threshold(z["clean"], thr_budget)
                                    - budget_threshold(z["bsrc"], thr_budget)) / G
    out["saturated_share"] = float(np.mean(np.abs(z["src"]) > 10))

    # --- mechanism ---
    out["weight_norm"] = weight_norm(model)
    g_src = input_gradients(model, sample_rows(d["bsrc"], GRAD_SAMPLE, 1))
    g_tgt = input_gradients(model, sample_rows(d["clean"], GRAD_SAMPLE, 2))
    out["input_grad_norm_source"] = float(np.mean(np.linalg.norm(g_src, axis=1)))
    out["input_grad_norm_target"] = float(np.mean(np.linalg.norm(g_tgt, axis=1)))
    out["input_grad_norm_source_rel"] = out["input_grad_norm_source"] / G
    w = np.mean(np.abs(g_src), axis=0)                       # weight per feature
    out["exposure_sources"] = float(np.sum(w * d["spread_sources"]) / np.sum(w))
    out["exposure_target"] = float(np.sum(w * np.abs(d["shift_target"])) / np.sum(w))
    pred = float(np.sum(np.mean(g_src, axis=0) * d["shift_target"]))
    out["predicted_shift"] = pred / G
    out["actual_shift_logit"] = float(np.median(z["clean"])) - med_b
    for i, v in enumerate(d["sources"]):
        m = d["src_vehicle"] == i
        yy = d["y_src"][m]
        out[f"auroc_source_{i}"] = (roc_auc_score(yy, z["src"][m])
                                    if 0 < yy.mean() < 1 else np.nan)
    return out


# ----------------------------------------------------------------------
# data for one held-out vehicle
# ----------------------------------------------------------------------

def selections():
    mlp = pd.read_csv(RESULTS / "mlp_selected.csv")
    out = {}
    for v in VEHICLES:
        m = dict(zip(mlp.loc[mlp["held_out"] == v, "parameter"],
                     mlp.loc[mlp["held_out"] == v, "chosen"]))
        out[v] = {"features": str(m["feature_set"]), "budget": str(m["budget"]),
                  "mu": float(m["fedprox_mu"])}
    return out


def prepare(held_out, data, parts, clean, evalset, s):
    sources = [v for v in VEHICLES if v != held_out]
    cols = feature_columns(data[VEHICLES[0]], s["features"])
    sc = rfd.Scaler(pd.concat([parts[v][0] for v in sources]), rfd.all_columns(data))
    clients = [(sc.X(parts[v][0], cols), rfd.labels(parts[v][0])) for v in sources]
    src_calib = pd.concat([parts[v][1] for v in sources], ignore_index=True)
    src_tests = [scored(parts[v][2]) for v in sources]
    src_test = pd.concat(src_tests, ignore_index=True)
    benign_src = heldout_benign(parts, sources)

    # feature shifts on the model's input scale (scaled, arcsinh)
    med = []
    for v in sources:
        fit = parts[v][0]
        med.append(np.median(sc.X(fit[fit["label"] == 0], cols).numpy(), axis=0))
    med = np.array(med)
    src_benign_all = pd.concat([parts[v][0][parts[v][0]["label"] == 0]
                                for v in sources])
    d = {
        "sources": sources, "cols": cols, "clients": clients,
        "calib": sc.X(src_calib, cols), "y_calib": src_calib["label"].to_numpy(),
        "bsrc": sc.X(benign_src, cols),
        "src": sc.X(src_test, cols), "y_src": src_test["label"].to_numpy(),
        "src_vehicle": np.concatenate([np.full(len(t), i)
                                       for i, t in enumerate(src_tests)]),
        "t": sc.X(evalset[held_out], cols), "y_t": evalset[held_out]["label"].to_numpy(),
        "clean": sc.X(clean[held_out], cols),
        "spread_sources": med.std(axis=0),
        "shift_target": (np.median(sc.X(clean[held_out], cols).numpy(), axis=0)
                         - np.median(sc.X(src_benign_all, cols).numpy(), axis=0)),
    }
    return d


# ----------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------

def saved_main():
    mlp = pd.read_csv(RESULTS / "mlp_results.csv")
    mlp["selected"] = mlp["selected"].astype(str).str.lower() == "true"
    mlp = mlp[mlp["selected"] & (mlp["setting"] == "cross") &
              mlp["arm"].isin(MAIN)]
    return mlp[["arm", "held_out", "seed", "auroc", "mcc_zs", "mcc_zsb", "mcc_cal"]]


def check_against_saved(runs):
    old = saved_main().rename(columns={"arm": "variant"})
    m = runs[runs["variant"].isin(MAIN)].merge(
        old, on=["variant", "held_out", "seed"], suffixes=("", "_saved"))
    rows = [{"metric": k, "runs_compared": len(m),
             "max_abs_difference": float((m[k] - m[f"{k}_saved"]).abs().max())}
            for k in ("auroc", "mcc_zs", "mcc_zsb", "mcc_cal")]
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "check.csv", index=False)
    print("\nCHECK: main arms against the saved results")
    print("  " + rep.to_string(index=False).replace("\n", "\n  "))
    ok = len(m) > 0 and rep["max_abs_difference"].max() <= 1e-3
    print("  OK: the main arms reproduce the thesis results." if ok else
          "  WARNING: not reproduced. Send this log before using the tables.")
    return ok


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    data, parts, clean, evalset = load_all()
    sel = selections()
    rows = []
    for held_out in VEHICLES:
        s = sel[held_out]
        d = prepare(held_out, data, parts, clean, evalset, s)
        print(f"\n--- hold out {held_out}: features {s['features']}, budget "
              f"{s['budget']}, mu {s['mu']} ---")
        for seed in SEEDS:
            for name in VARIANTS:
                t0 = time.time()
                model = train_variant(name, d["clients"], len(d["cols"]), seed, s["mu"])
                r = measure(model, d, BUDGETS[s["budget"]])
                rows.append({"variant": name, "held_out": held_out, "seed": seed,
                             "sources": "+".join(d["sources"]), **r})
                print(f"  seed {seed} {name:<24}{time.time() - t0:>4.0f}s  "
                      f"zs {r['mcc_zs']:.3f}  cal {r['mcc_cal']:.3f}  "
                      f"AUROC {r['auroc']:.3f}  flags {r['zs_flags_target_attack_free']:6.1%}"
                      f"  shift {r['shift']:+.2f}  headroom {r['headroom']:.2f}")
        pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)   # save as we go
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "runs.csv", index=False)
    check_against_saved(runs)
    print(f"\n  total {(time.time() - t_start) / 60:.0f} min")
    report()


def window_counts():
    """Attack and benign windows scored on each held-out vehicle."""
    _, _, _, evalset = load_all()
    return {v: (int(evalset[v]["label"].sum()),
                int((evalset[v]["label"] == 0).sum())) for v in VEHICLES}


def recall_from_mcc(mcc, fpr, n_pos, n_neg):
    """
    Recall at the zero-shot threshold, from the stored MCC and false-alarm
    rate. With the false positives fixed, MCC rises strictly with the number
    of true positives, so the recall is determined exactly.
    """
    fp = fpr * n_neg

    def m(tp):
        fn, tn = n_pos - tp, n_neg - fp
        d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return (tp * tn - fp * fn) / d if d > 0 else 0.0

    lo, hi = 1e-9, float(n_pos) - 1e-9
    for _ in range(200):                     # bisection
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if m(mid) < mcc else (lo, mid)
    return (lo + hi) / 2 / n_pos


def false_alarms_and_spread(runs, per_v):
    """Section 6 of the report: what the zero-shot loss consists of."""
    counts = window_counts()
    runs = runs.copy()
    runs["recall_zs"] = [recall_from_mcc(r.mcc_zs, r.fpr_zs, *counts[r.held_out])
                         for r in runs.itertuples()]
    runs["spread"] = runs["source_benign_sd_logit"] / runs["class_gap_logit"]
    pv = runs.groupby(["variant", "held_out"])[
        ["fpr_zs", "recall_zs", "spread", "headroom", "source_mcc_zs"]].mean()
    pv.reset_index().to_csv(OUT / "false_alarms_per_vehicle.csv", index=False)
    order = [v for v in VARIANTS if v in pv.index.get_level_values(0)]
    for col, title in [
            ("fpr_zs", "false-alarm rate at zs (benign windows of the attack recordings)"),
            ("recall_zs", "recall at zs (derived from MCC and false-alarm rate)"),
            ("spread", "spread of source benign logits (sd / G)"),
            ("headroom", "zs threshold above the source benign median (in G)")]:
        print(f"\n  {title}")
        print(f"  {'variant':<24}" + "".join(f"{v[:10]:>11}" for v in VEHICLES)
              + f"{'mean':>9}")
        for name in order:
            vals = [pv.loc[(name, v), col] if (name, v) in pv.index else np.nan
                    for v in VEHICLES]
            print(f"  {name:<24}" + "".join(f"{x:>11.3f}" for x in vals)
                  + f"{np.nanmean(vals):>9.3f}")


def report():
    runs = pd.read_csv(OUT / "runs.csv")
    num = [c for c in runs.columns if c not in ("variant", "held_out", "seed", "sources")]
    per_v = runs.groupby(["variant", "held_out"])[num].mean().reset_index()
    per_v["threshold_shift_mcc"] = per_v["mcc_cal"] - per_v["mcc_zsb"]
    summ = per_v.groupby("variant")[num + ["threshold_shift_mcc"]].mean()
    summ = summ.reindex([v for v in VARIANTS if v in summ.index])
    per_v.to_csv(OUT / "per_vehicle.csv", index=False)
    summ.to_csv(OUT / "summary.csv")
    n_seeds = runs["seed"].nunique()

    def table(title, cols, fmt):
        print(f"\n{title}")
        print(f"  {'variant':<24}" + "".join(f"{h:>12}" for h, _ in cols))
        for v, r in summ.iterrows():
            print(f"  {v:<24}" + "".join(f"{format(r[c], fmt):>12}" for _, c in cols))

    print("\n" + "=" * 100)
    print(f"WHY FEDAVG KEEPS ITS ZERO-SHOT THRESHOLD   mean over 4 held-out vehicles "
          f"({n_seeds} seeds averaged within each vehicle first)")
    print("=" * 100)
    table("1. Outcome on the new vehicle",
          [("src MCC", "source_mcc_zs"), ("AUROC", "auroc"), ("zs", "mcc_zs"),
           ("zsb", "mcc_zsb"), ("cal", "mcc_cal"), ("cal-zsb", "threshold_shift_mcc"),
           ("zs flags", "zs_flags_target_attack_free")], ".3f")
    table("2. Where the scores sit (logit scale, in units of the source class gap G)",
          [("G (logit)", "class_gap_logit"), ("headroom", "headroom"),
           ("shift", "shift"), ("tail shift", "tail_shift"), ("margin", "margin"),
           ("thr shift", "threshold_shift_logit")], ".2f")
    table("3. Mechanism",
          [("weight norm", "weight_norm"), ("grad src", "input_grad_norm_source"),
           ("grad tgt", "input_grad_norm_target"), ("grad/G", "input_grad_norm_source_rel"),
           ("expo src", "exposure_sources"), ("expo tgt", "exposure_target"),
           ("pred shift", "predicted_shift")], ".3f")

    print("\n4. Per vehicle: FedAvg minus pooled (seed means)")
    keys = ["mcc_zs", "zs_flags_target_attack_free", "shift", "headroom", "margin",
            "input_grad_norm_source_rel", "exposure_sources", "predicted_shift"]
    p = per_v.set_index(["variant", "held_out"])
    print(f"  {'held out':<12}" + "".join(f"{k[:12]:>13}" for k in keys))
    for v in VEHICLES:
        if ("fedavg", v) in p.index and ("pooled", v) in p.index:
            diff = p.loc[("fedavg", v), keys] - p.loc[("pooled", v), keys]
            print(f"  {v:<12}" + "".join(f"{x:>+13.3f}" for x in diff))

    print("\n5. Which single change moves pooled towards FedAvg? (zero-shot MCC, "
          "per vehicle)")
    order = [v for v in VARIANTS if (per_v["variant"] == v).any()]
    print(f"  {'variant':<24}" + "".join(f"{v[:10]:>11}" for v in VEHICLES) + f"{'mean':>9}")
    for name in order:
        vals = [p.loc[(name, v), "mcc_zs"] if (name, v) in p.index else np.nan
                for v in VEHICLES]
        print(f"  {name:<24}" + "".join(f"{x:>11.3f}" for x in vals)
              + f"{np.nanmean(vals):>9.3f}")
    print("\n6. What the zero-shot loss consists of, per vehicle")
    false_alarms_and_spread(runs, per_v)
    print("\n  Shift, headroom and margin are in units of G, so a model that only "
          "has larger or\n  smaller logits does not look different. Tables in "
          f"{OUT}/")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    {"run": run, "report": report}.get(what, lambda: print(__doc__))()
