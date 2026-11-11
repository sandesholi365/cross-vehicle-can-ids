"""
Single-vehicle, pooled, FedAvg and FedProx training under leave-one-vehicle-
out, with source-only selection and separate zero-shot and benign-calibrated
results.

All four arms use the SAME network, so the comparison is between training
strategies, not between models. In the federated arms each client is one
real vehicle.

What is run
-----------
For every held-out vehicle H:

1. SELECTION, on the three source vehicles only (common.py explains the
   inner loop). H is not touched.
     feature set  pooled MLP per feature set; highest mean inner AUPRC
     budget B     pooled MLP, selected feature set; highest mean inner MCC
                  at the target-calibrated rule
     FedProx mu   FedProx per mu, selected feature set; highest mean inner
                  AUPRC
   Each inner score is averaged over SELECTION_SEEDS, which are different
   from the evaluation seeds.

2. EVALUATION on H, for every arm and every seed in SEEDS, using the
   selected feature set and budget. FedProx is run at every mu so the
   sensitivity can be shown; only the selected mu is marked selected=True
   and used as "the" FedProx result.

Seeds measure how much a result moves when only the random initialisation
and batch order change. They are NOT independent vehicles: every seed sees
the same four vehicles. Statistics treat the vehicle as the unit
(analyse_results.py).

Settings fixed during development, not selected by the inner loop
-----------------------------------------------------------------
The network (28 -> 64 -> 32 -> 1 with LayerNorm), Adam with learning rate
3e-4, weight decay 1e-4, gradient clipping at 1.0, the positive-class
weight cap of 5, median/IQR scaling with arcsinh compression, 10 rounds of
2 local epochs, and the mu grid. These were fixed while getting training to
run stably (the earlier version diverged or produced constant outputs) and
were not tuned for accuracy. That development runs looked at all four
vehicles is a limitation to state in the thesis; nothing in this list is
chosen per held-out vehicle.

Scaling statistics come from the source vehicles' training windows. In a
real federation this assumes clients share per-feature medians and IQRs.
The zero-shot threshold of the federated arms is set on pooled source
calibration scores, which assumes clients can share model scores (not raw
data) on their calibration windows.

Outputs, in results/
  mlp_selection_log.csv  every inner-loop score behind every choice
  mlp_selected.csv       the choices made for each held-out vehicle
  mlp_results.csv        one row per arm, seed and held-out vehicle
  mlp_per_attack.csv     recall per attack type at zs, zsb, cal and oracle
  scores/mlp_<arm>_<vehicle>.npz  per-window scores (first seed), for the
                         score-distribution figure

Run from the project root:
    QUICK=1 python src/experiments/run_federated.py    (pipeline check)
    python src/experiments/run_federated.py            (the real run)
"""

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BUDGETS, FEATURE_SETS, QUICK, RESULTS, VEHICLES,  # noqa: E402
                    attack_types_trained_on, best_threshold, degenerate,
                    evaluate, feature_columns, heldout_benign, load_all,
                    per_attack_rows, pick_best, pick_budget, save_scores, scored,
                    threshold_rules)

HIDDEN = (64, 32)
ROUNDS = 10
LOCAL_EPOCHS = 2
CENTRAL_EPOCHS = ROUNDS * LOCAL_EPOCHS   # single and pooled see the same
BATCH = 512
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
POS_WEIGHT_CAP = 5.0
SHIFT_SD = 5.0
MU_VALUES = [0.01, 0.1, 1.0]

SEEDS = [0] if QUICK else list(range(10))
SELECTION_SEEDS = [100] if QUICK else [100, 101, 102]


# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------

class MLP(nn.Module):
    """LayerNorm, not BatchNorm: no running statistics to average."""

    def __init__(self, n_features):
        super().__init__()
        a, b = HIDDEN
        self.net = nn.Sequential(
            nn.Linear(n_features, a), nn.LayerNorm(a), nn.ReLU(),
            nn.Linear(a, b), nn.LayerNorm(b), nn.ReLU(),
            nn.Linear(b, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def new_model(n_features, seed):
    torch.manual_seed(seed)
    return MLP(n_features)


def pos_weight_for(y):
    n_pos = float(y.sum())
    raw = (len(y) - n_pos) / max(n_pos, 1.0)
    return torch.tensor(float(np.clip(raw, 1.0, POS_WEIGHT_CAP)),
                        dtype=torch.float32)


def train_local(model, X, y, epochs, global_params=None, mu=0.0, seed=0):
    """Minibatch training; with global_params, adds FedProx's proximal term."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_for(y))
    gen = torch.Generator().manual_seed(seed)
    last = float("nan")
    for _ in range(epochs):
        order = torch.randperm(len(X), generator=gen)
        for start in range(0, len(X), BATCH):
            idx = order[start:start + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            if global_params is not None and mu > 0:
                prox = sum(((p - g) ** 2).sum()
                           for p, g in zip(model.parameters(), global_params))
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            last = float(loss.detach())
    model.final_loss = last
    return model


def average_weights(states, weights):
    total = float(sum(weights))
    out = copy.deepcopy(states[0])
    for key in out:
        out[key] = sum(s[key] * (w / total) for s, w in zip(states, weights))
    return out


def client_drift(local, global_params):
    total = sum(((p - g) ** 2).sum().item()
                for p, g in zip(local.parameters(), global_params))
    return float(np.sqrt(total))


def train_federated(n_features, clients, mu=0.0, seed=0):
    """clients: list of (X, y) tensors, one per vehicle."""
    global_model = new_model(n_features, seed)
    sizes = [len(X) for X, _ in clients]
    drifts, losses = [], []
    for r in range(ROUNDS):
        global_params = [p.detach().clone() for p in global_model.parameters()]
        states = []
        for c, (X, y) in enumerate(clients):
            local = copy.deepcopy(global_model)
            train_local(local, X, y, LOCAL_EPOCHS,
                        global_params=global_params if mu > 0 else None, mu=mu,
                        seed=seed * 1000 + r * 10 + c)
            drifts.append(client_drift(local, global_params))
            losses.append(local.final_loss)
            states.append(local.state_dict())
        global_model.load_state_dict(average_weights(states, sizes))
    global_model.final_loss = float(np.mean(losses[-len(clients):]))
    return global_model, float(np.mean(drifts))


def train_arm(arm, clients, n_features, seed, mu=0.0):
    """Returns (model, mean client drift or nan)."""
    if arm == "pooled":
        X = torch.cat([c[0] for c in clients])
        y = torch.cat([c[1] for c in clients])
        return train_local(new_model(n_features, seed), X, y,
                           CENTRAL_EPOCHS, seed=seed), np.nan
    if arm == "fedavg":
        return train_federated(n_features, clients, mu=0.0, seed=seed)
    if arm == "fedprox":
        return train_federated(n_features, clients, mu=mu, seed=seed)
    raise ValueError(arm)


@torch.no_grad()
def predict(model, X):
    model.eval()
    if len(X) == 0:
        return np.array([])
    return np.concatenate([torch.sigmoid(model(X[s:s + 8192])).numpy()
                           for s in range(0, len(X), 8192)])


# ----------------------------------------------------------------------
# scaling, fitted on source training windows only
# ----------------------------------------------------------------------

class Scaler:
    """
    Median/IQR per column from the given training windows, then arcsinh.

    arcsinh keeps an unseen vehicle's extreme values (tens of source standard
    deviations away) small but still distinct; hard clipping collapsed them
    onto one value and produced constant outputs.
    """

    def __init__(self, fit_df, cols):
        a = fit_df[cols].to_numpy(dtype=np.float32)
        centre = np.median(a, axis=0)
        q75, q25 = np.percentile(a, [75, 25], axis=0)
        scale = (q75 - q25) / 1.349
        scale[scale <= 0] = 1.0
        self.centre = dict(zip(cols, centre.astype(np.float32)))
        self.scale = dict(zip(cols, scale.astype(np.float32)))

    def z(self, df, cols):
        c = np.array([self.centre[k] for k in cols], dtype=np.float32)
        s = np.array([self.scale[k] for k in cols], dtype=np.float32)
        return (df[cols].to_numpy(dtype=np.float32) - c) / s

    def X(self, df, cols):
        return torch.from_numpy(np.arcsinh(self.z(df, cols)).astype(np.float32))


def labels(df):
    return torch.from_numpy(df["label"].to_numpy(np.float32))


def all_columns(data):
    return feature_columns(data[VEHICLES[0]], "all")


# ----------------------------------------------------------------------
# 1. source-only selection
# ----------------------------------------------------------------------

def inner_eval(model, sc, cols, calib, pretend_clean, pretend_eval):
    thr_zs, _ = best_threshold(calib["label"].to_numpy(),
                               predict(model, sc.X(calib, cols)))
    rules = {"zs": thr_zs}
    p_clean = predict(model, sc.X(pretend_clean, cols))
    rules.update({k: v for k, v in threshold_rules(
        thr_zs, np.array([]), p_clean, 0.0).items() if k.startswith("cal_")})
    return evaluate(pretend_eval["label"].to_numpy(),
                    predict(model, sc.X(pretend_eval, cols)), rules)


def select(held_out, data, parts, clean, evalset):
    sources = [v for v in VEHICLES if v != held_out]
    everything = all_columns(data)
    log = []

    def note(kind, candidate, pretend, inner, seed, row, drift=np.nan):
        log.append({"held_out": held_out, "pretend_target": pretend,
                    "inner_sources": "+".join(inner), "kind": kind,
                    "candidate": candidate, "seed": seed,
                    "client_drift": drift, **row})

    # feature set, then budget, with the pooled model
    for pretend in sources:
        inner = [v for v in sources if v != pretend]
        sc = Scaler(pd.concat([parts[v][0] for v in inner]), everything)
        calib = pd.concat([parts[v][1] for v in inner], ignore_index=True)
        for group in FEATURE_SETS:
            cols = feature_columns(data[VEHICLES[0]], group)
            clients = [(sc.X(parts[v][0], cols), labels(parts[v][0]))
                       for v in inner]
            for sd in SELECTION_SEEDS:
                model, _ = train_arm("pooled", clients, len(cols), sd)
                row = inner_eval(model, sc, cols, calib, clean[pretend],
                                 evalset[pretend])
                note("feature_set", group, pretend, inner, sd, row)
            last = pd.DataFrame(log)
            last = last[(last["kind"] == "feature_set") &
                        (last["candidate"] == group) &
                        (last["pretend_target"] == pretend)]
            print(f"    inner: {'+'.join(inner):<20} -> {pretend:<10} "
                  f"{group:<8} AUPRC {last['auprc'].mean():.3f}")

    flog = pd.DataFrame(log)
    flog = flog[flog["kind"] == "feature_set"]
    group, _, auprc_means = pick_best(flog, "candidate", FEATURE_SETS, "auprc")
    budget, mcc_means = pick_budget(flog[flog["candidate"] == group])
    cols = feature_columns(data[VEHICLES[0]], group)

    # mu, with the selected feature set
    for pretend in sources:
        inner = [v for v in sources if v != pretend]
        sc = Scaler(pd.concat([parts[v][0] for v in inner]), everything)
        calib = pd.concat([parts[v][1] for v in inner], ignore_index=True)
        clients = [(sc.X(parts[v][0], cols), labels(parts[v][0])) for v in inner]
        for mu in MU_VALUES:
            aps = []
            for sd in SELECTION_SEEDS:
                model, drift = train_arm("fedprox", clients, len(cols), sd, mu)
                row = inner_eval(model, sc, cols, calib, clean[pretend],
                                 evalset[pretend])
                note("mu", mu, pretend, inner, sd, row, drift)
                aps.append(row["auprc"])
            print(f"    inner: {'+'.join(inner):<20} -> {pretend:<10} "
                  f"FedProx mu={mu:<5} AUPRC {np.mean(aps):.3f}")

    mlog = pd.DataFrame(log)
    mlog = mlog[mlog["kind"] == "mu"]
    mu, _, mu_means = pick_best(mlog, "candidate", MU_VALUES, "auprc")

    used = f"inner leave-one-source-out on {', '.join(sources)}"
    chosen = [
        {"family": "mlp", "held_out": held_out, "parameter": "feature_set",
         "chosen": group, "criterion": "mean inner AUPRC, pooled MLP",
         "scores": "; ".join(f"{k} {v:.4f}" for k, v in auprc_means.items()),
         "data_used": used},
        {"family": "mlp", "held_out": held_out, "parameter": "budget",
         "chosen": budget,
         "criterion": f"mean inner MCC at cal, pooled MLP ({group})",
         "scores": "; ".join(f"{k} {v:.4f}" for k, v in mcc_means.items()),
         "data_used": used},
        {"family": "mlp", "held_out": held_out, "parameter": "fedprox_mu",
         "chosen": mu, "criterion": f"mean inner AUPRC, FedProx ({group})",
         "scores": "; ".join(f"{k} {v:.4f}" for k, v in mu_means.items()),
         "data_used": used},
    ]
    print(f"  selected for {held_out}: feature set {group}, budget {budget} "
          f"({BUDGETS[budget]:.2%}), mu {mu}")
    return ({"feature_set": group, "budget": budget, "mu": mu},
            pd.DataFrame(log), chosen)


# ----------------------------------------------------------------------
# 2. evaluation
# ----------------------------------------------------------------------

def shift_report(z_source, z_target):
    return {
        "median_abs_z_source": float(np.median(np.abs(z_source))),
        "median_abs_z_target": float(np.median(np.abs(z_target))),
        "frac_target_out_of_range": float(np.mean(np.abs(z_target) > SHIFT_SD)),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    t_start = time.time()
    print("loading features ...")
    data, parts, clean, evalset = load_all()
    everything = all_columns(data)

    print("\n" + "=" * 84)
    print("1. SOURCE-ONLY SELECTION")
    print("=" * 84)
    sel, logs, chosen = {}, [], []
    for held_out in VEHICLES:
        print(f"\n  held out {held_out} (not used here)")
        t0 = time.time()
        sel[held_out], log, ch = select(held_out, data, parts, clean, evalset)
        logs.append(log)
        chosen += ch
        print(f"  ({(time.time() - t0) / 60:.1f} min)")
    pd.concat(logs).to_csv(RESULTS / "mlp_selection_log.csv", index=False)
    pd.DataFrame(chosen).to_csv(RESULTS / "mlp_selected.csv", index=False)

    print("\n" + "=" * 84)
    print("2. EVALUATION ON THE HELD-OUT VEHICLE")
    print("=" * 84)
    results, attack_rows = [], []

    for held_out in VEHICLES:
        s = sel[held_out]
        group, b, mu_sel = s["feature_set"], s["budget"], s["mu"]
        cols = feature_columns(data[VEHICLES[0]], group)
        sources = [v for v in VEHICLES if v != held_out]
        label = "+".join(sources)

        sc = Scaler(pd.concat([parts[v][0] for v in sources]), everything)
        clients = [(sc.X(parts[v][0], cols), labels(parts[v][0])) for v in sources]
        src_calib = pd.concat([parts[v][1] for v in sources], ignore_index=True)
        src_test = scored(pd.concat([parts[v][2] for v in sources],
                                    ignore_index=True))
        Xcal = sc.X(src_calib, cols)
        Xbsrc = sc.X(heldout_benign(parts, sources), cols)
        Xsrc_test = sc.X(src_test, cols)
        Xt = sc.X(evalset[held_out], cols)
        Xclean = sc.X(clean[held_out], cols)
        y_true = evalset[held_out]["label"].to_numpy()
        shift = shift_report(sc.z(pd.concat([parts[v][0] for v in sources]), cols),
                             sc.z(evalset[held_out], cols))
        seen_all = attack_types_trained_on(pd.concat([parts[v][0] for v in sources]))

        print(f"\n--- hold out {held_out}: features {group}, budget {b}, "
              f"mu {mu_sel}; {shift['frac_target_out_of_range']:.1%} of target "
              f"values beyond {SHIFT_SD:.0f} SD of the sources ---")

        runs = [("single", v, 0.0) for v in sources]
        runs += [("pooled", label, 0.0), ("fedavg", label, 0.0)]
        runs += [("fedprox", label, mu) for mu in MU_VALUES]

        for seed in SEEDS:
            print(f"  seed {seed}")
            for arm, train_label, mu in runs:
                t0 = time.time()
                if arm == "single":
                    i = sources.index(train_label)
                    model, drift = train_local(
                        new_model(len(cols), seed), *clients[i],
                        CENTRAL_EPOCHS, seed=seed), np.nan
                    calib_v = parts[train_label][1]
                    thr_zs, _ = best_threshold(
                        calib_v["label"].to_numpy(),
                        predict(model, sc.X(calib_v, cols)))
                    p_bsrc = predict(model, sc.X(
                        heldout_benign(parts, [train_label]), cols))
                    seen = attack_types_trained_on(parts[train_label][0])
                else:
                    model, drift = train_arm(arm, clients, len(cols), seed, mu)
                    thr_zs, _ = best_threshold(src_calib["label"].to_numpy(),
                                               predict(model, Xcal))
                    p_bsrc = predict(model, Xbsrc)
                    seen = seen_all

                prob = predict(model, Xt)
                p_clean = predict(model, Xclean)
                rules = threshold_rules(thr_zs, p_bsrc, p_clean, BUDGETS[b])
                mode = f"fedprox_mu{mu}" if arm == "fedprox" else arm
                is_sel = arm != "fedprox" or mu == mu_sel
                bad = degenerate(prob, y_true)

                info = {"family": "mlp", "feature_set": group, "mode": mode,
                        "arm": arm, "mu": mu if arm == "fedprox" else np.nan,
                        "train_vehicles": train_label, "test_vehicle": held_out,
                        "held_out": held_out, "setting": "cross", "seed": seed,
                        "selected": is_sel}
                row = {**info, "budget": b, "client_drift": drift,
                       "final_loss": float(getattr(model, "final_loss", np.nan)),
                       "degenerate": bad, **shift}
                row.update(evaluate(y_true, prob, rules))
                results.append(row)
                keep = {t: rules[t] for t in ("zs", "zsb", "cal")}
                keep["oracle"] = row["thr_oracle"]
                attack_rows += per_attack_rows(evalset[held_out], prob, keep,
                                               seen, info)

                if arm != "single":
                    p_src = predict(model, Xsrc_test)
                    src_rules = {"05": 0.5, "zs": thr_zs, "zsb": rules["zsb"]}
                    srow = {**info, "test_vehicle": "sources",
                            "setting": "within_sources", "budget": b,
                            "client_drift": drift, "degenerate": False}
                    srow.update(evaluate(src_test["label"].to_numpy(), p_src,
                                         src_rules))
                    results.append(srow)
                    if seed == SEEDS[0] and is_sel:
                        save_scores(
                            f"mlp_{arm}_{held_out}",
                            target_score=prob.astype(np.float32),
                            target_label=y_true.astype(np.int8),
                            target_attack=evalset[held_out]["attack_type"].to_numpy(str),
                            target_benign_score=p_clean.astype(np.float32),
                            source_score=p_src.astype(np.float32),
                            source_label=src_test["label"].to_numpy(np.int8),
                            thr_zs=thr_zs, thr_zsb=rules["zsb"],
                            thr_cal=rules["cal"], thr_oracle=row["thr_oracle"],
                            thr_oracle_source=srow["thr_oracle"],
                            feature_set=group, budget=b, mode=mode)

                name = f"{arm}:{train_label}" if arm == "single" else mode
                d_txt = "" if np.isnan(drift) else f"  drift {drift:4.2f}"
                print(f"    {name:<18}{time.time() - t0:>4.0f}s   "
                      f"zs {row['mcc_zs']:.3f}  zsb {row['mcc_zsb']:.3f}  "
                      f"cal {row['mcc_cal']:.3f}  oracle {row['mcc_oracle']:.3f}  "
                      f"AUPRC {row['auprc']:.3f}{d_txt}"
                      + ("  <-- DEGENERATE" if bad else ""))

    res = pd.DataFrame(results)
    res.to_csv(RESULTS / "mlp_results.csv", index=False)
    pd.DataFrame(attack_rows).to_csv(RESULTS / "mlp_per_attack.csv", index=False)

    summarise(res, sel)
    print(f"\ntotal runtime {(time.time() - t_start) / 60:.0f} min")
    print(f"saved mlp_results.csv, mlp_per_attack.csv, mlp_selection_log.csv, "
          f"mlp_selected.csv and scores/ in {RESULTS}/")


def summarise(res, sel):
    cross = res[(res["setting"] == "cross") & res["selected"].astype(bool)]
    n_bad = int(cross["degenerate"].astype(bool).sum())

    print(f"\n\n{'#' * 84}\nSUMMARY (MLP)   mean over {len(SEEDS)} seeds, "
          f"per held-out vehicle\n{'#' * 84}")
    if n_bad:
        print(f"\n  WARNING: {n_bad} run(s) gave constant scores; excluded below")
    cross = cross[~cross["degenerate"].astype(bool)]

    print("\n--- choices made on the source vehicles ---")
    for v in VEHICLES:
        s = sel[v]
        print(f"  held out {v:<10} feature set {s['feature_set']:<8} "
              f"budget {s['budget']:<5} mu {s['mu']}")

    for metric, title in [("mcc_zs", "MCC, zero-shot (zs)"),
                          ("mcc_cal", "MCC, benign-calibrated (cal)"),
                          ("auprc", "AUPRC")]:
        print(f"\n--- {title} ---")
        print(f"  {'arm':<10}" + "".join(f"{v[:9]:>11}" for v in VEHICLES)
              + f"{'MEAN':>9}")
        for arm in ["single", "pooled", "fedavg", "fedprox"]:
            per_v = []
            for v in VEHICLES:
                sub = cross[(cross["arm"] == arm) & (cross["held_out"] == v)]
                # single: average the three source vehicles within each seed,
                # then over seeds
                per_v.append(sub.groupby("seed")[metric].mean().mean())
            print(f"  {arm:<10}" + "".join(f"{x:>11.3f}" for x in per_v)
                  + f"{np.nanmean(per_v):>9.3f}")
    print("\n  Each cell averages the seeds, so there is one number per vehicle."
          "\n  With four vehicles, differences are described per vehicle, not"
          "\n  tested as if the seeds were independent (analyse_results.py).")


if __name__ == "__main__":
    main()
