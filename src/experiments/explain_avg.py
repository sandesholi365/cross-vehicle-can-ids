"""
Pre-check for a next direction: is the averaged model's wider band of normal
scores what an ENSEMBLE of the client models would also give?

Background (thesis, Section "Why Federated Training Keeps Its Zero-Shot
Threshold"): averaging three single-vehicle models once ("FedAvg, 1 round x
20 epochs") keeps the zero-shot threshold, while pooled training does not.
Diverse Weight Averaging (Rame et al., NeurIPS 2022) shows that averaging the
WEIGHTS of models that start from the same point approximates averaging their
PREDICTIONS (an ensemble). That gives three predictions to check:

  P1  an ensemble of the same three single-vehicle models behaves like their
      weight average (spread, headroom, false alarms on the new vehicle)
  P2  the band is wider where the three models DISAGREE more about the
      sources' normal windows
  P3  moving the weights step by step from the pooled model to the averaged
      model widens the band and lowers the false alarms gradually

CAN-Train-and-Test only, ROAD is not touched, nothing in the method changes:
the feature set, budget and mu are read from results/mlp_selected.csv.

For every held-out vehicle and seed:
  members     three single-vehicle models trained for 20 epochs from the
              same starting weights (exactly the clients of "FedAvg, 1 round
              x 20 epochs" in explain_arms.py)
  avg_weights their weight average, weighted by client size
              (= explain_arms "fedavg_20x1"; checked against runs.csv)
  ens_logit   ensemble: weighted mean of the members' logits
  ens_prob    ensemble: weighted mean of the members' probabilities
  pooled      the main pooled model (checked against mlp_results.csv)
  interp_a    weights (1-a)*pooled + a*avg_weights, a = 0.25, 0.5, 0.75
  member_k    each single-vehicle model on its own (for reference only:
              two of the three sources are new vehicles for it)

Run from the project root:
    python -u src/experiments/explain_avg.py run      (about 20-30 min)
    python -u src/experiments/explain_avg.py report   (seconds)
Outputs in results/explain_avg/: runs.csv, summary.csv, check.csv
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
from common import BUDGETS, QUICK, RESULTS, VEHICLES, load_all  # noqa: E402
import run_federated as rfd  # noqa: E402
import explain_arms as ea  # noqa: E402

OUT = RESULTS / ("explain_avg_quick" if QUICK else "explain_avg")
SEEDS = ea.SEEDS
LOCAL_EPOCHS = 20
ALPHAS = [0.25, 0.5, 0.75]
ORDER = ["pooled", "interp_0.25", "interp_0.5", "interp_0.75", "avg_weights",
         "ens_logit", "ens_prob", "member_0", "member_1", "member_2"]


class Ensemble(nn.Module):
    """Combines the members' outputs; returns a logit so that every
    measurement in explain_arms.measure works unchanged."""

    def __init__(self, members, weights, mode):
        super().__init__()
        self.members = nn.ModuleList(members)
        w = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("logw", torch.log(w / w.sum()))
        self.mode = mode

    def forward(self, x):
        z = torch.stack([m(x) for m in self.members])          # (M, N)
        if self.mode == "logit":
            return (self.logw.exp()[:, None] * z).sum(0)
        # mixture of probabilities, written as a logit in a numerically
        # stable way: log(sum w sigmoid(z)) - log(sum w sigmoid(-z))
        lw = self.logw[:, None]
        pos = torch.logsumexp(lw + nn.functional.logsigmoid(z), dim=0)
        neg = torch.logsumexp(lw + nn.functional.logsigmoid(-z), dim=0)
        return pos - neg


def train_members(clients, n_features, seed):
    """The clients of explain_arms 'fedavg_20x1': same start, one round."""
    start = rfd.new_model(n_features, seed)
    members = []
    for c, (X, y) in enumerate(clients):
        m = copy.deepcopy(start)
        rfd.train_local(m, X, y, LOCAL_EPOCHS, seed=seed * 1000 + 0 * 10 + c)
        members.append(m)
    return members


def from_state(n_features, state):
    m = rfd.new_model(n_features, 0)
    m.load_state_dict(state)
    return m


def mix(a_state, b_state, alpha):
    return {k: (1 - alpha) * a_state[k] + alpha * b_state[k] for k in a_state}


def disagreement(members, X, G):
    """Mean over windows of the spread of the members' logits, in units of G."""
    z = np.stack([ea.logits(m, X) for m in members])
    return float(np.mean(np.std(z, axis=0)) / G)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    data, parts, clean, evalset = load_all()
    sel = ea.selections()
    counts = {v: (int(evalset[v]["label"].sum()), int((evalset[v]["label"] == 0).sum()))
              for v in VEHICLES}
    rows = []
    for held_out in VEHICLES:
        s = sel[held_out]
        d = ea.prepare(held_out, data, parts, clean, evalset, s)
        nf = len(d["cols"])
        sizes = [len(X) for X, _ in d["clients"]]
        budget = BUDGETS[s["budget"]]
        print(f"\n--- hold out {held_out}: features {s['features']}, budget {s['budget']} ---")
        for seed in SEEDS:
            t0 = time.time()
            members = train_members(d["clients"], nf, seed)
            avg_state = rfd.average_weights([m.state_dict() for m in members], sizes)
            pooled = rfd.train_arm("pooled", d["clients"], nf, seed)[0]
            pooled_state = pooled.state_dict()
            models = {"pooled": pooled,
                      "avg_weights": from_state(nf, avg_state),
                      "ens_logit": Ensemble(members, sizes, "logit"),
                      "ens_prob": Ensemble(members, sizes, "prob")}
            for a in ALPHAS:
                models[f"interp_{a}"] = from_state(nf, mix(pooled_state, avg_state, a))
            for i, m in enumerate(members):
                models[f"member_{i}"] = m
            for name in ORDER:
                r = ea.measure(models[name], d, budget)
                r["recall_zs"] = ea.recall_from_mcc(r["mcc_zs"], r["fpr_zs"],
                                                    *counts[held_out])
                r["spread"] = r["source_benign_sd_logit"] / r["class_gap_logit"]
                if name == "avg_weights":
                    r["disagreement"] = disagreement(members, d["bsrc"],
                                                     r["class_gap_logit"])
                rows.append({"setting": name, "held_out": held_out, "seed": seed, **r})
            pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)
            got = {r["setting"]: r for r in rows[-len(ORDER):]}
            print(f"  seed {seed} {time.time() - t0:>4.0f}s   zs MCC / FPR zs / spread:"
                  + "".join(f"  {k} {got[k]['mcc_zs']:.2f}/{got[k]['fpr_zs']:.3f}/"
                            f"{got[k]['spread']:.2f}"
                            for k in ("pooled", "avg_weights", "ens_logit", "ens_prob")))
    check()
    print(f"\n  total {(time.time() - t_start) / 60:.0f} min")
    report()


def check():
    """avg_weights must equal explain_arms 'fedavg_20x1'; pooled the saved run."""
    runs = pd.read_csv(OUT / "runs.csv")
    rows = []
    prev = ea.OUT / "runs.csv"
    if prev.exists():
        old = pd.read_csv(prev)
        m = runs[runs["setting"] == "avg_weights"].merge(
            old[old["variant"] == "fedavg_20x1"], on=["held_out", "seed"],
            suffixes=("", "_old"))
        for k in ("auroc", "mcc_zs", "fpr_zs"):
            rows.append({"compared": "avg_weights vs explain_arms fedavg_20x1",
                         "metric": k, "runs": len(m),
                         "max_abs_difference": float((m[k] - m[f"{k}_old"]).abs().max())})
    saved = ea.saved_main()
    m = runs[runs["setting"] == "pooled"].merge(
        saved[saved["arm"] == "pooled"], on=["held_out", "seed"], suffixes=("", "_saved"))
    for k in ("auroc", "mcc_zs"):
        rows.append({"compared": "pooled vs saved mlp_results", "metric": k,
                     "runs": len(m),
                     "max_abs_difference": float((m[k] - m[f"{k}_saved"]).abs().max())})
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "check.csv", index=False)
    print("\nCHECK: re-trained models against earlier results")
    print("  " + rep.to_string(index=False).replace("\n", "\n  "))
    ok = len(rep) > 0 and (rep["runs"] > 0).all() and rep["max_abs_difference"].max() <= 1e-3
    print("  OK: same models as before." if ok else
          "  WARNING: not reproduced. Send this log before using the tables.")


def report():
    runs = pd.read_csv(OUT / "runs.csv")
    cols = ["source_mcc_zs", "auroc", "mcc_zs", "fpr_zs", "recall_zs", "spread",
            "headroom"]
    per_v = runs.groupby(["setting", "held_out"])[cols].mean()
    summ = per_v.groupby("setting").mean().reindex([o for o in ORDER
                                                    if o in per_v.index.get_level_values(0)])
    summ.to_csv(OUT / "summary.csv")
    print("\n" + "=" * 96)
    print(f"ENSEMBLE vs WEIGHT AVERAGE   mean over 4 held-out vehicles "
          f"({runs['seed'].nunique()} seeds averaged within each vehicle first)")
    print("=" * 96)
    heads = ["src MCC", "AUROC", "zs MCC", "FPR zs", "recall", "spread", "headroom"]
    print(f"  {'setting':<14}" + "".join(f"{h:>10}" for h in heads))
    for name, r in summ.iterrows():
        print(f"  {name:<14}" + "".join(f"{r[c]:>10.3f}" for c in cols))

    print("\n  P1: per vehicle, FPR zs and spread (avg_weights | ens_logit | ens_prob | pooled)")
    for v in VEHICLES:
        vals = []
        for s in ("avg_weights", "ens_logit", "ens_prob", "pooled"):
            if (s, v) in per_v.index:
                vals.append(f"{per_v.loc[(s, v), 'fpr_zs']:.3f}/{per_v.loc[(s, v), 'spread']:.2f}")
        print(f"    {v:<10}" + "   ".join(vals))

    print("\n  P2: does the averaged model's band widen with member disagreement?")
    a = runs[runs["setting"] == "avg_weights"][["held_out", "seed", "spread",
                                                  "disagreement", "fpr_zs"]]
    rho_all = a[["spread", "disagreement"]].corr(method="spearman").iloc[0, 1]
    print(f"    Spearman rho over all {len(a)} runs (spread vs disagreement): {rho_all:+.2f}")
    for v in VEHICLES:
        b = a[a["held_out"] == v]
        if len(b) >= 3:
            rho = b[["spread", "disagreement"]].corr(method="spearman").iloc[0, 1]
            print(f"    {v:<10} rho {rho:+.2f}  (n = {len(b)} seeds)   mean disagreement "
                  f"{b['disagreement'].mean():.2f}")

    print("\n  P3: from pooled (a = 0) to the averaged model (a = 1), per vehicle: FPR zs")
    path = ["pooled", "interp_0.25", "interp_0.5", "interp_0.75", "avg_weights"]
    print(f"    {'':<10}" + "".join(f"{p.replace('interp_', 'a=').replace('pooled', 'a=0').replace('avg_weights', 'a=1'):>9}" for p in path))
    for metric in ("fpr_zs", "spread", "source_mcc_zs"):
        print(f"    [{metric}]")
        for v in VEHICLES:
            vals = [per_v.loc[(p, v), metric] if (p, v) in per_v.index else np.nan
                    for p in path]
            print(f"    {v:<10}" + "".join(f"{x:>9.3f}" for x in vals))
    print(f"\n  Tables in {OUT}/  (member_k rows are for reference only: for a single-"
          f"\n  vehicle model two of the three sources are new vehicles)")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    {"run": run, "report": report, "check": check}.get(what, lambda: print(__doc__))()
