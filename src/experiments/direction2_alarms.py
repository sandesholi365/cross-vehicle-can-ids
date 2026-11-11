"""
Direction 2 (supervisor's email of 20 October): alarms built from several
consecutive windows instead of one.

CAN-Train-and-Test only. ROAD and CrySyS are NOT touched. The detectors are
not changed: the base detectors are the pooled and FedAvg MLPs of the thesis
at the zero-shot (zs) and the benign-calibrated (cal) threshold. Their
per-window scores for all ten seeds come from direction1_averaging.py
(results/direction1/scores/), whose re-run is checked against the saved
thesis results.

Two alarm rules only (fixed in the diary on 20 October)
-------------------------------------------------------
  k-of-n  alarm at window t when at least k of the last n windows of the
          same recording are at or above the threshold;
          n in {1, 3, 5, 10, 20}, 1 <= k <= n   (1-of-1 = the thesis today)
  CUSUM   on the logit scale, per recording:
          S_t = max(0, S_{t-1} + z_t - ref - kappa), alarm when S_t >= h,
          then S is reset to 0. ref is the base threshold on the logit scale
          (zs or cal); kappa and h are in units of the source class gap G:
          kappa in {-0.5, -0.25, 0, 0.25, 0.5, 1},
          h in {0.1, 0.25, 0.5, 1, 2, 4, 8, 16}

Fixed definitions
-----------------
  attack episode   attack windows whose benign gap is at most 1 s form one
                   episode (5 s is reported as a sensitivity check)
  detected         at least one alarm inside the episode
  delay            time from the start of the episode to the end of the first
                   alarm window inside it (seconds)
  false alarm      an alarm on a benign window at least 10 windows away from
                   any attack episode, whose evidence contains no attack
                   window (k-of-n: the last n windows; CUSUM: the windows
                   since S was last 0). Consecutive false alarms are one
                   false-alarm episode.
  driving hours    the summed duration of those benign windows

Choosing k, n, kappa and h on the source vehicles only
------------------------------------------------------
The thesis inner loop: for each held-out vehicle H, train on two of the
three sources (pooled and FedAvg, selection seeds 100-102), treat the third
as new, rotate. The thresholds are set exactly as in the thesis (zs on the
inner calibration windows, cal on the pretend-new vehicle's attack-free
recordings with H's selected budget). Criterion, fixed before any run:

  the highest mean share of detected attack episodes, among the candidates
  whose mean number of false-alarm episodes per driving hour is at most
  LIMIT; LIMIT = 1 (main) and 10 (second operating point). Ties within 0.005
  go to the lower false-alarm rate, then the shorter median delay, then the
  first candidate in the grid. If no candidate meets the limit, the one with
  the lowest false-alarm rate is taken and this is marked.

Each rule family (k-of-n, CUSUM) is chosen separately for each arm,
threshold and limit. H is not used in this step. The earlier 5-of-10
result (feasibility_alarms.py, 9 October) is exploratory only and plays no
part here.

Run from the project root, AFTER direction1_averaging.py run:
    QUICK=1 python -u src/experiments/direction2_alarms.py inner   (pipeline
    QUICK=1 python -u src/experiments/direction2_alarms.py select   check)
    QUICK=1 python -u src/experiments/direction2_alarms.py run
    python -u src/experiments/direction2_alarms.py inner    (about 15-20 min)
    python -u src/experiments/direction2_alarms.py select   (minutes)
    python -u src/experiments/direction2_alarms.py run      (about 20-60 min)
    python -u src/experiments/direction2_alarms.py report   (seconds)
Optional: "pip install numba" makes CUSUM much faster; results are the same.

Outputs in results/direction2/:
    inner_scores/        scores of the inner-loop models (pretend-new vehicles)
    inner_log.csv        every candidate on every inner run
    selected.csv         the rule chosen for each held-out vehicle, arm,
                         threshold, family and limit
    runs.csv             outer results, one row per choice and seed
    per_attack.csv       detected episodes per attack type
    check.csv            1-of-1 false-alarm rate against mlp_results.csv
    summary.csv          means over seeds, then over vehicles
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BUDGETS, QUICK, RESULTS, VEHICLES, best_threshold,  # noqa: E402
                    budget_threshold, feature_columns, load_all, scored)
from feasibility_alarms import episodes_of, runs_of  # noqa: E402

OUT = RESULTS / ("direction2_quick" if QUICK else "direction2")
D1_SCORES = RESULTS / ("direction1_quick" if QUICK else "direction1") / "scores"
INNER = OUT / "inner_scores"
ARMS = ["pooled", "fedavg"]
THRESHOLDS = ["zs", "cal"]
SEEDS = [0] if QUICK else list(range(10))
LIMITS = [1.0, 10.0]
GAPS_S = [1.0, 5.0]
SELECT_GAP_S = 1.0
CLEAR = 10          # benign windows this far from an attack episode count as normal
TIE = 0.005

KOFN = [(k, n) for n in (1, 3, 5, 10, 20) for k in range(1, n + 1)]
KAPPAS = [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
HS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
CUSUM = [(kp, h) for kp in KAPPAS for h in HS]


def rule_name(family, p):
    return f"{p[0]}-of-{p[1]}" if family == "kofn" else f"cusum k{p[0]:g} h{p[1]:g}"


# ----------------------------------------------------------------------
# alarm rules
# ----------------------------------------------------------------------

def kofn_alarms(exceed, k, n):
    """alarm[t] = at least k of exceed[t-n+1..t]; evidence starts at t-n+1."""
    c = np.concatenate([[0], np.cumsum(exceed.astype(np.int64))])
    t = np.arange(len(exceed))
    lo = np.maximum(0, t - n + 1)
    return (c[t + 1] - c[lo]) >= k, lo


def _cusum_py(inc, h):
    n = len(inc)
    alarm = np.zeros(n, np.bool_)
    start = np.zeros(n, np.int64)
    pos = np.flatnonzero(inc > 0)
    if len(pos) == 0:
        return alarm, start
    j, S, s0 = int(pos[0]), 0.0, int(pos[0])
    while j < n:
        if S == 0.0:
            s0 = j
        S += inc[j]
        if S >= h:
            alarm[j] = True
            start[j] = s0
            S = 0.0
        if S <= 0.0:
            S = 0.0
            k = np.searchsorted(pos, j + 1)
            if k >= len(pos):
                break
            j = int(pos[k])
            continue
        j += 1
    return alarm, start


try:                                   # optional speed-up, same results
    from numba import njit

    @njit(cache=True)
    def _cusum_nb(inc, h):
        n = len(inc)
        alarm = np.zeros(n, np.bool_)
        start = np.zeros(n, np.int64)
        S, s0 = 0.0, 0
        for j in range(n):
            if S == 0.0:
                s0 = j
            S += inc[j]
            if S >= h:
                alarm[j] = True
                start[j] = s0
                S = 0.0
            if S < 0.0:
                S = 0.0
        return alarm, start
    _cusum = _cusum_nb
except ImportError:                    # pragma: no cover
    _cusum = _cusum_py


def cusum_alarms(z, ref, kappa, h, G):
    inc = ((z.astype(np.float64) - ref) / G - kappa)
    return _cusum(inc, float(h))


# ----------------------------------------------------------------------
# episodes and counting
# ----------------------------------------------------------------------

def prepare_recordings(meta):
    """
    meta: one vehicle's scored windows in the ORDER of the score arrays, with
    recording, window_index, label, stat_duration, attack_type. Returns, per
    recording, the time order and everything that does not depend on the
    rule.
    """
    recs = []
    order_all = np.arange(len(meta))
    for rec, g in meta.groupby("recording", sort=True):
        idx = order_all[g.index.to_numpy()]
        idx = idx[np.argsort(g["window_index"].to_numpy(), kind="stable")]
        y = meta["label"].to_numpy()[idx].astype(bool)
        dur = np.clip(meta["stat_duration"].to_numpy(dtype=float)[idx], 0.0, None)
        per_gap = {}
        for gap in GAPS_S:
            eps = episodes_of(y, dur, gap)
            span = np.zeros(len(y), bool)
            for st, en in eps:
                span[st:en] = True
            c = np.concatenate([[0], np.cumsum(span.astype(np.int64))])
            t = np.arange(len(y))
            near = (c[np.minimum(len(y), t + CLEAR + 1)] - c[np.maximum(0, t - CLEAR)]) > 0
            clean = ~y & ~near
            per_gap[gap] = {"eps": eps, "span_cum": c, "clean": clean,
                            "clean_s": float(dur[clean].sum())}
        recs.append({"recording": rec, "idx": idx, "y": y, "dur": dur,
                     "attack_type": str(meta["attack_type"].iloc[idx[0]]),
                     "gaps": per_gap})
    return recs


def count(recs, alarm_fn, gaps=GAPS_S):
    """alarm_fn(rec) -> (alarm, evidence_start) in the recording's time order."""
    out = {gap: {"episodes": 0, "detected": 0, "delays": [], "fa_episodes": 0,
                 "clean_s": 0.0, "per_attack": {}} for gap in gaps}
    for r in recs:
        alarm, start = alarm_fn(r)
        for gap in gaps:
            gg, o = r["gaps"][gap], out[gap]
            c = gg["span_cum"]
            t = np.arange(len(alarm))
            evidence_clean = (c[t + 1] - c[start]) == 0
            false = alarm & gg["clean"] & evidence_clean
            o["fa_episodes"] += len(runs_of(false)[0])
            o["clean_s"] += gg["clean_s"]
            pa = o["per_attack"].setdefault(r["attack_type"], [0, 0])
            for st, en in gg["eps"]:
                o["episodes"] += 1
                pa[0] += 1
                hit = np.flatnonzero(alarm[st:en])
                if len(hit):
                    o["detected"] += 1
                    pa[1] += 1
                    o["delays"].append(float(r["dur"][st:st + hit[0] + 1].sum()))
    rows = {}
    for gap, o in out.items():
        hours = o["clean_s"] / 3600.0
        rows[gap] = {
            "attack_episodes": o["episodes"],
            "episode_detection": o["detected"] / o["episodes"] if o["episodes"] else np.nan,
            "median_delay_s": float(np.median(o["delays"])) if o["delays"] else np.nan,
            "p90_delay_s": float(np.quantile(o["delays"], 0.9)) if o["delays"] else np.nan,
            "fa_episodes_per_hour": o["fa_episodes"] / hours if hours > 0 else np.nan,
            "fa_episodes": o["fa_episodes"], "clean_hours": hours,
            "per_attack": o["per_attack"]}
    return rows


def all_rules(recs, prob, logit, thr_prob, thr_logit, G, gaps=GAPS_S):
    """Every candidate of both families on one set of scores."""
    exceed = prob >= thr_prob
    res = []
    for k, n in KOFN:
        r = count(recs, lambda rc: kofn_alarms(exceed[rc["idx"]], k, n), gaps)
        res.append(("kofn", (k, n), r))
    for kp, h in CUSUM:
        r = count(recs, lambda rc: cusum_alarms(logit[rc["idx"]], thr_logit, kp, h, G),
                  gaps)
        res.append(("cusum", (kp, h), r))
    return res


# ----------------------------------------------------------------------
# step 1: inner-loop models (pretend-new vehicles), scores saved
# ----------------------------------------------------------------------

def inner():
    import torch  # noqa: F401  (only needed here)
    import run_federated as rfd
    import explain_arms as ea
    INNER.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    data, parts, clean, evalset = load_all()
    sel = ea.selections()
    everything = rfd.all_columns(data)
    for held_out in VEHICLES:
        sources = [v for v in VEHICLES if v != held_out]
        cols = feature_columns(data[VEHICLES[0]], sel[held_out]["features"])
        budget = BUDGETS[sel[held_out]["budget"]]
        print(f"\n--- held out {held_out} (not used here) ---")
        for pretend in sources:
            inner_src = [v for v in sources if v != pretend]
            sc = rfd.Scaler(pd.concat([parts[v][0] for v in inner_src]), everything)
            calib = pd.concat([parts[v][1] for v in inner_src], ignore_index=True)
            src_test = scored(pd.concat([parts[v][2] for v in inner_src],
                                        ignore_index=True))
            clients = [(sc.X(parts[v][0], cols), rfd.labels(parts[v][0]))
                       for v in inner_src]
            Xcal, Xsrc = sc.X(calib, cols), sc.X(src_test, cols)
            Xt, Xclean = sc.X(evalset[pretend], cols), sc.X(clean[pretend], cols)
            for sd in rfd.SELECTION_SEEDS:
                for arm in ARMS:
                    f = INNER / f"{arm}_{held_out}_{pretend}_s{sd}.npz"
                    if f.exists():
                        continue
                    t0 = time.time()
                    model = rfd.train_arm(arm, clients, len(cols), sd)[0]
                    p_cal, p_t, p_clean = (rfd.predict(model, X) for X in (Xcal, Xt, Xclean))
                    z_cal, z_src, z_t, z_clean = (ea.logits(model, X)
                                                  for X in (Xcal, Xsrc, Xt, Xclean))
                    y_src = src_test["label"].to_numpy()
                    m0 = float(np.median(z_src[y_src == 0]))
                    G = float(np.median(z_src[y_src == 1]) - m0)
                    np.savez_compressed(
                        f, target_score=p_t.astype(np.float32),
                        target_logit=z_t.astype(np.float32),
                        target_label=evalset[pretend]["label"].to_numpy(np.int8),
                        thr_zs=best_threshold(calib["label"].to_numpy(), p_cal)[0],
                        thr_cal=budget_threshold(p_clean, budget),
                        thr_zs_logit=best_threshold(calib["label"].to_numpy(), z_cal)[0],
                        thr_cal_logit=budget_threshold(z_clean, budget),
                        class_gap_logit=G)
                    print(f"  {'+'.join(inner_src):<20} -> {pretend:<10} seed {sd} "
                          f"{arm:<7} {time.time() - t0:>4.0f}s")
    print(f"\n  total {(time.time() - t_start) / 60:.0f} min; scores in {INNER}/")


def load_scores(f, meta):
    z = np.load(f)
    if len(z["target_score"]) != len(meta) or not np.array_equal(
            z["target_label"].astype(int), meta["label"].to_numpy().astype(int)):
        raise SystemExit(f"{f}: scores do not line up with the window table. "
                         "Send this message.")
    return z


def meta_of(evalset, v):
    return evalset[v][["recording", "window_index", "label", "stat_duration",
                       "attack_type"]].reset_index(drop=True)


# ----------------------------------------------------------------------
# step 2: choose the rule on the inner runs
# ----------------------------------------------------------------------

def choose(log, limit):
    """log: rows of one (held_out, arm, threshold, family), all inner runs."""
    m = (log.groupby("rule", sort=False)[["episode_detection", "fa_episodes_per_hour",
                                           "median_delay_s"]].mean())
    ok = m[m["fa_episodes_per_hour"] <= limit]
    if len(ok) == 0:
        if m["fa_episodes_per_hour"].notna().any():
            best = m["fa_episodes_per_hour"].idxmin()
        else:   # no normal traffic far enough from attacks (only in QUICK mode)
            best = m["episode_detection"].fillna(-1).idxmax()
        return best, False, m.loc[best]
    top = ok["episode_detection"].max()
    cand = ok[ok["episode_detection"] >= top - TIE]
    cand = cand.sort_values(["fa_episodes_per_hour", "median_delay_s"],
                            kind="stable", na_position="last")
    best = cand.index[0]
    return best, True, m.loc[best]


def select():
    _, _, _, evalset = load_all()
    import run_federated as rfd
    rows = []
    t_start = time.time()
    for held_out in VEHICLES:
        sources = [v for v in VEHICLES if v != held_out]
        for pretend in sources:
            meta = meta_of(evalset, pretend)
            recs = prepare_recordings(meta)
            for sd in rfd.SELECTION_SEEDS:
                for arm in ARMS:
                    f = INNER / f"{arm}_{held_out}_{pretend}_s{sd}.npz"
                    if not f.exists():
                        raise SystemExit(f"{f} missing: run the 'inner' step first.")
                    z = load_scores(f, meta)
                    for thr in THRESHOLDS:
                        res = all_rules(recs, z["target_score"], z["target_logit"],
                                        float(z[f"thr_{thr}"]),
                                        float(z[f"thr_{thr}_logit"]),
                                        float(z["class_gap_logit"]),
                                        gaps=[SELECT_GAP_S])
                        for fam, p, r in res:
                            rr = r[SELECT_GAP_S]
                            rows.append({"held_out": held_out, "pretend_target": pretend,
                                         "seed": sd, "arm": arm, "threshold": thr,
                                         "family": fam, "rule": rule_name(fam, p),
                                         **{k: v for k, v in rr.items()
                                            if k != "per_attack"}})
            print(f"  inner runs scored: held out {held_out}, pretend {pretend} "
                  f"({(time.time() - t_start) / 60:.1f} min)")
    log = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    log.to_csv(OUT / "inner_log.csv", index=False)
    chosen = []
    for (h, arm, thr, fam), g in log.groupby(["held_out", "arm", "threshold", "family"],
                                             sort=False):
        for limit in LIMITS:
            best, met, vals = choose(g, limit)
            chosen.append({"held_out": h, "arm": arm, "threshold": thr, "family": fam,
                           "limit_per_hour": limit, "rule": best, "limit_met": met,
                           "inner_episode_detection": vals["episode_detection"],
                           "inner_fa_per_hour": vals["fa_episodes_per_hour"],
                           "inner_median_delay_s": vals["median_delay_s"]})
    ch = pd.DataFrame(chosen)
    ch.to_csv(OUT / "selected.csv", index=False)
    print("\nRules chosen on the source vehicles:")
    print("  " + ch.to_string(index=False).replace("\n", "\n  "))


# ----------------------------------------------------------------------
# step 3: the chosen rules on the held-out vehicle, all seeds
# ----------------------------------------------------------------------

def parse_rule(name):
    if name.startswith("cusum"):
        _, k, h = name.split()
        return "cusum", (float(k[1:]), float(h[1:]))
    k, n = name.split("-of-")
    return "kofn", (int(k), int(n))


def run():
    sel = pd.read_csv(OUT / "selected.csv")
    _, _, _, evalset = load_all()
    rows, att = [], []
    for held_out in VEHICLES:
        meta = meta_of(evalset, held_out)
        recs = prepare_recordings(meta)
        for seed in SEEDS:
            for arm in ARMS:
                f = D1_SCORES / f"{arm}_{held_out}_s{seed}.npz"
                if not f.exists():
                    raise SystemExit(f"{f} missing: run direction1_averaging.py run first.")
                z = load_scores(f, meta)
                for thr in THRESHOLDS:
                    tp, tl = float(z[f"thr_{thr}"]), float(z[f"thr_{thr}_logit"])
                    G = float(z["class_gap_logit"])
                    exceed = z["target_score"] >= tp
                    todo = [("single window", "kofn", (1, 1), np.nan, None)]
                    s = sel[(sel["held_out"] == held_out) & (sel["arm"] == arm) &
                            (sel["threshold"] == thr)]
                    for r in s.itertuples():
                        fam, p = parse_rule(r.rule)
                        todo.append((f"{r.family}, limit {r.limit_per_hour:g}/h", fam, p,
                                     r.limit_per_hour, r.limit_met))
                    for label, fam, p, limit, met in todo:
                        if fam == "kofn":
                            fn = lambda rc, p=p: kofn_alarms(exceed[rc["idx"]], *p)
                        else:
                            fn = lambda rc, p=p: cusum_alarms(
                                z["target_logit"][rc["idx"]], tl, p[0], p[1], G)
                        res = count(recs, fn)
                        for gap, rr in res.items():
                            base = {"held_out": held_out, "seed": seed, "arm": arm,
                                    "threshold": thr, "choice": label, "family": fam,
                                    "rule": rule_name(fam, p), "limit_per_hour": limit,
                                    "limit_met_inner": met, "episode_gap_s": gap}
                            rows.append({**base, **{k: v for k, v in rr.items()
                                                    if k != "per_attack"}})
                            for atk, (n_ep, n_det) in rr["per_attack"].items():
                                att.append({**base, "attack_type": atk,
                                            "episodes": n_ep, "detected": n_det})
            print(f"  {held_out:<10} seed {seed} done")
        pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)
        pd.DataFrame(att).to_csv(OUT / "per_attack.csv", index=False)
    check()
    report()


def check():
    """1-of-1 at zs: the share of benign windows flagged must equal fpr_zs."""
    if QUICK:
        print("\nCHECK skipped in QUICK mode (the saved thesis results use all windows).")
        return
    _, _, _, evalset = load_all()
    saved = pd.read_csv(RESULTS / "mlp_results.csv")
    saved["selected"] = saved["selected"].astype(str).str.lower() == "true"
    saved = saved[saved["selected"] & (saved["setting"] == "cross")]
    rows = []
    for v in VEHICLES:
        meta = meta_of(evalset, v)
        ben = meta["label"].to_numpy() == 0
        for seed in SEEDS:
            for arm in ARMS:
                z = load_scores(D1_SCORES / f"{arm}_{v}_s{seed}.npz", meta)
                for thr in THRESHOLDS:
                    mine = float(np.mean(z["target_score"][ben] >= float(z[f"thr_{thr}"])))
                    s = saved[(saved["arm"] == arm) & (saved["held_out"] == v) &
                              (saved["seed"] == seed)]
                    rows.append({"held_out": v, "seed": seed, "arm": arm,
                                 "threshold": thr, "fpr_here": mine,
                                 "fpr_saved": float(s[f"fpr_{thr}"].iloc[0]) if len(s) else np.nan})
    c = pd.DataFrame(rows)
    c["abs_diff"] = (c["fpr_here"] - c["fpr_saved"]).abs()
    c.to_csv(OUT / "check.csv", index=False)
    ok = c["abs_diff"].notna().all() and c["abs_diff"].max() <= 1e-6
    print(f"\nCHECK: single-window false-alarm rate against mlp_results.csv, "
          f"{len(c)} runs, largest difference {c['abs_diff'].max():.2e}")
    print("  OK: the base detectors are the thesis detectors." if ok else
          "  WARNING: not reproduced. Send this log before using any table.")


def report():
    runs = pd.read_csv(OUT / "runs.csv")
    keys = ["episode_gap_s", "arm", "threshold", "choice"]
    num = ["episode_detection", "median_delay_s", "p90_delay_s",
           "fa_episodes_per_hour", "attack_episodes", "clean_hours"]
    per_v = runs.groupby(keys + ["held_out"], sort=False)[num].mean()
    summ = per_v.groupby(level=list(range(len(keys))), sort=False).mean()
    per_v.reset_index().to_csv(OUT / "per_vehicle.csv", index=False)
    summ.to_csv(OUT / "summary.csv")
    print("\n" + "=" * 100)
    print(f"DIRECTION 2   mean over 4 held-out vehicles ({runs['seed'].nunique()} seeds "
          f"averaged within each vehicle first)")
    print("=" * 100)
    for gap in GAPS_S:
        print(f"\nAttack episodes merged over gaps up to {gap:g} s"
              + ("  (main)" if gap == SELECT_GAP_S else "  (sensitivity)"))
        print(f"  {'arm':<8}{'thr':<5}{'choice':<26}{'detected':>10}{'med delay':>11}"
              f"{'p90 delay':>11}{'FA/hour':>10}")
        for (g, arm, thr, choice), r in summ.iterrows():
            if g != gap:
                continue
            print(f"  {arm:<8}{thr:<5}{choice:<26}{r['episode_detection']:>10.3f}"
                  f"{r['median_delay_s']:>11.2f}{r['p90_delay_s']:>11.2f}"
                  f"{r['fa_episodes_per_hour']:>10.2f}")
    sel = pd.read_csv(OUT / "selected.csv")
    print("\nRules chosen on the source vehicles (per held-out vehicle):")
    for (arm, thr, fam, lim), g in sel.groupby(["arm", "threshold", "family",
                                                "limit_per_hour"], sort=False):
        print(f"  {arm:<7} {thr:<4} {fam:<6} limit {lim:>4g}/h: " + ", ".join(
            f"{r.held_out} {r.rule}{'' if r.limit_met else ' (limit not met)'}"
            for r in g.itertuples()))
    print(f"\n  Per vehicle, per attack type and every run in {OUT}/")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    {"inner": inner, "select": select, "run": run, "check": check,
     "report": report}.get(what, lambda: print(__doc__))()
