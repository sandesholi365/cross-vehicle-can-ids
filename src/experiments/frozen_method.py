"""
Freeze the complete method before any external dataset is used (supervisor's
email of 30 October: "freeze the complete methodology and save/document the
exact configuration", then CrySyS once, then ROAD).

The complete method = the thesis method + Direction 1 + Direction 2:
  thesis       Random Forest and MLP (single, pooled, FedAvg, FedProx),
               feature set, false-alarm budget and FedProx mu chosen by the
               inner leave-one-source-out loop, thresholds zs / zsb / cal
  Direction 1  one-shot weight average of the vehicle models, the logit and
               probability ensembles, the interpolation pooled -> averaged at
               an alpha chosen on the source vehicles, and the random-
               partition control (rand_avg_weights, rand_fedavg)
  Direction 2  k-of-n and CUSUM alarm rules chosen on the source vehicles
               (limits 1 and 10 false-alarm episodes per driving hour), for
               the pooled and FedAvg MLPs at zs and cal

For an external dataset the final models are trained on all four
CAN-Train-and-Test vehicles, so everything that has to be chosen is chosen by
the same inner loop run over all four (each in turn the pretend-new vehicle,
trained on the other three). This is exactly what run_road.py did for the
thesis; nothing here looks at CrySyS or ROAD.

Commands, from the project root, in this order:
    python -u src/experiments/frozen_method.py select   (CTT only; about 1-2 h,
                                                         continues if stopped)
    python -u src/experiments/frozen_method.py freeze   (seconds; writes the
                                                         frozen configuration)
    python -u src/experiments/frozen_method.py verify crysys   (or road)
    python -u src/experiments/frozen_method.py show     (prints the frozen file)

"freeze" needs the CrySyS windows (build_crysys_windows.py) to exist, so the
exact data is fixed too. It refuses to overwrite an existing freeze once an
external run has started.

Outputs in results/frozen/:
    rf_selection_log.csv, rf_selected.csv       RF feature set and budget
    mlp_selection_log.csv, mlp_selected.csv     MLP feature set, budget, mu
    alpha_selection_log.csv, alpha_selected.csv Direction 1 alpha
    inner_scores/, d2_inner_log.csv, d2_selected.csv   Direction 2 rules
    frozen_config.json    every setting, every choice, the CrySyS rules, and
                          the MD5 of every script and input file
    FROZEN_METHOD.txt     the same in plain words
"""

import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(HERE))
import common as cm  # noqa: E402
from common import (BUDGETS, QUICK, RESULTS, VEHICLES, best_threshold,  # noqa: E402
                    budget_threshold, feature_columns, load_all, scored)

FROZEN = RESULTS / ("frozen_quick" if QUICK else "frozen")
INNER = FROZEN / "inner_scores"
EXTERNAL = "external"           # held_out name: not a CTT vehicle -> all four are sources
FEATURES = Path("data/processed/features")
TARGET_FILES = {"crysys": FEATURES / "crysys_w100.csv", "road": FEATURES / "road_w100.csv"}
CRYSYS_LABELS = Path("data/processed/crysys_labels")

# scripts whose content decides any number of the external runs
CODE = ["experiments/common.py", "experiments/run_baselines.py",
        "experiments/run_federated.py", "experiments/explain_arms.py",
        "experiments/explain_avg.py", "experiments/direction1_averaging.py",
        "experiments/direction1_control.py", "experiments/direction2_alarms.py",
        "experiments/feasibility_alarms.py", "experiments/frozen_method.py",
        "experiments/run_external.py", "experiments/run_road.py",
        "features/build_windows.py", "features/build_road_windows.py",
        "features/build_crysys_windows.py", "data/crysys_labels.py"]


def md5_file(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def md5_folder(folder, pattern="*"):
    """One MD5 over the sorted file names and their MD5s."""
    h = hashlib.md5()
    files = sorted(p for p in Path(folder).glob(pattern) if p.is_file())
    for p in files:
        h.update(p.name.encode())
        h.update(md5_file(p).encode())
    return h.hexdigest(), len(files)


# ----------------------------------------------------------------------
# select: everything that is chosen, on CAN-Train-and-Test only
# ----------------------------------------------------------------------

def select():
    import torch  # noqa: F401
    import run_baselines as rb
    import run_federated as rfd
    import direction1_averaging as d1
    import direction2_alarms as d2
    FROZEN.mkdir(parents=True, exist_ok=True)
    INNER.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    data, parts, clean, evalset = load_all()

    # 1. RF and MLP: feature set, budget, mu (as run_road.py did)
    if not (FROZEN / "rf_selected.csv").exists():
        print("\n=== RF selection on the four CTT vehicles ===")
        _, log, chosen = rb.select(EXTERNAL, data, parts, clean, evalset)
        log.to_csv(FROZEN / "rf_selection_log.csv", index=False)
        pd.DataFrame(chosen).to_csv(FROZEN / "rf_selected.csv", index=False)
    if not (FROZEN / "mlp_selected.csv").exists():
        print("\n=== MLP selection on the four CTT vehicles ===")
        _, log, chosen = rfd.select(EXTERNAL, data, parts, clean, evalset)
        log.to_csv(FROZEN / "mlp_selection_log.csv", index=False)
        pd.DataFrame(chosen).to_csv(FROZEN / "mlp_selected.csv", index=False)
    sel = mlp_choice()
    cols = feature_columns(data[VEHICLES[0]], sel["feature_set"])
    nf = len(cols)
    budget = BUDGETS[sel["budget"]]
    everything = rfd.all_columns(data)
    print(f"\n  MLP: features {sel['feature_set']}, budget {sel['budget']}, mu {sel['mu']}")

    # 2. Direction 1 alpha and Direction 2 inner scores, same inner models
    alog_f = FROZEN / "alpha_selection_log.csv"
    alog = pd.read_csv(alog_f).to_dict("records") if alog_f.exists() else []
    done = {(r["pretend_target"], int(r["seed"])) for r in alog}
    print("\n=== Direction 1 alpha and Direction 2 inner models ===")
    for pretend in VEHICLES:
        inner = [v for v in VEHICLES if v != pretend]
        sc = rfd.Scaler(pd.concat([parts[v][0] for v in inner]), everything)
        calib = pd.concat([parts[v][1] for v in inner], ignore_index=True)
        src_test = scored(pd.concat([parts[v][2] for v in inner], ignore_index=True))
        clients = [(sc.X(parts[v][0], cols), rfd.labels(parts[v][0])) for v in inner]
        Xcal, Xsrc = sc.X(calib, cols), sc.X(src_test, cols)
        Xt, Xclean = sc.X(evalset[pretend], cols), sc.X(clean[pretend], cols)
        for sd in rfd.SELECTION_SEEDS:
            if (pretend, sd) in done and all(
                    (INNER / f"{a}_{pretend}_s{sd}.npz").exists() for a in d2.ARMS):
                continue
            t0 = time.time()
            pooled, _, _, avg_state = d1.make_models(clients, nf, sd)
            rows = []
            for a in d1.ALPHAS:
                model = d1.interp(nf, pooled, avg_state, a)
                r = rfd.inner_eval(model, sc, cols, calib, clean[pretend], evalset[pretend])
                rows.append({"held_out": EXTERNAL, "pretend_target": pretend,
                             "inner_sources": "+".join(inner), "seed": sd, "alpha": a, **r})
            alog = [r for r in alog if not (r["pretend_target"] == pretend
                                            and int(r["seed"]) == sd)] + rows
            pd.DataFrame(alog).to_csv(alog_f, index=False)
            models = {"pooled": pooled, "fedavg": rfd.train_arm("fedavg", clients, nf, sd)[0]}
            import explain_arms as ea
            for arm, model in models.items():
                p_cal, p_t, p_clean = (rfd.predict(model, X) for X in (Xcal, Xt, Xclean))
                z_cal, z_src, z_t, z_clean = (ea.logits(model, X)
                                              for X in (Xcal, Xsrc, Xt, Xclean))
                y_src = src_test["label"].to_numpy()
                m0 = float(np.median(z_src[y_src == 0]))
                G = float(np.median(z_src[y_src == 1]) - m0)
                np.savez_compressed(
                    INNER / f"{arm}_{pretend}_s{sd}.npz",
                    target_score=p_t.astype(np.float32), target_logit=z_t.astype(np.float32),
                    target_label=evalset[pretend]["label"].to_numpy(np.int8),
                    thr_zs=best_threshold(calib["label"].to_numpy(), p_cal)[0],
                    thr_cal=budget_threshold(p_clean, budget),
                    thr_zs_logit=best_threshold(calib["label"].to_numpy(), z_cal)[0],
                    thr_cal_logit=budget_threshold(z_clean, budget),
                    class_gap_logit=G)
            print(f"  {'+'.join(inner):<28} -> {pretend:<10} seed {sd} "
                  f"{time.time() - t0:>4.0f}s  inner zs MCC by alpha: "
                  + " ".join(f"{r['mcc_zs']:.2f}" for r in rows))

    lg = pd.DataFrame(alog)
    means = lg.groupby("alpha")["mcc_zs"].mean()
    best = float(means.max())
    a_sel = float(min(a for a in d1.ALPHAS if means[a] >= best - d1.ALPHA_TIE))
    pd.DataFrame([{"held_out": EXTERNAL, "alpha": a_sel,
                   "criterion": f"mean inner zero-shot MCC; ties within {d1.ALPHA_TIE} "
                                "to the smallest alpha",
                   "best_mean_inner_mcc_zs": best,
                   "chosen_mean_inner_mcc_zs": float(means[a_sel]),
                   "scores": "; ".join(f"{a} {means[a]:.4f}" for a in d1.ALPHAS),
                   "data_used": "inner leave-one-source-out on all four CTT vehicles"}]
                 ).to_csv(FROZEN / "alpha_selected.csv", index=False)
    print(f"\n  Direction 1: alpha {a_sel} (mean inner zs MCC {means[a_sel]:.3f})")

    # 3. Direction 2 rules on the inner scores
    rows = []
    for pretend in VEHICLES:
        meta = d2.meta_of(evalset, pretend)
        recs = d2.prepare_recordings(meta)
        for sd in rfd.SELECTION_SEEDS:
            for arm in d2.ARMS:
                z = d2.load_scores(INNER / f"{arm}_{pretend}_s{sd}.npz", meta)
                for thr in d2.THRESHOLDS:
                    res = d2.all_rules(recs, z["target_score"], z["target_logit"],
                                       float(z[f"thr_{thr}"]), float(z[f"thr_{thr}_logit"]),
                                       float(z["class_gap_logit"]), gaps=[d2.SELECT_GAP_S])
                    for fam, p, r in res:
                        rr = r[d2.SELECT_GAP_S]
                        rows.append({"held_out": EXTERNAL, "pretend_target": pretend,
                                     "seed": sd, "arm": arm, "threshold": thr,
                                     "family": fam, "rule": d2.rule_name(fam, p),
                                     **{k: v for k, v in rr.items() if k != "per_attack"}})
        print(f"  Direction 2 inner rules scored for {pretend}")
    log = pd.DataFrame(rows)
    log.to_csv(FROZEN / "d2_inner_log.csv", index=False)
    chosen = []
    for (arm, thr, fam), g in log.groupby(["arm", "threshold", "family"], sort=False):
        for limit in d2.LIMITS:
            best_rule, met, vals = d2.choose(g, limit)
            chosen.append({"held_out": EXTERNAL, "arm": arm, "threshold": thr,
                           "family": fam, "limit_per_hour": limit, "rule": best_rule,
                           "limit_met": met,
                           "inner_episode_detection": vals["episode_detection"],
                           "inner_fa_per_hour": vals["fa_episodes_per_hour"],
                           "inner_median_delay_s": vals["median_delay_s"]})
    ch = pd.DataFrame(chosen)
    ch.to_csv(FROZEN / "d2_selected.csv", index=False)
    print("\n  Direction 2 rules chosen on CTT:")
    print("  " + ch[["arm", "threshold", "family", "limit_per_hour", "rule",
                     "limit_met"]].to_string(index=False).replace("\n", "\n  "))

    # 4. the thesis ROAD run made the same RF/MLP choices?
    road = RESULTS / "road"
    for fam in ("rf", "mlp"):
        f = road / f"{fam}_selected.csv"
        if f.exists() and not QUICK:
            old = pd.read_csv(f)[["parameter", "chosen"]].astype(str)
            new = pd.read_csv(FROZEN / f"{fam}_selected.csv")[["parameter", "chosen"]].astype(str)
            same = old.reset_index(drop=True).equals(new.reset_index(drop=True))
            print(f"  {fam.upper()} choices equal to the thesis ROAD run: "
                  f"{'yes' if same else 'NO - send this log'}")
    print(f"\n  select done in {(time.time() - t_start) / 60:.0f} min; next: freeze")


def mlp_choice():
    m = pd.read_csv(FROZEN / "mlp_selected.csv").set_index("parameter")["chosen"]
    return {"feature_set": str(m["feature_set"]), "budget": str(m["budget"]),
            "mu": float(m["fedprox_mu"])}


def rf_choice():
    m = pd.read_csv(FROZEN / "rf_selected.csv").set_index("parameter")["chosen"]
    return {"feature_set": str(m["feature_set"]), "budget": str(m["budget"])}


def alpha_choice():
    return float(pd.read_csv(FROZEN / "alpha_selected.csv")["alpha"].iloc[0])


def d2_choice():
    return pd.read_csv(FROZEN / "d2_selected.csv")


# ----------------------------------------------------------------------
# freeze
# ----------------------------------------------------------------------

def settings():
    import run_baselines as rb
    import run_federated as rfd
    import explain_avg as eav
    import direction1_averaging as d1
    import direction1_control as d1c
    import direction2_alarms as d2
    sys.path.insert(0, str(SRC / "data"))
    import crysys_labels as cl
    return {
        "data": {"window_messages": cm.WINDOW, "split_seed": cm.SPLIT_SEED,
                 "calibration_fraction": cm.CALIB_FRACTION,
                 "threshold_candidates": cm.N_CANDIDATES,
                 "threshold_tie_tolerance": cm.THR_TOLERANCE,
                 "budgets": cm.BUDGETS, "feature_sets": cm.FEATURE_SETS,
                 "stat_scale_free": cm.STAT_SCALE_FREE, "ctt_vehicles": VEHICLES},
        "random_forest": {"trees": rb.N_TREES, "min_samples_leaf": rb.MIN_SAMPLES_LEAF,
                          "max_train_windows": rb.MAX_TRAIN, "seed": rb.RF_SEED,
                          "pooled_modes": rb.POOLED_MODES},
        "mlp": {"hidden": list(rfd.HIDDEN), "norm": "LayerNorm (no BatchNorm)",
                "rounds": rfd.ROUNDS, "local_epochs": rfd.LOCAL_EPOCHS,
                "central_epochs": rfd.CENTRAL_EPOCHS, "batch": rfd.BATCH, "lr": rfd.LR,
                "weight_decay": rfd.WEIGHT_DECAY, "grad_clip": rfd.GRAD_CLIP,
                "pos_weight_cap": rfd.POS_WEIGHT_CAP, "mu_values": rfd.MU_VALUES,
                "evaluation_seeds": rfd.SEEDS, "selection_seeds": rfd.SELECTION_SEEDS,
                "scaler": "median/IQR on the source training windows, then arcsinh"},
        "direction1": {"settings": ["pooled", "fedavg", "avg_weights", "ens_logit",
                                    "ens_prob", "interp_sel", "rand_avg_weights",
                                    "rand_fedavg"],
                       "member_epochs": eav.LOCAL_EPOCHS,
                       "same_start": "every model of a seed starts from new_model(n, seed)",
                       "alpha_grid": d1.ALPHAS, "alpha_tie": d1.ALPHA_TIE,
                       "ece_bins": d1.ECE_BINS, "benign_quantiles": d1.QUANTILES,
                       "random_partition_seed": f"{d1c.PARTITION_OFFSET} + seed",
                       "random_partition": "matched to the vehicle clients in windows "
                                           "and attack windows; every window once"},
        "direction2": {"arms": d2.ARMS, "thresholds": d2.THRESHOLDS,
                       "kofn_n": sorted({n for _, n in d2.KOFN}), "cusum_kappa": d2.KAPPAS,
                       "cusum_h": d2.HS, "limits_per_hour": d2.LIMITS,
                       "episode_gaps_s": d2.GAPS_S, "selection_gap_s": d2.SELECT_GAP_S,
                       "clear_windows": d2.CLEAR, "tie": d2.TIE},
        "crysys_labels": {"window": cl.WINDOW, "tol_s": cl.TOL, "late_frac": cl.LATE_FRAC,
                          "late_cap_s": cl.LATE_CAP, "inj_search_s": cl.INJ_SEARCH_S,
                          "inj_run": cl.INJ_RUN, "max_loss": cl.MAX_LOSS,
                          "fast_period_s": cl.FAST_PERIOD, "shift_run": cl.SHIFT_RUN,
                          "nontarget_window_s": cl.NT_WINDOW, "offset_try": cl.OFFSET_TRY,
                          "inj_window_s": cl.INJ_WINDOW, "offset_keep": cl.OFFSET_KEEP,
                          "offset_sample": cl.OFFSET_SAMPLE, "offset_same_s": cl.OFFSET_SAME,
                          "marker_tol_s": cl.MARKER_TOL,
                          "marker_search_before_s": cl.MARKER_SEARCH_BEFORE,
                          "marker_clip_s": cl.MARKER_CLIP, "f5_tol_periods": cl.F5_TOL_PERIODS},
    }


RULES_TEXT = [
    "Nothing is chosen, tuned or changed on CrySyS or ROAD. Every choice was made on "
    "CAN-Train-and-Test (CTT) by the inner leave-one-source-out loop over all four vehicles.",
    "Final models are trained on all four CTT vehicles (fit windows); the scaler is fitted "
    "on those windows only. MLPs use the ten evaluation seeds 0-9; the RF uses its one seed.",
    "Thresholds: zs = MCC-optimal on the CTT calibration windows; zsb = the selected budget "
    "B on CTT held-out attack-free windows; cal = budget B on the external dataset's "
    "attack-free traffic (CrySyS: the 26 benign traces; ROAD: the ambient captures); "
    "oracle = upper bound only.",
    "CrySyS only, reported next to cal and decided now: cal_loso = budget B on the benign "
    "traces of the OTHER scenarios, so the threshold never comes from the very trace that "
    "the attacked trace replays (every CrySyS attacked trace replays its scenario's benign "
    "trace, so plain cal is calibrated on the same normal traffic it is scored on).",
    "CrySyS labels: only traces that passed every check of crysys_labels.py (label_report."
    "csv, passed = True); frame labels from labels/<trace>.npy, never the JSON markers; "
    "window = attack if it holds at least one attack frame; uncertain windows are left out "
    "of every window-level result and count as attack windows for alarm episodes.",
    "Scored sets are evaluated separately: CrySyS fabrication and masquerade; ROAD "
    "fabrication and masquerade (as in the thesis). Attack-free recordings are never scored.",
    "Direction 1 settings: avg_weights, ens_logit, ens_prob (three vehicle members, 20 "
    "epochs, same start), interp_sel at the frozen alpha, rand_avg_weights and rand_fedavg "
    "(matched random partition of the four vehicles' fit windows), next to pooled and FedAvg.",
    "Direction 2: the frozen rule of each arm (pooled, FedAvg), threshold (zs, cal), family "
    "(k-of-n, CUSUM) and limit (1, 10 per hour) is applied unchanged; episodes merged over "
    "1 s (5 s as sensitivity); false alarm = alarm on a benign window at least 10 windows "
    "from an attack episode whose evidence holds no attack window.",
    "Each external dataset is run once. A finished run writes COMPLETED and is never "
    "repeated; an interrupted run continues where it stopped, with the same frozen files.",
]


def freeze():
    if QUICK:
        print("  QUICK mode: the freeze written here is for testing only.")
    f = FROZEN / "frozen_config.json"
    started = [t for t in TARGET_FILES
               if (RESULTS / (f"{'external_quick' if QUICK else 'external'}_{t}")).exists()]
    if f.exists() and started:
        raise SystemExit(f"An external run has already started ({', '.join(started)}): "
                         "the freeze cannot be changed any more.")
    if f.exists() and "--replace" not in sys.argv:
        raise SystemExit(f"{f} exists. Use 'freeze --replace' only if nothing has run yet.")
    for need in ("rf_selected.csv", "mlp_selected.csv", "alpha_selected.csv",
                 "d2_selected.csv"):
        if not (FROZEN / need).exists():
            raise SystemExit(f"{FROZEN / need} missing: run 'select' first.")
    if not TARGET_FILES["crysys"].exists():
        raise SystemExit(f"{TARGET_FILES['crysys']} missing: run build_crysys_windows.py "
                         "first, so the exact CrySyS data is frozen too.")

    print("  hashing scripts and input files (a few minutes for the large files) ...")
    code = {}
    for rel in CODE:
        p = SRC / rel
        code[rel] = md5_file(p) if p.exists() else "MISSING"
    missing = [k for k, v in code.items() if v == "MISSING"]
    if missing:
        raise SystemExit(f"scripts missing: {missing}")
    inputs = {str(FEATURES / f"{v}_w100.csv"): md5_file(FEATURES / f"{v}_w100.csv")
              for v in VEHICLES}
    for t, p in TARGET_FILES.items():
        if p.exists():
            inputs[str(p)] = md5_file(p)
    labels_md5, n_labels = md5_folder(CRYSYS_LABELS / "labels", "*.npy")
    rep = pd.read_csv(CRYSYS_LABELS / "label_report.csv")
    att = rep[rep["kind"].isin(["fabrication", "masquerade"])]
    passed = att[att["passed"].astype(str).str.lower() == "true"]
    selections = {n: md5_file(FROZEN / n) for n in
                  ("rf_selected.csv", "mlp_selected.csv", "alpha_selected.csv",
                   "d2_selected.csv", "rf_selection_log.csv", "mlp_selection_log.csv",
                   "alpha_selection_log.csv", "d2_inner_log.csv")
                  if (FROZEN / n).exists()}
    cfg = {
        "frozen_on": date.today().isoformat(),
        "quick_mode": QUICK,
        "rules": RULES_TEXT,
        "settings": settings(),
        "choices": {"rf": rf_choice(), "mlp": mlp_choice(), "alpha": alpha_choice(),
                    "d2_rules": d2_choice()[["arm", "threshold", "family",
                                             "limit_per_hour", "rule", "limit_met"]]
                    .to_dict("records")},
        "crysys": {"label_report_md5": md5_file(CRYSYS_LABELS / "label_report.csv"),
                   "label_files_md5": labels_md5, "label_files": n_labels,
                   "traces_passed": passed["kind"].value_counts().to_dict(),
                   "traces_failed": att[~att.index.isin(passed.index)]["kind"]
                   .value_counts().to_dict()},
        "md5": {"code": code, "inputs": inputs, "selections": selections},
    }
    text = json.dumps(cfg, indent=2, sort_keys=True, default=str)
    cfg_id = hashlib.md5(text.encode()).hexdigest()
    FROZEN.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    (FROZEN / "freeze_id.txt").write_text(cfg_id + "\n")
    write_text_summary(cfg, cfg_id)
    print(f"  frozen: {f}\n  freeze id {cfg_id}\n  plain-words copy: "
          f"{FROZEN / 'FROZEN_METHOD.txt'}")


def write_text_summary(cfg, cfg_id):
    c = cfg["choices"]
    lines = [f"FROZEN METHOD   (frozen on {cfg['frozen_on']}, freeze id {cfg_id})", "",
             "Rules:"]
    lines += [f"  - {r}" for r in cfg["rules"]]
    lines += ["", "Choices made on CAN-Train-and-Test only:",
              f"  RF:  feature set {c['rf']['feature_set']}, budget {c['rf']['budget']}",
              f"  MLP: feature set {c['mlp']['feature_set']}, budget {c['mlp']['budget']}, "
              f"FedProx mu {c['mlp']['mu']}",
              f"  Direction 1: alpha {c['alpha']}", "  Direction 2 rules:"]
    lines += [f"    {r['arm']:<7} {r['threshold']:<4} {r['family']:<6} limit "
              f"{r['limit_per_hour']:>4}/h: {r['rule']}"
              + ("" if r["limit_met"] else "  (limit not met on CTT)")
              for r in c["d2_rules"]]
    cr = cfg["crysys"]
    lines += ["", f"CrySyS traces used: {cr['traces_passed']}; left out: "
              f"{cr['traces_failed']}", "",
              "Settings:", json.dumps(cfg["settings"], indent=2, default=str), "",
              "MD5 of every script and input file: see frozen_config.json"]
    (FROZEN / "FROZEN_METHOD.txt").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# verify
# ----------------------------------------------------------------------

def verify(target, quiet=False):
    f = FROZEN / "frozen_config.json"
    if not f.exists():
        raise SystemExit(f"{f} not found: run 'select' and 'freeze' first.")
    text = f.read_text()
    cfg = json.loads(text)
    problems = []
    if hashlib.md5(text.encode()).hexdigest() != (FROZEN / "freeze_id.txt").read_text().strip():
        problems.append("frozen_config.json was edited after the freeze")
    for rel, h in cfg["md5"]["code"].items():
        p = SRC / rel
        if not p.exists() or md5_file(p) != h:
            problems.append(f"script changed: {rel}")
    for name, h in cfg["md5"]["selections"].items():
        if md5_file(FROZEN / name) != h:
            problems.append(f"frozen choice file changed: {name}")
    for v in VEHICLES:
        p = FEATURES / f"{v}_w100.csv"
        if md5_file(p) != cfg["md5"]["inputs"].get(str(p)):
            problems.append(f"CTT windows changed: {p}")
    tp = TARGET_FILES[target]
    if str(tp) not in cfg["md5"]["inputs"]:
        problems.append(f"{tp} was not frozen (build it before 'freeze')")
    elif not tp.exists() or md5_file(tp) != cfg["md5"]["inputs"][str(tp)]:
        problems.append(f"{target} windows changed or missing: {tp}")
    if bool(cfg.get("quick_mode")) != QUICK:
        problems.append("freeze and run differ in QUICK mode")
    if problems:
        print("VERIFY FAILED - nothing may run until this is explained:")
        for p in problems:
            print(f"  - {p}")
        return False, cfg
    if not quiet:
        print(f"VERIFY OK: scripts, choices and data are exactly as frozen on "
              f"{cfg['frozen_on']} (freeze id {(FROZEN / 'freeze_id.txt').read_text().strip()})")
    return True, cfg


def show():
    print((FROZEN / "FROZEN_METHOD.txt").read_text())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "verify":
        ok, _ = verify(sys.argv[2] if len(sys.argv) > 2 else "crysys")
        sys.exit(0 if ok else 1)
    {"select": select, "freeze": freeze, "show": show}.get(what, lambda: print(__doc__))()
