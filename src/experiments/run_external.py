"""
External validation with the FROZEN complete method (supervisor's email of
30 October): CrySyS once, as the new untouched external dataset; ROAD once
more afterwards, as a secondary check.

Nothing is chosen here. Every choice is read from results/frozen/ (made on
CAN-Train-and-Test only by frozen_method.py), and the run stops unless
"frozen_method.py verify" passes: same scripts, same choices, same data, MD5
for MD5. A finished run writes COMPLETED and refuses to run again; an
interrupted run continues where it stopped.

What is run (the complete method, see results/frozen/FROZEN_METHOD.txt)
  1. thesis models, exactly as run_road.py: Random Forest (single, pooled,
     pooled balanced, pooled matched; every feature set, the frozen one
     marked) and MLP (single, pooled, FedAvg, FedProx at every mu, the frozen
     one marked), trained on all four CTT vehicles; seeds 0-9 for the MLP
  2. Direction 1 settings on the same seeds: avg_weights, ens_logit,
     ens_prob, interp_sel (frozen alpha), rand_avg_weights, rand_fedavg, next
     to pooled and FedAvg, with the Direction 1 measurements (spread,
     headroom, benign quantiles, Brier, ECE, ...)
  3. Direction 2: the frozen alarm rules on the pooled and FedAvg scores

Thresholds zs, zsb, cal and oracle as in the thesis. For CrySyS also cal_loso
(budget B on the benign traces of the other scenarios), decided in the freeze.
CrySyS uncertain windows are left out of window-level results and count as
attack windows for alarm episodes. Scored sets are evaluated separately
(fabrication, masquerade).

Run from the project root:
    QUICK=1 python -u src/experiments/run_external.py run crysys   (pipeline
                                                  check, with a QUICK freeze)
    python -u src/experiments/run_external.py run crysys    (the real run, once)
    python -u src/experiments/run_external.py report crysys
and, only after CrySyS is analysed:
    python -u src/experiments/run_external.py run road
    python -u src/experiments/run_external.py report road   (includes a check
                                       against the thesis ROAD results)

Outputs in results/external_<target>/:
    rf_results.csv, rf_per_attack.csv     thesis RF rows
    mlp_results.csv, mlp_per_attack.csv   MLP rows (thesis arms and Direction 1
                                          settings, one row per set and seed)
    d1_runs.csv                           Direction 1 measurements
    d2_runs.csv, d2_per_attack.csv        Direction 2 alarms
    scores/, d2_scores/                   per-window scores
    check.csv (road only), COMPLETED, freeze_id.txt
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BUDGETS, FEATURE_SETS, QUICK, RESULTS, VEHICLES,  # noqa: E402
                    attack_types_trained_on, best_threshold, budget_threshold,
                    evaluate, feature_columns, heldout_benign, load_all,
                    load_vehicle, mcc_from_counts, per_attack_rows, scored,
                    threshold_rules)
import frozen_method as fm  # noqa: E402

TARGETS = ("crysys", "road")


def out_dir(target):
    return RESULTS / f"external{'_quick' if QUICK else ''}_{target}"


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------

def load_target(target):
    """attack-free windows, scored sets (window level), alarm sets (episodes)."""
    if target == "road":
        import run_road
        clean, sets = run_road.load_road()
        return clean, sets, sets
    df = load_vehicle("crysys")
    clean = df[df["attack_type"] == "attack-free"]
    att = scored(df)
    sets, alarm = {}, {}
    for k in ("fabrication", "masquerade"):
        s = att[att["crysys_set"] == k]
        if not len(s):
            continue
        sets[k] = s[s["uncertain"] == 0]
        a = s.copy()
        a.loc[a["uncertain"] == 1, "label"] = 1
        alarm[k] = a
    print(f"  crysys     attack-free {len(clean):,} windows ({clean['recording'].nunique()} "
          f"benign traces); scored: " + "; ".join(
              f"{k} {len(v):,} ({int(v['label'].sum()):,} attack, "
              f"{int((alarm[k]['uncertain'] == 1).sum())} uncertain left out)"
              for k, v in sets.items()))
    return clean, sets, alarm


def loso_eval(df, prob, benign_prob, benign_scen, budget):
    """cal_loso: per scenario, budget B on the benign traces of the OTHER scenarios."""
    scen = df["scenario"].to_numpy()
    thr = {}
    for s in np.unique(scen):
        other = benign_prob[benign_scen != s]
        thr[s] = budget_threshold(other, budget)
    t = np.array([thr[s] for s in scen])
    y = df["label"].to_numpy().astype(bool)
    pred = prob >= t
    tp = int(np.sum(y & pred))
    fp = int(np.sum(~y & pred))
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    fn, tn = n_pos - tp, n_neg - fp
    return {"mcc_cal_loso": mcc_from_counts(tp, tn, fp, fn),
            "recall_cal_loso": tp / n_pos if n_pos else np.nan,
            "fpr_cal_loso": fp / n_neg if n_neg else np.nan,
            "precision_cal_loso": tp / (tp + fp) if (tp + fp) else np.nan,
            "thr_cal_loso_median": float(np.median(list(thr.values())))}


def record(results, attack_rows, info, sets, probs, rules, seen, loso=None):
    rows = []
    for name, df in sets.items():
        row = {**info, "target_set": name}
        row.update(evaluate(df["label"].to_numpy(), probs[name], rules))
        if loso is not None:
            row.update(loso_eval(df, probs[name], *loso))
        results.append(row)
        keep = {t: rules[t] for t in ("zs", "zsb", "cal")}
        keep["oracle"] = row["thr_oracle"]
        attack_rows += per_attack_rows(df, probs[name], keep, seen,
                                       {**info, "target_set": name})
        rows.append(row)
    return rows


# ----------------------------------------------------------------------
# 1. Random Forest, as run_road.py, with the frozen choice
# ----------------------------------------------------------------------

def run_rf(target, OUT, data, parts, clean_t, sets):
    import run_baselines as rb
    MODE_NAMES = {"pooled_3": "pooled", "pooled_3_balanced": "pooled_balanced",
                  "pooled_3_matched": "pooled_matched"}
    if (OUT / "rf_done").exists():
        print("  RF already done")
        return
    sel = fm.rf_choice()
    b = sel["budget"]
    results, attack_rows = [], []
    src_calib = pd.concat([parts[v][1] for v in VEHICLES], ignore_index=True)
    src_test = scored(pd.concat([parts[v][2] for v in VEHICLES], ignore_index=True))
    benign_src = heldout_benign(parts, VEHICLES)
    for group in FEATURE_SETS:
        cols = feature_columns(data[VEHICLES[0]], group)
        is_sel = group == sel["feature_set"]
        print(f"\nRF feature set: {group}" + ("   <- frozen choice" if is_sel else ""))
        runs = [("single", v, parts[v][0], parts[v][1], heldout_benign(parts, [v]))
                for v in VEHICLES]
        for mode, train in rb.pooled_variants(parts, VEHICLES).items():
            runs.append((MODE_NAMES[mode], "+".join(VEHICLES), train, src_calib, benign_src))
        for mode, label, train, calib, benign in runs:
            t0 = time.time()
            model = rb.fit_model(train[cols], train["label"])

            def sc(df):
                return rb.score(model, df, cols)
            thr_zs, _ = best_threshold(calib["label"].to_numpy(), sc(calib))
            p_clean = sc(clean_t)
            rules = threshold_rules(thr_zs, sc(benign), p_clean, BUDGETS[b])
            probs = {k: sc(v) for k, v in sets.items()}
            info = {"family": "rf", "feature_set": group, "mode": mode,
                    "train_vehicles": label, "test_vehicle": target,
                    "setting": "external", "seed": rb.RF_SEED, "selected": is_sel,
                    "budget": b}
            loso = ((p_clean, clean_t["scenario"].to_numpy(), BUDGETS[b])
                    if target == "crysys" else None)
            rows = record(results, attack_rows, info, sets, probs, rules,
                          attack_types_trained_on(train), loso)
            print(f"  {mode:<16} {label:<34}{time.time() - t0:>4.0f}s  "
                  + "  ".join(f"{r['target_set'][:4]} zs {r['mcc_zs']:.3f} cal "
                              f"{r['mcc_cal']:.3f} AUROC {r['auroc']:.3f}" for r in rows))
            if mode == "pooled" and is_sel:
                p_src = sc(src_test)
                srow = {**info, "test_vehicle": "sources", "setting": "within_sources",
                        "target_set": ""}
                srow.update(evaluate(src_test["label"].to_numpy(), p_src,
                                     {"05": 0.5, "zs": thr_zs, "zsb": rules["zsb"]}))
                results.append(srow)
                for name, df in sets.items():
                    save_npz(OUT / "scores" / f"rf_pooled_{name}.npz",
                             target_score=probs[name].astype(np.float32),
                             target_label=df["label"].to_numpy(np.int8),
                             target_attack=df["attack_type"].to_numpy(str),
                             target_recording=df["recording"].to_numpy(str),
                             target_benign_score=p_clean.astype(np.float32),
                             thr_zs=thr_zs, thr_zsb=rules["zsb"], thr_cal=rules["cal"])
    pd.DataFrame(results).to_csv(OUT / "rf_results.csv", index=False)
    pd.DataFrame(attack_rows).to_csv(OUT / "rf_per_attack.csv", index=False)
    (OUT / "rf_done").write_text(date.today().isoformat() + "\n")


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


# ----------------------------------------------------------------------
# 2. MLP: thesis arms and Direction 1 settings, one seed at a time
# ----------------------------------------------------------------------

def run_mlp(target, OUT, data, parts, clean_t, sets, alarm_sets):
    import torch  # noqa: F401
    import run_federated as rfd
    import explain_arms as ea
    import explain_avg as eav
    import direction1_averaging as d1
    import direction1_control as d1c
    sel = fm.mlp_choice()
    alpha = fm.alpha_choice()
    group, b, mu_sel = sel["feature_set"], sel["budget"], sel["mu"]
    budget = BUDGETS[b]
    cols = feature_columns(data[VEHICLES[0]], group)
    nf = len(cols)
    sc_all = rfd.Scaler(pd.concat([parts[v][0] for v in VEHICLES]), rfd.all_columns(data))
    clients = [(sc_all.X(parts[v][0], cols), rfd.labels(parts[v][0])) for v in VEHICLES]
    sizes = [len(X) for X, _ in clients]
    src_calib = pd.concat([parts[v][1] for v in VEHICLES], ignore_index=True)
    src_test = scored(pd.concat([parts[v][2] for v in VEHICLES], ignore_index=True))
    benign_src = heldout_benign(parts, VEHICLES)
    X = {"calib": sc_all.X(src_calib, cols), "benign_src": sc_all.X(benign_src, cols),
         "src_test": sc_all.X(src_test, cols), "clean": sc_all.X(clean_t, cols)}
    X_sets = {k: sc_all.X(v, cols) for k, v in sets.items()}
    X_alarm = {k: sc_all.X(v, cols) for k, v in alarm_sets.items()}
    seen_all = attack_types_trained_on(pd.concat([parts[v][0] for v in VEHICLES]))
    label_all = "+".join(VEHICLES)
    loso_scen = clean_t["scenario"].to_numpy() if target == "crysys" else None
    d_base = {"calib": X["calib"], "y_calib": src_calib["label"].to_numpy(),
              "bsrc": X["benign_src"], "src": X["src_test"],
              "y_src": src_test["label"].to_numpy(), "clean": X["clean"]}

    f_res, f_att, f_d1 = (OUT / "mlp_results.csv", OUT / "mlp_per_attack.csv",
                          OUT / "d1_runs.csv")
    prog = OUT / "mlp_seeds_done.json"
    done = json.loads(prog.read_text()) if prog.exists() else []
    old = lambda f: (pd.read_csv(f).to_dict("records") if f.exists() else [])  # noqa: E731
    results = [r for r in old(f_res) if r.get("seed") in done]
    attack_rows = [r for r in old(f_att) if r.get("seed") in done]
    d1_rows = [r for r in old(f_d1) if r.get("seed") in done]
    if done:
        print(f"  continuing: MLP seeds already done {done}")
    print(f"\n  MLP: features {group}, budget {b}, mu {mu_sel}, alpha {alpha}")

    for seed in rfd.SEEDS:
        if seed in done:
            continue
        t0 = time.time()
        models = {}
        for v in VEHICLES:
            i = VEHICLES.index(v)
            models[f"single:{v}"] = rfd.train_local(rfd.new_model(nf, seed), *clients[i],
                                                    rfd.CENTRAL_EPOCHS, seed=seed)
        pooled = rfd.train_arm("pooled", clients, nf, seed)[0]
        models["pooled"] = pooled
        models["fedavg"] = rfd.train_arm("fedavg", clients, nf, seed)[0]
        for mu in rfd.MU_VALUES:
            models[f"fedprox_mu{mu}"] = rfd.train_arm("fedprox", clients, nf, seed, mu)[0]
        members = eav.train_members(clients, nf, seed)
        avg_state = rfd.average_weights([m.state_dict() for m in members], sizes)
        models["avg_weights"] = eav.from_state(nf, avg_state)
        models["ens_logit"] = eav.Ensemble(members, sizes, "logit")
        models["ens_prob"] = eav.Ensemble(members, sizes, "prob")
        models["interp_sel"] = d1.interp(nf, pooled, avg_state, alpha)
        rparts, _ = d1c.random_partition(clients, seed)
        rmembers = eav.train_members(rparts, nf, seed)
        models["rand_avg_weights"] = eav.from_state(nf, rfd.average_weights(
            [m.state_dict() for m in rmembers], sizes))
        models["rand_fedavg"] = rfd.train_arm("fedavg", rparts, nf, seed)[0]
        t_train = time.time() - t0

        for name, model in models.items():
            if name.startswith("single:"):
                v = name.split(":")[1]
                calib_v = parts[v][1]
                thr_zs, _ = best_threshold(calib_v["label"].to_numpy(),
                                           rfd.predict(model, sc_all.X(calib_v, cols)))
                p_bsrc = rfd.predict(model, sc_all.X(heldout_benign(parts, [v]), cols))
                seen = attack_types_trained_on(parts[v][0])
                arm, mode, label, mu = "single", "single", v, np.nan
            else:
                thr_zs, _ = best_threshold(src_calib["label"].to_numpy(),
                                           rfd.predict(model, X["calib"]))
                p_bsrc = rfd.predict(model, X["benign_src"])
                seen = seen_all
                mode, label = name, label_all
                arm = "fedprox" if name.startswith("fedprox") else name
                mu = float(name.split("mu")[1]) if arm == "fedprox" else np.nan
            p_clean = rfd.predict(model, X["clean"])
            rules = threshold_rules(thr_zs, p_bsrc, p_clean, budget)
            probs = {k: rfd.predict(model, x) for k, x in X_sets.items()}
            is_sel = arm != "fedprox" or mu == mu_sel
            info = {"family": "mlp", "feature_set": group, "mode": mode, "arm": arm,
                    "mu": mu, "train_vehicles": label, "test_vehicle": target,
                    "setting": "external", "seed": seed, "selected": is_sel, "budget": b,
                    "alpha": alpha if name == "interp_sel" else np.nan}
            loso = (p_clean, loso_scen, budget) if loso_scen is not None else None
            rows = record(results, attack_rows, info, sets, probs, rules, seen, loso)
            if seed == rfd.SEEDS[0] and name in ("pooled", "fedavg"):
                for nm, df in sets.items():
                    save_npz(OUT / "scores" / f"mlp_{name}_{nm}.npz",
                             target_score=probs[nm].astype(np.float32),
                             target_label=df["label"].to_numpy(np.int8),
                             target_attack=df["attack_type"].to_numpy(str),
                             target_recording=df["recording"].to_numpy(str),
                             target_benign_score=p_clean.astype(np.float32),
                             thr_zs=thr_zs, thr_zsb=rules["zsb"], thr_cal=rules["cal"])

            # Direction 1 measurements (the thesis single-vehicle models excluded)
            if name in d1c_settings():
                for nm in sets:
                    d = {**d_base, "t": X_sets[nm], "y_t": sets[nm]["label"].to_numpy()}
                    r, p, z, rr = d1.measure(model, d, budget)
                    if name in ("avg_weights", "ens_logit", "ens_prob", "rand_avg_weights"):
                        mem = members if name != "rand_avg_weights" else rmembers
                        dis_s = d1.member_disagreement(mem, d["bsrc"])
                        dis_t = d1.member_disagreement(mem, d["clean"])
                        r.update({"disagree_src_benign": dis_s, "disagree_tclean": dis_t,
                                  "disagree_src_benign_G": dis_s / r["class_gap_logit"],
                                  "disagree_tclean_G": dis_t / r["class_gap_logit"]})
                    d1_rows.append({"setting": name, "target_set": nm, "seed": seed,
                                    "alpha": alpha if name == "interp_sel" else np.nan,
                                    "budget": b, **r})
                    # Direction 2 inputs: pooled and FedAvg scores on the alarm sets
                    if name in ("pooled", "fedavg"):
                        xa = X_alarm[nm]
                        save_npz(OUT / "d2_scores" / f"{name}_{nm}_s{seed}.npz",
                                 target_score=rfd.predict(model, xa).astype(np.float32),
                                 target_logit=ea.logits(model, xa).astype(np.float32),
                                 target_label=alarm_sets[nm]["label"].to_numpy(np.int8),
                                 thr_zs=rr["zs"], thr_cal=rr["cal"],
                                 thr_zs_logit=r["thr_zs_logit"],
                                 thr_cal_logit=budget_threshold(z["clean"], budget),
                                 class_gap_logit=r["class_gap_logit"])
            if is_sel and seed == rfd.SEEDS[0] and not name.startswith("single:"):
                p_src = rfd.predict(model, X["src_test"])
                srow = {**info, "test_vehicle": "sources", "setting": "within_sources",
                        "target_set": ""}
                srow.update(evaluate(src_test["label"].to_numpy(), p_src,
                                     {"05": 0.5, "zs": thr_zs, "zsb": rules["zsb"]}))
                results.append(srow)

        pd.DataFrame(results).to_csv(f_res, index=False)
        pd.DataFrame(attack_rows).to_csv(f_att, index=False)
        pd.DataFrame(d1_rows).to_csv(f_d1, index=False)
        done.append(seed)
        prog.write_text(json.dumps(done))
        last = {(r["mode"], r["target_set"]): r for r in results
                if r.get("seed") == seed and r.get("setting") == "external"}
        print(f"  seed {seed}: trained in {t_train:.0f}s, total {time.time() - t0:.0f}s   zs MCC "
              + "  ".join(f"{m} " + "/".join(f"{last[(m, s)]['mcc_zs']:.2f}" for s in sets)
                          for m in ("pooled", "fedavg", "avg_weights", "interp_sel",
                                    "rand_avg_weights")))


def d1c_settings():
    return ("pooled", "fedavg", "avg_weights", "ens_logit", "ens_prob", "interp_sel",
            "rand_avg_weights", "rand_fedavg")


# ----------------------------------------------------------------------
# 3. Direction 2: the frozen alarm rules
# ----------------------------------------------------------------------

def run_d2(OUT, alarm_sets):
    import run_federated as rfd
    import direction2_alarms as d2
    sel = fm.d2_choice()
    rows, att = [], []
    for nm, df in alarm_sets.items():
        meta = df[["recording", "window_index", "label", "stat_duration",
                   "attack_type"]].reset_index(drop=True)
        recs = d2.prepare_recordings(meta)
        for seed in rfd.SEEDS:
            for arm in d2.ARMS:
                z = d2.load_scores(OUT / "d2_scores" / f"{arm}_{nm}_s{seed}.npz", meta)
                for thr in d2.THRESHOLDS:
                    tp, tl = float(z[f"thr_{thr}"]), float(z[f"thr_{thr}_logit"])
                    G = float(z["class_gap_logit"])
                    exceed = z["target_score"] >= tp
                    todo = [("single window", "kofn", (1, 1), np.nan, None)]
                    s = sel[(sel["arm"] == arm) & (sel["threshold"] == thr)]
                    for r in s.itertuples():
                        fam, p = d2.parse_rule(r.rule)
                        todo.append((f"{r.family}, limit {r.limit_per_hour:g}/h", fam, p,
                                     r.limit_per_hour, r.limit_met))
                    for label, fam, p, limit, met in todo:
                        if fam == "kofn":
                            fn = lambda rc, p=p: d2.kofn_alarms(exceed[rc["idx"]], *p)  # noqa: E731
                        else:
                            fn = lambda rc, p=p: d2.cusum_alarms(  # noqa: E731
                                z["target_logit"][rc["idx"]], tl, p[0], p[1], G)
                        for gap, rr in d2.count(recs, fn).items():
                            base = {"target_set": nm, "seed": seed, "arm": arm,
                                    "threshold": thr, "choice": label, "family": fam,
                                    "rule": d2.rule_name(fam, p), "limit_per_hour": limit,
                                    "limit_met_on_ctt": met, "episode_gap_s": gap}
                            rows.append({**base, **{k: v for k, v in rr.items()
                                                    if k != "per_attack"}})
                            for a, (n_ep, n_det) in rr["per_attack"].items():
                                att.append({**base, "attack_type": a, "episodes": n_ep,
                                            "detected": n_det})
        print(f"  Direction 2: {nm} done")
    pd.DataFrame(rows).to_csv(OUT / "d2_runs.csv", index=False)
    pd.DataFrame(att).to_csv(OUT / "d2_per_attack.csv", index=False)


# ----------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------

def run(target):
    OUT = out_dir(target)
    if (OUT / "COMPLETED").exists():
        raise SystemExit(f"{target} was already run to the end on "
                         f"{(OUT / 'COMPLETED').read_text().strip()}. It is run only once.")
    ok, cfg = fm.verify(target)
    if not ok:
        raise SystemExit("Stopped: the method is not exactly as frozen.")
    fid = (fm.FROZEN / "freeze_id.txt").read_text().strip()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "freeze_id.txt").exists() and (OUT / "freeze_id.txt").read_text().strip() != fid:
        raise SystemExit("This run was started under a different freeze. Send this message.")
    (OUT / "freeze_id.txt").write_text(fid + "\n")
    t_start = time.time()
    print("loading features ...")
    data, parts, clean, evalset = load_all()
    clean_t, sets, alarm_sets = load_target(target)
    run_rf(target, OUT, data, parts, clean_t, sets)
    run_mlp(target, OUT, data, parts, clean_t, sets, alarm_sets)
    run_d2(OUT, alarm_sets)
    (OUT / "COMPLETED").write_text(f"{date.today().isoformat()} (freeze id {fid})\n")
    print(f"\n  {target} done in {(time.time() - t_start) / 60:.0f} min; outputs in {OUT}/")
    report(target)


def check_road(OUT):
    """The thesis arms must reproduce the thesis ROAD run (same code, same choices)."""
    rows = []
    old_dir = RESULTS / "road"
    for fam, keys in (("rf", ["feature_set", "mode", "train_vehicles"]),
                      ("mlp", ["mode", "train_vehicles", "seed"])):
        fo, fn = old_dir / f"{fam}_results.csv", OUT / f"{fam}_results.csv"
        if not fo.exists() or not fn.exists():
            continue
        o = pd.read_csv(fo)
        o = o[o["setting"] == "external"].rename(columns={"road_set": "target_set"})
        n = pd.read_csv(fn)
        n = n[n["setting"] == "external"]
        m = n.merge(o, on=keys + ["target_set"], suffixes=("", "_thesis"))
        for k in ("auroc", "auprc", "mcc_zs", "mcc_zsb", "mcc_cal", "mcc_oracle"):
            if f"{k}_thesis" in m:
                rows.append({"family": fam, "metric": k, "rows_compared": len(m),
                             "max_abs_difference": float((m[k] - m[f"{k}_thesis"]).abs().max())
                             if len(m) else np.nan})
    c = pd.DataFrame(rows)
    if len(c):
        c.to_csv(OUT / "check.csv", index=False)
        print("\nCHECK: thesis arms against the thesis ROAD results")
        print("  " + c.to_string(index=False).replace("\n", "\n  "))
        ok = (c["rows_compared"] > 0).all() and c["max_abs_difference"].max() <= 1e-6
        print("  OK: same results as the thesis ROAD run." if ok else
              "  NOTE: differences found. Send this log.")


def report(target):
    OUT = out_dir(target)
    if target == "road" and not QUICK:
        check_road(OUT)
    print("\n" + "=" * 100)
    print(f"EXTERNAL VALIDATION: {target.upper()}  (frozen method; means over seeds)")
    print("=" * 100)
    f = OUT / "mlp_results.csv"
    if f.exists():
        r = pd.read_csv(f)
        r = r[(r["setting"] == "external") & (r["selected"].astype(str).str.lower() == "true")]
        r = r.assign(mode=np.where(r["arm"] == "single", "single (mean of 4)", r["mode"]))
        cols = [c for c in ("auroc", "auprc", "mcc_zs", "mcc_zsb", "mcc_cal", "mcc_cal_loso",
                            "mcc_oracle", "fpr_zs", "fpr_cal", "recall_zs") if c in r]
        for nm, g in r.groupby("target_set"):
            print(f"\n  MLP, {nm}")
            t = g.groupby("mode")[cols].mean()
            print("  " + t.round(3).to_string().replace("\n", "\n  "))
    f = OUT / "rf_results.csv"
    if f.exists():
        r = pd.read_csv(f)
        r = r[(r["setting"] == "external") & (r["selected"].astype(str).str.lower() == "true")]
        cols = [c for c in ("auroc", "auprc", "mcc_zs", "mcc_zsb", "mcc_cal", "mcc_cal_loso",
                            "mcc_oracle") if c in r]
        for nm, g in r.groupby("target_set"):
            print(f"\n  RF (frozen feature set), {nm}")
            print("  " + g.groupby(["mode", "train_vehicles"])[cols].mean().round(3)
                  .to_string().replace("\n", "\n  "))
    f = OUT / "d1_runs.csv"
    if f.exists():
        r = pd.read_csv(f)
        cols = [c for c in ("mcc_zs", "fpr_zs", "spread", "headroom", "auroc", "auprc",
                            "mcc_cal", "src_mcc_zs", "brier_src", "ece_src",
                            "disagree_src_benign_G") if c in r]
        for nm, g in r.groupby("target_set"):
            print(f"\n  Direction 1, {nm}")
            print("  " + g.groupby("setting")[cols].mean().reindex(
                [s for s in d1c_settings() if s in set(g["setting"])])
                .round(3).to_string().replace("\n", "\n  "))
    f = OUT / "d2_runs.csv"
    if f.exists():
        r = pd.read_csv(f)
        r = r[r["episode_gap_s"] == 1.0]
        for nm, g in r.groupby("target_set"):
            print(f"\n  Direction 2, {nm} (episodes merged over 1 s)")
            t = g.groupby(["arm", "threshold", "choice"], sort=False)[
                ["episode_detection", "median_delay_s", "fa_episodes_per_hour",
                 "attack_episodes", "clean_hours"]].mean()
            print("  " + t.round(3).to_string().replace("\n", "\n  "))
    print(f"\n  All rows in {OUT}/. Nothing here is tuned on {target}.")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    tgt = sys.argv[2] if len(sys.argv) > 2 else ""
    if what not in ("run", "report") or tgt not in TARGETS:
        print(__doc__)
        sys.exit(1)
    {"run": run, "report": report}[what](tgt)
