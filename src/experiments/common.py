"""
Shared pieces for the cross-vehicle experiments.

Both model families (Random Forest in run_baselines.py, the MLP in
run_federated.py) import from here, so they use exactly the same data
split, threshold rules, metrics and selection protocol.

EVALUATION REGIMES
------------------
Every cross-vehicle result is scored on the same rows: the held-out
vehicle's recordings that contain attacks. Its attack-free recordings are
never scored. Only the way the decision threshold is set changes. Source-
side results are scored the same way (attack recordings of the sources'
test half), so source and target numbers are like for like.

  zero-shot (no data at all from the held-out vehicle)
    zs    threshold that maximises MCC on the source vehicles' calibration
          windows ("source-selected")
    zsb   threshold that gives the selected false-alarm budget B on the
          source vehicles' attack-free windows. Same rule as "cal" below,
          but measured on source traffic instead of target traffic
    05    the default cut-off of 0.5, for reference

  benign target-calibrated (attack-free traffic from the held-out vehicle,
  no attack labels)
    cal   threshold that gives the selected false-alarm budget B on the
          held-out vehicle's attack-free recordings
    cal_tNNN  the same at every candidate budget, for sensitivity only

  oracle  the best possible threshold on the evaluation rows themselves.
          Not achievable in practice. An upper bound.

AUROC and AUPRC need no threshold and are reported for every run.

The difference cal - zsb isolates threshold shift: the rule and the budget
are identical, only the benign traffic used to place the threshold differs.

SOURCE-ONLY SELECTION (nested leave-one-vehicle-out)
----------------------------------------------------
For each held-out vehicle H, the three remaining vehicles are the sources.
Anything that has to be chosen is chosen by an inner leave-one-source-out
loop run on those three sources alone: train on two, treat the third as a
pretend unseen vehicle, and repeat so each source takes that role once.
H is not loaded into this loop at all.

  feature set    the one with the highest mean inner AUPRC
  budget B       the one with the highest mean inner MCC at "cal"
  FedProx mu     the one with the highest mean inner AUPRC (MLP only)

Ties go to the first candidate in the list. Every inner score is written to
a selection log, so each choice can be traced to the numbers behind it.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

FEATURES_DIR = Path("data/processed/features")
RESULTS = Path("results")
SCORES_DIR = RESULTS / "scores"

WINDOW = 100
VEHICLES = ["impala", "traverse", "silverado", "forester"]

SPLIT_SEED = 42          # fixed for the data split, shared by every run
CALIB_FRACTION = 0.15    # training windows held back for the "zs" threshold
N_CANDIDATES = 300       # threshold candidates drawn from the score quantiles
THR_TOLERANCE = 0.005    # MCC slack when breaking threshold ties

# candidate false-alarm budgets for the "cal" and "zsb" rules, as fractions
BUDGETS = {"t001": 0.0001, "t005": 0.0005, "t01": 0.0010, "t05": 0.0050}

FEATURE_SETS = ["raw", "stat", "stat_sf", "all"]
STAT_SCALE_FREE = [
    "stat_iat_cv", "stat_iat_max_ratio", "stat_iat_min_ratio",
    "stat_id_entropy_norm", "stat_unique_id_ratio", "stat_top_id_frac",
    "stat_repeat_id_frac", "stat_repeat_payload_frac", "stat_byte_change_rate",
]

# QUICK=1 runs everything on a small slice of the data, to check the whole
# pipeline end to end in a few minutes before starting the real run.
QUICK = os.environ.get("QUICK", "0") == "1"
QUICK_KEEP_EVERY = 25


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------

def load_vehicle(vehicle):
    df = pd.read_csv(FEATURES_DIR / f"{vehicle}_w{WINDOW}.csv")
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if QUICK:
        df = df.iloc[::QUICK_KEEP_EVERY].reset_index(drop=True)
    return df


def feature_columns(df, group):
    if group == "raw":
        return [c for c in df.columns if c.startswith("raw_")]
    if group == "stat":
        return [c for c in df.columns if c.startswith("stat_")]
    if group == "stat_sf":
        return [c for c in STAT_SCALE_FREE if c in df.columns]
    return [c for c in df.columns if c.startswith(("raw_", "stat_"))]


def split_recordings(df):
    """Per attack type, the first half of the sorted recordings train."""
    train_recs, test_recs = [], []
    for _, grp in df.groupby("attack_type"):
        recs = sorted(grp["recording"].unique())
        k = max(1, len(recs) // 2)
        train_recs += recs[:k]
        test_recs += recs[k:]
    return train_recs, test_recs


def prepare(df):
    """(fit, calib, test) for one vehicle, split at recording level."""
    train_recs, test_recs = split_recordings(df)
    train = df[df["recording"].isin(train_recs)]
    test = df[df["recording"].isin(test_recs)]
    mask = np.random.default_rng(SPLIT_SEED).random(len(train)) < CALIB_FRACTION
    return train[~mask], train[mask], test


def target_split(df):
    """A vehicle seen as a target: attack-free recordings, and the rest."""
    is_clean = df["attack_type"] == "attack-free"
    return df[is_clean], df[~is_clean]


def load_all():
    """Every vehicle, with its source-side and target-side splits."""
    data, parts, clean, evalset = {}, {}, {}, {}
    for v in VEHICLES:
        data[v] = load_vehicle(v)
        parts[v] = prepare(data[v])
        clean[v], evalset[v] = target_split(data[v])
        fit, calib, test = parts[v]
        print(f"  {v:<10} fit {len(fit):>8,}  calib {len(calib):>7,}  "
              f"test {len(test):>8,}   as target: attack-free "
              f"{len(clean[v]):>7,} / scored {len(evalset[v]):>8,} "
              f"({int(evalset[v]['label'].sum()):,} attack)")
    if QUICK:
        print(f"  QUICK mode: every {QUICK_KEEP_EVERY}th window only. "
              f"Results are for checking the pipeline, not for the thesis.")
    return data, parts, clean, evalset


def scored(df):
    """
    The rows a model is scored on: recordings that contain attacks.

    Attack-free recordings are never scored, on the target or on the
    sources, so every evaluation set has the same make-up and source-side
    and target-side numbers can be compared directly. Attack-free recordings
    are used only to place thresholds.
    """
    return df[df["attack_type"] != "attack-free"]


def heldout_benign(parts, sources):
    """
    Source attack-free windows that no model was fitted on (calibration and
    test portions). Used to place the zero-shot "zsb" threshold.
    """
    frames = []
    for v in sources:
        _, calib, test = parts[v]
        frames += [calib[calib["attack_type"] == "attack-free"],
                   test[test["attack_type"] == "attack-free"]]
    return pd.concat(frames, ignore_index=True)


def subsample(df, n, seed=SPLIT_SEED):
    """Deterministic subsample, so every feature set sees the same rows."""
    if n >= len(df):
        return df
    idx = np.random.default_rng(seed).choice(len(df), size=n, replace=False)
    return df.iloc[np.sort(idx)]


# ----------------------------------------------------------------------
# metrics and thresholds
# ----------------------------------------------------------------------

def mcc_from_counts(tp, tn, fp, fn):
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denom == 0 else (tp * tn - fp * fn) / denom


def _counts(y_bool, n_pos, n_neg, prob, thr):
    pred = prob >= thr
    tp = int(np.count_nonzero(y_bool & pred))
    fp = int(np.count_nonzero(pred)) - tp
    return tp, n_neg - fp, fp, n_pos - tp


def at_threshold(y_bool, n_pos, n_neg, prob, thr):
    tp, tn, fp, fn = _counts(y_bool, n_pos, n_neg, prob, thr)
    return {
        "mcc": mcc_from_counts(tp, tn, fp, fn),
        "f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "recall": tp / n_pos if n_pos else np.nan,
        "precision": tp / (tp + fp) if (tp + fp) else np.nan,
        "fpr": fp / n_neg if n_neg else np.nan,
    }


def candidate_thresholds(prob):
    """Candidates from the score quantiles, so the search is scale-free."""
    qs = np.quantile(prob, np.linspace(0.0, 1.0, N_CANDIDATES))
    return np.unique(np.concatenate([qs, [0.5]]))


def best_threshold(y, prob):
    """
    Threshold maximising MCC. Ties (within THR_TOLERANCE) are broken towards
    the middle of the flat region, so a flat MCC surface cannot push the
    cut-off to an extreme.
    """
    y_bool = np.asarray(y).astype(bool)
    n_pos = int(np.count_nonzero(y_bool))
    n_neg = len(y_bool) - n_pos
    grid = candidate_thresholds(prob)
    scores = np.array([mcc_from_counts(*_counts(y_bool, n_pos, n_neg, prob, t))
                       for t in grid])
    best = float(scores.max())
    return float(np.median(grid[scores >= best - THR_TOLERANCE])), best


def budget_threshold(benign_scores, budget):
    """Cut-off that flags `budget` of the given attack-free windows."""
    if len(benign_scores) == 0:
        return np.nan
    return float(np.quantile(benign_scores, 1.0 - budget))


def evaluate(y, prob, thresholds):
    """
    Threshold-free metrics, then every threshold rule in `thresholds`
    (a dict tag -> cut-off), then the oracle.
    """
    y_bool = np.asarray(y).astype(bool)
    n_pos = int(np.count_nonzero(y_bool))
    n_neg = len(y_bool) - n_pos
    two = n_pos > 0 and n_neg > 0

    row = {
        "n_windows": len(y_bool),
        "pos_rate": n_pos / len(y_bool) if len(y_bool) else np.nan,
        "auroc": roc_auc_score(y_bool, prob) if two else np.nan,
        "auprc": average_precision_score(y_bool, prob) if two else np.nan,
    }
    for tag, thr in thresholds.items():
        row[f"thr_{tag}"] = thr
        if thr is None or not np.isfinite(thr):
            continue
        for k, v in at_threshold(y_bool, n_pos, n_neg, prob, thr).items():
            row[f"{k}_{tag}"] = v

    thr_o, _ = best_threshold(y_bool, prob)
    row["thr_oracle"] = thr_o
    for k, v in at_threshold(y_bool, n_pos, n_neg, prob, thr_o).items():
        row[f"{k}_oracle"] = v
    return row


def threshold_rules(thr_zs, benign_source, benign_target, budget):
    """
    The full set of rules for one cross-vehicle evaluation.

    thr_zs         MCC-optimal cut-off on source calibration windows
    benign_source  model scores on source attack-free windows (held out)
    benign_target  model scores on the target's attack-free recordings
    budget         the selected false-alarm budget (a fraction)
    """
    rules = {"05": 0.5, "zs": thr_zs,
             "zsb": budget_threshold(benign_source, budget)}
    if benign_target is not None:
        rules["cal"] = budget_threshold(benign_target, budget)
        for tag, b in BUDGETS.items():
            rules[f"cal_{tag}"] = budget_threshold(benign_target, b)
    return rules


def degenerate(prob, y):
    """True if the scores carry no ranking information at all."""
    if float(np.std(prob)) < 1e-6:
        return True
    base = float(np.mean(y))
    ap = average_precision_score(y, prob) if 0 < base < 1 else np.nan
    return bool(np.isfinite(ap) and abs(ap - base) < 1e-3)


def per_attack_rows(test_df, prob, thresholds, seen_types, info):
    """Recall per attack type at each threshold in `thresholds`."""
    rows = []
    attacks = test_df["label"].to_numpy() == 1
    types = test_df["attack_type"].to_numpy()
    for atk in sorted(set(types[attacks])):
        m = attacks & (types == atk)
        row = {**info, "attack_type": atk, "n_attack_windows": int(m.sum()),
               "seen_in_training": atk in seen_types}
        for tag, thr in thresholds.items():
            row[f"recall_{tag}"] = (float(np.mean(prob[m] >= thr))
                                    if np.isfinite(thr) else np.nan)
        rows.append(row)
    return rows


def attack_types_trained_on(df):
    return set(df.loc[df["label"] == 1, "attack_type"].unique())


# ----------------------------------------------------------------------
# selection helpers
# ----------------------------------------------------------------------

def pick_best(log, key, candidates, criterion):
    """
    From the inner-loop log, the candidate with the highest mean criterion.
    Ties go to the first candidate in `candidates`.
    """
    means = log.groupby(key)[criterion].mean()
    best, best_val = None, -np.inf
    for c in candidates:
        if c in means.index and means[c] > best_val + 1e-12:
            best, best_val = c, float(means[c])
    return best, best_val, means


def pick_budget(log):
    """Budget with the highest mean inner MCC at the target-calibrated rule."""
    means = {tag: log[f"mcc_cal_{tag}"].mean() for tag in BUDGETS}
    best = None
    for tag in BUDGETS:
        if best is None or means[tag] > means[best] + 1e-12:
            best = tag
    return best, means


def save_scores(name, **arrays):
    """Per-window scores for the score-distribution figure."""
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SCORES_DIR / f"{name}.npz", **arrays)
