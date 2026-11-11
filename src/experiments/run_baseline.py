"""
First baseline experiments: within-vehicle vs cross-vehicle detection.

For every source vehicle, a model is trained once on that vehicle's
training recordings and then evaluated on:

  within-vehicle : the held-out recordings of the SAME vehicle
  cross-vehicle  : every window of each OTHER vehicle (never trained on)

The same fitted model is used for both, so the drop between them is a
direct measurement of the generalization gap.

The split is made at recording (file) level, never inside a recording,
so consecutive near-identical windows can never sit on both sides.

Three feature sets are compared:
  raw   vehicle-specific features only
  stat  vehicle-independent timing / distribution features only
  all   both together

Run from the project root:
    python src/experiments/run_baseline.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, roc_auc_score)

FEATURES = Path("data/processed/features")
RESULTS = Path("results")

WINDOW = 100
VEHICLES = ["impala", "traverse", "silverado", "forester"]
FEATURE_SETS = ["raw", "stat", "all"]

N_TREES = 200
SEED = 42

# XGBoost is optional. If it is not installed the script still runs.
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------

def load_vehicle(vehicle):
    df = pd.read_csv(FEATURES / f"{vehicle}_w{WINDOW}.csv")
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def feature_columns(df, group):
    if group == "raw":
        prefixes = ("raw_",)
    elif group == "stat":
        prefixes = ("stat_",)
    else:
        prefixes = ("raw_", "stat_")
    return [c for c in df.columns if c.startswith(prefixes)]


def split_recordings(df):
    """
    Split a vehicle's recordings into train and test at file level.

    Within each attack type the earlier recordings go to train and the
    later ones to test, so every attack type is represented on both
    sides while no recording is ever split.
    """
    train_recs, test_recs = [], []
    for _, grp in df.groupby("attack_type"):
        recs = sorted(grp["recording"].unique())
        k = max(1, len(recs) // 2)
        train_recs += recs[:k]
        test_recs += recs[k:]
    return train_recs, test_recs


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------

def score(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) > 1
    return {
        "mcc": matthews_corrcoef(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auprc": average_precision_score(y_true, y_prob) if both_classes else np.nan,
        "auroc": roc_auc_score(y_true, y_prob) if both_classes else np.nan,
        "recall": tp / (tp + fn) if (tp + fn) else np.nan,
        "precision": tp / (tp + fp) if (tp + fp) else np.nan,
        "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
        "n_windows": len(y_true),
        "pos_rate": float(np.mean(y_true)),
    }


def per_attack_recall(test, y_pred, train_attack_types):
    """Recall on the attack windows of each attack type separately."""
    out = []
    attacks = test[test["label"] == 1]
    pred = pd.Series(y_pred, index=test.index)
    for atk, grp in attacks.groupby("attack_type"):
        out.append({
            "attack_type": atk,
            "n_attack_windows": len(grp),
            "recall": float(pred.loc[grp.index].mean()),
            "seen_in_training": atk in train_attack_types,
        })
    return out


# ----------------------------------------------------------------------

def build_model(name, y_train):
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=N_TREES,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        )
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        scale_pos_weight=(n_neg / max(n_pos, 1)),
        n_jobs=-1,
        random_state=SEED,
        eval_metric="logloss",
    )


def main():
    RESULTS.mkdir(exist_ok=True)

    print("loading features ...")
    data = {v: load_vehicle(v) for v in VEHICLES}
    for v in VEHICLES:
        print(f"  {v:<10} {len(data[v]):>9,} windows, "
              f"{int(data[v]['label'].sum()):>7,} attack")

    models = ["RF"] + (["XGB"] if HAVE_XGB else [])
    if not HAVE_XGB:
        print("\nxgboost not installed, running Random Forest only")
        print("  (to add it later: pip install xgboost)")

    results, attack_rows, importance_rows = [], [], []

    for model_name in models:
        for group in FEATURE_SETS:
            for src in VEHICLES:
                df = data[src]
                train_recs, test_recs = split_recordings(df)
                train = df[df["recording"].isin(train_recs)]
                within = df[df["recording"].isin(test_recs)]

                cols = feature_columns(df, group)
                train_attack_types = set(
                    train.loc[train["label"] == 1, "attack_type"].unique())

                print(f"\n{model_name} | {group:<4} | train on {src} "
                      f"({len(train):,} windows, {len(cols)} features)")

                model = build_model(model_name, train["label"].to_numpy())
                model.fit(train[cols], train["label"])

                if hasattr(model, "feature_importances_"):
                    for c, imp in zip(cols, model.feature_importances_):
                        importance_rows.append({
                            "model": model_name, "feature_set": group,
                            "train_vehicle": src, "feature": c,
                            "importance": float(imp),
                        })

                targets = [(src, within, "within")] + [
                    (t, data[t], "cross") for t in VEHICLES if t != src]

                for tgt_name, test, setting in targets:
                    prob = model.predict_proba(test[cols])[:, 1]
                    pred = (prob >= 0.5).astype(int)

                    row = {
                        "model": model_name,
                        "feature_set": group,
                        "train_vehicle": src,
                        "test_vehicle": tgt_name,
                        "setting": setting,
                    }
                    row.update(score(test["label"].to_numpy(), pred, prob))
                    results.append(row)

                    print(f"    {setting:<6} -> {tgt_name:<10} "
                          f"MCC {row['mcc']:.3f}  F1 {row['f1']:.3f}  "
                          f"AUPRC {row['auprc']:.3f}  FPR {row['fpr']:.4f}")

                    for a in per_attack_recall(test, pred, train_attack_types):
                        a.update({"model": model_name, "feature_set": group,
                                  "train_vehicle": src,
                                  "test_vehicle": tgt_name,
                                  "setting": setting})
                        attack_rows.append(a)

    res = pd.DataFrame(results)
    res.to_csv(RESULTS / "baseline_cross_vehicle.csv", index=False)
    pd.DataFrame(attack_rows).to_csv(
        RESULTS / "baseline_per_attack_recall.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(
        RESULTS / "baseline_feature_importance.csv", index=False)

    # ---- summary ----------------------------------------------------
    for model_name in models:
        for group in FEATURE_SETS:
            sub = res[(res["model"] == model_name) & (res["feature_set"] == group)]
            print(f"\n=== MCC matrix | {model_name} | features = {group} ===")
            piv = sub.pivot(index="train_vehicle", columns="test_vehicle",
                            values="mcc").reindex(index=VEHICLES, columns=VEHICLES)
            print(piv.round(3).to_string())

            w = sub.loc[sub["setting"] == "within", "mcc"].mean()
            c = sub.loc[sub["setting"] == "cross", "mcc"].mean()
            print(f"  mean within-vehicle MCC : {w:.3f}")
            print(f"  mean cross-vehicle MCC  : {c:.3f}")
            print(f"  generalization gap      : {w - c:.3f}")

    print(f"\nSaved to {RESULTS}/:")
    print("  baseline_cross_vehicle.csv       one row per train/test pair")
    print("  baseline_per_attack_recall.csv   recall per attack type")
    print("  baseline_feature_importance.csv  feature importances")


if __name__ == "__main__":
    main()
