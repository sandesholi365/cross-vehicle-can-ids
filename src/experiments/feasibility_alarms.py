"""
Feasibility check for a next direction: would alarms built from several
consecutive windows separate attacks from the rare normal windows that score
high? CAN-Train-and-Test only. ROAD is NOT touched (any rule for ROAD would
have to be fixed on CAN-Train-and-Test first). Nothing is retrained.

Uses the per-window scores saved by run_federated.py for the first seed
(results/scores/mlp_<arm>_<vehicle>.npz) and puts them back in time order
with the window table (recording, window_index). Windows are
non-overlapping blocks of 100 messages.

For each arm, held-out vehicle and threshold (zs and cal, as saved) it reports:

  single-window picture
    share of false-alarm windows that are ISOLATED (no neighbouring window
    of the same recording above the threshold), and the mean length of runs
    of false-alarm windows, against the median length of attack segments
  k-of-n alarms (an alarm when at least k of the last n windows are above
  the threshold), for 1-of-1 (today), 2-of-3, 3-of-5 and 5-of-10
    false-alarm episodes per hour on clean benign traffic (benign windows at
    least n windows away from any attack window), counted as runs of
    consecutive alarms
    attack episodes: runs of attack windows merged when the benign gap between
    them is at most 1 s (and 5 s as a check); injected frames are spread out,
    so one attack leaves many windows without an injected frame
    event recall: share of attack episodes with at least one alarm inside
    median delay from the start of an episode to its first alarm (seconds)

Run from the project root:
    python -u src/experiments/feasibility_alarms.py
Output: results/feasibility_alarms/summary.csv and per_vehicle.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import QUICK, RESULTS, SCORES_DIR, VEHICLES, load_all  # noqa: E402

OUT = RESULTS / ("feasibility_alarms_quick" if QUICK else "feasibility_alarms")
ARMS = ["pooled", "fedavg", "fedprox"]
RULES = [(1, 1), (2, 3), (3, 5), (5, 10)]
GAPS_S = [1.0, 5.0]    # attack windows closer than this (s) form one attack episode


def runs_of(mask):
    """Start and length of each run of True values in a 1-D boolean array."""
    if not mask.any():
        return np.array([], int), np.array([], int)
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return starts, ends - starts


def k_of_n(exceed, k, n):
    """alarm[t] = at least k of exceed[t-n+1 .. t] (within one recording)."""
    c = np.concatenate([[0], np.cumsum(exceed.astype(int))])
    t = np.arange(len(exceed))
    lo = np.maximum(0, t - n + 1)
    return (c[t + 1] - c[lo]) >= k


def near_attack(label, n):
    """True for windows within n windows of any attack window."""
    att = label.astype(int)
    c = np.concatenate([[0], np.cumsum(att)])
    t = np.arange(len(att))
    lo, hi = np.maximum(0, t - n), np.minimum(len(att), t + n + 1)
    return (c[hi] - c[lo]) > 0


def episodes_of(y, dur, gap_s):
    """
    Attack episodes: runs of attack windows, merged when the benign gap
    between them is at most gap_s seconds. In CAN-Train-and-Test the injected
    frames are spread out, so one attack leaves many 100-message windows
    without an injected frame; without merging, one attack would count as
    dozens of tiny events.
    """
    starts, lens = runs_of(y)
    eps = []
    for st, ln in zip(starts, lens):
        en = st + ln
        if eps and float(dur[eps[-1][1]:st].sum()) <= gap_s:
            eps[-1][1] = en
        else:
            eps.append([st, en])
    return eps


def analyse(df, thr, gap_s):
    """df: one vehicle's scored windows in time order, with score and label."""
    out = {}
    iso, fa_run_len, seg_len, ep_dur = [], [], [], []
    per_rule = {r: {"episodes": 0, "clean_seconds": 0.0, "clean_windows": 0,
                    "clean_alarm_windows": 0, "attacks": 0, "detected": 0,
                    "delays": []} for r in RULES}
    for _, g in df.groupby("recording", sort=False):
        s, y = g["score"].to_numpy(), g["label"].to_numpy().astype(bool)
        dur = np.clip(g["stat_duration"].to_numpy(dtype=float), 0.0, None)
        exc = s >= thr
        fa = exc & ~y
        if fa.any():
            left = np.concatenate([[False], exc[:-1]])
            right = np.concatenate([exc[1:], [False]])
            iso += list(~(left | right)[fa])
            _, lens = runs_of(fa)
            fa_run_len += list(lens)
        seg_len += list(runs_of(y)[1])
        eps = episodes_of(y, dur, gap_s)
        span = np.zeros(len(y), bool)
        for st, en in eps:
            span[st:en] = True
            ep_dur.append(float(dur[st:en].sum()))
        for (k, n) in RULES:
            alarm = k_of_n(exc, k, n)
            clean = ~span & ~near_attack(span, n)
            pr = per_rule[(k, n)]
            pr["clean_seconds"] += float(dur[clean].sum())
            pr["clean_windows"] += int(clean.sum())
            pr["clean_alarm_windows"] += int((alarm & clean).sum())
            pr["episodes"] += len(runs_of(alarm & clean)[0])
            for st, en in eps:
                pr["attacks"] += 1
                hit = np.where(alarm[st:en])[0]
                if len(hit):
                    pr["detected"] += 1
                    pr["delays"].append(float(dur[st:st + hit[0] + 1].sum()))
    out["false_alarm_windows_isolated"] = float(np.mean(iso)) if iso else np.nan
    out["false_alarm_run_mean"] = float(np.mean(fa_run_len)) if fa_run_len else np.nan
    out["false_alarm_windows"] = int(len(iso))
    out["attack_segment_median"] = float(np.median(seg_len)) if seg_len else np.nan
    out["attack_episodes"] = len(ep_dur)
    out["attack_episode_median_s"] = float(np.median(ep_dur)) if ep_dur else np.nan
    rows = []
    for (k, n), pr in per_rule.items():
        hours = pr["clean_seconds"] / 3600.0
        rows.append({**out, "episode_gap_s": gap_s, "rule": f"{k}-of-{n}",
                     "false_alarm_episodes_per_hour": pr["episodes"] / hours if hours else np.nan,
                     "clean_window_alarm_rate": pr["clean_alarm_windows"] / max(pr["clean_windows"], 1),
                     "event_recall": pr["detected"] / max(pr["attacks"], 1),
                     "median_delay_s": float(np.median(pr["delays"])) if pr["delays"] else np.nan,
                     "clean_hours": hours})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, _, evalset = load_all()
    rows = []
    for v in VEHICLES:
        ev = evalset[v].reset_index(drop=True)
        for arm in ARMS:
            f = SCORES_DIR / f"mlp_{arm}_{v}.npz"
            if not f.exists():
                print(f"  missing {f}, skipped")
                continue
            z = np.load(f, allow_pickle=True)
            if len(z["target_score"]) != len(ev) or not np.array_equal(
                    z["target_label"].astype(int), ev["label"].to_numpy().astype(int)):
                raise SystemExit(f"{f}: scores do not line up with the window table. "
                                 "Send this message.")
            df = ev[["recording", "window_index", "label", "stat_duration"]].copy()
            df["score"] = z["target_score"]
            df = df.sort_values(["recording", "window_index"], kind="stable")
            for tag in ("zs", "cal"):
                thr = float(z[f"thr_{tag}"])
                for gap_s in GAPS_S:
                    for r in analyse(df, thr, gap_s):
                        rows.append({"arm": arm, "held_out": v, "threshold": tag, **r})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "per_vehicle.csv", index=False)
    cols = ["false_alarm_windows_isolated", "false_alarm_run_mean",
            "attack_segment_median", "attack_episodes", "attack_episode_median_s",
            "false_alarm_episodes_per_hour", "event_recall", "median_delay_s"]
    summ = res.groupby(["episode_gap_s", "arm", "threshold", "rule"], sort=False)[cols].mean()
    summ.to_csv(OUT / "summary.csv")

    print("\n" + "=" * 100)
    print("FEASIBILITY: alarms from several windows (CAN-Train-and-Test, first seed, "
          "mean over 4 held-out vehicles)")
    print("=" * 100)
    print("\n1. Are the high-scoring normal windows isolated, and how long are attacks?")
    one = res[(res["rule"] == "1-of-1") & (res["episode_gap_s"] == GAPS_S[0])]
    for (arm, tag), g in one.groupby(["arm", "threshold"], sort=False):
        print(f"  {arm:<8} {tag:<4} isolated false-alarm windows "
              f"{g['false_alarm_windows_isolated'].mean():6.1%}   mean false-alarm run "
              f"{g['false_alarm_run_mean'].mean():5.1f} windows ({int(g['false_alarm_windows'].sum()):,} in total)")
    for gap_s in GAPS_S:
        e = res[(res["episode_gap_s"] == gap_s) & (res["rule"] == "1-of-1") &
                (res["arm"] == ARMS[0]) & (res["threshold"] == "zs")]
        print(f"  attack episodes (gaps up to {gap_s:g} s merged): "
              + ", ".join(f"{r.held_out} {int(r.attack_episodes)} (median {r.attack_episode_median_s:.1f} s)"
                          for r in e.itertuples()))
    for gap_s in GAPS_S:
        print(f"\n2. k-of-n alarms, attack episodes merged over gaps up to {gap_s:g} s:"
              f"\n   false-alarm episodes per hour | attack episodes detected | median delay (s)")
        sub = summ.loc[gap_s]
        for (arm, tag), g in sub.groupby(level=[0, 1], sort=False):
            print(f"  {arm:<8} {tag:<4}" + "".join(
                f"   {rule}: {r['false_alarm_episodes_per_hour']:7.1f} | {r['event_recall']:.2f} | "
                f"{r['median_delay_s']:.2f}"
                for (_, _, rule), r in g.iterrows()))
    print("\n  Per vehicle in", OUT / "per_vehicle.csv")


if __name__ == "__main__":
    main()
