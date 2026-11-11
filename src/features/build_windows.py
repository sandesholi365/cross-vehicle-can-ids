"""
Turn the per-vehicle message CSVs into fixed-size windows with features.

Two feature groups are produced side by side in the same table:

  raw_*   vehicle-specific features: arbitration id statistics, payload
          byte means, DLC. These depend on how a particular vehicle
          numbers its ids and lays out its payloads.

  stat_*  vehicle-independent features: timing and distribution
          statistics (rates, ratios, entropies). These do not refer to
          any specific id value or byte position.

Windows are non-overlapping blocks of WINDOW_SIZE consecutive messages,
taken in file order (which is time order). A window is labelled attack
(1) if it contains at least one attack message.

Every window keeps its 'recording' name, so the train/test split can be
made at file level and never inside a recording.

Run from the project root:
    python src/features/build_windows.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

IN = Path("data/processed/can_train_and_test_by_vehicle")
OUT = Path("data/processed/features")

WINDOW_SIZE = 100        # messages per window
CHUNK_WINDOWS = 5000     # windows held in memory at once
N_BYTES = 8              # payload bytes kept per message

ID_HEX_WIDTH = 8         # enough for 29-bit extended ids
PAYLOAD_HEX_WIDTH = N_BYTES * 2


# ----------------------------------------------------------------------
# hex decoding (vectorised: one numpy pass, no per-row Python)
# ----------------------------------------------------------------------

# lookup table mapping the ASCII code of a hex character to its value
_LUT = np.zeros(256, dtype=np.uint8)
_LUT[np.frombuffer(b"0123456789", dtype=np.uint8)] = np.arange(10, dtype=np.uint8)
_LUT[np.frombuffer(b"ABCDEF", dtype=np.uint8)] = np.arange(10, 16, dtype=np.uint8)


def _nibbles(series, width):
    """Fixed-width uppercase hex text -> (n, width) array of nibble values."""
    text = series.fillna("").astype(str).str.upper()
    raw = np.frombuffer(text.values.astype(f"S{width}").tobytes(), dtype=np.uint8)
    return _LUT[raw.reshape(-1, width)]


def hex_to_int(series):
    """'0316' -> 790. Takes the last ID_HEX_WIDTH characters."""
    text = series.fillna("").astype(str).str.slice(-ID_HEX_WIDTH).str.zfill(ID_HEX_WIDTH)
    nib = _nibbles(text, ID_HEX_WIDTH).astype(np.int64)
    out = np.zeros(len(text), dtype=np.int64)
    for k in range(ID_HEX_WIDTH):
        out = (out << 4) | nib[:, k]
    return out


def hex_to_byte_matrix(series):
    """'0520EA...' -> (n, N_BYTES) byte values, right-padded with zeros."""
    text = (series.fillna("").astype(str)
            .str.slice(0, PAYLOAD_HEX_WIDTH)
            .str.ljust(PAYLOAD_HEX_WIDTH, "0"))
    nib = _nibbles(text, PAYLOAD_HEX_WIDTH)
    return (nib[:, 0::2] << 4) | nib[:, 1::2]


# ----------------------------------------------------------------------
# window features
# ----------------------------------------------------------------------

def window_features(df, vehicle, recording, attack_type, index_offset):
    """One row of features per complete window in df."""
    W = WINDOW_SIZE
    n = (len(df) // W) * W
    if n == 0:
        return None
    df = df.iloc[:n]
    nw = n // W

    ts = df["timestamp"].to_numpy(dtype=np.float64).reshape(nw, W)
    ids = hex_to_int(df["arbitration_id"]).reshape(nw, W)
    dlc = df["dlc"].to_numpy(dtype=np.float64).reshape(nw, W)
    lab = df["label"].to_numpy(dtype=np.int64).reshape(nw, W)
    by = hex_to_byte_matrix(df["data_field"]).reshape(nw, W, N_BYTES)

    feat = {}

    # ---- identification and label -----------------------------------
    feat["vehicle"] = vehicle
    feat["recording"] = recording
    feat["attack_type"] = attack_type
    feat["window_index"] = np.arange(nw) + index_offset
    feat["n_attack_msgs"] = lab.sum(axis=1)
    feat["label"] = (lab.sum(axis=1) > 0).astype(np.int8)

    # ---- id counts per window (needed by both feature groups) -------
    codes, _ = pd.factorize(ids.ravel())
    n_codes = int(codes.max()) + 1
    win_of_row = np.repeat(np.arange(nw, dtype=np.int64), W)
    counts = np.bincount(win_of_row * n_codes + codes,
                         minlength=nw * n_codes).reshape(nw, n_codes)
    n_unique = (counts > 0).sum(axis=1)

    # ---- raw / vehicle-specific features ----------------------------
    feat["raw_id_mean"] = ids.mean(axis=1)
    feat["raw_id_std"] = ids.std(axis=1)
    feat["raw_id_min"] = ids.min(axis=1)
    feat["raw_id_max"] = ids.max(axis=1)
    feat["raw_n_unique_ids"] = n_unique
    feat["raw_dlc_mean"] = dlc.mean(axis=1)
    for k in range(N_BYTES):
        feat[f"raw_byte{k}_mean"] = by[:, :, k].mean(axis=1)
    feat["raw_byte_mean_all"] = by.mean(axis=(1, 2))

    # ---- vehicle-independent timing / distribution features ---------
    duration = ts[:, -1] - ts[:, 0]
    feat["stat_duration"] = duration
    feat["stat_msg_rate"] = np.where(duration > 0, W / duration, 0.0)

    iat = np.clip(np.diff(ts, axis=1), 0, None)
    iat_mean = iat.mean(axis=1)
    iat_std = iat.std(axis=1)
    safe_mean = np.where(iat_mean > 0, iat_mean, np.nan)

    feat["stat_iat_mean"] = iat_mean
    feat["stat_iat_std"] = iat_std
    feat["stat_iat_cv"] = np.nan_to_num(iat_std / safe_mean)
    feat["stat_iat_max_ratio"] = np.nan_to_num(iat.max(axis=1) / safe_mean)
    feat["stat_iat_min_ratio"] = np.nan_to_num(iat.min(axis=1) / safe_mean)

    p = counts / W
    ent = -np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0.0).sum(axis=1)
    feat["stat_id_entropy_norm"] = ent / np.log2(np.maximum(n_unique, 2))
    feat["stat_unique_id_ratio"] = n_unique / W
    feat["stat_top_id_frac"] = counts.max(axis=1) / W

    same_id = ids[:, 1:] == ids[:, :-1]
    feat["stat_repeat_id_frac"] = same_id.mean(axis=1)

    same_payload = (by[:, 1:, :] == by[:, :-1, :]).all(axis=2)
    feat["stat_repeat_payload_frac"] = same_payload.mean(axis=1)

    byte_step = np.abs(by[:, 1:, :].astype(np.int16) - by[:, :-1, :].astype(np.int16))
    feat["stat_byte_change_rate"] = byte_step.mean(axis=(1, 2)) / 255.0

    return pd.DataFrame(feat)


# ----------------------------------------------------------------------

def build_for_vehicle(vehicle, manifest):
    """Window every recording of one vehicle and return one table."""
    rows = manifest[manifest["vehicle"] == vehicle]
    frames = []

    for _, rec in rows.iterrows():
        path = IN / vehicle / f"{rec['recording']}.csv"
        offset = 0
        for chunk in pd.read_csv(
            path,
            usecols=["timestamp", "arbitration_id", "dlc", "data_field", "label"],
            dtype={"arbitration_id": str, "data_field": str,
                   "dlc": "int16", "label": "int8"},
            chunksize=CHUNK_WINDOWS * WINDOW_SIZE,
        ):
            part = window_features(chunk, vehicle, rec["recording"],
                                   rec["attack_type"], offset)
            if part is not None:
                frames.append(part)
                offset += len(part)
        print(f"    {rec['recording']:<45} {offset:>7,} windows")

    return pd.concat(frames, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(IN / "manifest.csv")

    for vehicle in ["impala", "traverse", "silverado", "forester"]:
        print(f"\n{vehicle} ...")
        table = build_for_vehicle(vehicle, manifest)

        dest = OUT / f"{vehicle}_w{WINDOW_SIZE}.csv"
        table.to_csv(dest, index=False)

        n_attack = int(table["label"].sum())
        print(f"  -> {dest.name}: {len(table):,} windows, "
              f"{n_attack:,} attack ({100 * n_attack / len(table):.2f}%), "
              f"{table['recording'].nunique()} recordings")

    n_feat = len([c for c in table.columns if c.startswith(("raw_", "stat_"))])
    print(f"\nDone. {n_feat} features per window "
          f"({len([c for c in table.columns if c.startswith('raw_')])} raw, "
          f"{len([c for c in table.columns if c.startswith('stat_')])} stat). "
          f"Output is in {OUT}/")


if __name__ == "__main__":
    main()
