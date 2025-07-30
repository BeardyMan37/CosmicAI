#!/usr/bin/env python3
"""
merge_stats.py  –  attach scoring-algorithm statistics to the science Parquet table.

* Reads the original Parquet file.
* Gathers every  …/length_*/bandpass*_stat.csv  under a root directory.
* Keeps the four score columns:
      atmospheric_interference, score, win_start, win_end
* Matches rows on  (eb_uid, antenna, spw_name_ms, polarization).
* Writes a new Parquet file **containing only the rows that received stats**.

Run `--dry-run` to see all sanity numbers without creating the output file.

Example
-------
python merge_stats.py \
       --parquet science.parquet \
       --csv-root gazi_29jul25 \
       --output science_scored.parquet \
       --dry-run
"""
import argparse
from pathlib import Path

import pandas as pd

KEYS   = ["eb_uid", "antenna", "spw_name_ms", "polarization"]
STATS  = ["atmospheric_interference", "score", "win_start", "win_end"]
CSV_GLOB = "length_*/*.csv"

# ---------------------------------------------------------------------------
# normalisers for the key columns
# ---------------------------------------------------------------------------
def norm_pol(p: str) -> str:
    """'XX'->'X', 'YY'->'Y', keep anything else unchanged (None→None)."""
    if p is None:
        return None
    p = str(p).strip().upper()
    return {"XX": "X", "YY": "Y"}.get(p, p)

def norm_spw(s: str) -> str:
    """Remove 'spw' / 'spw_' prefix and leading zeros: 'spw_25'→'25'."""
    if s is None:
        return None
    s = str(s).lower().lstrip().replace("spw_", "").replace("spw", "")
    return s.lstrip("0") or "0"


# ---------------------------------------------------------------------------
def load_all_stats(csv_root: Path) -> pd.DataFrame:
    """Read every stats-CSV and return one concatenated DataFrame."""
    paths = sorted(csv_root.rglob(CSV_GLOB))
    if not paths:
        raise FileNotFoundError(f"No CSV files matched {csv_root / CSV_GLOB}")

    dfs = []
    for p in paths:
        df = pd.read_csv(
            p,
            usecols=lambda c: c in KEYS + STATS,   # ignore the huge arrays
            dtype={k: "string" for k in KEYS},     # keys as strings
        )
        # normalize poln and spw name:
        df["polarization"] = df["polarization"].map(norm_pol)
        df["spw_name_ms"]  = df["spw_name_ms"].map(norm_spw)
        dfs.append(df)

    stats = pd.concat(dfs, ignore_index=True)
    # remove exact duplicates that can appear across lengths
    stats = stats.drop_duplicates(subset=KEYS, keep="first")
    return stats


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--parquet", required=True, help="input science Parquet")
    ap.add_argument("--csv-root", required=True, help="root dir with length_*/ CSVs")
    ap.add_argument("--output", default="science_scored.parquet",
                    help="output Parquet (skipped in --dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="no file written")
    args = ap.parse_args(argv)

    # 1. science table -------------------------------------------------------
    sci = pd.read_parquet(args.parquet)
    #sci[KEYS] = sci[KEYS].astype("string")
    # ---- normalise key columns in science table ----------------------
    sci = sci.copy()
    sci["polarization"] = sci["polarization"].map(norm_pol)
    sci["spw_name_ms"]  = sci["spw_name_ms"].map(norm_spw)
    sci[KEYS] = sci[KEYS].astype("string")

    print(f"Science table : {len(sci):,} rows  |  {sci[KEYS].drop_duplicates().shape[0]:,} unique key-rows")

    # 2. stats CSVs ----------------------------------------------------------
    stats = load_all_stats(Path(args.csv_root))
    print(f"Stats rows    : {len(stats):,} rows  |  {stats[KEYS].drop_duplicates().shape[0]:,} unique key-rows")

    # 3. inner merge  --------------------------------------------------------
    merged = sci.merge(stats, on=KEYS, how="inner")
    print(f"Merged        : {len(merged):,} rows  |  {merged[KEYS].drop_duplicates().shape[0]:,} unique key-rows")

    # 4. unmatched report ----------------------------------------------------
    unmatched_keys = (
        sci[KEYS]
        .merge(stats[KEYS].drop_duplicates(), on=KEYS, how="left", indicator=True)
        .query('_merge == "left_only"')
        .drop(columns="_merge")
    )
    print(f"Unmatched science rows (dropped): {len(unmatched_keys):,}")
    if len(unmatched_keys):
        print(unmatched_keys.head(10).to_string(index=False))

    # 5. write or dry-run ----------------------------------------------------
    if args.dry_run:
        print("\nDry-run mode: no Parquet written.")
    else:
        merged.to_parquet(args.output, compression="zstd")
        print(f"\n✅  Wrote {args.output} ({len(merged):,} rows)")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

    
