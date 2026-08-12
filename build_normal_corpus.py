#!/usr/bin/env python
"""Build the known-normal training corpus used by semi-supervised TSB-AD models.

Semi-supervised detectors (USAD, TranAD, AnomalyTransformer, CNN, M2N2, ...)
need a training array.  TSB-AD's own protocol trains on a prefix of the series
being scored, which is unsafe here: a platforming anomaly can sit inside that
prefix.  Instead, build the corpus from labelled-negative rows.

    python build_normal_corpus.py \\
        --parquet data/qa2_labelled_dataset.parquet \\
        --n-rows 300 --out data/normal_corpus.npy

Then in your run script, before scoring:

    from helpers.deep_baselines import load_normal_corpus_npy
    load_normal_corpus_npy("data/normal_corpus.npy")

or set the env var and let score_experiment pick it up:

    export RSS_NORMAL_CORPUS=data/normal_corpus.npy
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _guess(df: pd.DataFrame, cands, kind: str) -> str:
    for c in cands:
        if c in df.columns:
            return c
    raise SystemExit(f"No {kind} column found. Tried {cands}; have {list(df.columns)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--value-col", default=None)
    p.add_argument("--label-col", default=None)
    p.add_argument("--split-col", default=None,
                   help="Use only rows where this column == --split-value "
                        "(keep the eval half clean)")
    p.add_argument("--split-value", default="tune")
    p.add_argument("--n-rows", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/normal_corpus.npy")
    args = p.parse_args()

    df = pd.read_parquet(args.parquet)
    vcol = args.value_col or _guess(
        df, ["amplitude", "amp", "values", "spectrum", "x", "signal"], "value")
    lcol = args.label_col or _guess(
        df, ["label", "is_anomaly", "platforming", "y"], "label")

    if args.split_col:
        df = df[df[args.split_col] == args.split_value]
        print(f"Restricted to {args.split_col}=={args.split_value!r}: {len(df)} rows")

    neg = df[df[lcol].astype(int) == 0]
    print(f"{len(neg)} negative rows available")
    if not len(neg):
        raise SystemExit("No negatives found -- check --label-col")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(neg), size=min(args.n_rows, len(neg)), replace=False)

    sigs = []
    for v in neg.iloc[idx][vcol].to_numpy():
        a = np.asarray(v, dtype=np.float64).ravel()
        if a.size >= 8:
            sigs.append(a)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, np.array(sigs, dtype=object), allow_pickle=True)
    print(f"Saved {len(sigs)} signals "
          f"(lengths {min(s.size for s in sigs)}-{max(s.size for s in sigs)}, "
          f"{sum(s.size for s in sigs)} timestamps) -> {args.out}")
    print(f"\nUse it with:  export RSS_NORMAL_CORPUS={args.out}")


if __name__ == "__main__":
    main()
