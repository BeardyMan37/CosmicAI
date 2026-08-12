#!/usr/bin/env python
"""Why can't the oracle recover the interval? Per-row ground-truth audit.

If ``diagnose_deep_scores.py`` reports an oracle ceiling below ~0.98, the
problem is upstream of every detector: the labelled interval and the array
being scored do not agree.  Nothing downstream can be interpreted until this
reads clean.

Checks each positive row for:
  * interval out of bounds  (a < 0, b >= n, a > b)
  * width exceeding the scan cap W
  * width below min_width
  * an apparent constant index offset  (the signature of a buffer trim applied
    to the arrays but not to the stored indices)
  * whether the labelled interval actually contains a step in the signal

Usage
-----
    python check_ground_truth.py --parquet data/qa2_labelled_dataset.parquet \\
        --interval-cols start,end --W 200
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _guess(df, cands, kind):
    for c in cands:
        if c in df.columns:
            return c
    raise SystemExit(f"No {kind} column. Tried {cands}; have {list(df.columns)}")


def step_offset(x: np.ndarray, a: int, b: int, search: int = 60):
    """Where is the strongest step near the labelled boundary?

    Returns (offset_at_a, offset_at_b): the shift, in channels, from the
    labelled boundary to the largest |first difference| within +/-search.
    A consistent non-zero value across rows means the indices are offset.
    """
    n = x.size
    d = np.abs(np.diff(x, prepend=x[0]))
    outs = []
    for edge in (a, b):
        lo, hi = max(0, edge - search), min(n, edge + search + 1)
        if hi - lo < 3:
            outs.append(np.nan)
            continue
        outs.append(int(lo + np.argmax(d[lo:hi]) - edge))
    return tuple(outs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", required=True)
    p.add_argument("--value-col", default=None)
    p.add_argument("--label-col", default=None)
    p.add_argument("--interval-cols", default="start,end")
    p.add_argument("--W", type=int, default=200)
    p.add_argument("--min-width", type=int, default=2)
    p.add_argument("--search", type=int, default=60)
    p.add_argument("--out", default="ground_truth_audit.csv")
    args = p.parse_args()

    df = pd.read_parquet(args.parquet)
    vcol = args.value_col or _guess(
        df, ["amplitude", "amp", "values", "spectrum", "x", "signal"], "value")
    lcol = args.label_col or _guess(
        df, ["label", "is_anomaly", "platforming", "y"], "label")
    c0, c1 = [c.strip() for c in args.interval_cols.split(",")]
    for c in (c0, c1):
        if c not in df.columns:
            raise SystemExit(f"Column {c!r} not found. Have: {list(df.columns)}")

    pos = df[df[lcol].astype(int) == 1]
    print(f"{len(pos)} positive rows; value={vcol!r} label={lcol!r} "
          f"interval=({c0},{c1})\n")

    recs = []
    for i, (_, r) in enumerate(pos.iterrows()):
        x = np.asarray(r[vcol], dtype=np.float64).ravel()
        n = x.size
        a, b = int(r[c0]), int(r[c1])
        w = b - a + 1
        reasons = []
        if a < 0 or b >= n:
            reasons.append("out_of_bounds")
        if a > b:
            reasons.append("inverted")
        if w > args.W:
            reasons.append("wider_than_W")
        if w < args.min_width:
            reasons.append("narrower_than_min_width")

        oa = ob = np.nan
        if not reasons:
            oa, ob = step_offset(x, a, b, args.search)

        # best achievable IoU given only the W / min_width caps
        if "out_of_bounds" in reasons or "inverted" in reasons:
            ceil = 0.0
        elif w > args.W:
            ceil = args.W / w
        elif w < args.min_width:
            ceil = w / args.min_width
        else:
            ceil = 1.0

        recs.append(dict(row=i, n=n, a=a, b=b, width=w,
                         reason=",".join(reasons) or "ok",
                         offset_a=oa, offset_b=ob, ceiling=ceil))

    au = pd.DataFrame(recs)
    au.to_csv(args.out, index=False)

    print("=" * 66)
    print("PER-ROW STATUS")
    print("=" * 66)
    print(au.reason.value_counts().to_string())

    print("\n" + "=" * 66)
    print("WIDTH vs CAPS")
    print("=" * 66)
    print(au.width.describe().round(1).to_string())
    print(f"\n  wider than W={args.W}        : {(au.width > args.W).sum()}")
    print(f"  narrower than {args.min_width}          : "
          f"{(au.width < args.min_width).sum()}")
    print(f"  implied ceiling from caps  : {au.ceiling.mean():.4f}")

    ok = au[au.reason == "ok"].dropna(subset=["offset_a", "offset_b"])
    if len(ok):
        print("\n" + "=" * 66)
        print("INDEX ALIGNMENT -- distance from labelled edge to nearest step")
        print("=" * 66)
        for col in ("offset_a", "offset_b"):
            v = ok[col]
            print(f"  {col}: median={v.median():+.1f}  mean={v.mean():+.1f}  "
                  f"|off|<=2 in {(v.abs() <= 2).mean():.1%} of rows")
        allo = np.concatenate([ok.offset_a.to_numpy(), ok.offset_b.to_numpy()])
        med = float(np.median(allo))
        # Robust spread: MAD is not dragged around by the rows where the
        # strongest step is some other instrumental artefact.
        mad = float(np.median(np.abs(allo - med)))
        agree = float(np.mean(np.abs(allo) <= 2))
        near_med = float(np.mean(np.abs(allo - med) <= 3))
        print(f"  median offset = {med:+.0f}  MAD = {mad:.1f}  "
              f"({near_med:.0%} of edges within 3 of the median)")
        print()
        if abs(med) >= 3 and mad <= 5 and near_med >= 0.5:
            print(f"  CONSISTENT OFFSET of about {med:+.0f} channels "
                  f"({near_med:.0%} of edges agree).")
            print( "  The arrays were almost certainly trimmed without shifting")
            print( "  the stored indices. Fix the dataset build, or add the offset")
            print( "  when loading -- do NOT compensate inside the scorers.")
        elif agree < 0.5:
            print(f"  Only {agree:.0%} of labelled edges sit within 2 channels of a")
            print( "  step. Either the labels are in different units (frequency")
            print( "  rather than channel index?) or many rows have several")
            print( "  instrumental issues and the labelled one is not the largest.")
        else:
            print(f"  Alignment looks fine ({agree:.0%} of edges within 2 channels).")
            print( "  The ceiling loss is coming from the width caps above, not")
            print( "  from indexing.")

    print(f"\nWritten to {args.out}")
    print("\nInspect the worst rows with:")
    print(f"  python -c \"import pandas as pd; d=pd.read_csv('{args.out}'); "
          f"print(d[d.ceiling<1].head(20))\"")


if __name__ == "__main__":
    main()
