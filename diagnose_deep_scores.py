#!/usr/bin/env python
"""Is the deep detector doing the work, or is the interval adapter?

Symptom that motivates this: several architecturally distinct detectors return
near-identical localization scores, and the numbers do not move between CPU and
GPU.  Both are consistent with ``scores_to_interval`` -- not the detector --
determining the predicted interval.

Three tests:

  1. NULL BASELINE.  Feed random / shuffled / constant score vectors through the
     same adapter.  If a random vector reaches the detector's IoU, the detector
     is contributing nothing and the number is an artifact of the adapter plus
     the anomaly's own geometry.  This is the localization analogue of the
     point-adjustment critique (Kim et al., AAAI 2022).

  2. AGREEMENT.  Rank-correlate the raw score vectors across detectors, and
     measure how often they return the same interval.  Near-identical intervals
     from weakly-correlated score vectors means the adapter is collapsing them.

  3. ORACLE CEILING.  Give the adapter the ground-truth indicator as its score
     vector.  That is the best any detector could do through this adapter; if it
     is far below 1.0, the adapter itself is the bottleneck.

Usage
-----
    python diagnose_deep_scores.py --parquet data/qa2_labelled_dataset.parquet \\
        --methods tsbad_usad,tsbad_tranad,tsbad_m2n2 --n-rows 30

    # no TSB-AD needed -- runs tests 1 and 3 only
    python diagnose_deep_scores.py --synthetic --n-rows 100
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.deep_baselines import scores_to_interval  # noqa: E402


# ---------------------------------------------------------------------------

def iou(p: Tuple[int, int], t: Tuple[int, int]) -> float:
    pa, pb = p
    ta, tb = t
    if pb <= pa:
        return 0.0
    inter = max(0, min(pb, tb) - max(pa, ta) + 1)
    if inter == 0:
        return 0.0
    return inter / (max(pb, tb) - min(pa, ta) + 1)


def make_synthetic(n: int, width: int, snr: float, rng) -> Tuple[np.ndarray, Tuple[int, int]]:
    phi1, phi2 = 1.985, -0.985056
    x = np.zeros(n)
    x[0], x[1] = rng.normal(scale=0.1), rng.normal(scale=0.1)
    for i in range(2, n):
        x[i] = phi1 * x[i - 1] + phi2 * x[i - 2] + rng.normal(scale=0.01)
    x = x / (x.std() + 1e-12) + rng.normal(scale=0.05, size=n)
    a = int(rng.integers(20, n - width - 20))
    b = a + width - 1
    x[a:b + 1] -= snr * x.std()
    return x, (a, b)


# --- null score vectors -----------------------------------------------------

def null_scores(kind: str, x: np.ndarray, rng) -> np.ndarray:
    n = x.size
    if kind == "uniform":
        return rng.random(n)
    if kind == "gaussian":
        return rng.normal(size=n)
    if kind == "constant":
        return np.ones(n)
    if kind == "brownian":            # smooth random walk -- a "plausible" shape
        return np.cumsum(rng.normal(size=n))
    if kind == "abs_resid":           # |x - mean|: no learning at all
        return np.abs(x - x.mean())
    if kind == "abs_diff":            # |first difference|: trivial edge detector
        d = np.abs(np.diff(x, prepend=x[0]))
        return d
    raise ValueError(kind)


NULLS = ("uniform", "gaussian", "constant", "brownian", "abs_resid", "abs_diff")


def oracle_scores(x: np.ndarray, truth: Tuple[int, int]) -> np.ndarray:
    s = np.zeros(x.size)
    s[truth[0]:truth[1] + 1] = 1.0
    return s


# ---------------------------------------------------------------------------

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size < 3:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 1e-12 else np.nan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", default=None)
    p.add_argument("--value-col", default=None)
    p.add_argument("--label-col", default=None)
    p.add_argument("--interval-cols", default="start,end",
                   help="Ground-truth interval columns, comma separated")
    p.add_argument("--methods", default="")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--n-rows", type=int, default=30)
    p.add_argument("--n", type=int, default=1920)
    p.add_argument("--width", type=int, default=50)
    p.add_argument("--snr", type=float, default=2.5)
    p.add_argument("--W", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="deep_score_diagnosis.csv")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---- assemble (signal, truth) pairs -----------------------------------
    rows: List[Tuple[np.ndarray, Tuple[int, int]]] = []
    if args.synthetic or not args.parquet:
        for _ in range(args.n_rows):
            rows.append(make_synthetic(args.n, args.width, args.snr, rng))
        print(f"Synthetic: {len(rows)} rows, n={args.n}, width={args.width}, SNR={args.snr}")
    else:
        df = pd.read_parquet(args.parquet)
        vcol = args.value_col or next(
            c for c in ["amplitude", "amp", "values", "spectrum", "x", "signal"]
            if c in df.columns)
        lcol = args.label_col or next(
            c for c in ["label", "is_anomaly", "platforming", "y"] if c in df.columns)
        c0, c1 = [c.strip() for c in args.interval_cols.split(",")]
        pos = df[df[lcol].astype(int) == 1].head(args.n_rows)
        for _, r in pos.iterrows():
            v = np.asarray(r[vcol], dtype=np.float64).ravel()
            rows.append((v, (int(r[c0]), int(r[c1]))))
        print(f"ALMA: {len(rows)} positive rows from {args.parquet}")

    records = []
    coverage: Dict[str, List[int]] = {}

    # ---- TEST 1: null baselines -------------------------------------------
    print("\n" + "=" * 70)
    print("TEST 1  TRIVIAL BASELINES -- what does the adapter achieve alone?")
    print("=" * 70)
    for kind in NULLS:
        ious = []
        for x, truth in rows:
            s = null_scores(kind, x, rng)
            _, ab = scores_to_interval(x, s, args.W)
            ious.append(iou(ab, truth))
        m = float(np.mean(ious))
        records.append(dict(source="trivial", name=kind, mean_iou=m))
        print(f"  {kind:<12} mean IoU = {m:.4f}")

    # ---- TEST 3: oracle ceiling -------------------------------------------
    print("\n" + "=" * 70)
    print("TEST 3  ORACLE CEILING -- best any detector could do via this adapter")
    print("=" * 70)
    ious = []
    for x, truth in rows:
        _, ab = scores_to_interval(x, oracle_scores(x, truth), args.W)
        ious.append(iou(ab, truth))
    oracle = float(np.mean(ious))
    records.append(dict(source="oracle", name="ground_truth_indicator", mean_iou=oracle))
    print(f"  oracle       mean IoU = {oracle:.4f}")
    if oracle < 0.9:
        print("  WARNING: the adapter cannot recover the interval even when handed")
        print("  the answer. Check --W against the true anomaly width, and min_width.")

    # ---- TEST 2: real detectors -------------------------------------------
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    raw: Dict[str, List[np.ndarray]] = {}
    if methods:
        print("\n" + "=" * 70)
        print("TEST 2  REAL DETECTORS")
        print("=" * 70)
        from helpers.deep_baselines import DEEP_METHODS, _BACKENDS, ensure_numpy2_compat
        ensure_numpy2_compat()

        for key in methods:
            cfg = DEEP_METHODS.get(key)
            if cfg is None:
                print(f"  {key:<26} not registered -- skipping")
                continue
            fn = _BACKENDS[cfg["backend"]]
            ious, vecs, ok_idx = [], [], []
            n_fail, first_err, fail_lens = 0, None, []
            for ri, (x, truth) in enumerate(rows):
                try:
                    s = fn(x, cfg["model"], seed=cfg.get("seed", 42))
                except Exception as exc:
                    # A single bad row must not discard the method.  Short
                    # signals commonly break windowed models (win_size > n).
                    n_fail += 1
                    fail_lens.append(x.size)
                    if first_err is None:
                        first_err = str(exc)[:160]
                    continue
                if s.size != x.size:
                    pad = x.size - s.size
                    s = (np.concatenate([np.full(pad, s[0]), s]) if pad > 0 else s[-x.size:])
                vecs.append(s)
                ok_idx.append(ri)
                _, ab = scores_to_interval(x, s, args.W)
                ious.append(iou(ab, truth))

            n_tot = len(rows)
            if n_fail:
                print(f"  {key:<26} {n_fail}/{n_tot} rows failed "
                      f"(n range {min(fail_lens)}-{max(fail_lens)})")
                print(f"  {'':<26} first error: {first_err}")
            if not ious:
                print(f"  {key:<26} no successful rows -- skipping")
                continue
            raw[key] = vecs
            coverage[key] = ok_idx
            m_ok = float(np.mean(ious))            # over rows it could score
            m_all = m_ok * len(ious) / n_tot       # failures counted as IoU 0
            records.append(dict(source="detector", name=key, mean_iou=m_ok,
                                mean_iou_all_rows=m_all,
                                coverage=len(ious) / n_tot))
            if n_fail:
                print(f"  {key:<26} mean IoU = {m_ok:.4f} (scored rows) | "
                      f"{m_all:.4f} (all rows, failures=0) | "
                      f"coverage {len(ious)/n_tot:.1%}")
            else:
                print(f"  {key:<26} mean IoU = {m_ok:.4f}")

    # ---- agreement between detectors --------------------------------------
    if len(raw) >= 2:
        print("\n  Pairwise Spearman correlation of raw score vectors:")
        keys = list(raw)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                cs = [spearman(a, b) for a, b in zip(raw[keys[i]], raw[keys[j]])]
                print(f"    {keys[i]:<22} vs {keys[j]:<22} rho = "
                      f"{np.nanmean(cs):+.3f}")

        print("\n  Fraction of rows where two detectors return the SAME interval:")
        ivs = {k: [scores_to_interval(x, s, args.W)[1]
                   for (x, _), s in zip(rows, v)] for k, v in raw.items()}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                same = np.mean([a == b for a, b in zip(ivs[keys[i]], ivs[keys[j]])])
                print(f"    {keys[i]:<22} vs {keys[j]:<22} {same:6.1%}")

    if len(coverage) >= 2:
        common = set.intersection(*(set(v) for v in coverage.values()))
        if 0 < len(common) < len(rows):
            print(f"\n  NOTE: methods cover different row subsets. "
                  f"{len(common)}/{len(rows)} rows scored by all of them.")
            print("  Report the common-subset comparison, or report coverage")
            print("  alongside each score -- otherwise a method that skips the")
            print("  hard rows looks better than one that attempts them.")

    # ---- verdict -----------------------------------------------------------
    res = pd.DataFrame(records)
    res.to_csv(args.out, index=False)
    # NOTE: the trivial-baseline rows are labelled "trivial", not "null" --
    # pandas parses the literal string "null" back as NaN on read.

    best_null = res[res.source == "trivial"].mean_iou.max()
    det = res[res.source == "detector"]
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  best null           : {best_null:.4f}")
    print(f"  oracle ceiling      : {oracle:.4f}")
    if len(det):
        best_det = det.mean_iou.max()
        print(f"  best real detector  : {best_det:.4f}")
        margin = best_det - best_null
        print(f"  detector - null     : {margin:+.4f}")
        print()
        if margin < 0.05:
            print("  The detectors are not beating a random/trivial score vector.")
            print("  Their reported IoU reflects the adapter and the anomaly's")
            print("  geometry, not learned structure. DO NOT report these as")
            print("  deep-learning results until this is resolved.")
        else:
            print("  Detectors clear the null by a meaningful margin -- the")
            print("  numbers reflect the models. Report the null row alongside")
            print("  them; it is the honest control and pre-empts the obvious")
            print("  reviewer question.")
    else:
        print("  Run with --methods to compare real detectors against these nulls.")
    print("=" * 70)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()