#!/usr/bin/env python
"""Fit one-class (OOD) deep detectors on known-normal signals.

This is the semi-supervised setting that actually matches the ALMA problem:
38,650 labelled-negative spectra are available as in-distribution training data,
and platforming is the out-of-distribution event.  Models are fit once here and
pickled; ``helpers/deep_baselines.py`` loads them at scan time.

The held-out split matters.  Fit only on rows in the *tuning* half so that the
evaluation half stays clean -- the same split you should be using to pick
tau_S, tau_I, r and w.

Examples
--------
    # Fit the default model set on 2000 negative rows, CPU, seed 42
    python fit_deep_models.py \
        --parquet data/qa2_labelled_dataset.parquet \
        --models TranAD,COUTA,TcnED \
        --n-train 2000 \
        --out-dir data/deep_models

    # GPU, longer training
    python fit_deep_models.py --parquet data/qa2_labelled_dataset.parquet \
        --models TranAD --device cuda --epochs 30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.deep_baselines import DEEPOD_MODELS, DeepModelStore  # noqa: E402

log = logging.getLogger("fit_deep_models")


def _extract_signals(df: pd.DataFrame, value_col: str) -> list:
    out = []
    for v in df[value_col].to_numpy():
        arr = np.asarray(v, dtype=np.float64).ravel()
        if arr.size:
            out.append(arr)
    return out


def _guess_column(df: pd.DataFrame, candidates, kind: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(
        f"Could not find a {kind} column. Looked for {candidates}; "
        f"available columns: {list(df.columns)}. Pass it explicitly."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", required=True,
                   help="Labelled dataset produced by create_dataset.py")
    p.add_argument("--models", default="TranAD,COUTA,TcnED",
                   help=f"Comma-separated DeepOD models. Available: {','.join(DEEPOD_MODELS)}")
    p.add_argument("--value-col", default=None,
                   help="Column holding the amplitude array (auto-detected if omitted)")
    p.add_argument("--label-col", default=None,
                   help="Binary label column (auto-detected if omitted)")
    p.add_argument("--split-col", default=None,
                   help="Optional column marking the tuning/eval split; only "
                        "rows equal to --split-value are used for fitting")
    p.add_argument("--split-value", default="tune")
    p.add_argument("--n-train", type=int, default=2000,
                   help="Number of negative rows to sample for fitting")
    p.add_argument("--seq-len", type=int, default=100)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="data/deep_models")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    df = pd.read_parquet(args.parquet)
    value_col = args.value_col or _guess_column(
        df, ["amplitude", "amp", "values", "spectrum", "x", "signal"], "value")
    label_col = args.label_col or _guess_column(
        df, ["label", "is_anomaly", "platforming", "y"], "label")
    log.info("Using value column %r, label column %r", value_col, label_col)

    if args.split_col:
        before = len(df)
        df = df[df[args.split_col] == args.split_value]
        log.info("Restricted to %s == %r: %d -> %d rows",
                 args.split_col, args.split_value, before, len(df))

    negatives = df[df[label_col].astype(int) == 0]
    log.info("%d negative rows available", len(negatives))
    if len(negatives) == 0:
        raise SystemExit("No negative rows found -- check --label-col")

    rng = np.random.default_rng(args.seed)
    n_take = min(args.n_train, len(negatives))
    idx = rng.choice(len(negatives), size=n_take, replace=False)
    signals = _extract_signals(negatives.iloc[idx], value_col)
    log.info("Sampled %d signals (lengths %d-%d)",
             len(signals),
             min(s.size for s in signals),
             max(s.size for s in signals))

    store = DeepModelStore(artifact_dir=args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = {
        "parquet": args.parquet,
        "value_col": value_col,
        "label_col": label_col,
        "split_col": args.split_col,
        "split_value": args.split_value if args.split_col else None,
        "n_train_rows": len(signals),
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "device": args.device,
        "seed": args.seed,
        "models": {},
    }

    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        if model not in DEEPOD_MODELS:
            log.warning("Skipping unknown model %r", model)
            continue
        log.info("=== %s ===", model)
        t0 = time.perf_counter()
        try:
            store.fit(
                model, signals,
                seq_len=args.seq_len,
                epochs=args.epochs,
                device=args.device,
                seed=args.seed,
            )
            manifest["models"][model] = {
                "status": "ok",
                "fit_seconds": round(time.perf_counter() - t0, 2),
                "artifact": store.path_for(model),
            }
        except Exception as exc:
            log.exception("%s failed", model)
            manifest["models"][model] = {"status": "failed", "error": str(exc)}

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("Wrote %s", manifest_path)
    log.info("Now run scoring with e.g. --methods deepod_tranad,deepod_couta "
             "(set RSS_DEEP_ARTIFACTS=%s if not using the default path)",
             args.out_dir)


if __name__ == "__main__":
    main()
