from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse, logging, time, os, math
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

from helpers.config import set_kernel_kind, ref_freq
from helpers.io_preprocess import load_data_by_length
from helpers.warmup import warmup_numba_and_caches
from helpers.superres import sr_factor, superresolve_ranges, superresolve

from benchmark_regressor_on_syth import (
    _run_one, ALL_METHODS, _resolve_methods, iou as compute_iou
)



# ---------------------------------------------------------------------------
# Per-row worker — called in a subprocess
# ---------------------------------------------------------------------------

def _score_one_row(
    args_tuple: tuple,
) -> List[dict]:
    """
    Score a single row with all requested methods.
    Returns a list of dicts (one per method).
    """
    from benchmark_regressor_on_syth import _run_one, ALL_METHODS
    from benchmark_regressor_on_syth import iou as compute_iou
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    (row_idx, x_list, freqs_list, gt_label, gt_start, gt_end,
     method_names, W, R, sr_factor_val) = args_tuple

    x     = np.asarray(x_list,     dtype=np.float64)
    freqs = np.asarray(freqs_list, dtype=np.float64)
    results = []
    for name in method_names:
        cfg = dict(ALL_METHODS[name])
        cfg["sr_factor"] = sr_factor_val

        t0 = time.perf_counter()
        try:
            score, (a, b) = _run_one(x, freqs, W, R, cfg)
        except Exception as e:
            import traceback
            print(f"[worker] row={row_idx} method={name} FAILED: {e}")
            traceback.print_exc()
            score, a, b = 0.0, 0, -1
        dt_ms = (time.perf_counter() - t0) * 1000.0

        iou_val = float(compute_iou(a, b, gt_start, gt_end)) \
                  if gt_label and b > a else 0.0

        results.append({
            "row_idx":    row_idx,
            "method":     name,
            "label":      gt_label,
            "gt_start":   gt_start,
            "gt_end":     gt_end,
            "pred_a":     int(a),
            "pred_b":     int(b),
            "score":      float(score),
            "iou_val":    iou_val,
            "time_ms":    round(dt_ms, 4),
            "n":          len(x),
            "W":          W,
            "R":          R,
            "sr_factor":  sr_factor_val,
        })
    return results


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

try:
    from helpers.deep_baselines import DEEP_METHODS as _DEEP_METHODS
    _DEEP_KEYS = set(_DEEP_METHODS)
except Exception:
    _DEEP_KEYS = set()

# stumpy: numba/OpenMP conflict in subprocesses.
# deep baselines: torch state does not survive fork reliably, and each worker
# would otherwise reload the model artifact.
SERIAL_METHODS = {"stumpy"} | _DEEP_KEYS

def run_score_experiment(
    *,
    parquet_path: str,
    method_names: List[str],
    out_dir: str = "data/score_experiment",
    max_rows: Optional[int] = None,
    max_workers: Optional[int] = None,
    balance: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Parallelised scoring experiment on a labelled real dataset.

    For every row in the dataset, runs all requested methods and records:
      - predicted interval (pred_a, pred_b)
      - epidemic score
      - IoU with ground truth interval (for positive rows)
      - runtime in ms
      - W, R, sr_factor used

    The output CSV contains one row per (spectrum, method) combination
    and can be used downstream for any TP/FP/TN/FN decision rule
    without re-running the methods.

    Parameters
    ----------
    parquet_path : str
        Path to the labelled dataset parquet.
    method_names : list of str
        Method keys from ALL_METHODS to evaluate.
    out_dir : str
        Output directory for CSV results.
    max_rows : int or None
        If set, subsample to this many rows (balanced if balance=True).
    max_workers : int or None
        Number of parallel workers. None = os.cpu_count().
    balance : bool
        If True and max_rows is set, sample equal positive/negative rows.
    seed : int
        Random seed for sampling.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load ----
    if parquet_path.endswith(".csv"):
        df = pd.read_csv(parquet_path)
        for col in ["amplitude", "frequency_array"]:
            df[col] = df[col].apply(
                lambda s: np.fromstring(s.strip("[]"), sep=" ")
                if isinstance(s, str) else np.asarray(s, dtype=np.float64)
            )
        df["label"] = df["label"].astype(bool)
    else:
        df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows  "
          f"(label=True: {df['label'].sum()}, "
          f"label=False: {(~df['label']).sum()})")

    if max_rows is not None:
        if balance:
            pos = df[df["label"] == True]
            neg = df[df["label"] == False]
            n_pos = min(len(pos), max_rows // 2)
            n_neg = min(len(neg), max_rows - n_pos)
            df = pd.concat([
                pos.sample(n=n_pos, random_state=seed),
                neg.sample(n=n_neg, random_state=seed),
            ]).sort_values("label", ascending=False).reset_index(drop=True)
            print(f"Balanced sample: {n_pos} positive + {n_neg} negative = {len(df)} rows")
        else:
            df = df.head(max_rows).reset_index(drop=True)

    # ---- Build argument tuples for parallel workers ----
    work_items = []
    for idx, row in df.iterrows():
        x     = np.asarray(row["amplitude"],       dtype=np.float64)
        freqs = np.asarray(row["frequency_array"] / 1e9, dtype=np.float64)
        n     = x.size

        freq_step = abs(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        r  = ref_freq / (freq_step if freq_step > 0 else 1.0)
        W  = int(round(max(3, min(r, n / 16))))
        R  = 3 * W
        sr = max(1, 2 ** math.ceil(
            math.log2(max(1, math.ceil((n + 1) / 450)))))

        work_items.append((
            int(idx),
            x.tolist(),
            freqs.tolist(),
            bool(row["label"]),
            int(row["start"]),
            int(row["end"]),
            method_names,
            W, R, sr,
        ))

    # ---- Parallel execution ----
    parallel_methods = [m for m in method_names if m not in SERIAL_METHODS]
    serial_methods   = [m for m in method_names if m in SERIAL_METHODS]

    t_start = time.perf_counter()
    all_results = []
    n_done = 0

    print(f"Scoring {len(work_items)} rows × {len(method_names)} methods "
          f"with {max_workers or os.cpu_count()} workers...")

    if parallel_methods:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_score_one_row, 
                                   (*item[:6], parallel_methods, *item[7:])): item[0]
                       for item in work_items}
            for future in as_completed(futures):
                row_idx = futures[future]
                try:
                    rows = future.result()
                    all_results.extend(rows)
                except Exception as e:
                    logging.warning("Row %d failed: %s", row_idx, e)
                n_done += 1
                if n_done % 100 == 0 or n_done == len(work_items):
                    elapsed = time.perf_counter() - t_start
                    print(f"  {n_done}/{len(work_items)} rows done  "
                        f"({elapsed:.1f}s elapsed)")

    total_time = time.perf_counter() - t_start
    print(f"Total time: {total_time:.2f}s")

    if serial_methods:
        print(f"Running serial methods: {serial_methods}")
        from benchmark_regressor_on_syth import _run_one, ALL_METHODS
        from benchmark_regressor_on_syth import iou as compute_iou

        for item in work_items:
            (row_idx, x_list, freqs_list, gt_label, gt_start, gt_end,
             _, W, R, sr_factor_val) = item

            x     = np.asarray(x_list,     dtype=np.float64)
            freqs = np.asarray(freqs_list, dtype=np.float64)

            for name in serial_methods:
                cfg = dict(ALL_METHODS[name])
                cfg["sr_factor"] = sr_factor_val

                t0 = time.perf_counter()
                try:
                    score, (a, b) = _run_one(x, freqs, W, R, cfg)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    score, a, b = 0.0, 0, -1
                dt_ms = (time.perf_counter() - t0) * 1000.0

                iou_val = float(compute_iou(a, b, gt_start, gt_end)) \
                          if gt_label and b > a else 0.0

                all_results.append({
                    "row_idx":   row_idx,
                    "method":    name,
                    "label":     gt_label,
                    "gt_start":  gt_start,
                    "gt_end":    gt_end,
                    "pred_a":    int(a),
                    "pred_b":    int(b),
                    "score":     float(score),
                    "iou_val":   iou_val,
                    "time_ms":   round(dt_ms, 4),
                    "n":         len(x),
                    "W":         W,
                    "R":         R,
                    "sr_factor": sr_factor_val,
                })

    # ---- Save ----
    results_df = pd.DataFrame(all_results).sort_values(
        ["row_idx", "method"]).reset_index(drop=True)

    out_path = os.path.join(out_dir, "score_experiment_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(results_df)} rows)")

    # ---- Quick summary ----
    summary = results_df.groupby("method").agg(
        mean_score    = ("score",   "mean"),
        mean_iou_pos  = ("iou_val", lambda x: x[results_df.loc[x.index, "label"]].mean()),
        mean_time_ms  = ("time_ms", "mean"),
        median_time_ms= ("time_ms", "median"),
    ).round(4)
    print("\n=== Summary ===")
    print(summary.to_string())

    summary_path = os.path.join(out_dir, "score_experiment_summary.csv")
    summary.to_csv(summary_path)
    print(f"Saved: {summary_path}")

    return results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parallelised scoring experiment on labelled real data."
    )
    p.add_argument("--parquet",      required=True,
                   help="Path to labelled dataset parquet.")
    p.add_argument("--methods",      default="mean,poly_deg1,poly_deg2,nwkr_gaussian,nwkr_laplace",
                   help="Comma-separated method keys.")
    p.add_argument("--out-dir",      default="data/score_experiment")
    p.add_argument("--max-rows",     type=int, default=None)
    p.add_argument("--workers",      type=int, default=None)
    p.add_argument("--no-balance",   action="store_true",
                   help="Do not balance positive/negative rows.")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--log-level",    default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    args   = _build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s: %(message)s")

    method_names = [m.strip() for m in args.methods.split(",")]
    unknown = [m for m in method_names if m not in ALL_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. "
                         f"Available: {sorted(ALL_METHODS)}")

    run_score_experiment(
        parquet_path = args.parquet,
        method_names = method_names,
        out_dir      = args.out_dir,
        max_rows     = args.max_rows,
        max_workers  = args.workers,
        balance      = not args.no_balance,
        seed         = args.seed,
    )