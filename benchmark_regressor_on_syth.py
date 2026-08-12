"""
benchmark_regressor_on_syth.py
================================
Benchmarking harness for CosmicAI scan methods on synthetic spectra.

Two benchmark modes
-------------------
runtime     -- measure per-row wall-clock time vs a chosen variable (n, w, r)
parameter   -- measure detection quality (localization score, coverage, parsimony)
               across a grid of (w, error_param) combinations

Usage examples
--------------
# Runtime scaling vs spectrum length n:
python benchmark_regressor_on_syth.py runtime --variable n --reps 5

# Runtime scaling vs window width w:
python benchmark_regressor_on_syth.py runtime --variable w --n 1000

# Parameter (detection quality) benchmark, all w values, all error params:
python benchmark_regressor_on_syth.py parameter

# Parameter benchmark, specific w and error params:
python benchmark_regressor_on_syth.py parameter --w 25 50 --snr 2.5 --noise 0.05

# Choose which methods to include (comma-separated):
python benchmark_regressor_on_syth.py parameter --methods mean,poly_deg1,nwkr_gaussian

# Save outputs to a specific directory:
python benchmark_regressor_on_syth.py runtime --variable n --out-dir results/runtime

Method keys (pass to --methods)
--------------------------------
  mean
  poly_deg1
  poly_deg2
  krr_gaussian
  krr_laplace
  nwkr_gaussian
  nwkr_laplace
  nwkr_gaussian_naive
  nwkr_laplace_naive
  ruptures_kernelcpd
  stumpy
  kats_cusum
"""
from __future__ import annotations

# import logging
# for logger_name in ["kats", "kats.detectors", "kats.models", "kats.utils"]:
#     logging.getLogger(logger_name).setLevel(logging.ERROR)

import argparse
import logging
import math
import multiprocessing as mp
import os
import queue
import time
from typing import Dict, List, Optional, Tuple
from math import lgamma

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stumpy
import ruptures as rpt
from helpers.superres import refine_all_windows_exact_for_length, superresolve
from gen_ar2_data import generate_ar2_dataset
# from kats.consts import TimeSeriesData
# from kats.detectors.cusum_detection import CUSUMDetector

from helpers.config import KernelKind, ref_freq, set_kernel_kind
from helpers.kernels import get_kernel_and_denom
from helpers.predictors import predict_on_idxs
import helpers.regressors as rg
from helpers.scan import scan_row_with_nwkr, scan_row_with_nwkr_naive
from helpers.scoring import calculate_gaussian_sra_with_nd, calculate_laplace_sra_fast
from gen_synth_data import SpectrumSpec, generate_synthetic_dataset



# ---------------------------------------------------------------------------
# Stumpy warmup (run once at import time)
# ---------------------------------------------------------------------------

def _warmup_stumpy(n: int = 100, m: int = 10) -> None:
    x = np.random.default_rng(0).normal(size=n).astype(np.float64)
    stumpy.stump(x, m=m, ignore_trivial=True, normalize=True)

_warmup_stumpy()


# ---------------------------------------------------------------------------
# All available method definitions
# ---------------------------------------------------------------------------

ALL_METHODS: Dict[str, dict] = {
    "mean": {
        "family": "mean",
        "params": {},
        "mode": "simple",
    },
    "poly_deg1": {
        "family": "poly",
        "params": {"degree": 1, "reg": 1e-8},
        "mode": "simple",
    },
    "poly_deg2": {
        "family": "poly",
        "params": {"degree": 2, "reg": 1e-8},
        "mode": "simple",
    },
    "krr_gaussian": {
        "family": "krr",
        "params": {"kernel": "rbf", "kernel_param_scale": 8.0, "reg": 1e-1},
        "mode": "simple",
    },
    "krr_laplace": {
        "family": "krr",
        "params": {"kernel": "laplace", "kernel_param_scale": 8.0, "reg": 1e-1},
        "mode": "simple",
    },
    "nwkr_gaussian": {
        "mode": "nwkr",
        "family": "optimized",
        "kernel_kind": "gaussian",
        "which": "unmasked_varlen",
        "buffer": 0,
        "sr_factor": 1,
    },
    "nwkr_laplace": {
        "mode": "nwkr",
        "family": "optimized",
        "kernel_kind": "laplace",
        "which": "unmasked_varlen",
        "buffer": 0,
        "sr_factor": 1,
    },
    "nwkr_gaussian_naive": {
        "mode": "nwkr",
        "family": "naive",
        "kernel_kind": "gaussian",
        "which": "unmasked_varlen",
        "buffer": 0,
        "sr_factor": 1,
    },
    "nwkr_laplace_naive": {
        "mode": "nwkr",
        "family": "naive",
        "kernel_kind": "laplace",
        "which": "unmasked_varlen",
        "buffer": 0,
        "sr_factor": 1,
    },
    "nwkr_gaussian_first_start": {
        "mode": "nwkr_first_start",
        "family": "optimized",
        "kernel_kind": "gaussian",
        "which": "unmasked_varlen",
    },
    "nwkr_laplace_first_start": {
        "mode": "nwkr_first_start",
        "family": "optimized",
        "kernel_kind": "laplace",
        "which": "unmasked_varlen",
    },
    "ruptures_kernelcpd": {
        "mode": "kernelcpd_rbf_c",
        "min_size": 5,
        "model": "l2",
    },
    "stumpy": {
        "mode": "stumpy",
        "m": None,
        "ignore_trivial": True,
    },
    # "kats_cusum": {
    #     "mode": "kats_cusum",
    #     "alpha": 0.01,
    # },
    "bocpd": {
        "mode": "bocpd",
        "hazard_lambda": 50.0,
    },
    "binseg": {
        "mode": "binseg",
        "min_size": 5,
        "model": "l2",
    },
    "clasp": {
        "mode": "clasp",
        "min_seg_size": 5,
        "n_cps": 2,
    },
    "lrt": {
        "mode": "lrt",
        "min_seg_size": 5,
    },
    "capa": {
        "mode": "capa",
        "beta_scale":       8.0,
        "beta_prime_scale": 6.0,
        "min_seg_len":      2,
    },
}

# --- Fast NWKR-background change-point baseline ------------------------------
try:
    from helpers.cpd_nwkr import CPDF_METHODS as _CPDF_METHODS
    ALL_METHODS.update(_CPDF_METHODS)
except Exception as _cpdf_exc:  # pragma: no cover
    logging.getLogger(__name__).debug("cpdf baselines unavailable: %s", _cpdf_exc)

# --- Deep / OOD baselines (TSB-AD, DeepOD) ------------------------------------
# Optional: only registered if the adapter module imports cleanly, so the file
# still works in environments without torch installed.
try:
    from helpers.deep_baselines import DEEP_METHODS as _DEEP_METHODS
    ALL_METHODS.update(_DEEP_METHODS)
    _DEEP_METHOD_KEYS = set(_DEEP_METHODS)
except Exception as _deep_exc:  # pragma: no cover
    _DEEP_METHOD_KEYS = set()
    logging.getLogger(__name__).debug("deep baselines unavailable: %s", _deep_exc)

# Methods that are known to be very slow and need a hard timeout guard
_NAIVE_METHODS = {"nwkr_gaussian_naive", "nwkr_laplace_naive"}

# Default method subsets for each benchmark mode
_DEFAULT_RUNTIME_METHODS = [
    "mean", "poly_deg1", "poly_deg2",
    "nwkr_gaussian", "nwkr_laplace",
    "nwkr_gaussian_naive", "nwkr_laplace_naive",
]
_DEFAULT_PARAMETER_METHODS = [
    "mean", "poly_deg1", "poly_deg2",
    "nwkr_gaussian", "nwkr_laplace",
]


# ---------------------------------------------------------------------------
# Utility / metric helpers
# ---------------------------------------------------------------------------

def interval_length(a: int, b: int) -> int:
    return max(0, b - a + 1)


def interval_intersection(ab: Tuple[int, int], cd: Tuple[int, int]) -> int:
    lo = max(ab[0], cd[0])
    hi = min(ab[1], cd[1])
    return max(0, hi - lo + 1)


def localization_metrics(
    pred_ab: Tuple[int, int],
    gt_ab: Tuple[int, int],
) -> Tuple[float, float, float]:
    """Return (recall/coverage, precision/parsimony, loc_score)."""
    inter = interval_intersection(pred_ab, gt_ab)
    gt_len = interval_length(*gt_ab)
    pred_len = interval_length(*pred_ab)

    if gt_len == 0 or pred_len == 0:
        return 0.0, 0.0, 0.0

    recall = inter / gt_len
    precision = inter / pred_len
    loc_score = (recall * precision) ** 0.5
    return recall, precision, loc_score


def _resolve_krr_params(kw: dict) -> dict:
    kw = dict(kw)
    if "kernel_param" not in kw and "kernel_param_scale" in kw:
        s = max(float(kw.pop("kernel_param_scale")), 1e-12)
        kw["kernel_param"] = 1.0 / s
    kw.pop("kernel_param_scale", None)
    return kw


# ---------------------------------------------------------------------------
# Per-family SSE helper
# ---------------------------------------------------------------------------

def _sse_subset_krr(
    x: np.ndarray, idxs: np.ndarray, K_full: Optional[np.ndarray], reg: float,
    kernel: str, kernel_param: float,
) -> float:
    if idxs.size == 0:
        return 0.0
    if K_full is not None:
        Kss = K_full[np.ix_(idxs, idxs)].astype(np.float64)
        A = Kss.copy()
        A.flat[:: A.shape[0] + 1] += reg
        alpha = np.linalg.solve(A, x[idxs])
        resid = x[idxs] - Kss @ alpha
        return float(resid @ resid)
    preds = rg.predict_on_idxs_krr(x, idxs, kernel=kernel, kernel_param=kernel_param, reg=reg)
    resid = x[idxs] - preds
    return float(resid @ resid)


def sse_subset(
    x: np.ndarray, idxs: np.ndarray,
    family: str, family_kwargs: dict,
    K_full: Optional[np.ndarray] = None,
) -> float:
    if idxs.size == 0:
        return 0.0
    fam = family.lower()
    if fam == "mean":
        mu = x[idxs].mean()
        resid = x[idxs] - mu
        return float(resid @ resid)
    if fam == "poly":
        degree = int(family_kwargs.get("degree", 1))
        reg = float(family_kwargs.get("reg", 1e-8))
        preds = rg.predict_on_idxs_poly(x, idxs, degree=degree, reg=reg)
        resid = x[idxs] - preds
        return float(resid @ resid)
    if fam == "krr":
        kw = _resolve_krr_params(family_kwargs)
        return _sse_subset_krr(
            x, idxs, K_full, float(kw.get("reg", 1e-1)),
            kw.get("kernel", "rbf"), float(kw.get("kernel_param", 1.0)),
        )
    raise ValueError(f"Unknown family {family!r}")


# ---------------------------------------------------------------------------
# Per-mode scan dispatchers
# ---------------------------------------------------------------------------

def scan_row_with_family(
    x: np.ndarray, W: int, family: str, family_kwargs: dict,
) -> Tuple[float, Tuple[int, int]]:
    n = x.size
    if n == 0:
        return -np.inf, (0, 0)

    fam = family.lower()
    max_W = int(min(W, n))
    if fam == "krr":
        family_kwargs = _resolve_krr_params(family_kwargs)

    if fam == "mean":
        pre = rg.mean_precompute(x)
        sra_all = rg.mean_window_sse(pre, 0, n - 1)
    elif fam == "poly":
        degree = int(family_kwargs.get("degree", 1))
        reg = float(family_kwargs.get("reg", 1e-8))
        pre = rg.poly_precompute(x, degree=degree, reg=reg)
        sra_all = rg.poly_window_sse(pre, 0, n - 1)
    else:
        sra_all, *_ = rg.build_regressor_sra(x, family, **family_kwargs)

    if sra_all <= 1e-18:
        return 0.0, (0, 0)

    best_score = -np.inf
    best_ab = (0, 0)
    idx_all = np.arange(n, dtype=np.int32)
    K_full_row = None

    for m in range(1, max_W + 1):
        for a in range(0, n - m + 1):
            b = a + m - 1

            if fam == "mean":
                sse_in  = rg.mean_window_sse(pre, a, b)
                sse_out = rg.mean_outside_sse(pre, a, b)
            elif fam == "poly":
                sse_in  = rg.poly_window_sse(pre, a, b)
                sse_out = rg.poly_outside_sse(pre, a, b)
            else:
                inside  = idx_all[a:b + 1]
                outside = np.concatenate([idx_all[:a], idx_all[b + 1:]])
                if fam == "krr" and K_full_row is None:
                    _, _, _, K_full_row, _, _, _ = rg.build_regressor_sra(x, "krr", **family_kwargs)
                sse_in  = sse_subset(x, inside,  fam, family_kwargs, K_full=K_full_row)
                sse_out = sse_subset(x, outside, fam, family_kwargs, K_full=K_full_row)

            score = 1.0 - (sse_in + sse_out) / sra_all
            if score > best_score:
                best_score = score
                best_ab = (a, b)

    return float(best_score), best_ab


def _make_nwkr_params(
    x: np.ndarray, freqs: np.ndarray, W: int, R: int,
    kernel_kind: str, which: str, buffer: int, sr_factor: int,
    naive: bool,
) -> tuple:
    if freqs is None:
        freq_step = float(ref_freq) / float(max(1, W))
        freqs = np.arange(x.size, dtype=np.float64) * freq_step
    set_kernel_kind(kernel_kind)
    # params = (0, x, [], [], freqs, buffer, sr_factor, None, None, None, False, True, False)
    params = (0, x, [], [], freqs, buffer, sr_factor, W, R, 2*W, False, True, False)
    return params


def _unpack_nwkr_result(x, freqs, buffer, out: tuple, which: str, sr_factor: int) -> Tuple[float, Tuple[int, int]]:
    windows_sr_masked = np.asarray([out[1]])
    windows_sr_unmasked = np.asarray([out[8]])
    windows_sr_fixed = np.asarray([out[13]])
    ws = np.asarray([out[6]])
    range_caps = np.asarray([out[7]])

    if sr_factor > 1:
        windows_exact_masked, windows_exact_unmasked, windows_exact_fixed = refine_all_windows_exact_for_length(
            x, freqs, windows_sr_masked, windows_sr_unmasked, windows_sr_fixed,
            [], ws, range_caps, sr_factor, buffer)
    else:
        windows_exact_masked   = windows_sr_masked
        windows_exact_unmasked = windows_sr_unmasked
        windows_exact_fixed    = windows_sr_fixed
    if which == "masked_varlen":
        return float(out[2]), (int(windows_exact_masked[0][0]), int(windows_exact_masked[0][1]))
    if which == "unmasked_varlen":
        return float(out[9]), (int(windows_exact_unmasked[0][0]), int(windows_exact_unmasked[0][1]))
    if which == "fixed":
        return float(out[14]), (int(windows_exact_fixed[0][0]), int(windows_exact_fixed[0][1]))
    raise ValueError(f"Unknown which={which!r}")


def scan_row_nwkr(
    x: np.ndarray, freqs: np.ndarray, W: int, R: int,
    kernel_kind: str, which: str, buffer: int, sr_factor: int,
    naive: bool = False,
) -> Tuple[float, Tuple[int, int]]:
    x = x.astype(np.float64)
    if x.size == 0:
        return -np.inf, (0, 0)
    x_2d     = x.reshape(1, -1)
    x_sr     = superresolve(x_2d,     factor=sr_factor)[0]

    if freqs is not None:
        freqs_2d = freqs.reshape(1, -1)
        freqs_sr = superresolve(freqs_2d, factor=sr_factor)[0]
    else:
        freqs_sr = freqs
    params = _make_nwkr_params(x_sr, freqs_sr, W, R, kernel_kind, which, buffer // sr_factor, sr_factor, naive)
    fn = scan_row_with_nwkr_naive if naive else scan_row_with_nwkr
    out = fn(params)
    return _unpack_nwkr_result(x_2d, freqs_2d, buffer, out, which, sr_factor)


def _epidemic_score(x: np.ndarray, a: int, b: int) -> float:
    """
    Score an interval [a,b] using the same formula as NWKR:
        1 - (SSE_in + SSE_out) / SRA
    where the baseline model is a constant (mean) fit on each segment.
    This makes the score directly comparable to the NWKR score in [0,1].
    SRA = SSE of global mean fit.
    """
    n = x.size
    if a > b or n == 0:
        return -np.inf
    mu_all = x.mean()
    sra = float(np.sum((x - mu_all) ** 2))
    if sra < 1e-18:
        return 0.0
    ins = x[a:b + 1]
    out = np.concatenate([x[:a], x[b + 1:]])
    sse_in  = float(np.sum((ins - ins.mean()) ** 2)) if ins.size > 0 else 0.0
    sse_out = float(np.sum((out - out.mean()) ** 2)) if out.size > 0 else 0.0
    return 1.0 - (sse_in + sse_out) / sra


def scan_row_kernelcpd(
    x: np.ndarray, W: int,
    min_size: int = 5,
    model: str = "l2",
) -> Tuple[float, Tuple[int, int]]:
    """
    Ruptures epidemic detector with NWKR-comparable score.

    Uses the l2 (least-squares) cost — equivalent to mean SSE, directly
    matching the mean-model baseline in _epidemic_score.  Searches all
    (cp1, cp2) pairs returned by ruptures within the W constraint, scores
    each with _epidemic_score, and returns the best.

    The l2 model is preferred over rbf for step detection because:
    - It directly minimises SSE, matching our score definition.
    - It's O(n) per segment vs O(n^2) for rbf.
    - Step signals are best described by piecewise-constant means, not kernels.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2 * min_size + 2:
        return 0.0, (0, -1)

    signal = x.reshape(-1, 1)
    algo = rpt.KernelCPD(kernel="linear", min_size=int(min_size), jump=1).fit(signal)
    # gamma = 1.0 / max(float(W), 1e-12)
    # algo = rpt.KernelCPD(kernel="rbf", min_size=int(min_size),
    #                      jump=1, params={"gamma": gamma}).fit(signal)

    best_score = -np.inf
    best_ab    = (0, -1)

    # Search a range of n_bkps to find the best 2-segment epidemic interval
    # n_bkps=2 gives exactly one foreground segment; try it first.
    for n_bkps in [2]:
        try:
            bkps = algo.predict(n_bkps=n_bkps)
        except Exception:
            continue

        # bkps = [cp1, cp2, ..., n] (exclusive right endpoints)
        # Each consecutive pair (bkps[i-1], bkps[i]-1) is a segment.
        starts = [0] + bkps[:-1]
        ends   = [b - 1 for b in bkps]

        for seg_a, seg_b in zip(starts, ends):
            if seg_b - seg_a + 1 > W:
                # Trim to W around the segment centre
                mid   = (seg_a + seg_b) // 2
                seg_a = max(0, mid - W // 2)
                seg_b = min(n - 1, seg_a + W - 1)
            if seg_b <= seg_a:
                continue
            sc = _epidemic_score(x, seg_a, seg_b)
            if sc > best_score:
                best_score = sc
                best_ab    = (seg_a, seg_b)

    if best_ab[1] <= best_ab[0]:
        return 0.0, (0, -1)
    return float(best_score), best_ab


def scan_row_stumpy(
    x: np.ndarray, W: int,
    m_list: Optional[List[int]] = None,
    ignore_trivial: bool = True,
    detrend_deg: Optional[int] = 1,
) -> Tuple[float, Tuple[int, int]]:
    """
    Stumpy discord detector with NWKR-comparable score.

    Key changes from original:
    - z_norm=True (correct stumpy usage — normalises each subsequence by its
      own mean and std before distance computation, making the profile
      shape-sensitive rather than amplitude-sensitive).
    - The discord position from the matrix profile is used as a *candidate
      centre*, then we score all windows of size m around it using
      _epidemic_score and return the best.
    - The final score is _epidemic_score (same formula as NWKR) rather than
      the raw matrix profile distance, making it directly comparable.
    - detrend_deg=1 (linear) rather than 2 — removes global trend without
      over-fitting local structure.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 10:
        return 0.0, (0, -1)
    
    detrend_deg = None

    # Linear detrend: removes baseline slope without removing step structure
    if detrend_deg is not None:
        t    = np.arange(n, dtype=np.float64)
        coef = np.polyfit(t, x, detrend_deg)
        x0   = x - np.polyval(coef, t)
    else:
        x0 = x.copy()

    if m_list is None:
        # Cover a range of subsequence lengths from ~half W to W
        base   = [max(4, int(round(W * f))) for f in (0.5, 0.75, 1.0)]
        m_list = sorted({m for m in base if 4 <= m <= min(W, n - 2)})
    if not m_list:
        return 0.0, (0, -1)

    best_score = -np.inf
    best_ab    = (0, -1)

    for m in m_list:
        if m >= n - 1:
            continue
        # z_norm=True: standard stumpy usage for shape-based discord detection
        mp_  = stumpy.stump(x0, m=int(m), ignore_trivial=ignore_trivial, normalize=False)
        prof = np.asarray(mp_[:, 0], dtype=np.float64)
        good = np.isfinite(prof)
        if not np.any(good):
            continue

        # Take top-3 discord candidates and score each with _epidemic_score
        # on the original (non-detrended) signal x for fair comparison
        n_cands = min(3, int(np.sum(good)))
        prof_masked = np.where(good, prof, -np.inf)
        for _ in range(n_cands):
            idx = int(np.argmax(prof_masked))
            prof_masked[max(0, idx - m // 2): idx + m // 2 + 1] = -np.inf
            a = idx; b = idx + int(m) - 1
            if b >= n:
                continue
            sc = _epidemic_score(x, a, b)
            if sc > best_score:
                best_score = sc
                best_ab    = (a, b)

    if not np.isfinite(best_score) or best_ab[1] <= best_ab[0]:
        return 0.0, (0, -1)
    return float(best_score), best_ab

# def scan_row_kats_cusum(
#     x: np.ndarray, W: int,
#     alpha: float = 0.01,
#     detrend_deg: Optional[int] = 1,
# ) -> Tuple[float, Tuple[int, int]]:
#     """
#     Two-CUSUM epidemic detector with NWKR-comparable score.

#     Original problem: CUSUMDetector finds a single permanent change-point,
#     which is wrong for epidemic (temporary excursion) detection.

#     Fix: run a forward CUSUM to find the start of the excursion, then run
#     a backward CUSUM on the segment after the start to find the end.  This
#     gives a (start, end) pair that defines the epidemic interval.  Score
#     with _epidemic_score for comparability with NWKR.

#     If Kats CUSUM finds no change-point, fall back to scanning all windows
#     of width W with _epidemic_score and returning the best (same as the mean
#     family baseline).
#     """
#     x = np.asarray(x, dtype=np.float64)
#     n = x.size
#     if n < 10 or W < 2:
#         return 0.0, (0, -1)

#     if detrend_deg is not None:
#         t  = np.arange(n, dtype=np.float64)
#         x0 = x - np.polyval(np.polyfit(t, x, detrend_deg), t)
#     else:
#         x0 = x.copy()

#     def _kats_cp(arr: np.ndarray) -> Optional[int]:
#         """Run CUSUMDetector on arr, return change-point index or None."""
#         try:
#             tt  = pd.date_range("2000-01-01", periods=len(arr), freq="s")
#             ts  = TimeSeriesData(pd.DataFrame({"time": tt, "value": arr}))
#             det = CUSUMDetector(ts)
#             res = det.detector(alpha=alpha)
#             items = res if isinstance(res, (list, tuple)) else ([res] if res is not None else [])
#             for item in items:
#                 if item is None:
#                     continue
#                 # Try known attribute names across Kats versions
#                 for attr in ("cp_index", "cp", "changepoint", "change_point", "start_time"):
#                     if hasattr(item, attr):
#                         v = getattr(item, attr)
#                         if isinstance(v, (int, np.integer)):
#                             return int(v)
#                         # Timestamp — map back to index
#                         try:
#                             v = pd.to_datetime(v)
#                             return int(np.argmin(np.abs((tt - v).to_numpy(
#                                 dtype="timedelta64[ns]").astype(np.int64))))
#                         except Exception:
#                             pass
#                 if isinstance(item, dict):
#                     for key in ("cp_index", "cp", "changepoint", "change_point"):
#                         if key in item:
#                             v = item[key]
#                             if isinstance(v, (int, np.integer)):
#                                 return int(v)
#         except Exception:
#             pass
#         return None

#     # --- Forward CUSUM: find start of excursion ---
#     tau_start = _kats_cp(x0)

#     if tau_start is not None:
#         tau_start = int(np.clip(tau_start, 1, n - 2))

#         # --- Backward CUSUM on the segment after tau_start: find end ---
#         # Run on the reversed tail so that a rising edge in the forward
#         # signal appears as a falling edge (detectable) in the reversed one.
#         tail     = x0[tau_start:]
#         tau_rel  = _kats_cp(tail[::-1])

#         if tau_rel is not None:
#             tau_end = tau_start + (len(tail) - 1 - int(tau_rel))
#             tau_end = int(np.clip(tau_end, tau_start + 1, n - 1))
#         else:
#             # No end detected — use W as the window length
#             tau_end = min(n - 1, tau_start + W - 1)

#         # Enforce W constraint
#         if tau_end - tau_start + 1 > W:
#             tau_end = tau_start + W - 1

#         a, b = tau_start, tau_end

#     else:
#         # CUSUM found nothing — fall back to best mean-score window of width W
#         # (equivalent to the mean baseline scanner)
#         best_sc = -np.inf
#         a = b = 0
#         ps = np.empty(n + 1); ps[0] = 0.0; ps[1:] = np.cumsum(x0)
#         ps2 = np.empty(n + 1); ps2[0] = 0.0; ps2[1:] = np.cumsum(x0 ** 2)
#         mu_all = x0.mean()
#         sra    = float(np.sum((x0 - mu_all) ** 2))
#         if sra < 1e-18:
#             return 0.0, (0, -1)
#         for aa in range(n - 1):
#             bb = min(n - 1, aa + W - 1)
#             if bb <= aa:
#                 continue
#             sc = _epidemic_score(x, aa, bb)
#             if sc > best_sc:
#                 best_sc = sc; a = aa; b = bb

#     if b <= a:
#         return 0.0, (0, -1)

#     # Score on original signal (not detrended) for fair comparison
#     return float(_epidemic_score(x, a, b)), (a, b)

def scan_row_bocpd(
    x: np.ndarray, W: int,
    hazard_lambda: float = 50.0,
    detrend_deg: Optional[int] = 1,
) -> Tuple[float, Tuple[int, int]]:
    """
    Bayesian Online Changepoint Detection (BOCPD) epidemic detector.

    Replaces the two-CUSUM approach with a principled probabilistic model
    that jointly reasons about both changepoint boundaries simultaneously,
    eliminating the fragile two-pass forward/backward heuristic.

    Algorithm (Adams & MacKay 2007):
      At each timestep t, BOCPD maintains a posterior distribution over the
      current "run length" r — how many steps since the last changepoint.
      A short run length means a changepoint just occurred. The posterior
      is updated recursively using a Gaussian likelihood with unknown mean
      and variance (Normal-Gamma conjugate prior) and a constant hazard
      function H(r) = 1/lambda that controls the expected interval between
      changepoints.

    Epidemic interval extraction:
      After running BOCPD on the full signal, the most probable run length
      at each timestep is read from the posterior. A sudden DROP in run
      length (from large to near-zero) marks the START of a new segment —
      i.e. a changepoint. We find all such drops, pair them up as
      (onset, return) boundaries, score each pair with _epidemic_score,
      and return the best.

      If no pair is found, fall back to the single most confident
      changepoint and extend W channels forward from it.

    Parameters
    ----------
    x : np.ndarray
        Input spectrum.
    W : int
        Maximum window cap — detected intervals wider than W are trimmed.
    hazard_lambda : float
        Expected number of steps between changepoints (prior). Set to a
        value roughly equal to n/2 so the prior expects ~2 changepoints
        per spectrum (the onset and return of the step).
    detrend_deg : int or None
        Degree of polynomial detrend applied before BOCPD. 1 = linear.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 10 or W < 2:
        return 0.0, (0, -1)

    if detrend_deg is not None:
        t  = np.arange(n, dtype=np.float64)
        x0 = x - np.polyval(np.polyfit(t, x, detrend_deg), t)
    else:
        x0 = x.copy()

    # ------------------------------------------------------------------
    # BOCPD with Normal-Gamma conjugate (Gaussian obs, unknown mean+var)
    # ------------------------------------------------------------------
    # Prior hyperparameters (weakly informative, data-driven scale)
    mu0    = float(np.mean(x0))
    kappa0 = 1.0
    alpha0 = 1.0
    beta0  = float(np.var(x0)) * alpha0   # prior scale matches data variance

    # Hazard function: constant H = 1/lambda
    H = 1.0 / max(float(hazard_lambda), 1.0)

    # R[t] = posterior distribution over run lengths at time t
    # We store it as a 1-D array of unnormalised log-probabilities,
    # one entry per possible run length 0..t.
    # For memory efficiency we keep only the current column.

    # Sufficient statistics for each run-length hypothesis:
    # mu_n, kappa_n, alpha_n, beta_n (Normal-Gamma posterior params)
    # updated incrementally as new observations arrive.
    max_rl  = n + 1
    mu_n    = np.full(max_rl, mu0)
    kappa_n = np.full(max_rl, kappa0)
    alpha_n = np.full(max_rl, alpha0)
    beta_n  = np.full(max_rl, beta0)

    # Log posterior over run lengths, shape (t+1,) at time t
    log_R = np.full(max_rl, -np.inf)
    log_R[0] = 0.0   # at t=0, run length 0 has probability 1

    # Store MAP run length at each timestep
    map_rl = np.zeros(n, dtype=int)

    for t in range(n):
        xt = float(x0[t])

        # --- predictive probability under each run-length hypothesis ---
        # Student-t predictive: t_{2*alpha}(mu, beta*(kappa+1)/(alpha*kappa))
        active = np.where(np.isfinite(log_R))[0]
        log_pred = np.full(max_rl, -np.inf)

        for r in active:
            a = alpha_n[r]; b = beta_n[r]; k = kappa_n[r]; m = mu_n[r]
            # Student-t log-pdf
            nu     = 2.0 * a
            scale2 = b * (k + 1.0) / (a * k)
            scale2 = max(scale2, 1e-12)
            diff   = xt - m
            log_pred[r] = (
                lgamma((nu + 1.0) / 2.0)
                - lgamma(nu / 2.0)
                - 0.5 * np.log(nu * np.pi * scale2)
                - ((nu + 1.0) / 2.0) * np.log(1.0 + diff * diff / (nu * scale2))
            )

        # --- update: growth probabilities (run length increases by 1) ---
        log_growth = log_R[active] + log_pred[active] + np.log(1.0 - H)

        # --- update: changepoint probabilities (run length resets to 0) ---
        log_cp = np.logaddexp.reduce(log_R[active] + log_pred[active]) + np.log(H)

        # --- new log_R ---
        new_log_R = np.full(max_rl, -np.inf)
        new_log_R[0] = log_cp
        for idx, r in enumerate(active):
            if r + 1 < max_rl:
                new_log_R[r + 1] = log_growth[idx]

        # --- normalise ---
        lse = np.logaddexp.reduce(new_log_R[np.isfinite(new_log_R)])
        new_log_R[np.isfinite(new_log_R)] -= lse
        log_R = new_log_R

        # --- MAP run length at this timestep ---
        map_rl[t] = int(np.argmax(log_R))

        # --- update sufficient statistics for next step ---
        # Growth hypotheses: r -> r+1
        for r in active:
            r1 = r + 1
            if r1 < max_rl:
                kn1           = kappa_n[r] + 1.0
                mu_n[r1]      = (kappa_n[r] * mu_n[r] + xt) / kn1
                kappa_n[r1]   = kn1
                alpha_n[r1]   = alpha_n[r] + 0.5
                diff          = xt - mu_n[r]
                beta_n[r1]    = beta_n[r] + 0.5 * kappa_n[r] / kn1 * diff * diff
        # Changepoint hypothesis: reset to prior at r=0
        kn1         = kappa0 + 1.0
        mu_n[0]     = (kappa0 * mu0 + xt) / kn1
        kappa_n[0]  = kn1
        alpha_n[0]  = alpha0 + 0.5
        diff        = xt - mu0
        beta_n[0]   = beta0 + 0.5 * kappa0 / kn1 * diff * diff

    # ------------------------------------------------------------------
    # Extract changepoints from MAP run length sequence
    # A changepoint at time t is indicated by a sharp DROP in map_rl:
    # map_rl[t] << map_rl[t-1], meaning the most probable run just reset.
    # We detect drops where map_rl[t] < threshold (near zero).
    # ------------------------------------------------------------------
    cp_threshold = max(3, W // 4)
    changepoints = []
    for t in range(1, n):
        if map_rl[t] <= cp_threshold and map_rl[t - 1] > cp_threshold:
            changepoints.append(t)

    best_score = -np.inf
    best_ab    = (0, -1)

    if len(changepoints) >= 2:
        # Try all pairs of consecutive changepoints as epidemic boundaries
        for i in range(len(changepoints) - 1):
            a = changepoints[i]
            b = changepoints[i + 1] - 1

            # Enforce W constraint
            if b - a + 1 > W:
                mid = (a + b) // 2
                a   = max(0, mid - W // 2)
                b   = min(n - 1, a + W - 1)

            if b <= a:
                continue

            sc = _epidemic_score(x, a, b)
            if sc > best_score:
                best_score = sc
                best_ab    = (a, b)

    if best_ab[1] <= best_ab[0]:
        # Fallback: use the single most confident changepoint + W
        if changepoints:
            a = changepoints[0]
            b = min(n - 1, a + W - 1)
            if b > a:
                best_score = _epidemic_score(x, a, b)
                best_ab    = (a, b)

    if best_ab[1] <= best_ab[0] or not np.isfinite(best_score):
        return 0.0, (0, -1)

    return float(best_score), best_ab

def scan_row_binseg(
    x: np.ndarray, W: int,
    min_size: int = 5,
    model: str = "l2",
) -> Tuple[float, Tuple[int, int]]:
    """
    Binary segmentation changepoint detector (ruptures.Binseg) with
    NWKR-comparable epidemic score.

    Binseg greedily finds the two best changepoints by recursively
    splitting the signal at the highest-cost breakpoint.  Unlike Dynp
    which finds the globally optimal pair, Binseg tends to find the
    higher-contrast boundary first, which is more robust when the two
    boundaries of the step anomaly have unequal contrast due to AR noise.

    The two detected changepoints define a candidate epidemic interval
    which is scored with _epidemic_score for direct comparability with
    NWKR.  Segments wider than W are trimmed to W around their centre.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2 * min_size + 2:
        return 0.0, (0, -1)

    signal = x.reshape(-1, 1)
    algo   = rpt.Binseg(model=model, min_size=int(min_size), jump=1).fit(signal)

    best_score = -np.inf
    best_ab    = (0, -1)

    try:
        bkps = algo.predict(n_bkps=2)
    except Exception:
        return 0.0, (0, -1)

    # bkps = [cp1, cp2, n] — exclusive right endpoints
    # segments: [0, cp1), [cp1, cp2), [cp2, n)
    starts = [0] + bkps[:-1]
    ends   = [b - 1 for b in bkps]

    for seg_a, seg_b in zip(starts, ends):
        if seg_b - seg_a + 1 > W:
            mid   = (seg_a + seg_b) // 2
            seg_a = max(0, mid - W // 2)
            seg_b = min(n - 1, seg_a + W - 1)
        if seg_b <= seg_a:
            continue
        sc = _epidemic_score(x, seg_a, seg_b)
        if sc > best_score:
            best_score = sc
            best_ab    = (seg_a, seg_b)

    if best_ab[1] <= best_ab[0]:
        return 0.0, (0, -1)
    return float(best_score), best_ab

def scan_row_clasp(
    x: np.ndarray, W: int,
    min_seg_size: int = 5,
    n_cps: int = 2,
) -> Tuple[float, Tuple[int, int]]:
    """
    ClaSP (Classification-based Segmentation) changepoint detector.

    ClaSP (Ermshaus et al. 2023) trains a binary time-series classifier
    to distinguish subsequences from either side of a candidate boundary.
    The boundary position that maximises classifier accuracy is selected
    as the changepoint.  This makes it robust to correlated backgrounds
    because the classifier adapts to the local temporal structure rather
    than assuming i.i.d. noise.

    Pipeline:
      1. Run ClaSP requesting n_cps=2 changepoints (onset + return).
      2. Use the two detected boundaries to define the epidemic interval.
      3. Score with _epidemic_score for direct comparability with NWKR.
      4. If only one boundary is found, extend W channels forward from it.
      5. Trim intervals wider than W to W around their centre.

    Parameters
    ----------
    x : np.ndarray
        Input spectrum.
    W : int
        Maximum window cap.
    min_seg_size : int
        Minimum segment size passed to ClaSP (default 5).
    n_cps : int
        Number of changepoints to request (default 2 for epidemic model).
    """
    try:
        from sktime.detection.clasp import ClaSPSegmentation
    except ImportError:
        raise ImportError("clasp not installed — run: pip install clasp")

    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2 * min_seg_size + 2 or W < 2:
        return 0.0, (0, -1)

    try:
        clasp = ClaSPSegmentation(
            period_length=max(min_seg_size, W // 2),
            n_change_points=n_cps,
        )
        clasp.fit(x)
        cps = sorted([int(cp) for cp in clasp.change_points_])
    except Exception:
        return 0.0, (0, -1)

    if len(cps) == 0:
        return 0.0, (0, -1)

    best_score = -np.inf
    best_ab    = (0, -1)

    if len(cps) >= 2:
        # Try every consecutive pair as epidemic (onset, return) boundaries
        for i in range(len(cps) - 1):
            a = cps[i]
            b = cps[i + 1] - 1

            if b - a + 1 > W:
                mid = (a + b) // 2
                a   = max(0, mid - W // 2)
                b   = min(n - 1, a + W - 1)

            if b <= a:
                continue

            sc = _epidemic_score(x, a, b)
            if sc > best_score:
                best_score = sc
                best_ab    = (a, b)

    if best_ab[1] <= best_ab[0]:
        # Fallback: single changepoint + W
        a = cps[0]
        b = min(n - 1, a + W - 1)
        if b > a:
            best_score = _epidemic_score(x, a, b)
            best_ab    = (a, b)

    if best_ab[1] <= best_ab[0] or not np.isfinite(best_score):
        return 0.0, (0, -1)

    return float(best_score), best_ab


def scan_row_lrt(
    x: np.ndarray, W: int,
    min_seg_size: int = 5,
    detrend_deg: Optional[int] = 1,
) -> Tuple[float, Tuple[int, int]]:
    """
    Sliding Window Likelihood Ratio Test for epidemic interval detection.

    For each candidate window [a, b], computes the log-likelihood ratio:

        LLR(a,b) = log p(x | two-segment: outside~N(mu_out, s_out^2),
                                           inside~N(mu_in,  s_in^2))
                 - log p(x | one-segment:  all  ~N(mu_all, s_all^2))

    Under the Gaussian model this simplifies to:

        LLR = (n_out/2) * log(s_all^2 / s_out^2)
            + (n_in /2) * log(s_all^2 / s_in^2)

    which is large when the inside segment has a distinctly different
    variance/mean from the outside.  The window maximising LLR is
    returned, scored with _epidemic_score for comparability with NWKR.

    Advantages over CUSUM and KernelCPD:
      - Finds both boundaries simultaneously (no two-pass fragility).
      - Variance-normalised: robust to heteroskedastic AR backgrounds.
      - O(n*W) with prefix sums — fast enough for n=500, W=25.
      - No hyperparameters beyond W and min_seg_size.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2 * min_seg_size + 2 or W < 2:
        return 0.0, (0, -1)

    if detrend_deg is not None:
        t  = np.arange(n, dtype=np.float64)
        x0 = x - np.polyval(np.polyfit(t, x, detrend_deg), t)
    else:
        x0 = x.copy()

    # Prefix sums for O(1) segment mean and variance
    ps1  = np.zeros(n + 1); ps1[1:]  = np.cumsum(x0)
    ps2  = np.zeros(n + 1); ps2[1:]  = np.cumsum(x0 ** 2)

    def seg_stats(lo: int, hi: int):
        """Mean and variance of x0[lo:hi+1] via prefix sums."""
        k    = hi - lo + 1
        s1   = ps1[hi + 1] - ps1[lo]
        s2   = ps2[hi + 1] - ps2[lo]
        mu   = s1 / k
        var  = max(s2 / k - mu * mu, 1e-12)
        return k, mu, var

    n_all, mu_all, var_all = seg_stats(0, n - 1)
    log_var_all = np.log(var_all)

    best_llr = -np.inf
    best_ab  = (0, -1)

    for a in range(min_seg_size, n - min_seg_size):
        for b in range(a, min(a + W, n - min_seg_size)):
            n_in,  mu_in,  var_in  = seg_stats(a, b)
            # outside = [0,a) + (b,n-1]
            n_out = n - n_in
            if n_out < min_seg_size:
                continue
            s1_out = ps1[a] + (ps1[n] - ps1[b + 1])
            s2_out = ps2[a] + (ps2[n] - ps2[b + 1])
            mu_out  = s1_out / n_out
            var_out = max(s2_out / n_out - mu_out * mu_out, 1e-12)

            llr = (0.5 * n_out * (log_var_all - np.log(var_out))
                 + 0.5 * n_in  * (log_var_all - np.log(var_in)))

            if llr > best_llr:
                best_llr = llr
                best_ab  = (a, b)

    if best_ab[1] <= best_ab[0]:
        return 0.0, (0, -1)

    sc = _epidemic_score(x, best_ab[0], best_ab[1])
    if not np.isfinite(sc):
        return 0.0, (0, -1)

    return float(sc), best_ab

def scan_row_capa(
    x: np.ndarray, W: int,
    beta_scale: float = 8.0,
    beta_prime_scale: float = 6.0,
    min_seg_len: int = 2,
    detrend_deg: Optional[int] = 1,
) -> Tuple[float, Tuple[int, int]]:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2 * min_seg_len + 2 or W < min_seg_len:
        return 0.0, (0, -1)

    if detrend_deg is not None:
        t  = np.arange(n, dtype=np.float64)
        x0 = x - np.polyval(np.polyfit(t, x, detrend_deg), t)
    else:
        x0 = x.copy()

    mu0    = float(np.median(x0))
    mad    = float(np.median(np.abs(x0 - mu0)))
    sigma0 = max(mad * 1.4826, 1e-12)
    var0   = sigma0 ** 2
    logvar0 = np.log(var0)
    log_n   = np.log(max(n, 2))
    beta   = beta_scale * (log_n + np.log(max(W, 2)))
    beta_p = beta_prime_scale * log_n
    gamma   = np.exp(-beta_p)

    ps1 = np.zeros(n + 1); ps1[1:] = np.cumsum(x0)
    ps2 = np.zeros(n + 1); ps2[1:] = np.cumsum(x0 ** 2)

    def seg_cost(a, b):
        k   = b - a + 1
        s1  = ps1[b + 1] - ps1[a]
        s2  = ps2[b + 1] - ps2[a]
        mu  = s1 / k
        var = max(s2 / k - mu * mu, 1e-12)
        return float(k * (np.log(var) + 1.0))

    def normal_cost(i):
        z = (x0[i] - mu0) / sigma0
        return float(logvar0 + z * z)

    def point_cost(i):
        return float(1.0 + np.log(gamma * var0 + (x0[i] - mu0) ** 2) + beta_p)

    C        = np.full(n + 1, np.inf)
    C[0]     = 0.0
    back     = np.full(n + 1, -1, dtype=int)
    is_point = np.zeros(n + 1, dtype=bool)
    seg_start = np.full(n + 1, -1, dtype=int)
    candidates = [0]

    for m in range(1, n + 1):
        obs_idx = m - 1
        cost_normal = C[m - 1] + normal_cost(obs_idx)
        cost_point  = C[m - 1] + point_cost(obs_idx)

        if cost_point < cost_normal:
            best = cost_point; best_back = m - 1
            best_is_point = True; best_seg = -1
        else:
            best = cost_normal; best_back = m - 1
            best_is_point = False; best_seg = -1

        max_start = max(0, m - W)
        for k in candidates:
            if k > m - min_seg_len or k < max_start:
                continue
            seg_len = m - k
            if seg_len < min_seg_len or seg_len > W:
                continue
            cost_anom = C[k] + seg_cost(k, m - 1) + beta
            if cost_anom < best:
                best = cost_anom; best_back = k
                best_is_point = False; best_seg = k

        C[m] = best; back[m] = best_back
        is_point[m] = best_is_point; seg_start[m] = best_seg
        candidates = [k for k in candidates if C[k] <= C[m]]
        candidates.append(m)

    # Backtrack — collect only genuine collective anomaly segments
    collective_segments = []
    m = n
    while m > 0:
        k = back[m]
        if seg_start[m] >= 0:
            a, b = k, m - 1
            # KEY FIX: only keep the segment if it beats treating those
            # points as normal by more than just the penalty rounding
            seg_len = b - a + 1
            cost_as_anom = seg_cost(a, b) + beta
            cost_as_norm = sum(normal_cost(i) for i in range(a, b + 1))
            saving = cost_as_norm - cost_as_anom
            # Require saving to exceed beta — filters spurious segments
            if saving > beta * 0.5:
                collective_segments.append((a, b))
        m = k

    if not collective_segments:
        return 0.0, (0, -1)

    best_score = -np.inf
    best_ab    = (0, -1)
    for (a, b) in collective_segments:
        if b - a + 1 > W:
            mid = (a + b) // 2
            a   = max(0, mid - W // 2)
            b   = min(n - 1, a + W - 1)
        if b <= a:
            continue
        sc = _epidemic_score(x, a, b)
        if sc > best_score:
            best_score = sc; best_ab = (a, b)

    if best_ab[1] <= best_ab[0] or not np.isfinite(best_score):
        return 0.0, (0, -1)

    return float(best_score), best_ab

# ---------------------------------------------------------------------------
# Unified single-row dispatcher
# ---------------------------------------------------------------------------

def _run_one(x: np.ndarray, freqs: np.ndarray, W: int, R: int, cfg: dict) -> Tuple[float, Tuple[int, int]]:
    mode = cfg.get("mode", "simple")
    if mode == "simple":
        fam    = cfg["family"]
        params = dict(cfg.get("params", {}))
        return scan_row_with_family(x, W, fam, params)

    if mode == "nwkr":
        naive = cfg.get("family", "optimized") == "naive"
        n     = x.size
        sr_factor = max(1, 2 ** math.ceil(math.log2(max(1, math.ceil((n + 1) / 450)))))

        return scan_row_nwkr(
            x, freqs=freqs, W=W, R=R,
            kernel_kind=cfg.get("kernel_kind", "gaussian"),
            which=cfg.get("which", "unmasked_varlen"),
            buffer=int(len(x)//20),
            sr_factor=sr_factor,
            # buffer=cfg.get("buffer", int(n//20)),
            # sr_factor=cfg.get("sr_factor", max(1, 2 ** math.ceil(math.log2(max(1, math.ceil((n + 1) / 450)))))),
            naive=naive,
        )

    if mode == "nwkr_first_start":
        from helpers.scan import set_fix_first_start
        set_fix_first_start(True)
        try:
            sub = dict(cfg); sub["mode"] = "nwkr"
            if not freqs:
                freqs_rt = np.linspace(220e9, 224e9, len(x))
            return _run_one(x, freqs_rt, W, R, sub)
        finally:
            set_fix_first_start(False)

    if mode == "kernelcpd_rbf_c":
        return scan_row_kernelcpd(
            x, W,
            min_size=int(cfg.get("min_size", 5)),
            model=cfg.get("model", "l2"),
        )

    if mode == "binseg":
        return scan_row_binseg(
            x, W,
            min_size=int(cfg.get("min_size", 5)),
            model=cfg.get("model", "l2"),
        )

    if mode == "bocpd":
        return scan_row_bocpd(
            x, W,
            hazard_lambda=float(cfg.get("hazard_lambda", 50.0)),
        )
    
    if mode == "clasp":
        return scan_row_clasp(
            x, W,
            min_seg_size=int(cfg.get("min_seg_size", 5)),
            n_cps=int(cfg.get("n_cps", 2)),
        )
    if mode == "lrt":
        return scan_row_lrt(
            x, W,
            min_seg_size=int(cfg.get("min_seg_size", 5)),
        )
    if mode == "stumpy":
        return scan_row_stumpy(
            x, W,
            m_list=cfg.get("m", None),
            ignore_trivial=bool(cfg.get("ignore_trivial", True)),
        )
    if mode == "capa":
        return scan_row_capa(
            x, W,
            beta_scale       = float(cfg.get("beta_scale",       4.0)),
            beta_prime_scale = float(cfg.get("beta_prime_scale", 3.0)),
            min_seg_len      = int(cfg.get("min_seg_len",         2)),
        )

    if mode == "cpdf":
        from helpers.cpd_nwkr import scan_row_cpd_fast
        from helpers.config import ref_freq
        fkw = dict(cfg.get("family_kwargs", {}))
        if str(cfg.get("family", "")).startswith("nwkr"):
            if freqs is not None and len(freqs) > 1:
                freq_step = abs(freqs[1] - freqs[0])
                L = len(freqs)
                R = ref_freq / (freq_step if freq_step > 0 else 1.)
                w_auto = int(round(max(3, min(R, L / 16))))
            else:
                w_auto = int(cfg.get("w", W))
            fkw["w"] = float(cfg.get("w", w_auto))
            width_cap = 3 * w_auto
        else:
            width_cap = cfg.get("width_cap", None)
        return scan_row_cpd_fast(
            x, W,
            family        = cfg.get("family", "nwkr_gaussian"),
            family_kwargs = fkw,
            algo          = cfg.get("algo", "kernelcpd"),
            min_size      = int(cfg.get("min_size", 3)),
            n_bkps        = int(cfg.get("n_bkps", 2)),
            width_cap     = width_cap,
            jump          = int(cfg.get("jump", 1)),
        )

    if mode == "deep":
        from helpers.deep_baselines import scan_row_deep
        return scan_row_deep(
            x, W,
            backend        = cfg.get("backend", "tsbad"),
            model          = cfg.get("model", "USAD"),
            seed           = int(cfg.get("seed", 42)),
            min_width      = int(cfg.get("min_width", 2)),
            top_k          = int(cfg.get("top_k", 3)),
            n_widths       = int(cfg.get("n_widths", 24)),
            refine         = bool(cfg.get("refine", True)),
            smooth         = int(cfg.get("smooth", 0)),
            select         = cfg.get("select", "deep"),
            backend_kwargs = cfg.get("backend_kwargs", None),
        )


    raise ValueError(f"Unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# Hard-timeout wrapper (for naive NWKR)
# ---------------------------------------------------------------------------

def _timeout_worker(q: mp.Queue, x, W, R, cfg) -> None:
    try:
        score, ab = _run_one(x, None, W, R, cfg)
        q.put(("ok", float(score), ab))
    except Exception as e:
        q.put(("err", repr(e), None))


def _run_with_timeout(
    x: np.ndarray, W: int, R: int, cfg: dict, timeout_s: float,
) -> dict:
    q = mp.Queue()
    p = mp.Process(target=_timeout_worker, args=(q, x, W, R, cfg))
    t0 = time.perf_counter()
    p.start()
    p.join(timeout=timeout_s)
    dt = time.perf_counter() - t0
    if p.is_alive():
        p.terminate()
        p.join()
        return {"status": "timeout", "time_s": dt, "score": np.nan, "ab": None}
    try:
        msg = q.get_nowait()
    except queue.Empty:
        return {"status": "error", "time_s": dt, "score": np.nan, "ab": None}
    if msg[0] == "ok":
        return {"status": "ok", "time_s": dt, "score": msg[1], "ab": msg[2]}
    return {"status": "error", "time_s": dt, "score": np.nan, "ab": None, "error": msg[1]}


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def _warmup_methods(x: np.ndarray, W: int, R: int, families: dict) -> None:
    for name, cfg in families.items():
        try:
            _run_one(x, None, W, R, cfg)
        except Exception as e:
            print(f"[warmup] {name}: {e}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_sra_sri(
    x: np.ndarray, freqs: np.ndarray,
    a: int, b: int,
    pred_sra: np.ndarray, pred_sri: np.ndarray,
    title: str, out_path: str,
) -> None:
    plt.figure(figsize=(10, 3))
    plt.plot(freqs, x, color="k", lw=1.0, label="data")
    plt.plot(freqs, pred_sra, color="C0", lw=1.5, label="SRA (global)")
    inside_idxs = np.arange(a, b + 1)[:len(pred_sri)]
    plt.plot(freqs[inside_idxs], pred_sri, color="C3", lw=2.0, label="SRI (inside)")
    plt.axvspan(freqs[a], freqs[b], color="C3", alpha=0.15)
    plt.xlabel("frequency")
    plt.ylabel("signal")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _extrapolate_tail_loglog(
    x_ok: np.ndarray, y_ok: np.ndarray, x_future: np.ndarray, degree: int = 2,
) -> np.ndarray:
    mask = (x_ok > 0) & (y_ok > 0) & np.isfinite(x_ok) & np.isfinite(y_ok)
    x_ok, y_ok = x_ok[mask], y_ok[mask]
    if len(x_ok) < degree + 1:
        return np.full_like(x_future, np.nan, dtype=float)
    coeffs = np.polyfit(np.log10(x_ok), np.log10(y_ok), deg=degree)
    return 10 ** np.polyval(coeffs, np.log10(x_future))


# ---------------------------------------------------------------------------
# BENCHMARK MODE 1: runtime scaling
# ---------------------------------------------------------------------------

def run_runtime_benchmark(
    *,
    variable: str = "n",
    n_fixed: int = 1000,
    w_fixed: int = 10,
    r_fixed: int = 30,
    families: dict,
    reps: int = 5,
    warmup: bool = True,
    out_dir: str = "data/runtime_scaling",
    csv_name: str = "runtimes.csv",
    fig_name: str = "runtimes.png",
    timeout_s: float = 30.0,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    if variable == "n":
        sweep = np.unique(np.round(np.logspace(np.log10(10), np.log10(10_000), 12)).astype(int)).tolist()
    elif variable == "w":
        sweep = np.unique(np.round(np.logspace(np.log10(10), np.log10(200), 12)).astype(int)).tolist()
    elif variable == "r":
        sweep = np.unique(np.round(np.logspace(np.log10(10), np.log10(300), 12)).astype(int)).tolist()
    else:
        raise ValueError(f"Unknown variable {variable!r}. Choose from: n, w, r")

    rows = []
    retired = set()
    timeout_counts: Dict[str, int] = {name: 0 for name in families}

    for elem in sweep:
        n = elem          if variable == "n" else n_fixed
        w = elem          if variable == "w" else w_fixed
        r = min(elem, n-1) if variable == "r" else r_fixed

        # Generate one synthetic row
        group = [SpectrumSpec(int(n), int(w), 1)]
        data = generate_synthetic_dataset(
            groups=group, seed=42,
            strong_rate=1.0, strong_kind="rect_step",
            exact_strong=True, error_params=(2.5, 0.05),
        )
        x = np.asarray(data["groups"][0]["spectra"][0], dtype=np.float64)

        if warmup:
            active = {k: v for k, v in families.items() if k not in retired}
            _warmup_methods(x, w, r, active)

        prev_times: List[float] = []

        for name, cfg in families.items():
            if name in retired:
                rows.append({variable: int(elem), "method": name,
                             "time_ms_median": np.nan, "score": np.nan,
                             "a": -1, "b": -1, "status": "retired"})
                continue

            use_timeout = name in _NAIVE_METHODS
            if prev_times:
                cutoff = max(5.0, min(timeout_s, 50.0 * float(np.median(prev_times))))
            else:
                cutoff = timeout_s

            times, last_score, last_ab, status = [], None, None, "ok"

            for _ in range(reps):
                if use_timeout:
                    res = _run_with_timeout(x, w, r, cfg, timeout_s=cutoff)
                    times.append(res["time_s"])
                    if res["status"] != "ok":
                        status = res["status"]
                        break
                    last_score, last_ab = res["score"], res["ab"]
                else:
                    t0 = time.perf_counter()
                    last_score, last_ab = _run_one(x, None, w, r, cfg)
                    times.append(time.perf_counter() - t0)

            dt_med = float(np.median(times)) if times else np.nan

            if status == "ok" and np.isfinite(dt_med):
                prev_times.append(dt_med)
            elif status == "timeout":
                timeout_counts[name] += 1
                if timeout_counts[name] >= 2:
                    retired.add(name)

            rows.append({
                variable: int(elem),
                "method":        name,
                "time_ms_median": dt_med * 1000.0 if np.isfinite(dt_med) else np.nan,
                "score":          float(last_score) if last_score is not None else np.nan,
                "a":              int(last_ab[0])   if last_ab is not None else -1,
                "b":              int(last_ab[1])   if last_ab is not None else -1,
                "status":         status,
            })

        print(f"  [{variable}={elem}] done")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # ---- Plot ----
    y_top = 1e4
    plt.figure(figsize=(8, 5))
    for method, sub in df.groupby("method"):
        sub = sub.sort_values(variable)
        xv = sub[variable].to_numpy(float)
        yv = sub["time_ms_median"].to_numpy(float)
        st = sub["status"].astype(str).to_numpy()

        y_meas = yv.copy()
        bad = np.where(st != "ok")[0]
        first_bad = bad[0] if len(bad) else None
        if first_bad is not None:
            y_meas[first_bad:] = np.nan

        line, = plt.plot(xv, y_meas, marker="o", linewidth=1.5, label=method)

        # Extrapolate for naive methods
        if method in _NAIVE_METHODS and first_bad is not None and first_bad >= 3:
            x_ok, y_ok = xv[:first_bad], yv[:first_bad]
            x_ext = np.logspace(np.log10(x_ok[-1]), np.log10(xv[-1] * 5), 200)
            y_ext = _extrapolate_tail_loglog(x_ok[-5:], y_ok[-5:], x_ext)
            good = np.isfinite(y_ext) & (y_ext > 0)
            x_ext, y_ext = x_ext[good], y_ext[good]
            if len(y_ext):
                hit = np.where(y_ext >= y_top)[0]
                if len(hit):
                    x_ext, y_ext = x_ext[:hit[0] + 1], y_ext[:hit[0] + 1]
                    y_ext[-1] = y_top
                plt.plot(x_ext, y_ext, linewidth=1.5, linestyle="--", color=line.get_color())

    plt.xscale("log"); plt.yscale("log")
    plt.ylim(1e-1, y_top)
    plt.xlabel(f"{variable} (log scale)")
    plt.ylabel("Median runtime per row (ms)")
    plt.title(f"Runtime scaling: variable = {variable}")
    plt.legend(fontsize=7)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, fig_name)
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved: {fig_path}")
    return df


# ---------------------------------------------------------------------------
# BENCHMARK MODE 2: parameter / detection quality
# ---------------------------------------------------------------------------

def run_parameter_benchmark(
    *,
    variable: str = "snr",
    w_list: List[int],
    error_params: List[Tuple[float, float]],
    n_spectrum: int = 500,
    n_rows: int = 100,
    strong_rate: float = 0.5,
    strong_kind: str = "rect_step",
    families: dict,
    seed: int = 123,
    out_dir: str = "data/method_benchmark",
    plot_dir: str = "images/method_benchmark",
    save_sra_sri: bool = False,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    all_rows = []

    for snr, noise in error_params:
        for w in w_list:
            print(f"  [snr={snr}, noise={noise}, w={w}]")
            group = [SpectrumSpec(n_spectrum, w, n_rows)]
            data = generate_synthetic_dataset(
                groups=group, seed=seed,
                strong_rate=strong_rate, strong_kind=strong_kind,
                exact_strong=True, error_params=(snr, noise),
            )
            g = data["groups"][0]
            X = g["spectra"]
            has_strong    = g["has_strong_anom"]
            strong_labels = g["strong_labels"]
            R = w

            for row_idx in range(X.shape[0]):
                x          = X[row_idx].astype(np.float64)
                is_strong  = bool(has_strong[row_idx])
                gt_ints    = strong_labels[row_idx]

                for name, cfg in families.items():
                    t0 = time.perf_counter()
                    score, (a, b) = _run_one(x, None, w, R, cfg)
                    dt = time.perf_counter() - t0

                    cov_best = pars_best = loc_best = 0.0
                    if is_strong and a < b:
                        for (s_gt, e_gt) in gt_ints:
                            cov, pars, loc = localization_metrics((a, b), (s_gt, e_gt))
                            if loc > loc_best:
                                cov_best, pars_best, loc_best = cov, pars, loc

                        if save_sra_sri and cfg.get("mode") == "nwkr":
                            _save_nwkr_sra_sri(x, a, b, w, cfg, name, row_idx, plot_dir)

                    all_rows.append({
                        "snr":         snr,
                        "noise":       noise,
                        "w":           w,
                        "method":      name,
                        "row_idx":     row_idx,
                        "is_strong":   is_strong,
                        "score":       float(score),
                        "a":           int(a),
                        "b":           int(b),
                        "coverage":    cov_best,
                        "parsimony":   pars_best,
                        "loc_score":   loc_best,
                        "time_s":      dt,
                    })

    df = pd.DataFrame(all_rows)

    # Add mean and std of loc_score per (method, w, snr, noise) group to CSV
    strong_df = df[df["is_strong"]]
    summary = (
        strong_df.groupby(["method", "w", "snr", "noise"])["loc_score"]
        .agg(mean_loc_score="mean", std_loc_score="std")
        .reset_index()
    )
    summary_path = os.path.join(out_dir, f"methods_summary_{variable}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    csv_path = os.path.join(out_dir, f"methods_results_{variable}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    _plot_parameter_results(df, plot_dir)
    return df

def run_parameter_benchmark_ar2(
    *,
    variable: str = "snr",
    w_list: List[int],
    error_params: List[Tuple[float, float]],
    n_spectrum: int = 500,
    n_rows: int = 100,
    strong_rate: float = 0.5,
    families: dict,
    seed: int = 123,
    out_dir: str = "data/parameter_benchmark_ar2",
    plot_dir: str = "images/parameter_benchmark_ar2",
    save_sra_sri: bool = False,
) -> pd.DataFrame:
    """
    Same as run_parameter_benchmark but uses generate_ar2_dataset
    instead of generate_synthetic_dataset.
    The error_params (snr, noise) map to:
        snr   → step_strength  (amplitude of step in units of local std)
        noise → step_width_frac (step width as fraction of n)
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    all_rows = []

    for snr, noise in error_params:
        for w in w_list:
            print(f"  [snr={snr}, noise={noise}, w={w}]")

            data = generate_ar2_dataset(
                n          = n_spectrum,
                W          = w,
                n_rows     = n_rows,
                strong_rate= strong_rate,
                seed       = seed,
                step_strength   = snr,
                step_width_frac = noise,
            )

            X             = data["spectra"]
            has_strong    = data["has_strong_anom"]
            strong_labels = data["strong_labels"]
            R = w

            for row_idx in range(X.shape[0]):
                x         = X[row_idx].astype(np.float64)
                is_strong = bool(has_strong[row_idx])
                gt_ints   = strong_labels[row_idx]

                for name, cfg in families.items():
                    t0 = time.perf_counter()
                    score, (a, b) = _run_one(x, None, w, R, cfg)
                    dt = time.perf_counter() - t0

                    cov_best = pars_best = loc_best = 0.0
                    if is_strong and a < b:
                        for (s_gt, e_gt) in gt_ints:
                            cov, pars, loc = localization_metrics((a, b), (s_gt, e_gt))
                            if loc > loc_best:
                                cov_best, pars_best, loc_best = cov, pars, loc

                    all_rows.append({
                        "snr":       snr,
                        "noise":     noise,
                        "w":         w,
                        "method":    name,
                        "row_idx":   row_idx,
                        "is_strong": is_strong,
                        "score":     float(score),
                        "a":         int(a),
                        "b":         int(b),
                        "coverage":  cov_best,
                        "parsimony": pars_best,
                        "loc_score": loc_best,
                        "time_s":    dt,
                    })

    df = pd.DataFrame(all_rows)

    strong_df = df[df["is_strong"]]
    summary = (
        strong_df.groupby(["method", "w", "snr", "noise"])["loc_score"]
        .agg(mean_loc_score="mean", std_loc_score="std")
        .reset_index()
    )
    summary_path = os.path.join(out_dir, f"methods_summary_{variable}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    csv_path = os.path.join(out_dir, f"methods_results_{variable}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    _plot_parameter_results(df, plot_dir)
    return df

def _save_nwkr_sra_sri(
    x: np.ndarray, a: int, b: int, W: int,
    cfg: dict, name: str, row_idx: int, plot_dir: str,
) -> None:
    try:
        freqs = np.linspace(0.0, 1.0, x.size)
        inside = np.arange(a, b + 1, dtype=np.int64)
        kind = cfg.get("kernel_kind", "gaussian")
        if kind == "gaussian":
            set_kernel_kind("gaussian")
            Wk, denom = get_kernel_and_denom(x.size, W, KernelKind.GAUSSIAN)
            _, _, pred_sra, _, _, _ = calculate_gaussian_sra_with_nd(x, Wk, denom)
            pred_sri = predict_on_idxs(x, inside, Wk, KernelKind.GAUSSIAN, float(W))
        else:
            set_kernel_kind("laplace")
            sigma = float(max(W, 1))
            _, _, pred_sra, _ = calculate_laplace_sra_fast(x, sigma)
            pred_sri = predict_on_idxs(x, inside, None, KernelKind.LAPLACE, sigma)
        d = os.path.join(plot_dir, "sra_sri", name)
        os.makedirs(d, exist_ok=True)
        plot_sra_sri(
            x, freqs, a, b, pred_sra, pred_sri,
            title=f"{name} | row={row_idx}",
            out_path=os.path.join(d, f"row_{row_idx}.png"),
        )
    except Exception as e:
        print(f"[sra_sri plot] {name} row={row_idx}: {e}")


def _plot_parameter_results(df: pd.DataFrame, plot_dir: str) -> None:
    methods = list(df["method"].unique())
    strong  = df[df["is_strong"]]
    w_vals  = sorted(df["w"].unique())
    n_w     = len(w_vals)
    n_m     = len(methods)

    # ---- 1) Bar chart: mean loc_score per method, one subplot per w ----
    # Error bars show std but are clipped so the top of the bar + std <= y-axis top.
    stats = (
        strong.groupby(["method", "w"])["loc_score"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    colors = [f"C{i}" for i in range(n_m)]
    x      = np.arange(n_m)
    width  = 0.65

    fig, axes = plt.subplots(
        1, n_w,
        figsize=(max(4, n_w * max(3, n_m * 0.7)), 4),
        sharey=True,
        squeeze=False,
    )

    for ci, w in enumerate(w_vals):
        ax   = axes[0][ci]
        sub  = stats[stats["w"] == w].set_index("method").reindex(methods)
        means = sub["mean"].fillna(0).values
        stds  = sub["std"].fillna(0).values

        bars = ax.bar(x, means, width, color=colors, edgecolor="k", linewidth=0.5)

        # Clip error bar top to 1.0 so it never leaves the plot
        err_top = np.minimum(stds, 1.0 - means)
        err_top = np.maximum(err_top, 0.0)
        ax.errorbar(
            x, means,
            yerr=[np.zeros(n_m), err_top],
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=3,
            capthick=1.2,
        )

        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=7)
        ax.set_title(f"w={w}", fontsize=9)
        if ci == 0:
            ax.set_ylabel("Mean localization score")

    fig.suptitle("Mean localization score ± std (strong rows only)", y=1.01)
    fig.tight_layout()
    path = os.path.join(plot_dir, "localization_by_method_w.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {path}")

    # ---- 2) Bar chart: mean loc_score per w, one subplot per method ----
    fig, axes = plt.subplots(
        1, n_m,
        figsize=(max(4, n_m * max(3, n_w * 0.7)), 4),
        sharey=True,
        squeeze=False,
    )
    x_w = np.arange(n_w)

    for ci, method in enumerate(methods):
        ax  = axes[0][ci]
        sub = stats[stats["method"] == method].set_index("w").reindex(w_vals)
        means = sub["mean"].fillna(0).values
        stds  = sub["std"].fillna(0).values

        ax.bar(x_w, means, width, color="C0", edgecolor="k", linewidth=0.5)

        err_top = np.minimum(stds, 1.0 - means)
        err_top = np.maximum(err_top, 0.0)
        ax.errorbar(
            x_w, means,
            yerr=[np.zeros(n_w), err_top],
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=3,
            capthick=1.2,
        )

        ax.set_ylim(0, 1.05)
        ax.set_xticks(x_w)
        ax.set_xticklabels([str(w) for w in w_vals], rotation=40, ha="right", fontsize=7)
        ax.set_title(method, fontsize=8)
        if ci == 0:
            ax.set_ylabel("Mean localization score")
        ax.set_xlabel("w")

    fig.suptitle("Mean localization score ± std by w (strong rows only)", y=1.01)
    fig.tight_layout()
    path = os.path.join(plot_dir, "loc_score_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {path}")

    # ---- 3) Runtime comparison ----
    med_times = (df.groupby("method")["time_s"].median() * 1000).reindex(methods)
    fig, ax = plt.subplots(figsize=(max(5, len(methods) * 1.2), 4))
    ax.bar(range(len(methods)), med_times.values)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Median runtime (ms)")
    ax.set_title("Runtime per method")
    fig.tight_layout()
    path = os.path.join(plot_dir, "runtime_comparison.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")

# ---------------------------------------------------------------------------
# BENCHMARK MODE 3: real data evaluation
# ---------------------------------------------------------------------------

def iou(a1: int, b1: int, a2: int, b2: int) -> float:
    """Intersection over Union for two intervals [a1,b1] and [a2,b2]."""
    inter = max(0, min(b1, b2) - max(a1, a2) + 1)
    union = max(b1, b2) - min(a1, a2) + 1
    if union <= 0:
        return 0.0
    return inter / union


def run_real_data_benchmark(
    *,
    parquet_path: str,
    families: dict,
    iou_thresh: float = 0.75,
    out_dir: str = "data/real_data_benchmark",
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    Run benchmark methods on the labelled real dataset and compute
    classification metrics using IoU >= iou_thresh as the TP criterion.

    For rows with label=True  (anomaly present):
        TP if predicted interval has IoU >= iou_thresh with [start, end]
        FN otherwise
    For rows with label=False (no anomaly):
        TN if predicted score is below a threshold (score < 0.5)
        FP if predicted score >= 0.5

    Parameters
    ----------
    parquet_path : str
        Path to the labelled dataset parquet.
    families : dict
        Method configs from ALL_METHODS.
    iou_thresh : float
        IoU threshold for a detection to count as TP. Default 0.75.
    out_dir : str
        Directory for CSV outputs.
    """
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    if max_rows is not None:
        # Balanced sample: 50% positive, 50% negative
        pos = df[df["label"] == True]
        neg = df[df["label"] == False]
        n_pos = min(len(pos), max_rows // 2)
        n_neg = min(len(neg), max_rows - n_pos)
        df = pd.concat([
            pos.sample(n=n_pos, random_state=42),
            neg.sample(n=n_neg, random_state=42),
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"[test mode] Balanced sample: {n_pos} positive + {n_neg} negative = {len(df)} rows")
    print(f"Loaded {len(df)} rows  "
          f"(label=True: {df['label'].sum()}, label=False: {(~df['label']).sum()})")
    

    all_rows = []

    for idx, row in df.iterrows():
        x   = np.asarray(row["amplitude"],       dtype=np.float64)
        freqs = np.asarray(row["frequency_array"] / 1e9,       dtype=np.float64)
        n   = x.size
        gt_label = bool(row["label"])
        gt_start = int(row["start"])
        gt_end   = int(row["end"])

        freq_step = abs(freqs[1] - freqs[0]); L = len(freqs)
        r = ref_freq / (freq_step if freq_step > 0 else 1.)
        W = int(round(max(3, min(r, L / 16))))
        R = 3*W

        for name, cfg in families.items():
            t0 = time.perf_counter()
            cfg_row = dict(cfg)
            cfg_row["buffer"] = int(n//20)
            cfg_row["sr_factor"] = max(1, 2 ** math.ceil(math.log2(max(1, math.ceil((n + 1) / 450)))))
            try:
                score, (a, b) = _run_one(x, freqs, W, R, cfg_row)
            except Exception:
                score, a, b = 0.0, 0, -1
            dt = time.perf_counter() - t0

            # Determine TP/FP/TN/FN
            if gt_label:
                # Positive ground truth — check IoU
                if a >= 0 and b > a:
                    overlap = iou(a, b, gt_start, gt_end)
                    tp = int(overlap >= iou_thresh)
                else:
                    tp = 0
                fn = 1 - tp
                fp = 0
                tn = 0
            else:
                # Negative ground truth — use score threshold
                predicted_positive = score >= 0.5
                tp = 0
                fn = 0
                fp = int(predicted_positive)
                tn = int(not predicted_positive)

            all_rows.append({
                "idx":       idx,
                "method":    name,
                "label":     gt_label,
                "score":     float(score),
                "pred_a":    int(a),
                "pred_b":    int(b),
                "gt_start":  gt_start,
                "gt_end":    gt_end,
                "iou":       float(iou(a, b, gt_start, gt_end)) if gt_label and b > a else 0.0,
                "tp":        tp,
                "fp":        fp,
                "tn":        tn,
                "fn":        fn,
                "time_s":    dt,
            })

        if idx % 500 == 0:
            print(f"  Processed {idx}/{len(df)} rows...")

    df_results = pd.DataFrame(all_rows)

    # --- Compute metrics per method ---
    summary_rows = []
    for method, sub in df_results.groupby("method"):
        tp = sub["tp"].sum()
        fp = sub["fp"].sum()
        tn = sub["tn"].sum()
        fn = sub["fn"].sum()

        total     = tp + fp + tn + fn
        accuracy  = (tp + tn) / total       if total > 0  else 0.0
        precision = tp / (tp + fp)          if tp + fp > 0 else 0.0
        recall    = tp / (tp + fn)          if tp + fn > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if precision + recall > 0 else 0.0)
        med_iou   = float(sub.loc[sub["label"], "iou"].median())

        summary_rows.append({
            "method":    method,
            "tp":        int(tp),
            "fp":        int(fp),
            "tn":        int(tn),
            "fn":        int(fn),
            "accuracy":  round(accuracy,  4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "median_iou": round(med_iou,  4),
            "mean_time_ms": round(sub["time_s"].mean() * 1000, 3),
        })

    summary = pd.DataFrame(summary_rows).sort_values("f1", ascending=False)

    print("\n=== Real Data Benchmark Results ===")
    print(summary.to_string(index=False))

    results_path = os.path.join(out_dir, "real_data_results.csv")
    summary_path = os.path.join(out_dir, "real_data_summary.csv")
    df_results.to_csv(results_path, index=False)
    summary.to_csv(summary_path,    index=False)
    print(f"\nSaved: {results_path}")
    print(f"Saved: {summary_path}")

    return summary

def run_iou_gridsearch(
    parquet_path: str,
    families: dict,
    iou_values: list = None,
    max_rows: int = 1000,
    out_dir: str = "data/real_data_benchmark",
) -> pd.DataFrame:
    import os
    os.makedirs(out_dir, exist_ok=True)

    if iou_values is None:
        iou_values = [round(v, 2) for v in np.arange(0.50, 1.01, 0.05)]

    # Load and balance
    df = pd.read_parquet(parquet_path)
    pos = df[df["label"] == True]
    neg = df[df["label"] == False]
    n_pos = min(len(pos), max_rows // 2)
    n_neg = min(len(neg), max_rows - n_pos)
    df = pd.concat([
        pos.sample(n=n_pos, random_state=42),
        neg.sample(n=n_neg, random_state=42),
    ]).sort_values("label", ascending=False).reset_index(drop=True)
    print(f"Dataset: {n_pos} positive + {n_neg} negative = {len(df)} rows")

    # Run all methods — store full per-row predictions including time and score
    print("Running methods...")
    raw_rows = []

    for idx, row in df.iterrows():
        x     = np.asarray(row["amplitude"],       dtype=np.float64)
        freqs = np.asarray(row["frequency_array"] / 1e9, dtype=np.float64)
        n     = x.size

        freq_step = abs(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        r  = ref_freq / (freq_step if freq_step > 0 else 1.0)
        W  = int(round(max(3, min(r, n / 16))))
        R  = 3 * W

        import math
        sr_factor = max(1, 2 ** math.ceil(
            math.log2(max(1, math.ceil((n + 1) / 450)))))

        gt_label = bool(row["label"])
        gt_start = int(row["start"])
        gt_end   = int(row["end"])

        for name, cfg in families.items():
            cfg_row = dict(cfg)
            cfg_row["sr_factor"] = sr_factor

            t0 = time.perf_counter()
            try:
                score, (a, b) = _run_one(x, freqs, W, R, cfg_row)
            except Exception:
                score, a, b = 0.0, 0, -1
            dt = time.perf_counter() - t0

            iou_val = float(iou(a, b, gt_start, gt_end)) \
                      if gt_label and b > a else 0.0

            raw_rows.append({
                "idx":        idx,
                "method":     name,
                "label":      gt_label,
                "gt_start":   gt_start,
                "gt_end":     gt_end,
                "pred_a":     int(a),
                "pred_b":     int(b),
                "score":      float(score),    # ← epidemic score
                "iou_val":    iou_val,         # ← IoU with GT
                "time_ms":    dt * 1000,       # ← runtime in ms
                "n":          n,
                "W":          W,
                "R":          R,
                "sr_factor":  sr_factor,
            })

        if idx % 10 == 0:
            print(f"  {idx}/{len(df)} rows...")

    raw_df = pd.DataFrame(raw_rows)

    # Save full per-row predictions immediately
    raw_path = os.path.join(out_dir, "iou_raw_predictions.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved raw predictions: {raw_path}")

    # Grid search over IoU thresholds
    print("Grid searching IoU thresholds...")
    summary_rows = []

    for thresh in iou_values:
        for method, sub in raw_df.groupby("method"):
            pos_rows = sub[sub["label"] == True]
            neg_rows = sub[sub["label"] == False]

            tp = int((pos_rows["iou_val"] >= thresh).sum())
            fn = int((pos_rows["iou_val"] <  thresh).sum())
            fp = int((neg_rows["score"]   >= 0.5).sum())
            tn = int((neg_rows["score"]   <  0.5).sum())

            total     = tp + fp + tn + fn
            accuracy  = (tp + tn) / total        if total > 0       else 0.0
            precision = tp / (tp + fp)           if tp + fp > 0     else 0.0
            recall    = tp / (tp + fn)           if tp + fn > 0     else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if precision + recall > 0 else 0.0)

            summary_rows.append({
                "iou_thresh":   thresh,
                "method":       method,
                "tp":           tp, "fp": fp, "tn": tn, "fn": fn,
                "accuracy":     round(accuracy,  4),
                "precision":    round(precision, 4),
                "recall":       round(recall,    4),
                "f1":           round(f1,        4),
                "mean_score_pos":   round(float(pos_rows["score"].mean()),   4),
                "mean_score_neg":   round(float(neg_rows["score"].mean()),   4),
                "mean_time_ms":     round(float(sub["time_ms"].mean()),      3),
                "median_iou_pos":   round(float(pos_rows["iou_val"].median()), 4),
            })

    results = pd.DataFrame(summary_rows)

    pivot = results.pivot(index="iou_thresh", columns="method", values="f1")
    print("\n=== F1 Score Grid ===")
    print(pivot.to_string())

    out_path = os.path.join(out_dir, "iou_gridsearch.csv")
    results.to_csv(out_path, index=False)
    print(f"Saved summary: {out_path}")

    # --- Plot window comparisons for NWKR on positive rows ---
    for method in [m for m in ["nwkr_gaussian", "nwkr_laplace"] if m in raw_df["method"].unique()]:
        pos_raw  = raw_df[(raw_df["method"] == method) & (raw_df["label"] == True)]
        plot_dir = os.path.join(out_dir, "window_plots", method)
        os.makedirs(plot_dir, exist_ok=True)

        for i, (_, prow) in enumerate(pos_raw.iterrows()):
            orig  = df.iloc[int(prow["idx"])]
            freq  = np.asarray(orig["frequency_array"], dtype=float)
            amp   = np.asarray(orig["amplitude"],       dtype=float)
            n_ch  = len(freq)

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(freq, amp, lw=0.8, color="C0", label="Spectrum")

            gs = int(prow["gt_start"]); ge = int(prow["gt_end"])
            if 0 <= gs < ge < n_ch:
                ax.axvspan(freq[gs], freq[ge], color="C2", alpha=0.25, label="GT window")

            pa = int(prow["pred_a"]); pb = int(prow["pred_b"])
            if 0 <= pa < pb < n_ch:
                ax.axvspan(freq[pa], freq[pb], color="C1", alpha=0.25, label="Pred window")

            ax.set_title(
                f"{method}  |  IoU={prow['iou_val']:.3f}  |  "
                f"GT=[{gs},{ge}]  Pred=[{pa},{pb}]  |  score={prow['score']:.4f}",
                fontsize=8)
            ax.set_xlabel("Frequency", fontsize=8)
            ax.set_ylabel("Amplitude", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, ncol=3)
            fig.tight_layout()

            safe_iou = f"{prow['iou_val']:.3f}".replace(".", "_")
            fig.savefig(
                os.path.join(plot_dir, f"row_{i:04d}_iou{safe_iou}.png"),
                dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"Saved window plots: {plot_dir}")

    return results

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CosmicAI benchmark harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ---- shared method selection argument ----
    def add_common(p):
        p.add_argument(
            "--methods", type=str, default=None,
            help=(
                "Comma-separated list of method keys to include. "
                f"Available: {', '.join(ALL_METHODS)}. "
                "Default depends on benchmark mode."
            ),
        )
        p.add_argument("--out-dir", type=str, default=None,
                       help="Directory for CSV and figure outputs.")
        p.add_argument("--no-warmup", action="store_true",
                       help="Skip JIT warmup pass before timing.")

    # ---- runtime sub-command ----
    rt = sub.add_parser("runtime", help="Runtime scaling benchmark.")
    add_common(rt)
    rt.add_argument(
        "--variable", choices=["n", "w", "r"], default="n",
        help="Which variable to sweep (spectrum length n, window width w, or range cap r).",
    )
    rt.add_argument("--n",    type=int, default=1000, help="Fixed n (when variable != n).")
    rt.add_argument("--w",    type=int, default=10,   help="Fixed w (when variable != w).")
    rt.add_argument("--r",    type=int, default=30,   help="Fixed r (when variable != r).")
    rt.add_argument("--reps", type=int, default=5,    help="Repetitions per (elem, method) pair.")
    rt.add_argument("--timeout", type=float, default=30.0,
                    help="Per-call timeout in seconds for naive methods.")

    # ---- parameter sub-command ----
    pm = sub.add_parser("parameter", help="Detection quality / parameter benchmark.")
    add_common(pm)
    pm.add_argument(
        "--variable", choices=["snr", "noise", "w"], default="snr",
        help="Which variable to sweep.",
    )
    pm.add_argument("--w",    type=int,   nargs="+", default=[5, 25, 50, 100, 200],
                    help="Window widths to sweep.")
    pm.add_argument("--snr",  type=float, nargs="+", default=[2.5],
                    help="Signal-to-noise ratio values.")
    pm.add_argument("--noise", type=float, nargs="+", default=[0.05],
                    help="Noise level values (paired positionally with --snr if single, "
                         "or as a cross-product).")
    pm.add_argument("--n-spectrum", type=int, default=500,
                    help="Spectrum length for each synthetic row.")
    pm.add_argument("--n-rows",     type=int, default=100,
                    help="Number of rows per (w, error_param) combination.")
    pm.add_argument("--strong-rate", type=float, default=0.5,
                    help="Fraction of rows that contain a strong anomaly.")
    pm.add_argument("--strong-kind", type=str, default="rect_step",
                    help="Anomaly shape, e.g. rect_step.")
    pm.add_argument("--seed", type=int, default=123)
    pm.add_argument("--save-sra-sri", action="store_true",
                    help="Save SRA/SRI diagnostic plots for NWKR methods on strong rows.")
    
    # ---- real data sub-command ----
    rd = sub.add_parser("realdata", help="Benchmark on labelled real dataset.")
    add_common(rd)
    rd.add_argument("--parquet",    required=True,
                    help="Path to labelled dataset parquet.")
    rd.add_argument("--iou-thresh", type=float, default=0.75,
                    help="IoU threshold for TP. Default 0.75.")
    rd.add_argument("--max-rows", type=int, default=None,
                help="Limit to first N rows for testing. Default: all rows.")

    return parser


def _resolve_methods(methods_str: Optional[str], default_keys: List[str]) -> dict:
    if methods_str is None:
        keys = default_keys
    else:
        keys = [k.strip() for k in methods_str.split(",") if k.strip()]
    unknown = [k for k in keys if k not in ALL_METHODS]
    if unknown:
        raise ValueError(
            f"Unknown method(s): {unknown}. "
            f"Available: {sorted(ALL_METHODS)}"
        )
    return {k: ALL_METHODS[k] for k in keys}


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "runtime":
        families = _resolve_methods(args.methods, _DEFAULT_RUNTIME_METHODS)
        out_dir  = args.out_dir or "data/runtime_scaling"
        print(f"Runtime benchmark | variable={args.variable} | methods={list(families)}")
        run_runtime_benchmark(
            variable=args.variable,
            n_fixed=args.n,
            w_fixed=args.w,
            r_fixed=args.r,
            families=families,
            reps=args.reps,
            warmup=not args.no_warmup,
            out_dir=out_dir,
            csv_name=f"runtimes_{args.variable}.csv",
            fig_name=f"runtimes_{args.variable}.png",
            timeout_s=args.timeout,
        )

    elif args.mode == "parameter":
        families = _resolve_methods(args.methods, _DEFAULT_PARAMETER_METHODS)
        out_dir  = args.out_dir or "data/parameter_benchmark"
        plot_dir = os.path.join(out_dir.replace("data/", "images/", 1), "plots") \
                   if "data/" in out_dir else os.path.join(out_dir, "plots")

        # Build error_params as cross-product of snr × noise
        snr_list   = args.snr
        noise_list = args.noise
        error_params = [(s, n) for s in snr_list for n in noise_list]

        print(
            f"Parameter benchmark | w={args.w} | "
            f"error_params={error_params} | methods={list(families)}"
        )
        run_parameter_benchmark_ar2(
            variable=args.variable,
            w_list=args.w,
            error_params=error_params,
            n_spectrum=args.n_spectrum,
            n_rows=args.n_rows,
            strong_rate=args.strong_rate,
            families=families,
            seed=args.seed,
            out_dir=out_dir,
            plot_dir=plot_dir,
        )
    
    elif args.mode == "realdata":
        families = _resolve_methods(
            args.methods,
            ["mean", "poly_deg1", "poly_deg2", "nwkr_gaussian", "nwkr_laplace"]
        )
        run_real_data_benchmark(
            parquet_path = args.parquet,
            families     = families,
            iou_thresh   = args.iou_thresh,
            out_dir      = args.out_dir or "data/real_data_benchmark",
            max_rows     = args.max_rows,
        )


if __name__ == "__main__":
    main()
