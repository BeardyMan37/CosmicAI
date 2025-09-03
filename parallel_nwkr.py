#!/usr/bin/env python3
"""
Spectral window scanning via NWKR (Nadaraya–Watson kernel regression).

This script:
1) Loads spectrogram arrays and metadata (CSV or Parquet).
2) Aligns frequency bins to an atmospheric transmission curve and estimates
   interference trough ranges.
3) (Optionally) super-resolves by block-averaging for a fast coarse scan.
4) Scans each spectrum to find the best contiguous window using a variance-based
   NWKR score.
5) Refines windows back at native resolution when super-resolution was used.
6) Saves ranked CSVs and plots top-K overlays with predicted curves.

Usage:
  python parallel_nwkr.py --data-path Data/spotcheck.csv --interference-path Data/full_spectrum.gzip
"""

from __future__ import annotations

import argparse
import ast
import logging
import math
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.signal import find_peaks, peak_widths
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------------------------------------------------------#
# Globals
# -----------------------------------------------------------------------------#

# Reference frequency step (GHz) used to derive kernel size scaling
ref_freq: float = 0.0625

# Cache for Gaussian kernels keyed by (length, width)
_kernel_cache: Dict[Tuple[int, float], np.ndarray] = {}


# -----------------------------------------------------------------------------#
# I/O & preprocessing
# -----------------------------------------------------------------------------#

def match_and_correct(
    freq_array: np.ndarray,
    trans_freqs: np.ndarray,
    trans_vals: np.ndarray,
) -> np.ndarray:
    """Map transmission values to the nearest transmission frequency sample.

    Assumes `trans_freqs` is sorted ascending.

    Args:
      freq_array: 1D array of frequencies (GHz) for one spectrum row.
      trans_freqs: 1D sorted array of transmission frequency samples (GHz).
      trans_vals: 1D array of transmission (%) aligned with `trans_freqs`.

    Returns:
      1D array of transmission (%) matched to `freq_array` by nearest neighbor.
    """
    idxs = np.searchsorted(trans_freqs, freq_array)
    idxs[idxs == len(trans_freqs)] = len(trans_freqs) - 1
    left = np.maximum(idxs - 1, 0)
    right = idxs
    dl = np.abs(freq_array - trans_freqs[left])
    dr = np.abs(trans_freqs[right] - freq_array)
    nearest = np.where(dl <= dr, left, right)
    return trans_vals[nearest]


def _parse_freqs(s: str) -> np.ndarray:
    """Parse a stringified list of Hz into GHz numpy array (CSV path).

    Args:
      s: String representing a Python list/tuple of float frequencies in Hz.

    Returns:
      1D numpy array of floats in GHz.
    """
    freqs = np.array(ast.literal_eval(s), dtype=float)
    return freqs / 1e9


def load_data_by_length(
    data_path: str,
    interference_path: str,
) -> Tuple[pd.DataFrame, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[Tuple[int, int]]]]]]:
    """Load spectra & metadata, annotate interference, and group by row length.

    For CSV, `frequency_array` and `amplitude_corr_tsys` are stringified arrays.
    For Parquet, they are assumed already materialized lists/arrays.

    Args:
      data_path: Path to input spectrogram table (.csv or .parquet).
      interference_path: Parquet file with columns:
                         'Frequency (GHz)', 'Transmission (%)'.

    Returns:
      df: Filtered DataFrame with added columns:
          - 'transmission_array' : matched transmission per row (np.ndarray)
          - 'atmospheric_interference' : list of (start_idx, end_idx) pairs
      groups: dict mapping spectrum length L to a tuple:
              (actual_specs_L, uid_L, ref_L, ant_L, pol_L, freqs_L, atm_intrf_L)
              where:
                actual_specs_L: (N,L) float32/float64 array
                uid_L, ref_L, ant_L, pol_L: 1D arrays (object/str)
                freqs_L: (N,L) float array in GHz
                atm_intrf_L: list of lists of (start,end) index tuples
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data_path not found: {data_path}")
    if not os.path.exists(interference_path):
        raise FileNotFoundError(f"interference_path not found: {interference_path}")

    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path, sep="|", dtype=str, header=0)
        df["uid"] = df.index
        df = df.reset_index(drop=True)
        df["frequency_array"] = df["frequency_array"].apply(_parse_freqs)
        trans_df = pd.read_parquet(interference_path)
        trans_freqs = trans_df["Frequency (GHz)"].to_numpy()
        trans_vals = trans_df["Transmission (%)"].to_numpy()

        df["transmission_array"] = df.apply(
            lambda row: match_and_correct(
                np.asarray(row["frequency_array"], dtype=float), trans_freqs, trans_vals
            ),
            axis=1,
        )

        # Estimate atmospheric trough ranges per row (closed intervals in index space)
        interference: List[List[Tuple[int, int]]] = []
        for index in df.index:
            freqs = np.asarray(df.at[index, "frequency_array"], dtype=float)
            trans = np.asarray(df.at[index, "transmission_array"], dtype=float)

            troughs, _ = find_peaks(-trans, prominence=1)
            _, _, left_ips, right_ips = peak_widths(-trans, troughs, rel_height=0.75)

            left_freqs = np.interp(left_ips, np.arange(len(freqs)), freqs)
            right_freqs = np.interp(right_ips, np.arange(len(freqs)), freqs)
            widths_freq = right_freqs - left_freqs

            trough_freqs = freqs[troughs]
            trough_ranges = np.column_stack(
                (trough_freqs - widths_freq / 2.0, trough_freqs + widths_freq / 2.0)
            )

            closest_idxs: List[Tuple[int, int]] = []
            for start_f, end_f in trough_ranges:
                s_idx = int(np.abs(freqs - start_f).argmin())
                e_idx = int(np.abs(freqs - end_f).argmin())
                closest_idxs.append((s_idx, e_idx))
            interference.append(closest_idxs)

        df["atmospheric_interference"] = interference

        actual_specs = [np.array(ast.literal_eval(s), dtype=float)
                        for s in df["amplitude_corr_tsys"]]
        freqs = [np.array(x, dtype=float) for x in df["frequency_array"].tolist()]

    elif data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
        df["uid"] = df.index
        df = df.reset_index(drop=True)
        # Hz->GHz
        df["frequency_array"] = df["frequency_array"].apply(lambda xs: [f / 1e9 for f in xs])
        trans_df = pd.read_parquet(interference_path)
        trans_freqs = trans_df["Frequency (GHz)"].to_numpy()
        trans_vals = trans_df["Transmission (%)"].to_numpy()

        df["transmission_array"] = df.apply(
            lambda row: match_and_correct(
                np.asarray(row["frequency_array"], dtype=float), trans_freqs, trans_vals
            ),
            axis=1,
        )

        interference = []
        for index in df.index:
            freqs = np.asarray(df.at[index, "frequency_array"], dtype=float)
            trans = np.asarray(df.at[index, "transmission_array"], dtype=float)

            troughs, _ = find_peaks(-trans, prominence=1)
            _, _, left_ips, right_ips = peak_widths(-trans, troughs, rel_height=0.75)

            left_freqs = np.interp(left_ips, np.arange(len(freqs)), freqs)
            right_freqs = np.interp(right_ips, np.arange(len(freqs)), freqs)
            widths_freq = right_freqs - left_freqs

            trough_freqs = freqs[troughs]
            trough_ranges = np.column_stack(
                (trough_freqs - widths_freq / 2.0, trough_freqs + widths_freq / 2.0)
            )

            closest_idxs: List[Tuple[int, int]] = []
            for start_f, end_f in trough_ranges:
                s_idx = int(np.abs(freqs - start_f).argmin())
                e_idx = int(np.abs(freqs - end_f).argmin())
                closest_idxs.append((s_idx, e_idx))
            interference.append(closest_idxs)

        df["atmospheric_interference"] = interference

        actual_specs = [np.asarray(x, dtype=float) for x in df["amplitude_corr_tsys"].tolist()]
        freqs = [np.asarray(x, dtype=float) for x in df["frequency_array"].tolist()]

    else:
        raise ValueError(f"Unsupported extension: {data_path!r}")

    # Drop all-zero rows and propagate consistent indexing
    df["_actual_spec"] = actual_specs
    df["_freqs"] = freqs
    df["_keep"] = [not np.all(s == 0.0) for s in df["_actual_spec"]]
    df = df[df["_keep"]].reset_index(drop=True)

    actual_specs = list(df["_actual_spec"])
    freqs = list(df["_freqs"])
    atm_intrf = list(df["atmospheric_interference"])
    uid = df["uid"].to_numpy()
    ref = df["ref_antenna_name"].to_numpy()
    ant = df["antenna"].to_numpy()
    pol = df["polarization"].to_numpy()

    # Group by spectrum length
    length_groups: Dict[int, List[int]] = {}
    for i, s in enumerate(actual_specs):
        L = s.shape[0]
        length_groups.setdefault(L, []).append(i)

    # Materialize grouped arrays
    groups: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[Tuple[int, int]]]]] = {}
    for L, idxs in length_groups.items():
        actual_specs_L = np.vstack([actual_specs[i] for i in idxs])
        freqs_L = np.vstack([freqs[i] for i in idxs])
        atm_intrf_L = [atm_intrf[i] for i in idxs]
        uid_L = uid[idxs]
        ref_L = ref[idxs]
        ant_L = ant[idxs]
        pol_L = pol[idxs]
        groups[L] = (actual_specs_L, uid_L, ref_L, ant_L, pol_L, freqs_L, atm_intrf_L)

    return df, groups


# -----------------------------------------------------------------------------#
# Kernel helpers
# -----------------------------------------------------------------------------#

def precompute_kernel(L: int, w: float) -> np.ndarray:
    """Precompute a dense Gaussian kernel K[i,j] = exp(-|i-j|^2 / w^2).

    Args:
      L: Kernel size (length of the row).
      w: Kernel width (in index units). Must be > 0.

    Returns:
      (L,L) numpy array.
    """
    if w <= 0:
        raise ValueError("Kernel width w must be positive.")
    idx = np.arange(L)
    D = np.abs(np.subtract.outer(idx, idx))
    return np.exp(-(D * D) / (w * w))


def _get_kernel(n: int, w: float) -> np.ndarray:
    """Return cached kernel for (n, w), computing if necessary."""
    key = (n, float(w))
    K = _kernel_cache.get(key)
    if K is None:
        K = precompute_kernel(n, w)
        _kernel_cache[key] = K
    return K


# -----------------------------------------------------------------------------#
# NWKR scoring (numba JIT)
# -----------------------------------------------------------------------------#

@njit(cache=True, fastmath=True, parallel=True)
def calculate_nwkr_sra(array: np.ndarray, W: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute NWKR predictions and sum of squared residuals (SSR).

    Args:
      array: 1D array of observations (length n).
      W: (n,n) kernel weight matrix (symmetric, positive).

    Returns:
      ssr: scalar sum of squared residuals.
      ssr_array: 1D per-index residual^2.
      pred_array: 1D predictions at each index.
    """
    n = array.shape[0]
    numer = np.empty(n, dtype=array.dtype)
    denom = np.empty(n, dtype=array.dtype)
    for i in prange(n):
        s_num = 0.0
        s_den = 0.0
        for j in range(n):
            w_ij = W[i, j]
            s_num += w_ij * array[j]
            s_den += w_ij
        numer[i] = s_num
        denom[i] = s_den
    ssr = 0.0
    ssr_array = np.empty(n, dtype=array.dtype)
    pred_array = np.empty(n, dtype=array.dtype)
    for i in prange(n):
        pred = numer[i] / denom[i] if denom[i] > 0.0 else 0.0
        pred_array[i] = pred
        diff = array[i] - pred
        ssr_array[i] = diff * diff
        ssr += ssr_array[i]
    return ssr, ssr_array, pred_array


@njit(cache=True, fastmath=True)
def predict_on_idxs(array: np.ndarray, idxs: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Predict NWKR values at a subset of indices using only the subset as support.

    NOTE: This uses *subset-local* support (weights from idxs only). If you
    intend to predict w.r.t. the full row, implement a global-support variant.

    Args:
      array: 1D row.
      idxs: 1D integer indices to predict at.
      W: full (n,n) kernel.

    Returns:
      1D predictions for positions in `idxs`.
    """
    m = idxs.shape[0]
    out = np.empty(m, dtype=array.dtype)
    for ii in range(m):
        i0 = idxs[ii]
        num = 0.0
        den = 0.0
        for jj in range(m):
            j0 = idxs[jj]
            w_ij = W[i0, j0]
            num += w_ij * array[j0]
            den += w_ij
        out[ii] = num / den if den > 1e-12 else 0.0
    return out


@njit(cache=True, fastmath=True, parallel=True)
def ssr_region(
    array: np.ndarray,
    idxs: np.ndarray,
    W: np.ndarray,
    ssr_array: np.ndarray,
    a: int,
    b: int,
    range_cap: int,
) -> float:
    """Compute SSR over a region and its near/far complements.

    Args:
      array: 1D row values.
      idxs: 1D indices to evaluate (either inside or outside set).
      W: (n,n) kernel.
      ssr_array: residuals^2 from full NWKR (for 'far' reuse).
      a: inclusive start of current hypothesis window in index space.
      b: inclusive end of current hypothesis window.
      range_cap: neighborhood half-width used to decide near/far.

    Returns:
      Scalar SSR contribution for this idx set.
    """
    m = idxs.shape[0]
    sri = 0.0
    sro_far = 0.0
    sro_near = 0.0
    low_cut = a - 2 * range_cap
    high_cut = b + 2 * range_cap

    for ii in prange(m):
        i0 = idxs[ii]
        if a <= i0 <= b:
            num = 0.0
            den = 0.0
            for jj in range(m):
                j0 = idxs[jj]
                w_ij = W[i0, j0]
                num += w_ij * array[j0]
                den += w_ij
            pred = num / den if den > 0.0 else 0.0
            diff = array[i0] - pred
            sri += diff * diff
        elif i0 < low_cut or i0 > high_cut:
            sro_far += ssr_array[i0]
        else:
            num = 0.0
            den = 0.0
            for jj in range(m):
                j0 = idxs[jj]
                w_ij = W[i0, j0]
                num += w_ij * array[j0]
                den += w_ij
            pred = num / den if den > 0.0 else 0.0
            diff = array[i0] - pred
            sro_near += diff * diff
    return sri + sro_near + sro_far


def score_variance_nwkr(
    array: np.ndarray,
    inside: np.ndarray,
    outside: np.ndarray,
    a: int,
    b: int,
    range_cap: int,
    W: np.ndarray,
    ssr_array: np.ndarray,
) -> float:
    """Variance score (negative SSR total) for a window [a,b].

    Returns:
      Negative total SSR (inside + outside components). Higher is better.
    """
    sri = ssr_region(array, inside, W, ssr_array, a, b, range_cap)
    sro = ssr_region(array, outside, W, ssr_array, a, b, range_cap)
    return -(sri + sro)


# -----------------------------------------------------------------------------#
# Row scan
# -----------------------------------------------------------------------------#

def _scan_row(params: Tuple[int, np.ndarray, List[Tuple[int, int]], np.ndarray, int, int]
              ) -> Tuple[int, Tuple[int, int], float, np.ndarray, np.ndarray | None, np.ndarray | None, int, int]:
    """Scan a single row to find the best contiguous window by NWKR score.

    Args:
      params: Tuple of (row_idx, row, ignore, freqs, buffer, sr_factor):
        row_idx: integer row id (for ordering).
        row: 1D spectrum values.
        ignore: list of (start_idx, end_idx) to ignore (interference).
        freqs: 1D frequency array (GHz), same length as row.
        buffer: number of channels to drop on each side.
        sr_factor: super-resolution block size used in the coarse pass.

    Returns:
      (row_idx,
       best_window_original,  # (start,end) indices (inclusive) in original (trimmed+buffer) coordinates
       best_score,            # normalized score
       pred_array,            # NWKR predictions for trimmed row
       best_sri_pred_idx_full,# inside indices (original coordinates) for best window (or None)
       best_sri_pred_vals,    # NWKR predictions on inside indices (or None)
       eff_kernel_size,       # kernel size in original scale (w*sr_factor)
       eff_range_cap)         # range cap in original scale (range_cap*sr_factor)
    """
    row_idx, row, ignore, freqs, buffer, sr_factor = params

    if len(freqs) < 2 or not np.isfinite(freqs[:2]).all():
        return (row_idx, (0, 0), -np.inf, np.array([]), None, None, 0, 0)

    freq_step = abs(freqs[1] - freqs[0])
    L = len(freqs)
    R = ref_freq / freq_step
    w = int(round(max(3, min(R / sr_factor, L / 16))))
    range_cap = 3 * w

    row_trimmed = row[buffer: len(row) - buffer]
    n_trimmed = row_trimmed.shape[0]
    if n_trimmed <= 0:
        return (row_idx, (0, 0), -np.inf, np.array([]), None, None, 0, 0)

    W_trimmed = _get_kernel(n_trimmed, w)
    sra, ssr_array, pred_array = calculate_nwkr_sra(row_trimmed, W_trimmed)
    sra = sra if sra > 1e-12 else 1e-12

    # Interference → mask
    ignore_trimmed: List[Tuple[int, int]] = []
    for (start, end) in ignore:
        s0 = max(start - buffer, 0)
        e0 = min(end - buffer, n_trimmed - 1)
        if s0 < e0:
            ignore_trimmed.append((s0, e0))
    mask = np.ones(n_trimmed, dtype=np.bool_)
    for s0, e0 in ignore_trimmed:
        mask[s0:e0 + 1] = False
    valid_trimmed = np.nonzero(mask)[0]
    all_ch_trimmed = np.arange(n_trimmed)

    best_score = -np.inf
    best_window = (0, 0)
    best_sri_pred_idx_full = None
    best_sri_pred_vals = None

    for pos_i, i in enumerate(valid_trimmed):
        # Enforce contiguous starts
        if pos_i < len(valid_trimmed) - 1 and (valid_trimmed[pos_i + 1] - i) > 1:
            continue
        stop = min(pos_i + 1 + range_cap, len(valid_trimmed))
        sub_valid_trimmed = valid_trimmed[pos_i + 1: stop]
        for pos_j, j in enumerate(sub_valid_trimmed):
            # Keep contiguity for j; prune on first gap
            if pos_j > 0 and (sub_valid_trimmed[pos_j] - sub_valid_trimmed[pos_j - 1]) > 1:
                break

            # Inside indices as a contiguous slice in valid_trimmed
            lo = pos_i
            hi = pos_i + 1 + pos_j
            inside = valid_trimmed[lo:hi + 1]
            outside = np.setdiff1d(all_ch_trimmed, inside, assume_unique=True)

            sc = score_variance_nwkr(row_trimmed, inside, outside, i, j, range_cap, W_trimmed, ssr_array)
            sc = sc / sra + 1.0

            if sc > best_score:
                best_score = sc
                best_window = (i, j)
                best_sri_pred_idx_full = inside + buffer
                best_sri_pred_vals = predict_on_idxs(row_trimmed, inside, W_trimmed)

    oi, oj = best_window
    best_window_original = (oi + buffer, oj + buffer)
    return (
        row_idx,
        best_window_original,
        best_score,
        pred_array,
        best_sri_pred_idx_full,
        best_sri_pred_vals,
        w * sr_factor,
        range_cap * sr_factor,
    )


# -----------------------------------------------------------------------------#
# Parallel scan driver
# -----------------------------------------------------------------------------#

def polynomial_scan_ranges_parallel(
    spec_arrays: np.ndarray,
    score_fn: Callable[[Tuple[Any, ...]], Tuple[Any, ...]],
    atm_interfs: List[List[Tuple[int, int]]],
    freq_arrays: np.ndarray,
    buffer: int,
    sr_factor: int,
    max_workers: int | None = None,
) -> Tuple[List[Tuple[int, int]], List[float], List[np.ndarray], List[np.ndarray | None], List[np.ndarray | None], List[int], List[int]]:
    """Run the row scanner in parallel over all spectra.

    Args:
      spec_arrays: (N,L) array of spectra.
      score_fn: Callable applied per row (usually `_scan_row`).
      atm_interfs: list (len N) of interference ranges per row.
      freq_arrays: (N,L) array of frequencies (GHz).
      buffer: number of channels trimmed on each side.
      sr_factor: super-resolution block factor used in this pass.
      max_workers: process pool size (defaults to os.cpu_count()).

    Returns:
      Tuple of lists (each of len N):
        windows, scores, sra_preds, sri_idxs, sri_vals, ws, range_caps
    """
    n_rows, _ = spec_arrays.shape
    params = [
        (i, spec_arrays[i], atm_interfs[i], freq_arrays[i], buffer, sr_factor)
        for i in range(n_rows)
    ]

    results: List[Tuple[Any, ...]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(score_fn, p) for p in params]
        for f in as_completed(futures):
            results.append(f.result())

    results.sort(key=lambda x: x[0])
    windows = [r[1] for r in results]
    scores = [r[2] for r in results]
    sra_preds = [r[3] for r in results]
    sri_idxs = [r[4] for r in results]
    sri_vals = [r[5] for r in results]
    ws = [r[6] for r in results]
    range_caps = [r[7] for r in results]
    return windows, scores, sra_preds, sri_idxs, sri_vals, ws, range_caps


# -----------------------------------------------------------------------------#
# Visualization & post-processing
# -----------------------------------------------------------------------------#

def plot_top_k(
    df: pd.DataFrame,
    actual_spec_arrays: np.ndarray,
    windows: List[Tuple[int, int]],
    atm_interfs: List[List[Tuple[int, int]]],
    scores: List[float],
    meta: Dict[str, Any],
    ws: List[int],
    k: int = 10,
    per_fig: int = 10,
    buffer: int = 10,
    out_dir: str = "Images",
    data_dir: str = "Data",
    sra_preds: List[np.ndarray] | None = None,
    sri_idxs: List[np.ndarray | None] | None = None,
    sri_vals: List[np.ndarray | None] | None = None,
    sr_factor: int = 1,
) -> None:
    """Save ranked CSV and plot top-K rows with overlays.

    Args:
      df: Source dataframe (must contain 'uid').
      actual_spec_arrays: (N,L) spectra at native resolution.
      windows: best windows at native resolution.
      atm_interfs: interference ranges at native resolution.
      scores: ranking scores (higher is better).
      meta: dict with 'uid', 'ref', 'ant', 'pol', 'freq' arrays.
      ws: effective kernel sizes (native scale).
      k: top-K to plot (and include in summary CSV tail).
      per_fig: number of rows per PNG figure.
      buffer: native buffer used at plotting time.
      out_dir: directory to emit figures.
      data_dir: directory to emit CSV.
      sra_preds: list of NWKR predictions for (trimmed SR) rows (optional).
      sri_idxs: list of index arrays for inside-window predictions (optional).
      sri_vals: list of predicted values for inside-window indices (optional).
      sr_factor: upsampling factor to map SR predictions back to native.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    buf_orig = buffer

    scores_np = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores_np)
    indexes = np.where(finite)[0]
    top_all = indexes[np.argsort(scores_np[finite])[:]][::-1]

    top_uids = np.asarray(meta["uid"])[top_all]
    top_scores = scores_np[top_all]
    top_windows = np.asarray(windows, dtype=object)[top_all]
    top_ws = np.asarray(ws, dtype=object)[top_all]

    sub_df = df.loc[df["uid"].isin(top_uids)].copy()
    score_map = dict(zip(top_uids, top_scores))
    window_map = dict(zip(top_uids, top_windows))
    kernel_map = dict(zip(top_uids, top_ws))

    sub_df["score"] = sub_df["uid"].map(score_map)
    sub_df["kernel_size"] = sub_df["uid"].map(kernel_map)
    sub_df[["win_start", "win_end"]] = sub_df["uid"].map(window_map).apply(pd.Series)
    sub_df = sub_df.sort_values("score", ascending=False).reset_index(drop=True)
    sub_df.insert(0, "uid", sub_df.pop("uid"))

    out_csv = os.path.join(
        data_dir,
        f"bandpass_qa0_no_partitions_labelled_filt_scan_stat_length_{actual_spec_arrays.shape[1]}.csv",
    )
    sub_df.to_csv(out_csv, index=False)
    logging.info("Wrote summary CSV: %s", out_csv)

    # Slice top-k for plotting
    top = indexes[np.argsort(scores_np[finite])[-k:]][::-1]
    df_slice = df.iloc[top].copy()
    df_slice["orig_idx"] = top
    df_slice["score"] = scores_np[top]
    df_slice["kernel_size"] = [ws[i] for i in top]
    df_slice["win_start"] = [windows[i][0] for i in top]
    df_slice["win_end"] = [windows[i][1] for i in top]

    n_figs = math.ceil(len(df_slice) / per_fig)
    for fig_i in range(n_figs):
        chunk = df_slice.iloc[fig_i * per_fig : (fig_i + 1) * per_fig]
        fig_k = len(chunk)
        fig, axes = plt.subplots(fig_k, 1, figsize=(10, 3 * fig_k))
        if fig_k == 1:
            axes = [axes]

        for ax, row in zip(axes, chunk.itertuples()):
            i0 = row.orig_idx
            spec = actual_spec_arrays[i0]
            a, b = row.win_start, row.win_end

            for (c, d) in atm_interfs[i0]:
                ax.axvspan(c, d, color="C9", alpha=0.2)

            x = np.arange(len(spec))
            ax.plot(x, spec, color="C0", label="Actual")

            if buf_orig > 0:
                ax.axvspan(0, buf_orig - 1, color="gray", alpha=0.2)
                ax.axvspan(len(spec) - buf_orig, len(spec) - 1, color="gray", alpha=0.2)

            ax.axvspan(a, b, color="C1", alpha=0.3)

            if sra_preds is not None and sra_preds[i0] is not None and len(sra_preds[i0]) > 0:
                pred_full = np.full(len(spec), np.nan)
                sra_sr = sra_preds[i0]
                sra_up = np.repeat(sra_sr, sr_factor)
                end = len(spec) - buf_orig
                pred_full[buf_orig:end] = sra_up[: max(0, end - buf_orig)]
                ax.plot(np.arange(len(spec)), pred_full, ".", ms=2, label="SRA pred")

            if (sri_idxs is not None) and (sri_vals is not None) and (sri_idxs[i0] is not None):
                sri_full = np.full(len(spec), np.nan)
                idx_sr = np.asarray(sri_idxs[i0], dtype=int)
                val_sr = np.asarray(sri_vals[i0], dtype=float)
                idx_orig_start = idx_sr * sr_factor
                for p, v in zip(idx_orig_start, val_sr):
                    p_end = min(p + sr_factor, len(spec))
                    sri_full[p:p_end] = v
                ax.plot(np.arange(len(spec)), sri_full, ".", ms=2, label="SRI pred")

            ax.set_title(
                f"UID={row.orig_idx}  Score={row.score:.2f}  "
                f"Range=[{a},{b}] Kernel size={row.kernel_size}"
            )
            ax.set_xlabel("Channel")
            ax.set_ylabel("Amplitude")
            ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.suptitle(f"Items {fig_i * per_fig + 1}–{fig_i * per_fig + fig_k} of Top {k}", y=0.98)
        outpath = os.path.join(out_dir, f"top_{k}_fig{fig_i + 1}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logging.info("Wrote figure: %s", outpath)


# -----------------------------------------------------------------------------#
# Super-resolution and refinement
# -----------------------------------------------------------------------------#

def superresolve_ranges(ranges_list: List[List[Tuple[int, int]]], factor: int) -> List[List[Tuple[int, int]]]:
    """Downsample index ranges by integer factor and merge contiguous segments.

    Args:
      ranges_list: list of per-row lists of (start,end) integer indices.
      factor: positive integer block size.

    Returns:
      New list with each row's ranges downsampled and merged.
    """
    if factor < 1:
        raise ValueError("superresolve_ranges: factor must be >= 1")

    def merge(rs: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for s, e in rs:
            if not out or s > out[-1][1] + 1:
                out.append((s, e))
            else:
                out[-1] = (out[-1][0], max(out[-1][1], e))
        return out

    new: List[List[Tuple[int, int]]] = []
    for sub in ranges_list:
        adjusted = [(s // factor, e // factor) for s, e in sub]
        adjusted = sorted(set(adjusted))
        merged = merge(adjusted)
        new.append(merged)
    return new


def superresolve(specs: np.ndarray, factor: int) -> np.ndarray:
    """Block-average along the channel axis by an integer factor.

    Args:
      specs: (N,L) array.
      factor: positive integer block size.

    Returns:
      (N, floor(L/factor)) array of block means.
    """
    if factor < 1:
        raise ValueError("superresolve: factor must be >= 1")
    n_rows, n_ch = specs.shape
    n_blk = n_ch // factor
    if n_blk == 0:
        return np.empty((n_rows, 0), dtype=specs.dtype)
    trimmed = specs[:, : n_blk * factor]
    return trimmed.reshape(n_rows, n_blk, factor).mean(axis=2)


def refine_windows_exact_for_length(
    spec_arrays: np.ndarray,
    windows_sr: List[Tuple[int, int]],
    atm_interfs: List[List[Tuple[int, int]]],
    ws: List[int],
    range_caps: List[int],
    sr_factor: int,
    buffer: int,
) -> List[Tuple[int, int]]:
    """Refine coarse (SR) windows on the native-resolution spectra.

    Args:
      spec_arrays: (N,L) native-resolution spectra.
      windows_sr: list of (start,end) on SR grid (inclusive).
      atm_interfs: native-resolution interference ranges.
      ws: per-row kernel sizes to use at native resolution.
      range_caps: per-row range caps (native scale).
      sr_factor: SR factor used in coarse step.
      buffer: native-resolution buffer trimmed at plotting/scoring time.

    Returns:
      List of (start,end) at native resolution for each row.
    """
    n_rows, n_ch = spec_arrays.shape
    n_trimmed = n_ch - 2 * buffer
    if n_trimmed <= 0:
        return [(0, 0)] * n_rows

    refined: List[Tuple[int, int]] = []

    for i in range(n_rows):
        W_trimmed = _get_kernel(n_trimmed, ws[i])
        range_cap = range_caps[i]
        row_trimmed = spec_arrays[i, buffer: n_ch - buffer]

        sra, ssr_array, _ = calculate_nwkr_sra(row_trimmed, W_trimmed)
        sra = sra if sra > 1e-12 else 1e-12

        # Build valid index mask
        mask = np.ones(n_trimmed, dtype=np.bool_)
        for (s, e) in atm_interfs[i]:
            s0 = max(s - buffer, 0)
            e0 = min(e - buffer, n_trimmed - 1)
            if s0 <= e0:
                mask[s0:e0 + 1] = False
        valid = np.nonzero(mask)[0]

        x_sr, y_sr = windows_sr[i]
        a_lo = max(x_sr * sr_factor - buffer, 0)
        a_hi = min((x_sr + 1) * sr_factor - 1 - buffer, n_trimmed - 1)
        b_lo = max(y_sr * sr_factor - buffer, 0)
        b_hi = min((y_sr + 1) * sr_factor - 1 - buffer, n_trimmed - 1)

        best_sc = -np.inf
        best_ab = (x_sr, y_sr)

        for a in range(a_lo, a_hi + 1):
            start_b = max(a, b_lo)
            for b in range(start_b, b_hi + 1):
                inside = valid[(valid >= a) & (valid <= b)]
                if inside.size == 0:
                    continue
                outside = np.setdiff1d(valid, inside, assume_unique=False)
                sc = score_variance_nwkr(row_trimmed, inside, outside, a, b, range_cap, W_trimmed, ssr_array)
                sc = sc / sra + 1.0
                if sc > best_sc:
                    best_sc = sc
                    best_ab = (a, b)

        a_t, b_t = best_ab
        refined.append((a_t + buffer, b_t + buffer))

    return refined


# -----------------------------------------------------------------------------#
# CLI / main
# -----------------------------------------------------------------------------#

def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    p = argparse.ArgumentParser(description="Scan spectrograms for best windows using NWKR.")
    p.add_argument("--data-path", required=True, help="Input table (.csv with '|' sep or .parquet).")
    p.add_argument("--interference-path", required=True, help="Parquet with columns 'Frequency (GHz)', 'Transmission (%)'.")
    p.add_argument("--top-k", type=int, default=100, help="Top-K rows to plot and include in summary tail.")
    p.add_argument("--per-fig", type=int, default=10, help="How many rows per figure.")
    p.add_argument("--buffer-coeff", type=int, default=20, help="BUFFER = length // buffer_coeff.")
    p.add_argument("--workers", type=int, default=None, help="Process pool size (default: os.cpu_count()).")
    p.add_argument("--out-root", default="Images", help="Root folder for figures (per-length subdirs will be created).")
    p.add_argument("--data-root", default="Data", help="Root folder for emitted CSVs (per-length subdirs).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    return p


def main() -> None:
    """Entrypoint: parse args, load data, run scan, refine, and plot."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    t0 = time.perf_counter()
    df, groups = load_data_by_length(args.data_path, args.interference_path)
    t1 = time.perf_counter()
    logging.info("Loaded & grouped data in %.3fs", (t1 - t0))
    logging.info("Found lengths: %s", sorted(groups.keys()))

    # Typical SR factors per channel length (fallback=1 if missing).
    length_SR_FACTOR_map: Dict[int, int] = {
        64: 1, 120: 1, 128: 1, 240: 1, 256: 1,
        480: 2, 512: 2, 960: 4, 1024: 4,
        1920: 8, 2048: 8, 3840: 16,
    }

    for length in sorted(groups):
        BUFFER = length // args.buffer_coeff
        actual_specs, uid, ref, ant, pol, freqs, atm_interfs = groups[length]
        n_rows, row_len = actual_specs.shape
        logging.info("Before Preprocessing: Length=%d: %d rows, %d channels", length, n_rows, row_len)

        SR_FACTOR = length_SR_FACTOR_map.get(length, 1)
        atm_interfs_sr = superresolve_ranges(atm_interfs, factor=SR_FACTOR)
        actual_specs_sr = superresolve(actual_specs, factor=SR_FACTOR)
        freqs_sr = superresolve(freqs, factor=SR_FACTOR)

        n_rows, row_len = actual_specs_sr.shape
        logging.info("After Preprocessing: Length=%d: %d rows, %d channels, SR_factor %d", length, n_rows, row_len, SR_FACTOR)

        meta = {"uid": uid, "ref": ref, "ant": ant, "pol": pol, "freq": freqs}

        t2 = time.perf_counter()
        windows_sr, scores, sra_preds, sri_idxs, sri_vals, ws, range_caps = polynomial_scan_ranges_parallel(
            spec_arrays=actual_specs_sr,
            score_fn=_scan_row,
            atm_interfs=atm_interfs_sr,
            freq_arrays=freqs_sr,
            buffer=BUFFER // SR_FACTOR,
            sr_factor=SR_FACTOR,
            max_workers=args.workers,
        )
        t3 = time.perf_counter()
        logging.info("  Scan time: %.3fs", (t3 - t2))

        out_dir = os.path.join(args.out_root, f"length_{length}")
        data_dir = os.path.join(args.data_root, f"length_{length}")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        if SR_FACTOR > 1:
            windows_exact = refine_windows_exact_for_length(
                actual_specs, windows_sr, atm_interfs, ws, range_caps, SR_FACTOR, BUFFER
            )
        else:
            windows_exact = windows_sr

        plot_top_k(
            df=df,
            actual_spec_arrays=actual_specs,
            windows=windows_exact,
            atm_interfs=atm_interfs,
            scores=scores,
            meta=meta,
            ws=ws,
            k=min(args.top_k, n_rows),
            per_fig=args.per_fig,
            buffer=BUFFER,
            out_dir=out_dir,
            data_dir=data_dir,
            sra_preds=sra_preds,
            sri_idxs=sri_idxs,
            sri_vals=sri_vals,
            sr_factor=SR_FACTOR,
        )


if __name__ == "__main__":
    main()
