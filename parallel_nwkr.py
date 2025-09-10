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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
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
        df["uid"] = df.index.copy()
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
        df["uid"] = df.index.copy()
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
              ) -> Tuple[
    # masked var-len (existing)
    int, Tuple[int, int], float, np.ndarray, np.ndarray | None, np.ndarray | None, int, int,
    # unmasked var-len (new) + overlap
    Tuple[int, int], float, float, int, np.ndarray | None, np.ndarray | None,
    # fixed-len linear (new) + overlap
    Tuple[int, int], float, float, int, np.ndarray | None, np.ndarray | None, int
]:
    """Scan a row in three modes: masked var-len, unmasked var-len, fixed-len linear.

    Returns (in order):
      A) masked var-len:
         row_idx, best_win_masked, best_sc_masked, pred_array_trimmed,
         sri_idx_masked, sri_vals_masked, eff_kernel_native, eff_range_cap_native
      B) unmasked var-len:
         best_win_unmasked, best_sc_unmasked, overlap_pct_unmasked, overlap_cnt_unmasked,
         sri_idx_unmasked, sri_vals_unmasked
      C) fixed-len linear (span ≈ ref_freq, unmasked):
         best_win_fixed, best_sc_fixed, overlap_pct_fixed, overlap_cnt_fixed,
         sri_idx_fixed, sri_vals_fixed, fixed_bins_native
    """
    row_idx, row, ignore, freqs, buffer, sr_factor = params

    def _overlap_stats(a_orig: int, b_orig: int, ignore_ranges: List[Tuple[int, int]]) -> Tuple[float, int]:
        if b_orig < a_orig:
            return 0.0, 0
        win_len = (b_orig - a_orig + 1)
        if win_len <= 0:
            return 0.0, 0
        overlap = 0
        for s, e in ignore_ranges:
            lo = max(a_orig, s)
            hi = min(b_orig, e)
            if hi >= lo:
                overlap += (hi - lo + 1)
        return overlap / win_len

    def _safe_freq_step(fs: np.ndarray) -> float:
        d = np.diff(fs)
        d = d[np.isfinite(d)]
        return float(np.nanmedian(d)) if d.size else 0.0

    # Guards
    if len(freqs) < 2 or not np.isfinite(freqs[:2]).all():
        return (
            row_idx, (0,0), -np.inf, np.array([]), None, None, 0, 0,
            (0,0), -np.inf, 0.0, 0, None, None,
            (0,0), -np.inf, 0.0, 0, None, None, 0
        )

    # Kernel/trim setup (shared)
    freq_step = abs(freqs[1] - freqs[0])
    L = len(freqs)
    R = ref_freq / (freq_step if freq_step > 0 else 1.0)
    w = int(round(max(3, min(R / sr_factor, L / 16))))
    range_cap = 3 * w

    row_trimmed = row[buffer: len(row) - buffer]
    n_trimmed = row_trimmed.shape[0]
    if n_trimmed <= 0:
        return (
            row_idx, (0,0), -np.inf, np.array([]), None, None, 0, 0,
            (0,0), -np.inf, 0.0, 0, None, None,
            (0,0), -np.inf, 0.0, 0, None, None, 0
        )

    W_trimmed = _get_kernel(n_trimmed, w)
    sra, ssr_array, pred_array = calculate_nwkr_sra(row_trimmed, W_trimmed)
    sra = sra if sra > 1e-12 else 1e-12

    # Build ignore mask for masked mode (trimmed coords)
    ignore_trimmed: List[Tuple[int, int]] = []
    for (start, end) in ignore:
        s0 = max(start - buffer, 0)
        e0 = min(end - buffer, n_trimmed - 1)
        if s0 < e0:
            ignore_trimmed.append((s0, e0))

    mask = np.ones(n_trimmed, dtype=np.bool_)
    for s0, e0 in ignore_trimmed:
        mask[s0:e0 + 1] = False

    all_trimmed = np.arange(n_trimmed)
    valid_masked = np.nonzero(mask)[0]

    # ---------- variable-length search helper ----------
    def _varlen_search(valid: np.ndarray) -> Tuple[Tuple[int,int], float, np.ndarray | None, np.ndarray | None]:
        best_sc = -np.inf
        best_win = (0, 0)
        best_idx_full = None
        best_vals = None

        n_valid = valid.shape[0]
        for pos_i in range(n_valid):
            i = valid[pos_i]
            # enforce contiguous start
            if pos_i < n_valid - 1 and (valid[pos_i + 1] - i) > 1:
                continue
            stop = min(pos_i + 1 + range_cap, n_valid)
            sub = valid[pos_i + 1: stop]
            for pos_j in range(sub.shape[0]):
                j = sub[pos_j]
                # keep contiguity for j; prune on first gap
                if pos_j > 0 and (sub[pos_j] - sub[pos_j - 1]) > 1:
                    break
                lo = pos_i
                hi = pos_i + 1 + pos_j
                inside = valid[lo:hi + 1]
                outside = np.setdiff1d(all_trimmed, inside, assume_unique=True)
                sc = score_variance_nwkr(row_trimmed, inside, outside, i, j, range_cap, W_trimmed, ssr_array)
                sc = sc / sra + 1.0
                if sc > best_sc:
                    best_sc = sc
                    best_win = (i, j)
                    best_idx_full = inside + buffer
                    best_vals = predict_on_idxs(row_trimmed, inside, W_trimmed)
        oi, oj = best_win
        return (oi + buffer, oj + buffer), best_sc, best_idx_full, best_vals

    # ---------- fixed-length linear sweep (span ~ ref_freq, unmasked) ----------
    # step_med = _safe_freq_step(freqs)
    # window_bins = int(np.floor(ref_freq / (step_med if step_med > 0 else 1.0)))
    # window_bins = max(1, min(window_bins, n_trimmed))
    window_bins = min(int(round(R)), n_trimmed)

    def _fixedlen_sweep() -> Tuple[Tuple[int,int], float, np.ndarray | None, np.ndarray | None]:
        best_sc = -np.inf
        best_win = (0, 0)
        best_idx_full = None
        best_vals = None

        max_start = max(0, n_trimmed - window_bins)
        for i in range(max_start + 1):
            inside = np.arange(i, i + window_bins, dtype=np.int64)
            outside = np.setdiff1d(all_trimmed, inside, assume_unique=False)
            j = i + window_bins - 1
            sc = score_variance_nwkr(row_trimmed, inside, outside, i, j, range_cap, W_trimmed, ssr_array)
            sc = sc / sra + 1.0
            if sc > best_sc:
                best_sc = sc
                best_win = (i, j)
                best_idx_full = inside + buffer
                best_vals = predict_on_idxs(row_trimmed, inside, W_trimmed)
        oi, oj = best_win
        return (oi + buffer, oj + buffer), best_sc, best_idx_full, best_vals

    # A) masked var-len
    best_win_masked, best_sc_masked, sri_idx_masked, sri_vals_masked = _varlen_search(valid_masked)

    # B) unmasked var-len + overlap
    best_win_unmasked, best_sc_unmasked, sri_idx_unmasked, sri_vals_unmasked = _varlen_search(all_trimmed)
    overlap_pct_unmasked = _overlap_stats(best_win_unmasked[0], best_win_unmasked[1], ignore)

    # C) fixed-len linear (unmasked) + overlap
    best_win_fixed, best_sc_fixed, sri_idx_fixed, sri_vals_fixed = _fixedlen_sweep()
    overlap_pct_fixed = _overlap_stats(best_win_fixed[0], best_win_fixed[1], ignore)

    return (
        # masked var-len
        row_idx, best_win_masked, best_sc_masked, pred_array, sri_idx_masked, sri_vals_masked, w * sr_factor, range_cap * sr_factor,
        # unmasked var-len
        best_win_unmasked, best_sc_unmasked, overlap_pct_unmasked, sri_idx_unmasked, sri_vals_unmasked,
        # fixed-len linear
        best_win_fixed, best_sc_fixed, overlap_pct_fixed, sri_idx_fixed, sri_vals_fixed, window_bins * sr_factor
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
    
    # masked var-len
    windows_masked   = [r[1]  for r in results]
    scores_masked    = [r[2]  for r in results]
    sra_preds        = [r[3]  for r in results]
    sri_idxs_masked  = [r[4]  for r in results]
    sri_vals_masked  = [r[5]  for r in results]
    ws               = [r[6]  for r in results]
    range_caps       = [r[7]  for r in results]

    # unmasked var-len
    windows_unmasked = [r[8]  for r in results]
    scores_unmasked  = [r[9]  for r in results]
    overlap_pct_unm  = [r[10] for r in results]
    sri_idxs_unm     = [r[11] for r in results]
    sri_vals_unm     = [r[12] for r in results]

    # fixed-len linear
    windows_fixed    = [r[13] for r in results]
    scores_fixed     = [r[14] for r in results]
    overlap_pct_fix  = [r[15] for r in results]
    sri_idxs_fixed   = [r[16] for r in results]
    sri_vals_fixed   = [r[17] for r in results]
    fixed_bins_nat   = [r[18] for r in results]

    return (windows_masked, scores_masked, sra_preds, sri_idxs_masked, sri_vals_masked, ws, range_caps,
            windows_unmasked, scores_unmasked, overlap_pct_unm, sri_idxs_unm, sri_vals_unm,
            windows_fixed, scores_fixed, overlap_pct_fix, sri_idxs_fixed, sri_vals_fixed, fixed_bins_nat)


# -----------------------------------------------------------------------------#
# Visualization & post-processing
# -----------------------------------------------------------------------------#

def plot_top_k(
    df: pd.DataFrame,
    actual_spec_arrays: np.ndarray,
    # windows (native coords, inclusive)
    windows_masked: List[Tuple[int, int]],
    windows_unmasked: List[Tuple[int, int]],
    windows_fixed: List[Tuple[int, int]],
    # scores
    scores_masked: List[float],
    scores_unmasked: List[float],
    scores_fixed: List[float],
    # overlap stats (fractions in 0..1 and counts) for unmasked + fixed
    overlap_unmasked_pct: List[float] | None,
    overlap_fixed_pct:    List[float] | None,
    # misc/meta
    atm_interfs: List[List[Tuple[int, int]]],
    meta: Dict[str, Any],
    ws: List[int],
    # viz/IO
    k: int = 10,
    per_fig: int = 10,
    buffer: int = 10,
    out_dir: str = "Images",
    data_dir: str = "Data",
    # predictions
    sra_preds: List[np.ndarray] | None = None,  # shared SRA (trimmed) per row
    sri_idxs_masked:  List[np.ndarray | None] | None = None,
    sri_vals_masked:  List[np.ndarray | None] | None = None,
    sri_idxs_unmasked:List[np.ndarray | None] | None = None,
    sri_vals_unmasked:List[np.ndarray | None] | None = None,
    sri_idxs_fixed:   List[np.ndarray | None] | None = None,
    sri_vals_fixed:   List[np.ndarray | None] | None = None,
    sr_factor: int = 1,
    fixed_bins_nat: List[int] | None = None,
    # ranking
    rank_by: str = "masked",  # "masked" | "unmasked" | "fixed"
) -> None:
    """Save ranked CSV and plot top-K rows with three window overlays.

    Ranks rows by `rank_by` score, but CSV contains all windows/scores/overlaps.

    Args:
      df: Source dataframe (must contain 'uid').
      actual_spec_arrays: (N,L) spectra at native resolution.
      windows_*: best windows at native resolution for each scan type.
      scores_*: scores for each scan type.
      overlap_*: overlap stats (only used/available for unmasked and fixed).
      atm_interfs: interference ranges at native resolution.
      meta: dict with 'uid', 'ref', 'ant', 'pol', 'freq' arrays.
      ws: effective kernel sizes (native scale).
      k, per_fig, buffer, out_dir, data_dir: plotting/IO knobs.
      sra_preds: NWKR predictions on trimmed SR rows (for overlay).
      sri_idxs_*/sri_vals_*: per-window inside predictions (optional overlays).
      sr_factor: SR upsampling for SRA/SRI overlays.
      fixed_bins_nat: optional native bin length for fixed windows (diagnostic).
      rank_by: which score to rank by: "masked" | "unmasked" | "fixed".
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    buf_orig = buffer

    # Choose score vector for ranking
    score_map_all = {
        "masked":   np.asarray(scores_masked, dtype=float),
        "unmasked": np.asarray(scores_unmasked, dtype=float),
        "fixed":    np.asarray(scores_fixed, dtype=float),
    }
    if rank_by not in score_map_all:
        raise ValueError("rank_by must be one of: 'masked', 'unmasked', 'fixed'")
    scores_rank = score_map_all[rank_by]

    # Top ordering (desc)
    finite = np.isfinite(scores_rank)
    idx_all = np.where(finite)[0]
    order_desc = idx_all[np.argsort(scores_rank[finite])[:]][::-1]

    # Build CSV (for *all* rows in the filtered ordering)
    top_uids = np.asarray(meta["uid"])[order_desc]
    kernel_sizes = np.asarray(ws, dtype=object)[order_desc]

    sub_df = df.loc[df["uid"].isin(top_uids)].copy()

    # Map all columns by uid
    def _mk_map(vec): return dict(zip(top_uids, np.asarray(vec, dtype=object)[order_desc]))

    sub_df["score_masked"]   = sub_df["uid"].map(_mk_map(scores_masked))
    sub_df["score_unmasked"] = sub_df["uid"].map(_mk_map(scores_unmasked))
    sub_df["score_fixed"]    = sub_df["uid"].map(_mk_map(scores_fixed))
    sub_df["kernel_size"]    = sub_df["uid"].map(_mk_map(kernel_sizes))

    # Windows
    sub_df[["win_masked_start", "win_masked_end"]]     = sub_df["uid"].map(_mk_map(windows_masked)).apply(pd.Series)
    sub_df[["win_unmasked_start", "win_unmasked_end"]] = sub_df["uid"].map(_mk_map(windows_unmasked)).apply(pd.Series)
    sub_df[["win_fixed_start", "win_fixed_end"]]       = sub_df["uid"].map(_mk_map(windows_fixed)).apply(pd.Series)

    # Overlaps (optional)
    if overlap_unmasked_pct is not None:
        sub_df["overlap_unmasked_pct"] = sub_df["uid"].map(_mk_map(overlap_unmasked_pct))
    if overlap_fixed_pct is not None:
        sub_df["overlap_fixed_pct"] = sub_df["uid"].map(_mk_map(overlap_fixed_pct))
    if fixed_bins_nat is not None:
        sub_df["fixed_bins_native"] = sub_df["uid"].map(_mk_map(fixed_bins_nat))

    # Sort by chosen score
    sub_df["rank_score"] = sub_df["uid"].map(_mk_map(scores_rank))
    sub_df = sub_df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    sub_df.insert(0, "uid", sub_df.pop("uid"))

    out_csv = os.path.join(
        data_dir,
        f"bandpass_qa0_no_partitions_labelled_filt_scan_stat_length_{actual_spec_arrays.shape[1]}.csv",
    )
    sub_df.to_csv(out_csv, index=False)
    logging.info("Wrote summary CSV (all windows): %s", out_csv)

    # ---- Plot top-k with all windows overlaid ----
    top = order_desc[:min(k, len(order_desc))]
    n_figs = math.ceil(len(top) / per_fig)
    for fig_i in range(n_figs):
        batch = top[fig_i * per_fig : (fig_i + 1) * per_fig]
        fig_k = len(batch)
        fig, axes = plt.subplots(fig_k, 1, figsize=(10, 3 * fig_k))
        if fig_k == 1:
            axes = [axes]

        for ax, i0 in zip(axes, batch):
            spec = actual_spec_arrays[i0]
            a_m, b_m = windows_masked[i0]
            a_u, b_u = windows_unmasked[i0]
            a_f, b_f = windows_fixed[i0]

            # Atmospheric interference shading
            for (c, d) in atm_interfs[i0]:
                ax.axvspan(c, d, color="C9", alpha=0.15, label=None)

            # Raw spectrum
            x = np.arange(len(spec))
            ax.plot(x, spec, color="C0", label="Actual")

            # Buffer shading
            if buf_orig > 0:
                ax.axvspan(0, buf_orig - 1, color="gray", alpha=0.15, label=None)
                ax.axvspan(len(spec) - buf_orig, len(spec) - 1, color="gray", alpha=0.15, label=None)

            # Window overlays (distinct alphas)
            ax.axvspan(a_m, b_m, color="C1", alpha=0.35, label=None)
            ax.axvspan(a_u, b_u, color="C2", alpha=0.25, label=None)
            ax.axvspan(a_f, b_f, color="C3", alpha=0.25, label=None)

            # SRA predicted curve (shared)
            if sra_preds is not None and sra_preds[i0] is not None and len(sra_preds[i0]) > 0:
                pred_full = np.full(len(spec), np.nan)
                sra_up = np.repeat(sra_preds[i0], sr_factor)
                end = len(spec) - buf_orig
                pred_full[buf_orig:end] = sra_up[: max(0, end - buf_orig)]
                ax.plot(np.arange(len(spec)), pred_full, ".", ms=2, label="SRA pred")

            # SRI overlays per window (optional)
            def _plot_sri(idx_list, val_list, label: str):
                if (idx_list is None) or (val_list is None):
                    return
                if (idx_list[i0] is None) or (val_list[i0] is None):
                    return
                sri_full = np.full(len(spec), np.nan)
                idx_sr  = np.asarray(idx_list[i0], dtype=int)
                val_sr  = np.asarray(val_list[i0], dtype=float)
                idx_orig_start = idx_sr * sr_factor
                for p, v in zip(idx_orig_start, val_sr):
                    p_end = min(p + sr_factor, len(spec))
                    sri_full[p:p_end] = v
                ax.plot(np.arange(len(spec)), sri_full, ".", ms=2, label=label)

            _plot_sri(sri_idxs_masked,   sri_vals_masked,   "SRI masked")
            _plot_sri(sri_idxs_unmasked, sri_vals_unmasked, "SRI unmasked")
            _plot_sri(sri_idxs_fixed,    sri_vals_fixed,    "SRI fixed")

            # Title with all scores (+overlaps if available)
            parts = [
                f"UID={i0}",
                f"S_masked={scores_masked[i0]:.2f}",
                f"S_unmasked={scores_unmasked[i0]:.2f}",
                f"S_fixed={scores_fixed[i0]:.2f}",
            ]
            if overlap_unmasked_pct is not None:
                parts.append(f"ovl_unm={100*overlap_unmasked_pct[i0]:.1f}%")
            if overlap_fixed_pct is not None:
                parts.append(f"ovl_fix={100*overlap_fixed_pct[i0]:.1f}%")
            if fixed_bins_nat is not None:
                parts.append(f"fixed_bins={fixed_bins_nat[i0]}")
            parts.append(f"W={ws[i0]}")
            ax.set_title("  ".join(parts))

            ax.set_xlabel("Channel")
            ax.set_ylabel("Amplitude")

        legend_elements = [
            Line2D([0], [0], color="C0", label="Actual"),
            Line2D([0], [0], marker=".", linestyle="None", label="SRA pred"),
            Line2D([0], [0], marker=".", linestyle="None", label="SRI masked"),
            Line2D([0], [0], marker=".", linestyle="None", label="SRI unmasked"),
            Line2D([0], [0], marker=".", linestyle="None", label="SRI fixed"),
            Patch(facecolor="C1", alpha=0.35, label="Masked window"),
            Patch(facecolor="C2", alpha=0.25, label="Unmasked window"),
            Patch(facecolor="C3", alpha=0.25, label="Fixed window"),
            Patch(facecolor="C9", alpha=0.15, label="Interference"),
            Patch(facecolor="gray", alpha=0.15, label="Buffer"),
        ]

        # Let tight_layout do its thing for subplots first
        plt.tight_layout()

        # Add a compact legend just below the top edge (inside the figure box)
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            ncol=5,
            frameon=True,
            bbox_to_anchor=(0.5, 0.985),  # pull this down if you still see crowding
        )

        # Now reclaim top space but keep a little headroom for legend + title
        fig.subplots_adjust(top=0.92)  # try 0.92–0.95 depending on your fonts

        # Keep the title but don’t push everything down
        fig.suptitle(
            f"Top {min(k, len(order_desc))} ranked by {rank_by} — batch {fig_i+1}/{n_figs}",
            y=0.995
        )

        outpath = os.path.join(out_dir, f"top_{min(k, len(order_desc))}_by_{rank_by}_fig{fig_i + 1}.png")
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


def refine_all_windows_exact_for_length(
    spec_arrays: np.ndarray,
    windows_masked_sr: List[Tuple[int, int]],
    windows_unmasked_sr: List[Tuple[int, int]],
    windows_fixed_sr: List[Tuple[int, int]],
    atm_interfs: List[List[Tuple[int,int]]],
    ws: List[int],
    range_caps: List[int],
    sr_factor: int,
    buffer: int,
) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[Tuple[int,int]]]:
    """Refine all coarse (SR) windows on native-resolution spectra.

    Inputs:
      spec_arrays        : (N, L_native)
      windows_masked_sr  : best var-len windows on SR grid (masked scan), len N
      windows_unmasked_sr: best var-len windows on SR grid (unmasked scan), len N
      windows_fixed_sr   : best fixed-len windows on SR grid (unmasked fixed scan), len N
      atm_interfs        : per-row native interference ranges [(s,e), ...]
      ws, range_caps     : per-row kernel size and range cap (both in native scale;
                           i.e., returned as w*sr_factor and range_cap*sr_factor by the SR scan)
      sr_factor          : SR block size used to produce the coarse windows
      buffer             : native channels trimmed on both sides for scoring

    Returns:
      (windows_masked_exact, windows_unmasked_exact, windows_fixed_exact)
      — each a list of (start,end) in native coordinates (inclusive), length N.
    """
    N, L = spec_arrays.shape
    n_trimmed = L - 2 * buffer
    if n_trimmed <= 0:
        z = [(0, 0)] * N
        return z, z, z

    out_masked   : List[Tuple[int,int]] = []
    out_unmasked : List[Tuple[int,int]] = []
    out_fixed    : List[Tuple[int,int]] = []

    for i in range(N):
        W_trimmed = _get_kernel(n_trimmed, ws[i])      # ws[i] is native-scale
        range_cap = range_caps[i]                      # native-scale
        row_trimmed = spec_arrays[i, buffer:L-buffer]

        sra, ssr_array, _ = calculate_nwkr_sra(row_trimmed, W_trimmed)
        sra = sra if sra > 1e-12 else 1e-12

        # Build native valid mask for masked refinement
        mask = np.ones(n_trimmed, dtype=np.bool_)
        for (s, e) in atm_interfs[i]:
            s0 = max(s - buffer, 0)
            e0 = min(e - buffer, n_trimmed - 1)
            if s0 <= e0:
                mask[s0:e0+1] = False
        valid_all = np.arange(n_trimmed, dtype=np.int64)
        valid_masked = valid_all[mask]

        # ---- helpers ----
        def _score_varlen(a: int, b: int, valid: np.ndarray) -> float:
            # contiguous inside over [a,b] intersected with valid
            inside = valid[(valid >= a) & (valid <= b)]
            if inside.size == 0:
                return -np.inf
            outside = np.setdiff1d(valid_all, inside, assume_unique=False)
            sc = score_variance_nwkr(row_trimmed, inside, outside, a, b, range_cap, W_trimmed, ssr_array)
            return sc / sra + 1.0

        def _refine_varlen_from_sr(x_sr: int, y_sr: int, valid: np.ndarray) -> Tuple[int,int]:
            # SR -> native candidate box
            a_lo = max(x_sr * sr_factor - buffer, 0)
            a_hi = min((x_sr + 1) * sr_factor - 1 - buffer, n_trimmed - 1)
            b_lo = max(y_sr * sr_factor - buffer, 0)
            b_hi = min((y_sr + 1) * sr_factor - 1 - buffer, n_trimmed - 1)

            best_sc, best_ab = -np.inf, (a_lo, max(a_lo, b_lo))
            for a in range(a_lo, a_hi + 1):
                b_start = max(a, b_lo)
                for b in range(b_start, b_hi + 1):
                    sc = _score_varlen(a, b, valid)
                    if sc > best_sc:
                        best_sc, best_ab = sc, (a, b)
            a_t, b_t = best_ab
            return (a_t + buffer, b_t + buffer)  # back to native coords

        def _refine_fixed_from_sr(x_sr: int, y_sr: int) -> Tuple[int,int]:
            # fixed window native length
            fixed_bins_native = (y_sr - x_sr + 1) * sr_factor
            fixed_bins_native = max(1, min(fixed_bins_native, n_trimmed))

            # derive feasible a range from SR box; enforce fixed len
            a_lo = max(x_sr * sr_factor - buffer, 0)
            a_hi = min((x_sr + 1) * sr_factor - 1 - buffer, n_trimmed - fixed_bins_native)
            b_from_a = lambda a: a + fixed_bins_native - 1

            best_sc, best_a = -np.inf, a_lo
            for a in range(a_lo, a_hi + 1):
                b = b_from_a(a)
                # unmasked fixed refinement: use all valid indices
                sc = _score_varlen(a, b, valid_all)
                if sc > best_sc:
                    best_sc, best_a = sc, a
            a_t, b_t = best_a, b_from_a(best_a)
            return (a_t + buffer, b_t + buffer)

        # ---- refine each coarse window ----
        xm, ym = windows_masked_sr[i]
        xu, yu = windows_unmasked_sr[i]
        xf, yf = windows_fixed_sr[i]

        out_masked.append(_refine_varlen_from_sr(xm, ym, valid_masked))
        out_unmasked.append(_refine_varlen_from_sr(xu, yu, valid_all))
        out_fixed.append(_refine_fixed_from_sr(xf, yf))

    return out_masked, out_unmasked, out_fixed


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
        (windows_sr_masked, scores_masked, sra_preds, sri_idxs, sri_vals, ws, range_caps,
        windows_sr_unmasked, scores_unmasked, overlap_unm_pct, sri_idxs_unm, sri_vals_unm,
        windows_sr_fixed, scores_fixed, overlap_fix_pct, sri_idxs_fix, sri_vals_fix, fixed_bins_nat
        ) = polynomial_scan_ranges_parallel(
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
            windows_exact_masked, windows_exact_unmasked, windows_exact_fixed = refine_all_windows_exact_for_length(
                actual_specs,
                windows_sr_masked,
                windows_sr_unmasked,
                windows_sr_fixed,
                atm_interfs,
                ws,
                range_caps,
                SR_FACTOR,
                BUFFER,
            )
        else:
            windows_exact_masked   = windows_sr_masked
            windows_exact_unmasked = windows_sr_unmasked
            windows_exact_fixed    = windows_sr_fixed

        plot_top_k(
            df=df,
            actual_spec_arrays=actual_specs,
            windows_masked=windows_exact_masked,
            windows_unmasked=windows_exact_unmasked,
            windows_fixed=windows_exact_fixed,
            scores_masked=scores_masked,
            scores_unmasked=scores_unmasked,
            scores_fixed=scores_fixed,
            overlap_unmasked_pct=overlap_unm_pct,
            overlap_fixed_pct=overlap_fix_pct,
            atm_interfs=atm_interfs,
            meta=meta,
            ws=ws,
            k=min(args.top_k, n_rows),
            per_fig=args.per_fig,
            buffer=BUFFER,
            out_dir=out_dir,
            data_dir=data_dir,
            sra_preds=sra_preds,
            sri_idxs_masked=sri_idxs,
            sri_vals_masked=sri_vals,
            sri_idxs_unmasked=sri_idxs_unm,
            sri_vals_unmasked=sri_vals_unm,
            sri_idxs_fixed=sri_idxs_fix,
            sri_vals_fixed=sri_vals_fix,
            sr_factor=SR_FACTOR,
            fixed_bins_nat=fixed_bins_nat,
            rank_by="masked",
        )


if __name__ == "__main__":
    main()
