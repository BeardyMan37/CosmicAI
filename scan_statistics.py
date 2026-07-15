"""
scan_statistics.py
==================
Self-contained NWKR scan statistics pipeline for spectral line detection
in ALMA bandpass calibration data.

Library usage
-------------
    from scan_statistics import compute_scan_statistics_scores, Input, Output

    results = compute_scan_statistics_scores({
        "baseline_1": Input(amplitude=amp, frequency=freq, flag_array=flags),
    })
    out = results["baseline_1"]["masked"]
    print(out.score, out.win_start, out.win_end)
"""

from __future__ import annotations

import argparse
import math
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import groupby
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple, TypeAlias

import numpy as np
from numba import njit

# ===========================================================================
# 1. Constants (mirror helpers/config.py)
# ===========================================================================

_REF_FREQ: float = 0.0625          # GHz — reference spectral-line channel width
_BUFFER_DIVISOR: int = 20          # buffer = len(frequency) // _BUFFER_DIVISOR
_DEFAULT_KERNEL: str = "gaussian"
_SUPER_RESOLVE_BASE: int = 450     # spectra longer than this get superresolved

logger = logging.getLogger(__name__)

# ===========================================================================
# 2. Public types
# ===========================================================================

class Input(NamedTuple):
    """Input data for a single bandpass calibration solution."""
    amplitude: np.ndarray
    frequency: np.ndarray
    flag_array: np.ndarray
    atm_ranges: Optional[List[Tuple[int, int]]] = None


class Output(NamedTuple):
    """Scan statistic result for one scan mode."""
    score: float
    win_start: int
    win_end: int
    overlap_pct: float = 0.0


ScanMode: TypeAlias = Literal["masked", "unmasked", "fixed"]
ScanResult: TypeAlias = Dict[ScanMode, Output]

# ===========================================================================
# 3. Atmospheric detection
# ===========================================================================

def load_transmission(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load an atmospheric transmission table from parquet."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for loading transmission tables.")
    df = pd.read_parquet(path)
    freqs = df["Frequency (GHz)"].to_numpy(dtype=np.float64)
    vals = df["Transmission (%)"].to_numpy(dtype=np.float64)
    order = np.argsort(freqs)
    return freqs[order], vals[order]


def detect_atm_ranges(
    freq_array: np.ndarray,
    trans_freqs: np.ndarray,
    trans_vals: np.ndarray,
    prominence: float = 1.0,
    rel_height: float = 0.75,
) -> List[Tuple[int, int]]:
    """Detect atmospheric absorption ranges by finding troughs in matched transmission."""
    try:
        from scipy.signal import find_peaks, peak_widths
    except ImportError:
        raise ImportError("scipy is required for atmospheric detection.")

    freq_array = np.asarray(freq_array, dtype=np.float64)
    idxs = np.clip(np.searchsorted(trans_freqs, freq_array), 0, len(trans_freqs) - 1)
    left = np.maximum(idxs - 1, 0)
    right = idxs
    dl = np.abs(freq_array - trans_freqs[left])
    dr = np.abs(trans_freqs[right] - freq_array)
    nearest = np.where(dl <= dr, left, right)
    trans_matched = trans_vals[nearest]

    troughs, _ = find_peaks(-trans_matched, prominence=prominence)
    if len(troughs) == 0:
        return []

    _, _, left_ips, right_ips = peak_widths(-trans_matched, troughs, rel_height=rel_height)
    freqs = freq_array
    left_freqs = np.interp(left_ips, np.arange(len(freqs)), freqs)
    right_freqs = np.interp(right_ips, np.arange(len(freqs)), freqs)
    widths_freq = right_freqs - left_freqs
    trough_freqs = freqs[troughs]
    trough_ranges = np.column_stack(
        (trough_freqs - widths_freq / 2.0, trough_freqs + widths_freq / 2.0)
    )

    ranges: List[Tuple[int, int]] = []
    for start_f, end_f in trough_ranges:
        s_idx = int(np.abs(freqs - start_f).argmin())
        e_idx = int(np.abs(freqs - end_f).argmin())
        ranges.append((min(s_idx, e_idx), max(s_idx, e_idx)))
    return ranges


# ===========================================================================
# 4. Superresolution
# ===========================================================================

def _sr_factor(L: int, base: int = _SUPER_RESOLVE_BASE, r: int = 2, q: int = 2, cap=None) -> int:
    """
    Compute superresolution downsampling factor.
    """
    s = math.ceil((L + 1) / base)
    k = math.ceil(math.log(s, r)) if s > 1 else 0
    f = q ** k
    return min(f, cap) if cap is not None else f


def _superresolve(specs: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample a 2-D array (n_rows × n_ch) by averaging blocks of `factor`
    channels.

    Input must be 2-D.  Trailing channels that don't fill a block are dropped.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    n_rows, n_ch = specs.shape
    n_blk = n_ch // factor
    if n_blk == 0:
        return np.empty((n_rows, 0), dtype=specs.dtype)
    trimmed = specs[:, :n_blk * factor]
    return trimmed.reshape(n_rows, n_blk, factor).mean(axis=2)


def _superresolve_ranges(
    ranges_list: List[List[Tuple[int, int]]], factor: int,
) -> List[List[Tuple[int, int]]]:
    """
    Map per-row range lists from native to SR coordinates and merge overlaps.
    
    Input: list of per-row range lists [[(s,e), ...], ...]
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")

    def merge(rs):
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


# ===========================================================================
# 5. Kernel primitives (Numba JIT)
# ===========================================================================

@njit(cache=True, fastmath=True)
def _truncated_kernel_vector(w: float, r: int, kind_is_gaussian: bool) -> np.ndarray:
    """Build k[d] for d = 0..r.  Gaussian: exp(-(d²)/(w²));  Laplace: exp(-d/w)."""
    d = np.arange(r + 1, dtype=np.float64)
    if kind_is_gaussian:
        return np.exp(-(d * d) / (w * w))
    else:
        sigma = w if w > 1e-12 else 1e-12
        return np.exp(-d / sigma)


# ===========================================================================
# 6. NWKR SRA computation (Numba JIT)
# ===========================================================================

@njit(cache=True, fastmath=True)
def _calculate_sra_trunc(array: np.ndarray, k: np.ndarray):
    """
    Truncated-kernel NWKR SRA (global SSE baseline).
    Returns (sra, pred_array, numer_all, denom_all).
    Works for both Gaussian and Laplace since k already encodes the kernel shape.
    Mirrors helpers.scoring.calculate_gaussian_sra_trunc.
    """
    n = array.shape[0]
    r = k.shape[0] - 1
    eps = 1e-12

    numer = np.empty(n, dtype=np.float64)
    denom = np.empty(n, dtype=np.float64)
    pred = np.empty(n, dtype=np.float64)

    for i in range(n):
        j0 = max(0, i - r)
        j1 = min(n - 1, i + r)
        num = 0.0
        den = 0.0
        for j in range(j0, j1 + 1):
            d = i - j
            if d < 0:
                d = -d
            wgt = k[d]
            den += wgt
            num += wgt * array[j]
        numer[i] = num
        denom[i] = den
        pi = num / (den if den > eps else eps)
        pred[i] = pi

    sra = 0.0
    for i in range(n):
        di = array[i] - pred[i]
        sra += di * di

    return sra, pred, numer, denom


# ===========================================================================
# 7. Incremental state management (Numba JIT)
#    Direct port of helpers/kernel_optimized_state.py
# ===========================================================================

@njit(cache=True, fastmath=True)
def _is_inside(buf_idxs: np.ndarray, m: int, idx: int) -> bool:
    lo = 0
    hi = m
    while lo < hi:
        mid = (lo + hi) >> 1
        if buf_idxs[mid] < idx:
            lo = mid + 1
        else:
            hi = mid
    return lo < m and buf_idxs[lo] == idx


@njit(cache=True, fastmath=True)
def _nin_din_init_full(
    x: np.ndarray, idxs_in: np.ndarray, k: np.ndarray,
    nin: np.ndarray, din: np.ndarray,
) -> None:
    n = x.shape[0]
    r = k.shape[0] - 1
    for i in range(n):
        nin[i] = 0.0
        din[i] = 0.0
    for p in range(idxs_in.shape[0]):
        j = int(idxs_in[p])
        xj = x[j]
        i = j
        while i >= 0:
            d = j - i
            if d > r:
                break
            nin[i] += k[d] * xj
            din[i] += k[d]
            i -= 1
        i = j + 1
        while i < n:
            d = i - j
            if d > r:
                break
            nin[i] += k[d] * xj
            din[i] += k[d]
            i += 1


@njit(cache=True, fastmath=True)
def _nin_din_add(x: np.ndarray, nin: np.ndarray, din: np.ndarray,
                 idx: int, k: np.ndarray) -> None:
    r = k.shape[0] - 1
    n = nin.shape[0]
    xj = x[idx]
    i = idx
    while i >= 0:
        d = idx - i
        if d > r:
            break
        nin[i] += k[d] * xj
        din[i] += k[d]
        i -= 1
    i = idx + 1
    while i < n:
        d = i - idx
        if d > r:
            break
        nin[i] += k[d] * xj
        din[i] += k[d]
        i += 1


@njit(cache=True, fastmath=True)
def _nin_din_remove(x: np.ndarray, nin: np.ndarray, din: np.ndarray,
                    idx: int, k: np.ndarray) -> None:
    r = k.shape[0] - 1
    n = nin.shape[0]
    xj = x[idx]
    i = idx
    while i >= 0:
        d = idx - i
        if d > r:
            break
        nin[i] -= k[d] * xj
        din[i] -= k[d]
        i -= 1
    i = idx + 1
    while i < n:
        d = i - idx
        if d > r:
            break
        nin[i] -= k[d] * xj
        din[i] -= k[d]
        i += 1


@njit(cache=True, fastmath=True)
def _sse_out_from_nin_din(
    x: np.ndarray, numer_all: np.ndarray, denom_all: np.ndarray,
    nin: np.ndarray, din: np.ndarray,
    buf_idxs: np.ndarray, m: int,
) -> float:
    n = x.shape[0]
    eps = 1e-12
    sse = 0.0
    p = 0
    for i in range(n):
        while p < m and buf_idxs[p] < i:
            p += 1
        if p < m and buf_idxs[p] == i:
            continue
        den_out = denom_all[i] - din[i]
        num_out = numer_all[i] - nin[i]
        pred = num_out / den_out if den_out > eps else 0.0
        sse += (x[i] - pred) ** 2
    return sse


@njit(cache=True, fastmath=True)
def _sse_out_add(
    x: np.ndarray, numer_all: np.ndarray, denom_all: np.ndarray,
    nin: np.ndarray, din: np.ndarray,
    buf_idxs: np.ndarray, m_new: int,
    new_idx: int, k: np.ndarray, sse_out: float,
) -> float:
    r = k.shape[0] - 1
    eps = 1e-12
    n = nin.shape[0]
    x_new = x[new_idx]

    nin_old = nin[new_idx] - k[0] * x_new
    din_old = din[new_idx] - k[0]
    den_out_old = denom_all[new_idx] - din_old
    num_out_old = numer_all[new_idx] - nin_old
    pred_old = num_out_old / den_out_old if den_out_old > eps else 0.0
    sse_out -= (x_new - pred_old) ** 2

    i = new_idx - 1
    while i >= 0:
        d = new_idx - i
        if d > r:
            break
        if not _is_inside(buf_idxs, m_new, i):
            w = k[d]
            nin_old_i = nin[i] - w * x_new
            din_old_i = din[i] - w
            den_out_old = denom_all[i] - din_old_i
            num_out_old = numer_all[i] - nin_old_i
            pred_old_i = num_out_old / den_out_old if den_out_old > eps else 0.0
            den_out_new = denom_all[i] - din[i]
            num_out_new = numer_all[i] - nin[i]
            pred_new_i = num_out_new / den_out_new if den_out_new > eps else 0.0
            sse_out -= (x[i] - pred_old_i) ** 2
            sse_out += (x[i] - pred_new_i) ** 2
        i -= 1

    i = new_idx + 1
    while i < n:
        d = i - new_idx
        if d > r:
            break
        if not _is_inside(buf_idxs, m_new, i):
            w = k[d]
            nin_old_i = nin[i] - w * x_new
            din_old_i = din[i] - w
            den_out_old = denom_all[i] - din_old_i
            num_out_old = numer_all[i] - nin_old_i
            pred_old_i = num_out_old / den_out_old if den_out_old > eps else 0.0
            den_out_new = denom_all[i] - din[i]
            num_out_new = numer_all[i] - nin[i]
            pred_new_i = num_out_new / den_out_new if den_out_new > eps else 0.0
            sse_out -= (x[i] - pred_old_i) ** 2
            sse_out += (x[i] - pred_new_i) ** 2
        i += 1

    return sse_out


@njit(cache=True, fastmath=True)
def _sse_out_remove(
    x: np.ndarray, numer_all: np.ndarray, denom_all: np.ndarray,
    nin: np.ndarray, din: np.ndarray,
    buf_idxs: np.ndarray, m_new: int,
    rem_idx: int, k: np.ndarray, sse_out: float,
) -> float:
    r = k.shape[0] - 1
    eps = 1e-12
    n = nin.shape[0]
    x_rem = x[rem_idx]

    den_out_new = denom_all[rem_idx] - din[rem_idx]
    num_out_new = numer_all[rem_idx] - nin[rem_idx]
    pred_new = num_out_new / den_out_new if den_out_new > eps else 0.0
    sse_out += (x_rem - pred_new) ** 2

    i = rem_idx - 1
    while i >= 0:
        d = rem_idx - i
        if d > r:
            break
        if not _is_inside(buf_idxs, m_new, i):
            w = k[d]
            nin_old_i = nin[i] + w * x_rem
            din_old_i = din[i] + w
            den_out_old = denom_all[i] - din_old_i
            num_out_old = numer_all[i] - nin_old_i
            pred_old_i = num_out_old / den_out_old if den_out_old > eps else 0.0
            den_out_new = denom_all[i] - din[i]
            num_out_new = numer_all[i] - nin[i]
            pred_new_i = num_out_new / den_out_new if den_out_new > eps else 0.0
            sse_out -= (x[i] - pred_old_i) ** 2
            sse_out += (x[i] - pred_new_i) ** 2
        i -= 1

    i = rem_idx + 1
    while i < n:
        d = i - rem_idx
        if d > r:
            break
        if not _is_inside(buf_idxs, m_new, i):
            w = k[d]
            nin_old_i = nin[i] + w * x_rem
            din_old_i = din[i] + w
            den_out_old = denom_all[i] - din_old_i
            num_out_old = numer_all[i] - nin_old_i
            pred_old_i = num_out_old / den_out_old if den_out_old > eps else 0.0
            den_out_new = denom_all[i] - din[i]
            num_out_new = numer_all[i] - nin[i]
            pred_new_i = num_out_new / den_out_new if den_out_new > eps else 0.0
            sse_out -= (x[i] - pred_old_i) ** 2
            sse_out += (x[i] - pred_new_i) ** 2
        i += 1

    return sse_out


# ---- buf: sorted buffer of inside indices + local NWKR stats ----

@njit(cache=True, fastmath=True)
def _buf_init(
    x: np.ndarray, idxs_in: np.ndarray, k: np.ndarray,
    buf_idxs: np.ndarray, buf_num: np.ndarray, buf_den: np.ndarray,
) -> Tuple[int, float]:
    m0 = idxs_in.shape[0]
    r = k.shape[0] - 1
    eps = 1e-12
    for t in range(m0):
        buf_idxs[t] = idxs_in[t]
    for u in range(m0):
        i = int(buf_idxs[u])
        s_num = 0.0
        s_den = 0.0
        v = u
        while v >= 0:
            d = i - int(buf_idxs[v])
            if d > r:
                break
            s_num += k[d] * x[int(buf_idxs[v])]
            s_den += k[d]
            v -= 1
        v = u + 1
        while v < m0:
            d = int(buf_idxs[v]) - i
            if d > r:
                break
            s_num += k[d] * x[int(buf_idxs[v])]
            s_den += k[d]
            v += 1
        buf_num[u] = s_num
        buf_den[u] = s_den
    sse_in = 0.0
    for u in range(m0):
        d = buf_den[u]
        pred = buf_num[u] / d if d > eps else 0.0
        sse_in += (x[int(buf_idxs[u])] - pred) ** 2
    return m0, float(sse_in)


@njit(cache=True, fastmath=True)
def _buf_add(
    x: np.ndarray, new_idx: int, k: np.ndarray,
    buf_idxs: np.ndarray, buf_num: np.ndarray, buf_den: np.ndarray,
    m: int, sse_in: float,
) -> Tuple[int, float]:
    r = k.shape[0] - 1
    eps = 1e-12
    x_new = x[new_idx]
    lo = 0
    hi = m
    while lo < hi:
        mid = (lo + hi) >> 1
        if buf_idxs[mid] < new_idx:
            lo = mid + 1
        else:
            hi = mid
    ins = lo
    for t in range(m - 1, ins - 1, -1):
        buf_idxs[t + 1] = buf_idxs[t]
        buf_num[t + 1] = buf_num[t]
        buf_den[t + 1] = buf_den[t]
    buf_idxs[ins] = new_idx
    m_new = m + 1

    v = ins - 1
    while v >= 0:
        d = new_idx - int(buf_idxs[v])
        if d > r:
            break
        w = k[d]
        od = buf_den[v]
        op = buf_num[v] / od if od > eps else 0.0
        sse_in -= (x[int(buf_idxs[v])] - op) ** 2
        buf_num[v] += w * x_new
        buf_den[v] += w
        nd = buf_den[v]
        np_ = buf_num[v] / nd if nd > eps else 0.0
        sse_in += (x[int(buf_idxs[v])] - np_) ** 2
        v -= 1
    v = ins + 1
    while v < m_new:
        d = int(buf_idxs[v]) - new_idx
        if d > r:
            break
        w = k[d]
        od = buf_den[v]
        op = buf_num[v] / od if od > eps else 0.0
        sse_in -= (x[int(buf_idxs[v])] - op) ** 2
        buf_num[v] += w * x_new
        buf_den[v] += w
        nd = buf_den[v]
        np_ = buf_num[v] / nd if nd > eps else 0.0
        sse_in += (x[int(buf_idxs[v])] - np_) ** 2
        v += 1

    s_num = k[0] * x_new
    s_den = k[0]
    v = ins - 1
    while v >= 0:
        d = new_idx - int(buf_idxs[v])
        if d > r:
            break
        s_num += k[d] * x[int(buf_idxs[v])]
        s_den += k[d]
        v -= 1
    v = ins + 1
    while v < m_new:
        d = int(buf_idxs[v]) - new_idx
        if d > r:
            break
        s_num += k[d] * x[int(buf_idxs[v])]
        s_den += k[d]
        v += 1
    buf_num[ins] = s_num
    buf_den[ins] = s_den
    pred_new = s_num / s_den if s_den > eps else 0.0
    sse_in += (x_new - pred_new) ** 2
    return m_new, float(sse_in)


@njit(cache=True, fastmath=True)
def _buf_remove(
    x: np.ndarray, rem_idx: int, k: np.ndarray,
    buf_idxs: np.ndarray, buf_num: np.ndarray, buf_den: np.ndarray,
    m: int, sse_in: float,
) -> Tuple[int, float]:
    r = k.shape[0] - 1
    eps = 1e-12
    x_rem = x[rem_idx]
    lo = 0
    hi = m
    while lo < hi:
        mid = (lo + hi) >> 1
        if buf_idxs[mid] < rem_idx:
            lo = mid + 1
        else:
            hi = mid
    pos = lo
    if pos >= m or buf_idxs[pos] != rem_idx:
        return m, sse_in

    od = buf_den[pos]
    op = buf_num[pos] / od if od > eps else 0.0
    sse_in -= (x_rem - op) ** 2

    v = pos - 1
    while v >= 0:
        d = rem_idx - int(buf_idxs[v])
        if d > r:
            break
        w = k[d]
        od = buf_den[v]
        op = buf_num[v] / od if od > eps else 0.0
        sse_in -= (x[int(buf_idxs[v])] - op) ** 2
        buf_num[v] -= w * x_rem
        buf_den[v] -= w
        nd = buf_den[v]
        np_ = buf_num[v] / nd if nd > eps else 0.0
        sse_in += (x[int(buf_idxs[v])] - np_) ** 2
        v -= 1
    v = pos + 1
    while v < m:
        d = int(buf_idxs[v]) - rem_idx
        if d > r:
            break
        w = k[d]
        od = buf_den[v]
        op = buf_num[v] / od if od > eps else 0.0
        sse_in -= (x[int(buf_idxs[v])] - op) ** 2
        buf_num[v] -= w * x_rem
        buf_den[v] -= w
        nd = buf_den[v]
        np_ = buf_num[v] / nd if nd > eps else 0.0
        sse_in += (x[int(buf_idxs[v])] - np_) ** 2
        v += 1

    for t in range(pos, m - 1):
        buf_idxs[t] = buf_idxs[t + 1]
        buf_num[t] = buf_num[t + 1]
        buf_den[t] = buf_den[t + 1]
    return m - 1, float(sse_in)


# ---- Truncated-kernel prediction on a subset ----

@njit(cache=True, fastmath=True)
def _predict_on_idxs_trunc(array: np.ndarray, idxs: np.ndarray,
                            k: np.ndarray) -> np.ndarray:
    m = idxs.shape[0]
    r = k.shape[0] - 1
    out = np.empty(m, dtype=np.float64)
    for ii in range(m):
        i0 = idxs[ii]
        num = 0.0
        den = 0.0
        jj = ii
        while jj >= 0:
            d = i0 - idxs[jj]
            if d > r:
                break
            w = k[d]
            num += w * array[idxs[jj]]
            den += w
            jj -= 1
        jj = ii + 1
        while jj < m:
            d = idxs[jj] - i0
            if d > r:
                break
            w = k[d]
            num += w * array[idxs[jj]]
            den += w
            jj += 1
        out[ii] = num / den if den > 1e-12 else 0.0
    return out


# ===========================================================================
# 8. Core scan functions (use incremental state)
#    These operate on data at whatever resolution they are given.
# ===========================================================================

def _derive_params(frequency: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Derive kernel/window params from frequency grid.
    Returns (w, kernel_cap, range_cap, window_bins).
    """
    L = len(frequency)
    freq_step = abs(float(frequency[1] - frequency[0]))
    if freq_step <= 0.0:
        freq_step = _REF_FREQ
    R = _REF_FREQ / freq_step
    w = int(round(max(3.0, min(R, L / 16.0))))
    kernel_cap = 2 * w
    range_cap = 3 * w
    window_bins = int(math.floor(R)) + 1
    return w, kernel_cap, range_cap, window_bins


def _overlap_fraction(a: int, b: int, ranges: List[Tuple[int, int]]) -> float:
    if b < a or not ranges:
        return 0.0
    wl = b - a + 1
    ov = 0
    for s, e in ranges:
        lo = max(a, s)
        hi = min(b, e)
        if hi >= lo:
            ov += hi - lo + 1
    return ov / wl


def _scan_row(
    row: np.ndarray,
    ignore_ranges: List[Tuple[int, int]],
    flag_ranges: List[Tuple[int, int]],
    frequency: np.ndarray,
    buffer: int,
    sr_factor_val: int,
    kernel_kind: str,
) -> Tuple:
    is_gaussian = kernel_kind.lower() == "gaussian"
    n = row.shape[0]
    _bad_win = (0, -1)
    _bad = (0, _bad_win, 0., np.array([]), None, None, 0, 0,
            _bad_win, 0., 0., None, None, _bad_win, 0., 0., None, None, 0)
 
    if len(frequency) < 2 or not np.isfinite(frequency[:2]).all():
        return _bad
 
    w, kernel_cap, range_cap, window_bins = _derive_params(frequency)
    k_vector = _truncated_kernel_vector(float(w), kernel_cap, is_gaussian)
 
    _, pred_array, _, _ = _calculate_sra_trunc(row.astype(np.float64), k_vector)
 
    buffer_ranges: List[Tuple[int, int]] = []
    if buffer > 0:
        buffer_ranges = [(0, buffer - 1), (n - buffer, n - 1)]
 
    def _ranges_to_mask(rngs):
        msk = np.ones(n, dtype=np.bool_)
        for s, e in rngs:
            s0 = max(s, 0); e0 = min(e, n - 1)
            if s0 <= e0:
                msk[s0:e0 + 1] = False
        return msk
 
    keep_masked   = np.nonzero(_ranges_to_mask(list(ignore_ranges) + list(flag_ranges) + buffer_ranges))[0]
    keep_unmasked = np.nonzero(_ranges_to_mask(buffer_ranges + list(flag_ranges)))[0]
    keep_fixed = np.nonzero(_ranges_to_mask(list(flag_ranges)))[0]
 
    REFRESH = max(1, range_cap)
 
    def _stats(srow):
        s, _, num, den = _calculate_sra_trunc(srow.astype(np.float64), k_vector)
        return max(s, 1e-12), num, den
 
    def _varlen_search(srow, snumer, sdenom, ssra, keep):
        best_sc   = 0.0
        best_win  = (0, -1)
        best_idx  = None
        best_vals = None
 
        nf = srow.shape[0]
        if nf < 2:
            return best_win, best_sc, best_idx, best_vals
 
        valid   = np.arange(nf, dtype=np.int64)
        n_valid = nf
 
        cap      = range_cap + 2
        buf_idxs = np.empty(cap, dtype=np.int64)
        buf_num  = np.empty(cap, dtype=np.float64)
        buf_den  = np.empty(cap, dtype=np.float64)
 
        nin = np.zeros(nf, dtype=np.float64)
        din = np.zeros(nf, dtype=np.float64)
 
        carry_valid    = False
        carry_left_idx = -1
        steps_since_refresh = 0
 
        for pos_i in range(n_valid - 1):
            i = int(valid[pos_i])
 
            if keep[pos_i + 1] != keep[pos_i] + 1:
                carry_valid = False
                continue
 
            use_carry = carry_valid and carry_left_idx == i - 1
 
            if use_carry:
                _nin_din_remove(srow, nin, din, i - 1, k_vector)
            else:
                for ii in range(nf):
                    nin[ii] = 0.0
                    din[ii] = 0.0
                steps_since_refresh = 0
 
            g_in_initialized = False
            m      = 0
            sse_in = 0.0
            sse_out = 0.0
 
            max_k = min(pos_i + range_cap, n_valid - 1)
 
            for kk in range(pos_i + 1, max_k + 1):
                if keep[kk] != keep[kk - 1] + 1:
                    break
                j = int(valid[kk])
 
                if not g_in_initialized:
                    _nin_din_init_full(
                        srow, np.array([i, j], dtype=np.int64),
                        k_vector, nin, din)
                    m, sse_in = _buf_init(
                        srow, np.array([i, j], dtype=np.int64),
                        k_vector, buf_idxs, buf_num, buf_den)
                    sse_out = _sse_out_from_nin_din(
                        srow, snumer, sdenom, nin, din, buf_idxs, m)
                    steps_since_refresh = 0
                    g_in_initialized = True
                else:
                    _nin_din_add(srow, nin, din, j, k_vector)
                    m, sse_in = _buf_add(
                        srow, j, k_vector, buf_idxs, buf_num, buf_den, m, sse_in)
                    sse_out = _sse_out_add(
                        srow, snumer, sdenom, nin, din,
                        buf_idxs, m, j, k_vector, sse_out)
                    steps_since_refresh += 1
                    if steps_since_refresh >= REFRESH:
                        sse_out = _sse_out_from_nin_din(
                            srow, snumer, sdenom, nin, din, buf_idxs, m)
                        steps_since_refresh = 0
 
                sc = 1.0 - (sse_in + sse_out) / ssra
                if sc > best_sc:
                    best_sc   = sc
                    best_win  = (i, j)
                    best_idx  = buf_idxs[:m].copy()
                    best_vals = _predict_on_idxs_trunc(
                        srow, buf_idxs[:m].copy(), k_vector)
 
            if g_in_initialized:
                carry_valid    = True
                carry_left_idx = int(i)
            else:
                carry_valid = False
 
        return best_win, best_sc, best_idx, best_vals
 
    def _fixedlen_sweep(srow, snumer, sdenom, ssra, keep):
        best_sc   = 0.0
        best_win  = (0, -1)
        best_idx  = None
        best_vals = None
 
        nf = srow.shape[0]
        if window_bins <= 0 or window_bins > nf:
            return best_win, best_sc, best_idx, best_vals
 
        cap      = window_bins + 1
        buf_idxs = np.empty(cap, dtype=np.int64)
        buf_num  = np.empty(cap, dtype=np.float64)
        buf_den  = np.empty(cap, dtype=np.float64)
        nin      = np.zeros(nf, dtype=np.float64)
        din      = np.zeros(nf, dtype=np.float64)
 
        m = 0; sse_in = 0.0; sse_out = 0.0
        g_in_initialized = False; steps = 0
 
        for i in range(nf - window_bins + 1):
            j = i + window_bins - 1
 
            if keep[j] - keep[i] != window_bins - 1:
                g_in_initialized = False
                continue
 
            inside = np.arange(i, i + window_bins, dtype=np.int64)
 
            if not g_in_initialized:
                _nin_din_init_full(srow, inside, k_vector, nin, din)
                m, sse_in = _buf_init(srow, inside, k_vector,
                                      buf_idxs, buf_num, buf_den)
                sse_out = _sse_out_from_nin_din(
                    srow, snumer, sdenom, nin, din, buf_idxs, m)
                steps = 0; g_in_initialized = True
            else:
                rem = np.int64(i - 1); add = np.int64(j)
                _nin_din_remove(srow, nin, din, rem, k_vector)
                m, sse_in = _buf_remove(srow, rem, k_vector,
                                        buf_idxs, buf_num, buf_den, m, sse_in)
                sse_out = _sse_out_remove(
                    srow, snumer, sdenom, nin, din,
                    buf_idxs, m, rem, k_vector, sse_out)
                _nin_din_add(srow, nin, din, add, k_vector)
                m, sse_in = _buf_add(srow, add, k_vector,
                                     buf_idxs, buf_num, buf_den, m, sse_in)
                sse_out = _sse_out_add(
                    srow, snumer, sdenom, nin, din,
                    buf_idxs, m, add, k_vector, sse_out)
                steps += 1
                if steps >= REFRESH:
                    sse_out = _sse_out_from_nin_din(
                        srow, snumer, sdenom, nin, din, buf_idxs, m)
                    steps = 0
 
            sc = 1.0 - (sse_in + sse_out) / ssra
            if sc > best_sc:
                best_sc   = sc
                best_win  = (i, j)
                best_idx  = inside.copy()
                best_vals = _predict_on_idxs_trunc(srow, inside, k_vector)
 
        return best_win, best_sc, best_idx, best_vals
 
    def _run(keep, search_fn):
        if keep.shape[0] < 2:
            return (0, -1), 0., None, None
        srow = row[keep].astype(np.float64)
        ssra, snum, sden = _stats(srow)
        (oi, oj), sc, idx_f, vals = search_fn(srow, snum, sden, ssra, keep)
        if oj < oi:
            return (0, -1), sc, None, None
        win      = (int(keep[oi]), int(keep[oj]))
        idx_orig = keep[idx_f] if idx_f is not None else None
        return win, sc, idx_orig, vals
 
    # Run all three
    wm, sm, im, vm = _run(keep_masked, _varlen_search)
    wu, su, iu, vu = _run(keep_unmasked, _varlen_search)
    ovl_u = _overlap_fraction(wu[0], wu[1], ignore_ranges)
    wf, sf, if_, vf = _run(keep_fixed, _fixedlen_sweep)
    ovl_f = _overlap_fraction(wf[0], wf[1], ignore_ranges)
 
    return (0, wm, sm, pred_array, im, vm,
            w * sr_factor_val, range_cap * sr_factor_val,
            wu, su, ovl_u, iu, vu,
            wf, sf, ovl_f, if_, vf,
            window_bins * sr_factor_val)

# ===========================================================================
# 9. Superresolution refinement
# ===========================================================================

def _refine_all_windows(
    spec_native: np.ndarray,
    freq_native: np.ndarray,
    win_masked_sr: Tuple[int, int],
    win_unmasked_sr: Tuple[int, int],
    win_fixed_sr: Tuple[int, int],
    atm_ranges: List[Tuple[int, int]],
    w_native: int,
    range_cap_native: int,
    sr: int,
    buffer: int,
    kernel_kind: str,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Refine SR-resolution windows back to native resolution.

    Uses w_native and range_cap_native (= w*SR, range_cap*SR from the scan)
    as the kernel parameters for refinement
    which uses range_cap as the kernel truncation radius r.
    """
    is_gaussian = kernel_kind.lower() == "gaussian"
    L = len(spec_native)
    n_trimmed = L - 2 * buffer

    if n_trimmed <= 0:
        return (0, 0), (0, 0), (0, 0)

    row_trimmed = np.asarray(spec_native[buffer:L - buffer], dtype=np.float64)

    k_vector = _truncated_kernel_vector(float(w_native), range_cap_native, is_gaussian)
    sra, _, _, _ = _calculate_sra_trunc(row_trimmed, k_vector)
    sra = max(float(sra), 1e-12)

    # Build valid sets
    mask = np.ones(n_trimmed, dtype=np.bool_)
    for s, e in atm_ranges:
        s0 = max(s - buffer, 0)
        e0 = min(e - buffer, n_trimmed - 1)
        if s0 <= e0:
            mask[s0:e0 + 1] = False
    valid_all = np.arange(n_trimmed, dtype=np.int64)
    valid_masked = valid_all[mask]

    def _subset_sse(idxs: np.ndarray) -> float:
        if idxs.size == 0:
            return 0.0
        preds = _predict_on_idxs_trunc(row_trimmed, idxs, k_vector)
        diff = row_trimmed[idxs] - preds
        return float(np.dot(diff, diff))

    def _score_varlen(a: int, b: int, valid: np.ndarray) -> float:
        if b < a:
            return 0
        inside = valid[(valid >= a) & (valid <= b)]
        if inside.size == 0:
            return 0
        outside = np.setdiff1d(valid_all, inside, assume_unique=False)
        sri = _subset_sse(inside)
        sro = _subset_sse(outside)
        return 1.0 - (sri + sro) / sra

    def _refine_varlen_from_sr(x_sr: int, y_sr: int, valid: np.ndarray) -> Tuple[int, int]:
        a_lo = max(x_sr * sr - buffer, 0)
        a_hi = min((x_sr + 1) * sr - 1 - buffer, n_trimmed - 1)
        b_lo = max(y_sr * sr - buffer, 0)
        b_hi = min((y_sr + 1) * sr - 1 - buffer, n_trimmed - 1)

        best_sc = 0
        best_ab = (a_lo, max(a_lo, b_lo))

        for a in range(a_lo, a_hi + 1):
            b_start = max(a, b_lo)
            for b in range(b_start, b_hi + 1):
                sc = _score_varlen(a, b, valid)
                if sc > best_sc:
                    best_sc = sc
                    best_ab = (a, b)

        a_t, b_t = best_ab
        return (a_t + buffer, b_t + buffer)

    def _refine_fixed_from_sr(x_sr: int, y_sr: int) -> Tuple[int, int]:
        freq_step = abs(float(freq_native[1] - freq_native[0]))
        R = _REF_FREQ / (freq_step if freq_step > 0 else 1.0)
        fixed_bins_native = int(math.floor(R)) + 1
        fixed_bins_native = max(1, min(fixed_bins_native, n_trimmed))

        a_lo = max(x_sr * sr - buffer, 0)
        a_hi = min((x_sr + 1) * sr - 1 - buffer, n_trimmed - fixed_bins_native)

        best_sc = 0
        best_a = a_lo

        for a in range(a_lo, a_hi + 1):
            b = a + fixed_bins_native - 1
            sc = _score_varlen(a, b, valid_all)
            if sc > best_sc:
                best_sc = sc
                best_a = a

        b_t = best_a + fixed_bins_native - 1
        return (best_a + buffer, b_t + buffer)

    xm, ym = win_masked_sr
    xu, yu = win_unmasked_sr
    xf, yf = win_fixed_sr

    out_masked = _refine_varlen_from_sr(xm, ym, valid_masked)
    out_unmasked = _refine_varlen_from_sr(xu, yu, valid_all)
    out_fixed = _refine_fixed_from_sr(xf, yf)

    return out_masked, out_unmasked, out_fixed


# ===========================================================================
# 10. Helpers
# ===========================================================================

def _flag_array_to_ranges(flag_array: np.ndarray) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    idx = 0
    for val, grp in groupby(flag_array.tolist()):
        length = sum(1 for _ in grp)
        if val:
            ranges.append((idx, idx + length - 1))
        idx += length
    return ranges


# ===========================================================================
# 11. Full single-row pipeline (with superresolution)
# ===========================================================================

def _scan_single_row(
    amplitude: np.ndarray,
    frequency: np.ndarray,
    atm_ranges: List[Tuple[int, int]],
    flag_ranges: List[Tuple[int, int]],
    buffer: int,
    kernel_kind: str,
) -> ScanResult:
    """
    Run all three scan modes:
    1. Compute SR factor from spectrum length
    2. Superresolve data, ranges, frequencies BEFORE scanning
    3. Scan at SR resolution → get scores and SR-coordinate windows
    4. If SR > 1, refine windows to native resolution
    5. Return scores from SR scan + refined windows
    """
    _BAD = Output(score=0., win_start=0, win_end=0, overlap_pct=0.0)
    _BAD_RESULT: ScanResult = {"masked": _BAD, "unmasked": _BAD, "fixed": _BAD}

    n = amplitude.shape[0]
    if n < 4 or len(frequency) < 2:
        return _BAD_RESULT

    SR = _sr_factor(n)

    amp_2d = amplitude.reshape(1, -1)
    freq_2d = frequency.reshape(1, -1)

    if SR > 1:
        amp_sr_2d = _superresolve(amp_2d, SR)
        freq_sr_2d = _superresolve(freq_2d, SR)
        amp_sr = amp_sr_2d[0]
        freq_sr = freq_sr_2d[0]
        atm_sr = _superresolve_ranges([atm_ranges], SR)[0]
        flag_sr = _superresolve_ranges([flag_ranges], SR)[0]
        buffer_sr = buffer // SR
    else:
        amp_sr = amplitude
        freq_sr = frequency
        atm_sr = atm_ranges
        flag_sr = flag_ranges
        buffer_sr = buffer

    out = _scan_row(
        row=amp_sr.astype(np.float64),
        ignore_ranges=atm_sr,
        flag_ranges=flag_sr,
        frequency=freq_sr,
        buffer=buffer_sr,
        sr_factor_val=SR,
        kernel_kind=kernel_kind,
    )

    win_m_sr = out[1]      # masked window in SR coordinates
    score_m = out[2]       # masked score (from SR scan)
    w_native = out[6]      # w * SR
    rc_native = out[7]     # range_cap * SR
    win_u_sr = out[8]      # unmasked window in SR coordinates
    score_u = out[9]       # unmasked score
    ovl_u = out[10]        # unmasked overlap
    win_f_sr = out[13]     # fixed window in SR coordinates
    score_f = out[14]      # fixed score
    ovl_f = out[15]        # fixed overlap

    if SR > 1:
        win_m, win_u, win_f = _refine_all_windows(
            spec_native=amplitude.astype(np.float64),
            freq_native=frequency.astype(np.float64),
            win_masked_sr=win_m_sr,
            win_unmasked_sr=win_u_sr,
            win_fixed_sr=win_f_sr,
            atm_ranges=atm_ranges,
            w_native=w_native,
            range_cap_native=rc_native,
            sr=SR,
            buffer=buffer,
            kernel_kind=kernel_kind,
        )
    else:
        win_m = win_m_sr
        win_u = win_u_sr
        win_f = win_f_sr

    all_ignore = atm_ranges + flag_ranges
    ovl_m = _overlap_fraction(win_m[0], win_m[1], all_ignore)
    ovl_u_nat = _overlap_fraction(win_u[0], win_u[1], all_ignore) if SR > 1 else ovl_u
    ovl_f_nat = _overlap_fraction(win_f[0], win_f[1], all_ignore) if SR > 1 else ovl_f

    return {
        "masked": Output(score=score_m, win_start=win_m[0], win_end=win_m[1],
                         overlap_pct=ovl_m),
        "unmasked": Output(score=score_u, win_start=win_u[0], win_end=win_u[1],
                           overlap_pct=ovl_u_nat),
        "fixed": Output(score=score_f, win_start=win_f[0], win_end=win_f[1],
                        overlap_pct=ovl_f_nat),
    }


# ===========================================================================
# 12. Public entry point
# ===========================================================================

def _process_single_key(
    key: str,
    amplitude: np.ndarray,
    frequency: np.ndarray,
    flag_array: np.ndarray,
    atm_ranges: Optional[List[Tuple[int, int]]],
    kernel_kind: str,
    buffer_divisor: int,
) -> Tuple[str, ScanResult]:
    """
    Worker function for a single key.  Pickle-friendly (top-level function).
    """
    _BAD = Output(score=0., win_start=0, win_end=0, overlap_pct=0.0)
    _BAD_RESULT: ScanResult = {"masked": _BAD, "unmasked": _BAD, "fixed": _BAD}

    try:
        amplitude = np.asarray(amplitude, dtype=np.float64)
        frequency = np.asarray(frequency, dtype=np.float64)
        flag_array = np.asarray(flag_array, dtype=bool)
    except Exception:
        return key, _BAD_RESULT

    if amplitude.ndim != 1 or frequency.ndim != 1 or flag_array.ndim != 1:
        return key, _BAD_RESULT
    if not (amplitude.shape == frequency.shape == flag_array.shape):
        return key, _BAD_RESULT
    if np.all(amplitude == 0.0) or flag_array.all():
        return key, _BAD_RESULT

    buffer = max(1, len(frequency) // buffer_divisor)
    flag_ranges = _flag_array_to_ranges(flag_array)
    atm = atm_ranges if atm_ranges is not None else []

    try:
        result = _scan_single_row(
            amplitude, frequency, atm, flag_ranges,
            buffer, kernel_kind,
        )
        return key, result
    except Exception:
        return key, _BAD_RESULT


def compute_scan_statistics_scores(
    keyed_input: Dict[str, Input],
    *,
    kernel_kind: str = _DEFAULT_KERNEL,
    buffer_divisor: int = _BUFFER_DIVISOR,
    max_workers: Optional[int] = None,
) -> Dict[str, ScanResult]:
    """
    Entry point for computing NWKR scan statistics on batched inputs.

    Parameters
    ----------
    keyed_input : Dict[str, Input]
        Unique key → Input(amplitude, frequency, flag_array, atm_ranges).
    kernel_kind : str
        ``"gaussian"`` (default) or ``"laplace"``.
    buffer_divisor : int
        Edge buffer = ``len(frequency) // buffer_divisor``.  Default 20.
    max_workers : int or None
        Number of parallel worker processes.
        ``None`` or ``0`` → sequential (no multiprocessing).
        ``1`` → sequential (no multiprocessing).
        ``>1`` → parallel with that many workers.
        ``-1`` → use ``os.cpu_count()`` workers.

    Returns
    -------
    Dict[str, ScanResult]

    Notes
    -----
    Score = 1 - (SSE_inside + SSE_outside) / SRA_global.

    Scores are computed at superresolved resolution (matching calculate_stats.py).
    Windows are refined back to native resolution.

    For large batches (thousands of rows), use ``max_workers=-1`` to parallelize
    across CPU cores, matching the ``ProcessPoolExecutor`` approach in
    ``calculate_stats.py``.
    """
    if not keyed_input:
        return {}

    # Resolve worker count
    n_workers = max_workers
    if n_workers is None or n_workers == 0:
        n_workers = 1
    elif n_workers == -1:
        n_workers = os.cpu_count() or 1

    # Build work items
    work_items = []
    _BAD = Output(score=0., win_start=0, win_end=0, overlap_pct=0.0)
    _BAD_RESULT: ScanResult = {"masked": _BAD, "unmasked": _BAD, "fixed": _BAD}
    results: Dict[str, ScanResult] = {}

    for key, inp in keyed_input.items():
        try:
            amplitude = np.asarray(inp.amplitude, dtype=np.float64)
            frequency = np.asarray(inp.frequency, dtype=np.float64)
            flag_array = np.asarray(inp.flag_array, dtype=bool)
        except Exception as exc:
            logger.warning("[%s] Could not coerce arrays: %s", key, exc)
            results[key] = _BAD_RESULT
            continue

        if amplitude.ndim != 1 or frequency.ndim != 1 or flag_array.ndim != 1:
            raise ValueError(
                f"[{key}] All arrays must be 1-D. Got shapes "
                f"amplitude={amplitude.shape}, frequency={frequency.shape}, "
                f"flag_array={flag_array.shape}."
            )
        if not (amplitude.shape == frequency.shape == flag_array.shape):
            raise ValueError(
                f"[{key}] All arrays must have equal length."
            )

        atm_ranges = inp.atm_ranges if inp.atm_ranges is not None else []
        work_items.append((key, amplitude, frequency, flag_array, atm_ranges))

    if not work_items:
        return results

    # ---- Sequential path ----
    if n_workers <= 1 or len(work_items) == 1:
        for key, amp, freq, flags, atm in work_items:
            k, r = _process_single_key(
                key, amp, freq, flags, atm, kernel_kind, buffer_divisor,
            )
            results[k] = r
        return results

    # ---- Parallel path ----
    logger.info("Running %d rows across %d workers", len(work_items), n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for key, amp, freq, flags, atm in work_items:
            fut = pool.submit(
                _process_single_key,
                key, amp, freq, flags, atm, kernel_kind, buffer_divisor,
            )
            futures[fut] = key

        done_count = 0
        for fut in as_completed(futures):
            done_count += 1
            try:
                k, r = fut.result()
                results[k] = r
            except Exception as exc:
                k = futures[fut]
                logger.warning("[%s] Worker raised: %s", k, exc)
                results[k] = _BAD_RESULT
            if done_count % 500 == 0:
                logger.info("  %d/%d done", done_count, len(work_items))

    return results


# ===========================================================================
# 13. CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run NWKR scan statistics on a single spectrum.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--amplitude", required=True,
                   help=".npy file: 1-D float64 amplitude array.")
    p.add_argument("--frequency", required=True,
                   help=".npy file: 1-D float64 frequency array (GHz).")
    p.add_argument("--flag-array", default=None,
                   help=".npy file: 1-D bool flag array. Optional.")
    p.add_argument("--interference", default=None,
                   help="Transmission parquet for atmospheric detection. Optional.")
    p.add_argument("--key", default="spectrum")
    p.add_argument("--kernel", default="gaussian", choices=["gaussian", "laplace"])
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> Dict[str, ScanResult]:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    amp = np.load(args.amplitude).astype(np.float64)
    freq = np.load(args.frequency).astype(np.float64)
    n = len(amp)

    flag_array = np.load(args.flag_array).astype(bool) if args.flag_array else np.zeros(n, dtype=bool)

    atm_ranges: List[Tuple[int, int]] = []
    if args.interference:
        trans_freqs, trans_vals = load_transmission(args.interference)
        atm_ranges = detect_atm_ranges(freq, trans_freqs, trans_vals)

    keyed_input = {
        args.key: Input(amplitude=amp, frequency=freq,
                        flag_array=flag_array, atm_ranges=atm_ranges)
    }
    results = compute_scan_statistics_scores(keyed_input, kernel_kind=args.kernel)

    for key, scan_result in results.items():
        for mode, out in scan_result.items():
            print(f"[{key}] {mode:10s}  score={out.score:.6f}  "
                  f"window=[{out.win_start}, {out.win_end}]  "
                  f"overlap={out.overlap_pct:.2%}")

    return results


if __name__ == "__main__":
    main()