"""
scan_tool.py
============
General-purpose scan statistics tool.

Supports three regression families:
  - mean   (F₀): constant model, O(nw) via prefix sums
  - poly   (Fₐ): polynomial of degree d, O(nwd³) via Vandermonde prefix sums
  - nwkr        : Nadaraya-Watson kernel regression, O(nwr) via incremental state

Usage examples
--------------
    # NWKR on a CSV column
    python scan_tool.py --input data.csv --column amplitude --method nwkr

    # Polynomial degree-2 with buffer and ignore regions
    python scan_tool.py --input data.npy --method poly --degree 2 \
        --buffer 20 --ignore 50:60 100:110

    # Mean model on a text file (one value per line)
    python scan_tool.py --input values.txt --method mean --max-window 80

    # Output to JSON
    python scan_tool.py --input data.npy --method nwkr --kernel gaussian \
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit


# ═══════════════════════════════════════════════════════════════════════════
# 1. Kernel vector
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def _truncated_kernel_vector(w: float, r: int, kind_is_gaussian: bool) -> np.ndarray:
    d = np.arange(r + 1, dtype=np.float64)
    if kind_is_gaussian:
        return np.exp(-(d * d) / (w * w))
    else:
        sigma = w if w > 1e-12 else 1e-12
        return np.exp(-d / sigma)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Mean scan statistic  — O(nw)
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def _scan_mean(x: np.ndarray, max_w: int, keep: np.ndarray) -> Tuple[float, int, int]:
    """
    Scan using constant-mean model with prefix sums.
    Returns (score, best_i, best_j) in filtered-row coordinates.
    """
    n = x.shape[0]
    if n < 2:
        return 0.0, 0, -1

    # prefix sums: ps[i] = sum(x[0..i-1]), ps2[i] = sum(x[0..i-1]^2)
    ps = np.empty(n + 1, dtype=np.float64)
    ps2 = np.empty(n + 1, dtype=np.float64)
    ps[0] = 0.0
    ps2[0] = 0.0
    for i in range(n):
        ps[i + 1] = ps[i] + x[i]
        ps2[i + 1] = ps2[i] + x[i] * x[i]

    total_sum = ps[n]
    total_sum2 = ps2[n]

    def _sse_range(a: int, b: int) -> float:
        # SSE of x[a..b] under mean model = sum(x^2) - (sum(x))^2 / count
        cnt = b - a + 1
        if cnt <= 0:
            return 0.0
        s = ps[b + 1] - ps[a]
        s2 = ps2[b + 1] - ps2[a]
        return s2 - (s * s) / cnt

    sra = _sse_range(0, n - 1)
    if sra < 1e-12:
        return 0.0, 0, -1

    best_sc = 0.0
    best_i = 0
    best_j = -1

    for i in range(n - 1):
        # check original contiguity
        if keep[i + 1] != keep[i] + 1:
            continue
        j_max = min(i + max_w - 1, n - 1)
        for j in range(i + 1, j_max + 1):
            if keep[j] != keep[j - 1] + 1:
                break
            sri = _sse_range(i, j)
            # outside = [0..i-1] + [j+1..n-1]
            sro_left = _sse_range(0, i - 1) if i > 0 else 0.0
            sro_right = _sse_range(j + 1, n - 1) if j < n - 1 else 0.0
            sro = sro_left + sro_right
            sc = 1.0 - (sri + sro) / sra
            if sc > best_sc:
                best_sc = sc
                best_i = i
                best_j = j

    return best_sc, best_i, best_j


# ═══════════════════════════════════════════════════════════════════════════
# 3. Polynomial scan statistic — O(nwd³)
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def _poly_sse(x: np.ndarray, a: int, b: int, degree: int) -> float:
    """SSE of polynomial fit of given degree on x[a..b]."""
    cnt = b - a + 1
    if cnt <= degree:
        return 0.0

    d1 = degree + 1
    # Build V^T V and V^T x using the segment
    VTV = np.zeros((d1, d1), dtype=np.float64)
    VTx = np.zeros(d1, dtype=np.float64)
    ss = 0.0  # sum of x^2

    for idx in range(a, b + 1):
        t = float(idx - a)  # local coordinate
        xi = x[idx]
        ss += xi * xi
        powers = np.empty(d1, dtype=np.float64)
        powers[0] = 1.0
        for k in range(1, d1):
            powers[k] = powers[k - 1] * t
        for r in range(d1):
            VTx[r] += powers[r] * xi
            for c in range(d1):
                VTV[r, c] += powers[r] * powers[c]

    # Solve VTV @ alpha = VTx via Gaussian elimination
    A = np.empty((d1, d1 + 1), dtype=np.float64)
    for r in range(d1):
        for c in range(d1):
            A[r, c] = VTV[r, c]
        A[r, d1] = VTx[r]

    for col in range(d1):
        pivot = col
        for r in range(col + 1, d1):
            if abs(A[r, col]) > abs(A[pivot, col]):
                pivot = r
        if pivot != col:
            for c in range(d1 + 1):
                A[col, c], A[pivot, c] = A[pivot, c], A[col, c]
        if abs(A[col, col]) < 1e-15:
            return 0.0
        for r in range(col + 1, d1):
            factor = A[r, col] / A[col, col]
            for c in range(col, d1 + 1):
                A[r, c] -= factor * A[col, c]

    alpha = np.empty(d1, dtype=np.float64)
    for r in range(d1 - 1, -1, -1):
        s = A[r, d1]
        for c in range(r + 1, d1):
            s -= A[r, c] * alpha[c]
        alpha[r] = s / A[r, r]

    # SSE = sum(x^2) - 2 * alpha^T @ VTx + alpha^T @ VTV @ alpha
    sse = ss
    for r in range(d1):
        sse -= 2.0 * alpha[r] * VTx[r]
        for c in range(d1):
            sse += alpha[r] * alpha[c] * VTV[r, c]

    return max(sse, 0.0)


@njit(cache=True, fastmath=True)
def _scan_poly(x: np.ndarray, max_w: int, degree: int, keep: np.ndarray) -> Tuple[float, int, int]:
    n = x.shape[0]
    if n < 2:
        return 0.0, 0, -1

    sra = _poly_sse(x, 0, n - 1, degree)
    if sra < 1e-12:
        return 0.0, 0, -1

    best_sc = 0.0
    best_i = 0
    best_j = -1

    for i in range(n - 1):
        if keep[i + 1] != keep[i] + 1:
            continue
        j_max = min(i + max_w - 1, n - 1)
        for j in range(i + 1, j_max + 1):
            if keep[j] != keep[j - 1] + 1:
                break
            sri = _poly_sse(x, i, j, degree)
            sro = 0.0
            if i > 0:
                sro += _poly_sse(x, 0, i - 1, degree)
            if j < n - 1:
                sro += _poly_sse(x, j + 1, n - 1, degree)
            sc = 1.0 - (sri + sro) / sra
            if sc > best_sc:
                best_sc = sc
                best_i = i
                best_j = j

    return best_sc, best_i, best_j


# ═══════════════════════════════════════════════════════════════════════════
# 4. NWKR scan statistic — O(nwr) via incremental state
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def _calculate_sra_trunc(array: np.ndarray, k: np.ndarray):
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
            if d < 0: d = -d
            den += k[d]
            num += k[d] * array[j]
        numer[i] = num
        denom[i] = den
        pred[i] = num / (den if den > eps else eps)
    sra = 0.0
    for i in range(n):
        di = array[i] - pred[i]
        sra += di * di
    return sra, pred, numer, denom


@njit(cache=True, fastmath=True)
def _is_inside(buf_idxs: np.ndarray, m: int, idx: int) -> bool:
    lo, hi = 0, m
    while lo < hi:
        mid = (lo + hi) >> 1
        if buf_idxs[mid] < idx: lo = mid + 1
        else: hi = mid
    return lo < m and buf_idxs[lo] == idx


@njit(cache=True, fastmath=True)
def _nin_din_init(x, idxs_in, k, nin, din):
    n = x.shape[0]; r = k.shape[0] - 1
    for i in range(n): nin[i] = 0.0; din[i] = 0.0
    for p in range(idxs_in.shape[0]):
        j = int(idxs_in[p]); xj = x[j]
        i = j
        while i >= 0:
            d = j - i
            if d > r: break
            nin[i] += k[d]*xj; din[i] += k[d]; i -= 1
        i = j + 1
        while i < n:
            d = i - j
            if d > r: break
            nin[i] += k[d]*xj; din[i] += k[d]; i += 1


@njit(cache=True, fastmath=True)
def _nin_din_add(x, nin, din, idx, k):
    r = k.shape[0]-1; n = nin.shape[0]; xj = x[idx]
    i = idx
    while i >= 0:
        d = idx-i
        if d > r: break
        nin[i] += k[d]*xj; din[i] += k[d]; i -= 1
    i = idx+1
    while i < n:
        d = i-idx
        if d > r: break
        nin[i] += k[d]*xj; din[i] += k[d]; i += 1


@njit(cache=True, fastmath=True)
def _nin_din_remove(x, nin, din, idx, k):
    r = k.shape[0]-1; n = nin.shape[0]; xj = x[idx]
    i = idx
    while i >= 0:
        d = idx-i
        if d > r: break
        nin[i] -= k[d]*xj; din[i] -= k[d]; i -= 1
    i = idx+1
    while i < n:
        d = i-idx
        if d > r: break
        nin[i] -= k[d]*xj; din[i] -= k[d]; i += 1


@njit(cache=True, fastmath=True)
def _sse_out_full(x, numer_all, denom_all, nin, din, buf_idxs, m):
    n = x.shape[0]; eps = 1e-12; sse = 0.0; p = 0
    for i in range(n):
        while p < m and buf_idxs[p] < i: p += 1
        if p < m and buf_idxs[p] == i: continue
        den = denom_all[i]-din[i]; num = numer_all[i]-nin[i]
        pred = num/den if den > eps else 0.0
        sse += (x[i]-pred)**2
    return sse


@njit(cache=True, fastmath=True)
def _sse_out_add(x, numer_all, denom_all, nin, din, buf_idxs, m_new, new_idx, k, sse_out):
    r = k.shape[0]-1; eps = 1e-12; n = nin.shape[0]; xn = x[new_idx]
    nin_o = nin[new_idx]-k[0]*xn; din_o = din[new_idx]-k[0]
    do = denom_all[new_idx]-din_o; no = numer_all[new_idx]-nin_o
    sse_out -= (xn - (no/do if do > eps else 0.0))**2
    for direction in (np.int64(-1), np.int64(1)):
        i = new_idx + direction
        while 0 <= i < n:
            d = abs(i - new_idx)
            if d > r: break
            if not _is_inside(buf_idxs, m_new, i):
                w = k[d]
                nio = nin[i]-w*xn; dio = din[i]-w
                po = (numer_all[i]-nio)/(denom_all[i]-dio) if (denom_all[i]-dio) > eps else 0.0
                pn = (numer_all[i]-nin[i])/(denom_all[i]-din[i]) if (denom_all[i]-din[i]) > eps else 0.0
                sse_out -= (x[i]-po)**2; sse_out += (x[i]-pn)**2
            i += direction
    return sse_out


@njit(cache=True, fastmath=True)
def _sse_out_remove(x, numer_all, denom_all, nin, din, buf_idxs, m_new, rem_idx, k, sse_out):
    r = k.shape[0]-1; eps = 1e-12; n = nin.shape[0]; xr = x[rem_idx]
    dn = denom_all[rem_idx]-din[rem_idx]; nn = numer_all[rem_idx]-nin[rem_idx]
    sse_out += (xr - (nn/dn if dn > eps else 0.0))**2
    for direction in (np.int64(-1), np.int64(1)):
        i = rem_idx + direction
        while 0 <= i < n:
            d = abs(i - rem_idx)
            if d > r: break
            if not _is_inside(buf_idxs, m_new, i):
                w = k[d]
                nio = nin[i]+w*xr; dio = din[i]+w
                po = (numer_all[i]-nio)/(denom_all[i]-dio) if (denom_all[i]-dio) > eps else 0.0
                pn = (numer_all[i]-nin[i])/(denom_all[i]-din[i]) if (denom_all[i]-din[i]) > eps else 0.0
                sse_out -= (x[i]-po)**2; sse_out += (x[i]-pn)**2
            i += direction
    return sse_out


@njit(cache=True, fastmath=True)
def _buf_init(x, idxs_in, k, buf_idxs, buf_num, buf_den):
    m0 = idxs_in.shape[0]; r = k.shape[0]-1; eps = 1e-12
    for t in range(m0): buf_idxs[t] = idxs_in[t]
    for u in range(m0):
        i = int(buf_idxs[u]); sn = 0.0; sd = 0.0
        v = u
        while v >= 0:
            d = i-int(buf_idxs[v])
            if d > r: break
            sn += k[d]*x[int(buf_idxs[v])]; sd += k[d]; v -= 1
        v = u+1
        while v < m0:
            d = int(buf_idxs[v])-i
            if d > r: break
            sn += k[d]*x[int(buf_idxs[v])]; sd += k[d]; v += 1
        buf_num[u] = sn; buf_den[u] = sd
    sse = 0.0
    for u in range(m0):
        p = buf_num[u]/buf_den[u] if buf_den[u] > eps else 0.0
        sse += (x[int(buf_idxs[u])]-p)**2
    return m0, float(sse)


@njit(cache=True, fastmath=True)
def _buf_add(x, new_idx, k, buf_idxs, buf_num, buf_den, m, sse_in):
    r = k.shape[0]-1; eps = 1e-12; xn = x[new_idx]
    lo, hi = 0, m
    while lo < hi:
        mid = (lo+hi)>>1
        if buf_idxs[mid] < new_idx: lo = mid+1
        else: hi = mid
    ins = lo
    for t in range(m-1, ins-1, -1):
        buf_idxs[t+1]=buf_idxs[t]; buf_num[t+1]=buf_num[t]; buf_den[t+1]=buf_den[t]
    buf_idxs[ins] = new_idx; m_new = m+1
    for direction in (np.int64(-1), np.int64(1)):
        v = ins + direction
        while 0 <= v < m_new:
            d = abs(new_idx - int(buf_idxs[v]))
            if d > r: break
            w = k[d]; od = buf_den[v]
            sse_in -= (x[int(buf_idxs[v])] - (buf_num[v]/od if od > eps else 0.0))**2
            buf_num[v] += w*xn; buf_den[v] += w; nd = buf_den[v]
            sse_in += (x[int(buf_idxs[v])] - (buf_num[v]/nd if nd > eps else 0.0))**2
            v += direction
    sn = k[0]*xn; sd = k[0]
    for direction in (np.int64(-1), np.int64(1)):
        v = ins + direction
        while 0 <= v < m_new:
            d = abs(new_idx - int(buf_idxs[v]))
            if d > r: break
            sn += k[d]*x[int(buf_idxs[v])]; sd += k[d]; v += direction
    buf_num[ins] = sn; buf_den[ins] = sd
    sse_in += (xn - (sn/sd if sd > eps else 0.0))**2
    return m_new, float(sse_in)


@njit(cache=True, fastmath=True)
def _buf_remove(x, rem_idx, k, buf_idxs, buf_num, buf_den, m, sse_in):
    r = k.shape[0]-1; eps = 1e-12; xr = x[rem_idx]
    lo, hi = 0, m
    while lo < hi:
        mid = (lo+hi)>>1
        if buf_idxs[mid] < rem_idx: lo = mid+1
        else: hi = mid
    pos = lo
    if pos >= m or buf_idxs[pos] != rem_idx: return m, sse_in
    sse_in -= (xr - (buf_num[pos]/buf_den[pos] if buf_den[pos] > eps else 0.0))**2
    for direction in (np.int64(-1), np.int64(1)):
        v = pos + direction
        while 0 <= v < m:
            d = abs(rem_idx - int(buf_idxs[v]))
            if d > r: break
            w = k[d]; od = buf_den[v]
            sse_in -= (x[int(buf_idxs[v])] - (buf_num[v]/od if od > eps else 0.0))**2
            buf_num[v] -= w*xr; buf_den[v] -= w; nd = buf_den[v]
            sse_in += (x[int(buf_idxs[v])] - (buf_num[v]/nd if nd > eps else 0.0))**2
            v += direction
    for t in range(pos, m-1):
        buf_idxs[t]=buf_idxs[t+1]; buf_num[t]=buf_num[t+1]; buf_den[t]=buf_den[t+1]
    return m-1, float(sse_in)


def _scan_nwkr(
    srow: np.ndarray, keep: np.ndarray,
    max_w: int, kernel_width: float, kernel_cap: int,
    kind_is_gaussian: bool,
) -> Tuple[float, int, int]:
    
    max_w = max_w if max_w is not None else max(2, nf // 10)
    kernel_width = kernel_width if kernel_width is not None else max(3.0, float(w) / 3.0)
    kernel_cap = kernel_cap if kernel_cap is not None else int(2 * kernel_width)

    nf = srow.shape[0]
    if nf < 2:
        return 0.0, 0, -1

    k_vector = _truncated_kernel_vector(kernel_width, kernel_cap, kind_is_gaussian)
    sra, _, numer_all, denom_all = _calculate_sra_trunc(srow, k_vector)
    if sra < 1e-12:
        return 0.0, 0, -1

    REFRESH = max(1, max_w)
    cap = max_w + 2
    buf_idxs = np.empty(cap, dtype=np.int64)
    buf_num = np.empty(cap, dtype=np.float64)
    buf_den = np.empty(cap, dtype=np.float64)
    nin = np.zeros(nf, dtype=np.float64)
    din = np.zeros(nf, dtype=np.float64)

    best_sc = 0.0
    best_i = 0
    best_j = -1

    carry_valid = False
    carry_left = -1
    steps = 0

    for pos_i in range(nf - 1):
        i = pos_i
        if keep[pos_i + 1] != keep[pos_i] + 1:
            carry_valid = False
            continue

        if carry_valid and carry_left == i - 1:
            _nin_din_remove(srow, nin, din, i - 1, k_vector)
        else:
            for ii in range(nf):
                nin[ii] = 0.0; din[ii] = 0.0
            steps = 0

        g_init = False
        m = 0; sse_in = 0.0; sse_out = 0.0
        max_k = min(pos_i + max_w, nf - 1)

        for kk in range(pos_i + 1, max_k + 1):
            if keep[kk] != keep[kk - 1] + 1:
                break
            j = kk

            if not g_init:
                _nin_din_init(srow, np.array([i, j], dtype=np.int64), k_vector, nin, din)
                m, sse_in = _buf_init(srow, np.array([i, j], dtype=np.int64),
                                      k_vector, buf_idxs, buf_num, buf_den)
                sse_out = _sse_out_full(srow, numer_all, denom_all, nin, din, buf_idxs, m)
                steps = 0; g_init = True
            else:
                _nin_din_add(srow, nin, din, j, k_vector)
                m, sse_in = _buf_add(srow, j, k_vector, buf_idxs, buf_num, buf_den, m, sse_in)
                sse_out = _sse_out_add(srow, numer_all, denom_all, nin, din,
                                       buf_idxs, m, j, k_vector, sse_out)
                steps += 1
                if steps >= REFRESH:
                    sse_out = _sse_out_full(srow, numer_all, denom_all, nin, din, buf_idxs, m)
                    steps = 0

            sc = 1.0 - (sse_in + sse_out) / sra
            if sc > best_sc:
                best_sc = sc; best_i = i; best_j = j

        if g_init:
            carry_valid = True; carry_left = i
        else:
            carry_valid = False

    return best_sc, best_i, best_j


# ═══════════════════════════════════════════════════════════════════════════
# 5. Unified scan interface
# ═══════════════════════════════════════════════════════════════════════════

def scan(
    signal: np.ndarray,
    *,
    method: str = "nwkr",
    kernel: str = "gaussian",
    degree: int = 2,
    max_window: Optional[int] = None,
    kernel_width: Optional[float] = None,
    buffer: int = 0,
    ignore: Optional[List[Tuple[int, int]]] = None,
) -> Dict:
    """
    Run scan statistics on a 1-D signal.

    Parameters
    ----------
    signal      : 1-D float array.
    method      : "mean", "poly", or "nwkr".
    kernel      : "gaussian" or "laplace" (only for nwkr).
    degree      : polynomial degree (only for poly).
    max_window  : max window width. Default: len(signal) // 5.
    kernel_width: NWKR bandwidth w. Default: auto from max_window.
    buffer      : number of edge samples to ignore on each side.
    ignore      : list of (start, end) inclusive ranges to ignore.

    Returns
    -------
    dict with keys: score, win_start, win_end, method, filtered_len
    """
    x = np.asarray(signal, dtype=np.float64)
    n = x.shape[0]

    # Build keep set
    mask = np.ones(n, dtype=np.bool_)
    if buffer > 0:
        mask[:buffer] = False
        mask[max(0, n - buffer):] = False
    for s, e in (ignore or []):
        s0 = max(s, 0); e0 = min(e, n - 1)
        if s0 <= e0:
            mask[s0:e0 + 1] = False

    keep = np.nonzero(mask)[0]
    nf = keep.shape[0]
    if nf < 2:
        return {"score": 0.0, "win_start": 0, "win_end": -1,
                "method": method, "filtered_len": nf}

    srow = x[keep]
    w = max_window if max_window is not None else max(2, nf // 5)

    if method == "mean":
        sc, oi, oj = _scan_mean(srow, w, keep)
    elif method == "poly":
        sc, oi, oj = _scan_poly(srow, w, degree, keep)
    elif method == "nwkr":
        kw = kernel_width if kernel_width is not None else max(3.0, float(w) / 3.0)
        kc = int(2 * kw)
        sc, oi, oj = _scan_nwkr(srow, keep, w, kw, kc, kernel.lower() == "gaussian")
    else:
        raise ValueError(f"Unknown method: {method!r}")

    if oj < oi:
        return {"score": 0.0, "win_start": 0, "win_end": -1,
                "method": method, "filtered_len": nf}

    return {
        "score": float(sc),
        "win_start": int(keep[oi]),
        "win_end": int(keep[oj]),
        "win_start_filtered": int(oi),
        "win_end_filtered": int(oj),
        "method": method,
        "filtered_len": nf,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. I/O helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_signal(path: str, column: Optional[str] = None,
                 per_row: bool = False, separator: str = ",") -> List[np.ndarray]:
    """
    Returns a list of 1-D arrays.
    per_row=False → one element (whole column flattened).
    per_row=True  → one element per row (each cell is a list/array).
    """
    if path.endswith(".npy"):
        data = np.load(path, allow_pickle=True)
        if per_row and data.ndim == 2:
            return [data[i].astype(np.float64) for i in range(data.shape[0])]
        elif per_row and data.dtype == object:
            return [np.asarray(row, dtype=np.float64) for row in data]
        else:
            return [data.astype(np.float64).ravel()]

    elif path.endswith(".npz"):
        f = np.load(path, allow_pickle=True)
        key = column or list(f.keys())[0]
        data = f[key]
        if per_row and data.ndim == 2:
            return [data[i].astype(np.float64) for i in range(data.shape[0])]
        elif per_row and data.dtype == object:
            return [np.asarray(row, dtype=np.float64) for row in data]
        else:
            return [data.astype(np.float64).ravel()]

    elif path.endswith(".csv") or path.endswith(".tsv") or path.endswith(".parquet"):
        import pandas as pd
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        elif path.endswith(".csv") or path.endswith(".tsv"):
            import pandas as pd
            sep = separator if separator != "tab" else "\t"
            df = pd.read_csv(path, sep=sep)
        col = column or df.columns[0]
        if per_row:
            signals = []
            for val in df[col]:
                if isinstance(val, (list, np.ndarray)):
                    signals.append(np.asarray(val, dtype=np.float64))
                elif isinstance(val, str):
                    # try parsing "[1.0, 2.0, ...]" or "1.0,2.0,..."
                    cleaned = val.strip().strip("[]")
                    signals.append(np.fromstring(cleaned, sep=",", dtype=np.float64))
                else:
                    signals.append(np.array([float(val)], dtype=np.float64))
            return signals
        else:
            return [df[col].to_numpy(dtype=np.float64)]
    else:
        data = np.loadtxt(path, dtype=np.float64)
        if per_row and data.ndim == 2:
            return [data[i] for i in range(data.shape[0])]
        else:
            return [data.ravel()]


def _parse_ranges(range_strs: Optional[List[str]]) -> List[Tuple[int, int]]:
    if not range_strs:
        return []
    ranges = []
    for r in range_strs:
        parts = r.split(":")
        if len(parts) == 2:
            ranges.append((int(parts[0]), int(parts[1])))
        else:
            idx = int(parts[0])
            ranges.append((idx, idx))
    return ranges


# ═══════════════════════════════════════════════════════════════════════════
# 7. CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="General-purpose scan statistics tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
        python scan_tool.py --input data.csv --column price --method nwkr
        python scan_tool.py --input signal.npy --method mean --buffer 10 --ignore 50:60
        python scan_tool.py --input data.csv --column value --method poly --degree 2 --max-window 80
        python scan_tool.py --input data.npy --method nwkr --kernel laplace --output result.json""",
    )
    p.add_argument("--input", "-i", required=True,
                   help="Input file (.npy, .npz, .csv, .parquet, or plain text)")
    p.add_argument("--column", "-c", default=None,
                   help="Column name (for CSV/parquet/npz). Default: first column.")
    p.add_argument("--method", "-m", default="nwkr", choices=["mean", "poly", "nwkr"],
                   help="Regression family. Default: nwkr.")
    p.add_argument("--kernel", "-k", default="gaussian", choices=["gaussian", "laplace"],
                   help="Kernel type for NWKR. Default: gaussian.")
    p.add_argument("--degree", "-d", type=int, default=2,
                   help="Polynomial degree (for --method poly). Default: 2.")
    p.add_argument("--max-window", "-w", type=int, default=None,
                   help="Max window width. Default: len(signal) // 10.")
    p.add_argument("--kernel-width", type=float, default=None,
                   help="NWKR bandwidth w. Default: auto.")
    p.add_argument("--buffer", "-b", type=int, default=0,
                   help="Ignore this many samples on each edge. Default: 0.")
    p.add_argument("--ignore", nargs="*", default=None,
                   help="Ranges to ignore, as start:end pairs (inclusive). "
                        "Example: --ignore 0:49 100:110")
    p.add_argument("--output", "-o", default=None,
                   help="Write JSON result to this file. Default: print to stdout.")
    p.add_argument("--per-row", action="store_true",
                   help="Treat each row of the column as a separate signal "
                        "(each cell contains a list/array). Default: treat "
                        "the entire column as one signal.")
    p.add_argument("--sep", default=",",
                   help="CSV delimiter. Default: ','. Use 'tab' for TSV.")

    args = p.parse_args()
    signals = _load_signal(args.input, args.column, args.per_row, args.sep)
    ignore_ranges = _parse_ranges(args.ignore)

    print(f"Loaded {len(signals)} signal(s)", file=sys.stderr)
    if args.per_row:
        lengths = [len(s) for s in signals]
        print(f"  lengths: min={min(lengths)}, max={max(lengths)}", file=sys.stderr)
    else:
        print(f"  length: {len(signals[0])} samples", file=sys.stderr)
    if args.buffer > 0:
        print(f"Buffer: {args.buffer} on each edge", file=sys.stderr)
    if ignore_ranges:
        print(f"Ignore ranges: {ignore_ranges}", file=sys.stderr)

    results = []
    for idx, sig in enumerate(signals):
        if len(sig) < 2:
            results.append({"row": idx, "score": 0.0, "win_start": 0, "win_end": -1,
                            "method": args.method, "filtered_len": len(sig)})
            continue
        r = scan(
            sig,
            method=args.method,
            kernel=args.kernel,
            degree=args.degree,
            max_window=args.max_window,
            kernel_width=args.kernel_width,
            buffer=args.buffer,
            ignore=ignore_ranges,
        )
        if args.per_row:
            r["row"] = idx
        results.append(r)

        if args.per_row and (idx + 1) % 500 == 0:
            print(f"  processed {idx + 1}/{len(signals)}", file=sys.stderr)

    output = results if args.per_row else results[0]
    result_str = json.dumps(output, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result_str + "\n")
        print(f"Result written to {args.output}", file=sys.stderr)
    else:
        print(result_str)

if __name__ == "__main__":
    main()