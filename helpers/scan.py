from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple
from .config import ref_freq, get_kernel_kind, KernelKind
from .kernels import _kernel_matrix, truncated_kernel_vector, get_kernel as get_kernel
from .kernel_optimized_state import (
    buf_init, buf_add, buf_remove,
    nin_din_init_full, nin_din_add, nin_din_remove,
    sse_out_from_nin_din, sse_out_add, sse_out_remove,
)
from .scoring import (
    _nwkr_predict_subset,
    _nwkr_sse_on_subset,
    calculate_gaussian_sra_trunc,
    calculate_laplace_sra_fast_trunc,
)
from .regressors import (
    build_regressor_sra,
    predict_on_idxs_mean,
    predict_on_idxs_poly,
    predict_on_idxs_krr,
)
from .predictors import predict_on_idxs_trunc

FIX_FIRST_START = False

def set_fix_first_start(flag: bool):
    global FIX_FIRST_START
    FIX_FIRST_START = bool(flag)


def predict_subset_using_regressor(x, idxs, family, **kwargs):
    family = family.lower()
    if family == "mean":   return predict_on_idxs_mean(x, idxs)
    if family == "poly":   return predict_on_idxs_poly(x, idxs,
                               degree=int(kwargs.get("degree", 1)), reg=float(kwargs.get("reg", 1e-8)))
    if family == "krr":    return predict_on_idxs_krr(x, idxs,
                               kernel=kwargs.get("kernel", "rbf"),
                               kernel_param=float(kwargs.get("kernel_param", 1.0)),
                               reg=float(kwargs.get("reg", 1e-3)))
    raise ValueError(f"Unknown family {family!r}")


def scan_row_with_regressor(x, w, family="mean", **kwargs):
    x = x.astype(np.float64); n = x.shape[0]
    sra_all, _, _, _, _, _, _ = build_regressor_sra(x, family, **kwargs)
    best_score = -np.inf; best_a = best_b = -1
    for a in range(n):
        for b in range(a, min(n - 1, a + w - 1) + 1):
            inside = np.arange(a, b + 1, dtype=np.int64)
            if a == 0 and b == n - 1: continue
            outside = (np.arange(b + 1, n, dtype=np.int64) if a == 0 else
                       np.arange(0, a, dtype=np.int64) if b == n - 1 else
                       np.concatenate((np.arange(0, a, dtype=np.int64),
                                       np.arange(b + 1, n, dtype=np.int64))))
            ri = x[inside]  - predict_subset_using_regressor(x, inside,  family, **kwargs)
            ro = x[outside] - predict_subset_using_regressor(x, outside, family, **kwargs)
            sc = sra_all - float(ri @ ri) - float(ro @ ro)
            if sc > best_score: best_score = sc; best_a = a; best_b = b
    return best_score, best_a, best_b


def scan_row_with_nwkr_naive(params: Tuple):
    row_idx, row, ignore, flags, freqs, buffer, sr_factor = params[:7]
    w_override          = params[7]  if len(params) >= 8  else None
    range_cap_override  = params[8]  if len(params) >= 9  else None
    fixed_bins_override = params[9]  if len(params) >= 10 else None
    row = np.asarray(row, dtype=np.float64); n = row.size
    _bad = (row_idx,(0,-1),0.,np.array([]),None,None,0,0,
            (0,-1),0.,0.,None,None,(0,-1),0.,0.,None,None,0)
    if len(freqs) < 2 or not np.isfinite(freqs[:2]).all(): return _bad
    freq_step = abs(freqs[1] - freqs[0]); L = len(freqs)
    R = ref_freq / (freq_step if freq_step > 0 else ref_freq)
    w_auto = int(round(max(3, min(R, L / 16)))); range_cap_auto = 3 * w_auto
    window_bins_auto = int(math.floor(R)) + 1
    w           = int(w_override)           if (w_override           is not None and int(w_override)           > 0) else w_auto
    range_cap   = int(range_cap_override)   if (range_cap_override   is not None and int(range_cap_override)   > 0) else range_cap_auto
    window_bins = int(fixed_bins_override)  if (fixed_bins_override  is not None and int(fixed_bins_override)  > 0) else window_bins_auto
    row_trimmed = row[buffer: n - buffer]; n_trimmed = row_trimmed.size
    if n_trimmed <= 2: return _bad
    kernel_kind = get_kernel_kind()
    kind = "gaussian" if kernel_kind == KernelKind.GAUSSIAN else "laplace"
    K_full = _kernel_matrix(n, float(w), kind); K_trim = _kernel_matrix(n_trimmed, float(w), kind)
    all_full = np.arange(n, dtype=np.int64); all_trim = np.arange(n_trimmed, dtype=np.int64)
    sra_full = max(_nwkr_sse_on_subset(row, K_full, all_full), 1e-12)
    sra_trim = max(_nwkr_sse_on_subset(row_trimmed, K_trim, all_trim), 1e-12)
    pred_array = _nwkr_predict_subset(row_trimmed, K_trim, all_trim)
    ignore_trimmed = []
    for s, e in (ignore or []):
        s0 = max(s - buffer, 0); e0 = min(e - buffer, n_trimmed - 1)
        if s0 <= e0: ignore_trimmed.append((s0, e0))
    for s, e in (flags or []):
        s0 = max(s - buffer, 0); e0 = min(e - buffer, n_trimmed - 1)
        if s0 <= e0: ignore_trimmed.append((s0, e0))
    mask = np.ones(n_trimmed, dtype=np.bool_)
    for s0, e0 in ignore_trimmed: mask[s0:e0 + 1] = False
    valid_idxs_masked = np.nonzero(mask)[0]
    def _ov(a, b, rngs):
        if b <= a: return 0.
        return sum(max(0, min(b,e)-max(a,s)+1) for s,e in rngs)/(b-a+1)
    def _score(i, j):
        ins = np.arange(i, j+1, dtype=np.int64); out = np.setdiff1d(all_trim, ins, assume_unique=True)
        sc = 1. - ((_nwkr_sse_on_subset(row_trimmed,K_trim,ins)+_nwkr_sse_on_subset(row_trimmed,K_trim,out))/sra_trim)
        return sc, ins, _nwkr_predict_subset(row_trimmed, K_trim, ins)
    def _varlen(valid):
        best_sc=-np.inf; best_win=(0,0); best_idx=None; best_vals=None
        if valid.size<2: return best_win,0.,None,None
        for pi in range(valid.size):
            i=int(valid[pi])
            if pi+1<valid.size and valid[pi+1]!=i+1: continue
            for kk in range(pi+1,min(pi+range_cap,valid.size-1)+1):
                if valid[kk]!=valid[kk-1]+1: break
                j=int(valid[kk]); sc,ins,vals=_score(i,j)
                if sc>best_sc: best_sc=sc;best_win=(i,j);best_idx=ins+buffer;best_vals=vals
        oi,oj=best_win; return (oi+buffer,oj+buffer),float(best_sc),best_idx,best_vals
    def _fixed():
        best_sc=-np.inf; best_win=(0,0); best_idx=None; best_vals=None
        if window_bins<=1 or window_bins>n: return best_win,0.,None,None
        for i in range(n-window_bins+1):
            j=i+window_bins-1; ins=np.arange(i,j+1,dtype=np.int64); out=np.setdiff1d(all_full,ins,assume_unique=True)
            sc=1.-((_nwkr_sse_on_subset(row,K_full,ins)+_nwkr_sse_on_subset(row,K_full,out))/sra_full)
            if sc>best_sc: best_sc=sc;best_win=(i,j);best_idx=ins;best_vals=_nwkr_predict_subset(row,K_full,ins)
        return best_win,float(best_sc),best_idx,best_vals
    wm,sm,im,vm=_varlen(valid_idxs_masked)
    wu,su,iu,vu=_varlen(all_trim); ovl_u=_ov(wu[0],wu[1],ignore or [])
    wf,sf,if_,vf=_fixed(); ovl_f=_ov(wf[0],wf[1],ignore or [])
    return (row_idx,wm,sm,pred_array,im,vm,w*sr_factor,range_cap*sr_factor,
            wu,su,ovl_u,iu,vu,wf,sf,ovl_f,if_,vf,window_bins*sr_factor)


def scan_row_with_nwkr(params: Tuple):
    row_idx, row, ignore, flags, freqs, buffer, sr_factor = params[:7]
    w_override          = params[7]  if len(params) >= 8 and params[7] is not None  else None
    kernel_cap_override  = params[8]  if len(params) >= 9 and params[8] is not None else None
    fixed_bins_override = params[9]  if len(params) >= 10 and params[9] is not None else None
    masked_search_flag = params[10]  if len(params) >= 11 and params[10] is not None else True
    unmasked_search_flag = params[11]  if len(params) >= 12 and params[11] is not None else True
    fixed_search_flag = params[12]  if len(params) >= 13 and params[12] is not None else True

    n = row.shape[0]; kernel_kind = get_kernel_kind()
    _bad = (row_idx,(0,-1),0.,np.array([]),None,None,0,0,
            (0,-1),0.,0.,None,None,(0,-1),0.,0.,None,None,0)

    def _ov(a, b, rngs):
        if b <= a: return 0.
        wl = b-a+1; ov = 0
        for s,e in rngs:
            lo=max(a,s); hi=min(b,e)
            if hi>=lo: ov+=hi-lo+1
        return ov/wl

    if len(freqs) < 2 or not np.isfinite(freqs[:2]).all(): return _bad

    freq_step = abs(freqs[1] - freqs[0]); L = len(freqs)
    R = ref_freq / (freq_step if freq_step > 0 else 1.)
    w_auto = int(round(max(3, min(R, L / 16))))
    kernel_cap_auto = 2 * w_auto
    range_cap_auto = 3 * w_auto
    window_bins_auto = int(math.floor(R)) + 1

    w           = int(w_override)           if (w_override           is not None and int(w_override)           > 0) else w_auto
    range_cap = int(w_override * 3)   if (w_override   is not None and int(w_override)   > 0) else range_cap_auto
    kernel_cap   = int(kernel_cap_override)   if (kernel_cap_override   is not None and int(kernel_cap_override)   > 0) else kernel_cap_auto
    window_bins = int(fixed_bins_override)  if (fixed_bins_override  is not None and int(fixed_bins_override)  > 0) else window_bins_auto

    if kernel_kind == KernelKind.LAPLACE:
        from .scoring import _laplace_accum_trunc_1d
        sigma = float(max(w, 1))
    k_vector = truncated_kernel_vector(
        w=float(w), r=kernel_cap,
        kind="gaussian" if kernel_kind == KernelKind.GAUSSIAN else "laplace")

    # global prediction on the full row (for output/plotting only)
    if kernel_kind == KernelKind.GAUSSIAN:
        _,_,pred_array,_,_,_ = calculate_gaussian_sra_trunc(row, k_vector)
    else:
        _,_,pred_array,_ = calculate_laplace_sra_fast_trunc(row, sigma, kernel_cap)

    def _ranges_to_mask(rngs):
        msk = np.ones(n, dtype=np.bool_)
        for s, e in rngs:
            s0 = max(s, 0); e0 = min(e, n - 1)
            if s0 <= e0: msk[s0:e0+1] = False
        return msk

    buffer_ranges: List[Tuple[int,int]] = []
    if buffer > 0:
        buffer_ranges = [(0, buffer - 1), (n - buffer, n - 1)]

    keep_masked   = np.nonzero(_ranges_to_mask(list(ignore) + list(flags) + buffer_ranges))[0]
    keep_unmasked = np.nonzero(_ranges_to_mask(list(flags) + buffer_ranges))[0]
    keep_fixed = np.nonzero(_ranges_to_mask(list(flags)))[0]

    REFRESH = max(1, range_cap)

    # ---- compute SRA / numer / denom on a filtered row ----
    def _stats(srow):
        nf = srow.shape[0]
        if kernel_kind == KernelKind.GAUSSIAN:
            s,_,_,num,den,_ = calculate_gaussian_sra_trunc(srow, k_vector)
        else:
            s,_,_,_ = calculate_laplace_sra_fast_trunc(srow, sigma, kernel_cap)
            num = _laplace_accum_trunc_1d(srow.astype(np.float64), sigma, kernel_cap)
            den = _laplace_accum_trunc_1d(np.ones(nf, dtype=np.float64), sigma, kernel_cap)
        return max(s, 1e-12), num, den

    def _varlen_search(srow, snumer, sdenom, ssra, keep):
        best_sc_var       = 0.
        best_win_var      = (0, -1)
        best_idx_var      = None
        best_vals_var     = None

        nf = srow.shape[0]
        if nf < 2:
            return best_win_var, best_sc_var, best_idx_var, best_vals_var

        valid   = np.arange(nf, dtype=np.int64)
        n_valid = nf

        # cap      = range_cap + 2
        cap      = n_valid + 1
        buf_idxs = np.empty(cap, dtype=np.int64)
        buf_num  = np.empty(cap, dtype=np.float64)
        buf_den  = np.empty(cap, dtype=np.float64)

        nin = np.zeros(nf, dtype=np.float64)
        din = np.zeros(nf, dtype=np.float64)

        carry_valid     = False
        carry_left_idx  = -1
        carry_right_idx = -1
        carry_sse_out   = 0.
        steps_since_refresh = 0

        start_iter = range(0, 1) if FIX_FIRST_START else range(0, n_valid - 1)
        for pos_i in start_iter:
        # for pos_i in range(n_valid - 1):
            i = int(valid[pos_i])

            if keep[pos_i + 1] != keep[pos_i] + 1:
                carry_valid = False
                continue

            use_carry = carry_valid and carry_left_idx == i - 1

            if use_carry:
                nin_din_remove(srow, nin, din, i - 1, k_vector)
            else:
                for ii in range(nf):
                    nin[ii] = 0.0; din[ii] = 0.0
                carry_sse_out = 0.
                steps_since_refresh = 0

            g_in_initialized = False
            m      = 0
            sse_in = 0.
            sse_out = 0.

            # max_k = min(pos_i + range_cap, n_valid - 1)
            max_k = n_valid - 1 if FIX_FIRST_START else min(pos_i + range_cap, nf, n_valid - 1)

            for kk in range(pos_i + 1, max_k + 1):
                if keep[kk] != keep[kk - 1] + 1:
                    break
                j = int(valid[kk])

                if not g_in_initialized:
                    nin_din_init_full(
                        srow, np.array([i, j], dtype=np.int64), k_vector, nin, din)
                    m, sse_in = buf_init(
                        srow, np.array([i, j], dtype=np.int64),
                        k_vector, buf_idxs, buf_num, buf_den)
                    sse_out = sse_out_from_nin_din(
                        srow, snumer, sdenom, nin, din, buf_idxs, m)
                    steps_since_refresh = 0
                    g_in_initialized = True
                else:
                    nin_din_add(srow, nin, din, j, k_vector)
                    m, sse_in = buf_add(
                        srow, j, k_vector, buf_idxs, buf_num, buf_den, m, sse_in)
                    sse_out = sse_out_add(
                        srow, snumer, sdenom, nin, din,
                        buf_idxs, m, j, k_vector, sse_out)
                    steps_since_refresh += 1
                    if steps_since_refresh >= REFRESH:
                        sse_out = sse_out_from_nin_din(
                            srow, snumer, sdenom, nin, din, buf_idxs, m)
                        steps_since_refresh = 0

                sc = 1. - (sse_in + sse_out) / ssra
                if sc > best_sc_var:
                    best_sc_var   = sc
                    best_win_var  = (i, j)
                    best_idx_var  = buf_idxs[:m].copy()
                    best_vals_var = predict_on_idxs_trunc(
                        srow, buf_idxs[:m].copy(), k_vector)

            if g_in_initialized:
                carry_valid     = True
                carry_left_idx  = int(i)
                carry_right_idx = int(buf_idxs[m - 1]) if m > 0 else i
                carry_sse_out   = sse_out
            else:
                carry_valid = False

        return best_win_var, best_sc_var, best_idx_var, best_vals_var

    def _fixedlen_sweep(srow, snumer, sdenom, ssra, keep):
        best_sc_fix       = 0.
        best_win_fix      = (0, -1)
        best_idx_fix      = None
        best_vals_fix     = None

        nf = srow.shape[0]
        if window_bins <= 0 or window_bins > nf:
            return best_win_fix, best_sc_fix, best_idx_fix, best_vals_fix

        cap      = window_bins + 1
        buf_idxs = np.empty(cap, dtype=np.int64)
        buf_num  = np.empty(cap, dtype=np.float64)
        buf_den  = np.empty(cap, dtype=np.float64)
        nin      = np.zeros(nf, dtype=np.float64)
        din      = np.zeros(nf, dtype=np.float64)

        m = 0; sse_in = 0.; sse_out = 0.
        g_in_initialized = False; steps = 0

        for i in range(nf - window_bins + 1):
            j = i + window_bins - 1

            # window spans an original gap → skip and reset state
            if keep[j] - keep[i] != window_bins - 1:
                g_in_initialized = False
                continue

            inside = np.arange(i, i + window_bins, dtype=np.int64)

            if not g_in_initialized:
                nin_din_init_full(srow, inside, k_vector, nin, din)
                m, sse_in = buf_init(srow, inside, k_vector, buf_idxs, buf_num, buf_den)
                sse_out = sse_out_from_nin_din(
                    srow, snumer, sdenom, nin, din, buf_idxs, m)
                steps = 0; g_in_initialized = True
            else:
                rem = np.int64(i - 1); add = np.int64(j)
                nin_din_remove(srow, nin, din, rem, k_vector)
                m, sse_in = buf_remove(srow, rem, k_vector, buf_idxs, buf_num, buf_den, m, sse_in)
                sse_out = sse_out_remove(
                    srow, snumer, sdenom, nin, din, buf_idxs, m, rem, k_vector, sse_out)
                nin_din_add(srow, nin, din, add, k_vector)
                m, sse_in = buf_add(srow, add, k_vector, buf_idxs, buf_num, buf_den, m, sse_in)
                sse_out = sse_out_add(
                    srow, snumer, sdenom, nin, din, buf_idxs, m, add, k_vector, sse_out)
                steps += 1
                if steps >= REFRESH:
                    sse_out = sse_out_from_nin_din(
                        srow, snumer, sdenom, nin, din, buf_idxs, m)
                    steps = 0

            sc = 1. - (sse_in + sse_out) / ssra
            if sc > best_sc_fix:
                best_sc_fix   = sc
                best_win_fix  = (i, j)
                best_idx_fix  = inside.copy()
                best_vals_fix = predict_on_idxs_trunc(srow, inside, k_vector)

        return best_win_fix, best_sc_fix, best_idx_fix, best_vals_fix

    # ------------------------------------------------------------------
    # _run — now passes keep to the search function
    # ------------------------------------------------------------------
    def _run(keep, search_fn):
        if keep.shape[0] < 2:
            return (0, -1), 0., None, None
        srow = row[keep]
        ssra, snum, sden = _stats(srow)
        (oi, oj), sc, idx_f, vals = search_fn(srow, snum, sden, ssra, keep)
        if oj < oi:
            return (0, -1), sc, None, None
        win      = (int(keep[oi]), int(keep[oj]))
        idx_orig = keep[idx_f] if idx_f is not None else None
        return win, sc, idx_orig, vals

    if masked_search_flag:
        wm, sm, im, vm = _run(keep_masked, _varlen_search)
    else:
        wm, sm, im, vm = (0, -1), 0., None, None
    if unmasked_search_flag:
        wu, su, iu, vu = _run(keep_unmasked, _varlen_search)
        ovl_u = _ov(wu[0], wu[1], ignore)
    else:
        wu, su, iu, vu = (0, -1), 0., None, None
        ovl_u = 0.
    if fixed_search_flag:
        wf, sf, if_, vf = _run(keep_fixed, _fixedlen_sweep)
        ovl_f = _ov(wf[0], wf[1], ignore)
    else:
        wf, sf, if_, vf = (0, -1), 0., None, None
        ovl_f = 0.

    return (row_idx, wm, sm, pred_array, im, vm, w * sr_factor, range_cap * sr_factor,
            wu, su, ovl_u, iu, vu, wf, sf, ovl_f, if_, vf, window_bins * sr_factor)