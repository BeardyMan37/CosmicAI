import ast
import time
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
import numba
from numba import njit, prange
from scipy.signal import chirp, find_peaks, peak_widths
from concurrent.futures import ProcessPoolExecutor, as_completed


import ast
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List


ref_freq = 31.25e6

def match_and_correct(
    freq_array: np.ndarray,
    trans_freqs: np.ndarray,
    trans_vals: np.ndarray
) -> np.ndarray:
    idxs = np.searchsorted(trans_freqs, freq_array)
    idxs[idxs == len(trans_freqs)] = len(trans_freqs) - 1
    left  = np.maximum(idxs - 1, 0)
    right = idxs
    dl = np.abs(freq_array - trans_freqs[left])
    dr = np.abs(trans_freqs[right] - freq_array)
    nearest = np.where(dl <= dr, left, right)
    mt = trans_vals[nearest]
    return mt

def load_data_by_length(data_path: str, 
                        interference_path: str
) -> Tuple[pd.DataFrame, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """Load spectrograms + metadata, group by the length of each spectrum."""
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path, sep='|', dtype=str, header=0)
        df['uid'] = df.index
        df = df.reset_index(drop=True)
        df['frequency_array'] = df['frequency_array'].apply(lambda s: np.array(ast.literal_eval(s), dtype=float))
        df['frequency_array'] = df['frequency_array'].apply(lambda freqs: [f/1e9 for f in freqs])
        trans_df = pd.read_parquet(interference_path)
        trans_freqs = trans_df['Frequency (GHz)'].values
        trans_vals  = trans_df['Transmission (%)'].values
        
        results = df.apply(
            lambda row: match_and_correct(
                np.array(row['frequency_array'], dtype=float),
                trans_freqs,
                trans_vals
            ),
            axis=1
        )

        df['transmission_array'] = results
        
        interference = []
        for index in df.index:
            freqs = np.array(df.loc[index, 'frequency_array'], dtype=float)
            trans = np.array(df.loc[index, 'transmission_array'], dtype=float)

            troguhs, props = find_peaks(-trans, prominence=1)
            _, _, left_ips, right_ips = peak_widths(-trans, troguhs, rel_height=0.75)

            left_freqs  = np.interp(left_ips,  np.arange(len(freqs)), freqs)
            right_freqs = np.interp(right_ips, np.arange(len(freqs)), freqs)
            widths_freq = right_freqs - left_freqs

            trough_freqs  = freqs[troguhs]
            trough_ranges = []
            for i in range(len(trough_freqs)):
                trough_ranges.append((trough_freqs[i] - widths_freq[i] / 2, trough_freqs[i] + widths_freq[i] / 2))
            trough_ranges = np.array(trough_ranges)

            closest_idxs = []
            for troguhs_range in trough_ranges:
                start, end = troguhs_range[0], troguhs_range[1]
                closest_start_idx = int(np.abs(freqs - start).argmin())
                closest_end_idx = int(np.abs(freqs - end).argmin())
                closest_idxs.append((closest_start_idx, closest_end_idx))
            interference.append(closest_idxs)

        df['atmospheric_interference'] = interference
        actual_specs = [np.array(ast.literal_eval(s), dtype=float)
                 for s in df['spec_arrays']]
        freqs = [np.array(x, dtype=float)
                 for x in df['frequency_array'].tolist()]
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
        df['uid'] = df.index
        df = df.reset_index(drop=True)
        df['frequency_array'] = df['frequency_array'].apply(lambda freqs: [f/1e9 for f in freqs])
        trans_df = pd.read_parquet(interference_path)
        trans_freqs = trans_df['Frequency (GHz)'].values
        trans_vals  = trans_df['Transmission (%)'].values
        
        results = df.apply(
            lambda row: match_and_correct(
                np.array(row['frequency_array'], dtype=float),
                trans_freqs,
                trans_vals
            ),
            axis=1
        )

        df['transmission_array'] = results
        
        interference = []
        for index in df.index:
            freqs = np.array(df.loc[index, 'frequency_array'], dtype=float)
            trans = np.array(df.loc[index, 'transmission_array'], dtype=float)

            troguhs, props = find_peaks(-trans, prominence=1)
            _, _, left_ips, right_ips = peak_widths(-trans, troguhs, rel_height=0.75)

            left_freqs  = np.interp(left_ips,  np.arange(len(freqs)), freqs)
            right_freqs = np.interp(right_ips, np.arange(len(freqs)), freqs)
            widths_freq = right_freqs - left_freqs

            trough_freqs  = freqs[troguhs]
            trough_ranges = []
            for i in range(len(trough_freqs)):
                trough_ranges.append((trough_freqs[i] - widths_freq[i] / 2, trough_freqs[i] + widths_freq[i] / 2))
            trough_ranges = np.array(trough_ranges)

            closest_idxs = []
            for troguhs_range in trough_ranges:
                start, end = troguhs_range[0], troguhs_range[1]
                closest_start_idx = int(np.abs(freqs - start).argmin())
                closest_end_idx = int(np.abs(freqs - end).argmin())
                closest_idxs.append((closest_start_idx, closest_end_idx))
            interference.append(closest_idxs)

        df['atmospheric_interference'] = interference
        actual_specs = [np.array(x, dtype=float)
                 for x in df['amplitude_corr_tsys'].tolist()]
        freqs = [np.array(x, dtype=float)
                 for x in df['frequency_array'].tolist()]
        
    else:
        raise ValueError(f"Unsupported extension: {data_path!r}")

    keep = [not np.all(s == 0.0) for s in actual_specs]
    actual_specs = [s for s,k in zip(actual_specs, keep) if k]
    freqs = [f for f,k in zip(freqs, keep) if k]
    atm_intrf = df['atmospheric_interference'].values[keep]

    uid = df['uid'].values[keep]
    ref = df['ref_antenna_name'].values[keep]
    ant = df['antenna'].values[keep]
    pol = df['polarization'].values[keep]

    length_groups: Dict[int, List[int]] = {}
    for i, s in enumerate(actual_specs):
        L = s.shape[0]
        length_groups.setdefault(L, []).append(i)

    result: Dict[int, Tuple[np.ndarray, ...]] = {}
    for L, idxs in length_groups.items():
        actual_specs_L = np.vstack([actual_specs[i] for i in idxs])
        freqs_L = np.vstack([freqs[i] for i in idxs])
        atm_intrf_L = atm_intrf[idxs]
        uid_L   = uid[idxs]
        ref_L   = ref[idxs]
        ant_L   = ant[idxs]
        pol_L   = pol[idxs]
        result[L] = (actual_specs_L, uid_L, ref_L, ant_L, pol_L, freqs_L, atm_intrf_L)

    return df, result


def precompute_kernel(L: int, w: float) -> np.ndarray:
    index = np.arange(L)
    D = np.abs(np.subtract.outer(index, index))
    return np.exp(-(D * D) / (w * w))

_kernel_cache = {}
def _get_kernel(n: int, w: float) -> np.ndarray:
    key = (n, float(w))
    K = _kernel_cache.get(key)
    if K is None:
        K = precompute_kernel(n, w)
        _kernel_cache[key] = K
    return K

@njit(cache=True, fastmath=True, parallel=True)
def calculate_nwkr_sra(array: np.ndarray, W: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
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
def ssr_region(array: np.ndarray, idxs: np.ndarray, W: np.ndarray, ssr_array: np.ndarray, a: int, b: int, range_cap: int) -> float:
    m = idxs.shape[0]

    sri = 0.0
    sro_far  = 0.0
    sro_near = 0.0

    low_cut  = a - 2 * range_cap
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

def score_variance_nwkr(array: np.ndarray, inside: np.ndarray, outside: np.ndarray, a: int, b: int, range_cap: int, W: np.ndarray, ssr_array: np.ndarray) -> float:
    sri = ssr_region(array, inside,  W, ssr_array, a, b, range_cap)
    sro = ssr_region(array, outside, W, ssr_array, a, b, range_cap)
    # print(a, "...", b, "inside:", inside)
    # print(a, "...", b, "sri:", sri)
    # print(a, "...", b, "outside:", outside)
    # print(a, "...", b, "sro:", sro)
    return -(sri + sro)

def _scan_row(params):
    row_idx, row, ignore, freqs, buffer, max_w = params

    freq_step = abs(freqs[1] - freqs[0])
    kernel_factor = math.floor(math.log2(ref_freq / freq_step))
    w         = min(3 + 2 * kernel_factor, max_w)
    range_cap = 3 * w


    row_trimmed = row[buffer: len(row) - buffer]
    n_trimmed = row_trimmed.shape[0]
    W_trimmed = _get_kernel(n_trimmed, w)

    # sra, ssr_array = calculate_nwkr_sra(row, W)
    sra, ssr_array, pred_array = calculate_nwkr_sra(row_trimmed, W_trimmed)
    # print('ssr_array shape:',ssr_array.shape)

    best_score = -np.inf
    best_window = (0, 0)
    best_sri_pred_idx_full = None
    best_sri_pred_vals = None
    
    ignore_trimmed = []
    for (start, end) in ignore:
        s0 = max(start - buffer, 0)
        e0 = min(end - buffer, n_trimmed - 1)
        if s0 < e0:
            ignore_trimmed.append((s0, e0))
    
    all_ch_trimmed = list(range(n_trimmed))
    forbidden   = set()
    for s0,e0 in ignore_trimmed:
        forbidden.update(range(s0, e0 + 1))
    valid_trimmed = [i for i in all_ch_trimmed if i not in forbidden]
    
    for pos_i, i in enumerate(valid_trimmed):
        if pos_i < len(valid_trimmed) - 1 and valid_trimmed[pos_i+1] - i > 1:
            continue
        sub_valid_trimmed = valid_trimmed[pos_i + 1 : pos_i + 1 + range_cap]
        for pos_j, j in enumerate(sub_valid_trimmed):
            if pos_j > 0 and sub_valid_trimmed[pos_j] - sub_valid_trimmed[pos_j - 1] > 1:
                break
            inside  = np.asarray([k for k in valid_trimmed if i <= k <= j], dtype=np.int64)
            outside = np.asarray([k for k in all_ch_trimmed if k not in inside], dtype=np.int64)

            sc = score_variance_nwkr(
                row_trimmed,
                inside,
                outside,
                i, j,
                range_cap,
                W_trimmed,
                ssr_array
            )
            sc = sc / sra + 1

            if sc > best_score:
                best_score, best_window = sc, (i, j)
                sri_pred_vals_trim = predict_on_idxs(row_trimmed, inside, W_trimmed)
                best_sri_pred_idx_full = (inside + buffer)
                best_sri_pred_vals = sri_pred_vals_trim

    oi, oj = best_window
    best_window_original = (oi + buffer, oj + buffer)

    return (row_idx, best_window_original, best_score, pred_array, best_sri_pred_idx_full, best_sri_pred_vals, w, range_cap)

def polynomial_scan_ranges_parallel(
    spec_arrays: np.ndarray,
    score_fn,
    atm_interfs: List[List[Tuple[int,int]]],
    freq_arrays: np.ndarray,
    buffer: int,
    max_w: int,

):
    n_rows, _ = spec_arrays.shape

    params = [
        (i, spec_arrays[i], atm_interfs[i], freq_arrays[i], buffer, max_w)
        for i in range(n_rows)
    ]

    results = []
    with ProcessPoolExecutor() as exe:
        futures = [exe.submit(score_fn, p) for p in params]
        for f in as_completed(futures):
            results.append(f.result())

    results.sort(key=lambda x: x[0])
    windows = [r[1] for r in results]
    scores  = [r[2] for r in results]
    sra_preds = [r[3] for r in results]
    sri_idxs  = [r[4] for r in results]
    sri_vals  = [r[5] for r in results]
    ws = [r[6] for r in results]
    range_caps = [r[7] for r in results]
    return windows, scores, sra_preds, sri_idxs, sri_vals, ws, range_caps

def plot_top_k(
    df: pd.DataFrame,
    actual_spec_arrays: np.ndarray,
    windows: list,
    atm_interfs:list,
    scores: list,
    meta: dict,
    ws: list,
    k: int = 10,
    per_fig: int = 10,
    buffer: int = 10,
    out_dir: str = "Images",
    data_dir: str = "Data",
    sra_preds: list = None,
    sri_idxs: list = None,
    sri_vals: list = None,
    sr_factor: int = 1,
):
    os.makedirs(out_dir, exist_ok=True)
    buf_orig = buffer * sr_factor

    scores_np = np.array(scores)
    finite = np.isfinite(scores_np)
    indexes = np.where(finite)[0]
    top = indexes[np.argsort(scores_np[finite])[:]][::-1]

    top_uids = np.array(meta['uid'])[top]
    top_scores = np.asarray(scores)[top]
    top_windows = np.asarray(windows)[top]
    top_ws = np.asarray(ws)[top]
    
    sub_df = df.loc[df["uid"].isin(top_uids)].copy()
    score_map  = dict(zip(top_uids, top_scores))
    window_map = dict(zip(top_uids, top_windows))
    kernel_map = dict(zip(top_uids, top_ws))

    sub_df["score"]  = sub_df["uid"].map(score_map)
    sub_df["kernel_size"]  = sub_df["uid"].map(kernel_map)
    sub_df[["win_start", "win_end"]] = (
        sub_df["uid"].map(window_map).apply(pd.Series)
    )
    sub_df = sub_df.sort_values("score", ascending=False).reset_index(drop=True)

    sub_df.insert(0, "uid", sub_df.pop("uid"))

    sub_df.to_csv(os.path.join(data_dir, "bandpass_qa0_no_partitions_labelled_filt_scan_stat.csv"), index=False)

    top = indexes[np.argsort(scores_np[finite])[-k:]][::-1]
    
    df_slice = df.iloc[top].copy()

    df_slice["orig_idx"] = top
    df_slice["score"]    = scores_np[top]
    df_slice["kernel_size"] = [ws[i] for i in top]
    df_slice["win_start"] = [windows[i][0] for i in top]
    df_slice["win_end"]   = [windows[i][1] for i in top]

    n_figs = math.ceil(len(df_slice) / per_fig)
    for fig_i in range(n_figs):
        chunk = df_slice.iloc[fig_i*per_fig:(fig_i+1)*per_fig]
        fig_k = len(chunk)
        fig, axes = plt.subplots(fig_k, 1, figsize=(10, 3*fig_k))
        if fig_k == 1:
            axes = [axes]

        for ax, row in zip(axes, chunk.itertuples()):
            i0 = row.orig_idx
            spec = actual_spec_arrays[i0]
            a, b = row.win_start, row.win_end

            for (c,d) in atm_interfs[i0]:
                ax.axvspan(c, d, color='C9', alpha=0.2)

            x = np.arange(len(spec))
            ax.plot(x, spec, color='C0', label="Actual")

            if buf_orig > 0:
                ax.axvspan(0, buf_orig - 1, color='gray', alpha=0.2)
                ax.axvspan(len(spec) - buf_orig, len(spec) - 1, color='gray', alpha=0.2) 

            ax.axvspan(a, b, color='C1', alpha=0.3)

            if sra_preds is not None:
                pred_full = np.full(len(spec), np.nan)
                sra_sr = sra_preds[i0]
                sra_up = np.repeat(sra_sr, sr_factor)
                end = len(spec) - buf_orig
                pred_full[buf_orig:end] = sra_up[:end-buf_orig]
                ax.plot(np.arange(len(spec)), pred_full, '.', ms=2, label="SRA pred")

            if (sri_idxs is not None) and (sri_vals is not None):
                sri_full = np.full(len(spec), np.nan)
                idx_sr  = np.asarray(sri_idxs[i0], dtype=int)
                val_sr  = np.asarray(sri_vals[i0], dtype=float)
                idx_orig_start = idx_sr * sr_factor
                for p, v in zip(idx_orig_start, val_sr):
                    p_end = min(p + sr_factor, len(spec))
                    sri_full[p:p_end] = v
                ax.plot(np.arange(len(spec)), sri_full, '.', ms=2, label="SRI pred")

            ax.set_title(
                f"UID={row.orig_idx}  Score={row.score:.2f}  "
                f"Range=[{a},{b}] Kernel size={row.kernel_size}"
            )
            ax.set_xlabel("Channel")
            ax.set_ylabel("Amplitude")
            ax.legend()

        plt.tight_layout(rect=[0,0,1,0.92])
        plt.suptitle(f"Items {fig_i*per_fig+1}–{fig_i*per_fig+fig_k} of Top {k}", y=0.98)
        outpath = os.path.join(out_dir, f"top_{k}_fig{fig_i+1}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)

def superresolve_ranges(ranges_list: list, factor: int = 4) -> list:
    def merge(rs):
        out = []
        for s,e in rs:
            if not out or s > out[-1][1] + 1:
                out.append((s,e))
            else:
                out[-1] = (out[-1][0], max(out[-1][1], e))
        return out

    new = []
    for sub in ranges_list:
        adjusted = [(s//factor, e//factor) for s,e in sub]
        adjusted = sorted(set(adjusted))
        merged   = merge(adjusted)
        new.append(merged)
    return new

def superresolve(specs: np.ndarray, factor: int = 4) -> np.ndarray:
    n_rows, n_ch = specs.shape
    n_blk = n_ch // factor
    trimmed = specs[:, :n_blk * factor]
    return trimmed.reshape(n_rows, n_blk, factor).mean(axis=2)

def refine_windows_exact_for_length(
    spec_arrays: np.ndarray,
    windows_sr: List[Tuple[int, int]],
    atm_interfs: List[List[Tuple[int,int]]],
    ws: list,
    range_caps: list,
    sr_factor: int,
    buffer: int,
) -> List[Tuple[int,int]]:
    n_rows, n_ch = spec_arrays.shape
    n_trimmed = n_ch - 2 * buffer
    if n_trimmed <= 0:
        return [(0, 0)] * n_rows

    refined = []

    for i in range(n_rows):
        W_trimmed = _get_kernel(n_trimmed, ws[i])
        range_cap = range_caps[i]
        spec_array = spec_arrays[i]
        row_trimmed = spec_array[buffer : n_ch - buffer]

        sra, ssr_array, _ = calculate_nwkr_sra(row_trimmed, W_trimmed)

        forbidden = set()
        for (s, e) in atm_interfs[i]:
            s0 = max(s - buffer, 0)
            e0 = min(e - buffer, n_trimmed - 1)
            if s0 <= e0:
                forbidden.update(range(s0, e0 + 1))
        valid = [ix for ix in range(n_trimmed) if ix not in forbidden]

        x_sr, y_sr = windows_sr[i]

        a_lo = max(x_sr * sr_factor - buffer, 0)
        a_hi = min((x_sr + 1) * sr_factor - 1 - buffer, n_trimmed - 1)
        b_lo = max(y_sr * sr_factor - buffer, 0)
        b_hi = min((y_sr + 1) * sr_factor - 1 - buffer, n_trimmed - 1)

        best_sc = -np.inf
        best_ab = (a_lo, max(a_lo, b_lo))

        for a in range(a_lo, a_hi + 1):
            start_b = max(a, b_lo)
            for b in range(start_b, b_hi + 1):
                inside = np.fromiter((k for k in valid if a <= k <= b), dtype=np.int64)
                if inside.size == 0:
                    continue
                outside = np.fromiter((k for k in valid if k < a or k > b), dtype=np.int64)

                sc = score_variance_nwkr(
                    row_trimmed, inside, outside, a, b,
                    range_cap, W_trimmed, ssr_array
                )
                sc = sc / sra + 1.0
                if sc > best_sc:
                    best_sc = sc
                    best_ab = (a, b)

        a_t, b_t = best_ab
        refined.append((a_t + buffer, b_t + buffer))

    return refined

def main():
    DATA_PATH = "Data/bandpass_qa0_no_partitions_labelled_filt.parquet"
    INTERFERENCE_PATH = "Data/full_spectrum.gzip"
    TOP_K     = 100
    PER_FIG   = 10
    BUFFER_COEFF = 20
    length_SR_FACTOR_map = {64 : 1,
                            120 : 1,
                            128 : 1,
                            240 : 1,
                            256 : 1,
                            480 : 2,
                            512 : 2,
                            960 : 4,
                            1024 : 4,
                            1920 : 8,
                            2048 : 8,
                            3840 : 16}

    length_w_map = {64 : 3,
                    120 : 3,
                    128 : 3,
                    240 : 7,
                    256 : 7,
                    480 : 15,
                    512 : 15,
                    960 : 31,
                    1024 : 31,
                    1920 : 63,
                    2048 : 63,
                    3840 : 127}

    t0 = time.perf_counter()
    df, groups = load_data_by_length(DATA_PATH, INTERFERENCE_PATH)
    t1 = time.perf_counter()
    print(f"Loaded & grouped data in {t1-t0:.3f}s")
    print(f"Found lengths: {sorted(groups.keys())}")

    for length in sorted(groups):
        BUFFER = length // BUFFER_COEFF
        actual_specs, uid, ref, ant, pol, freqs, atm_interfs = groups[length]
        n_rows, row_len = actual_specs.shape
        print(f"\nBefore Preprocessing: Length={length}: {n_rows} rows, {row_len} channels")

        SR_FACTOR = length_SR_FACTOR_map[length]
        atm_interfs_sr = superresolve_ranges(atm_interfs, factor=SR_FACTOR)

        actual_specs_sr = superresolve(actual_specs, factor=SR_FACTOR)

        n_rows, row_len = actual_specs_sr.shape
        print(f"\nAfter Preprocessing: Length={length}: {n_rows} rows, {row_len} channels, SR_factor {SR_FACTOR}")

        meta = {'uid': uid, 'ref': ref, 'ant': ant, 'pol': pol, 'freq': freqs}

        t2 = time.perf_counter()
        windows_sr, scores, sra_preds, sri_idxs, sri_vals, ws, range_caps = polynomial_scan_ranges_parallel(
            spec_arrays=actual_specs_sr,
            score_fn=_scan_row,
            atm_interfs=atm_interfs_sr,
            freq_arrays=freqs,
            buffer=BUFFER // SR_FACTOR,
            max_w=length_w_map[length],
        )
        t3 = time.perf_counter()
        print(f"  Scan time: {t3-t2:.3f}s")

        out_dir = os.path.join("Images", f"length_{length}")
        os.makedirs(out_dir, exist_ok=True)

        data_dir = os.path.join("Data", f"length_{length}")
        os.makedirs(data_dir, exist_ok=True)

        if SR_FACTOR > 1:
            windows_exact = refine_windows_exact_for_length(
                actual_specs,
                windows_sr,
                atm_interfs,
                ws,
                range_caps,
                SR_FACTOR,
                BUFFER
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
            k=min(TOP_K, n_rows),
            per_fig=PER_FIG,
            buffer=BUFFER // SR_FACTOR,
            out_dir=out_dir,
            data_dir=data_dir,
            sra_preds=sra_preds,
            sri_idxs=sri_idxs,
            sri_vals=sri_vals,
            sr_factor=SR_FACTOR,
        )


if __name__ == "__main__":
    main()
