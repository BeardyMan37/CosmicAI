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

numba.config.BOUNDSCHECK = True


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

@njit(parallel=True)
def calculate_nwkr_sra(array: np.ndarray, W: np.ndarray) -> float:
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
    for i in prange(n):
        pred = numer[i] / denom[i] if denom[i] > 0.0 else 0.0
        diff = array[i] - pred
        ssr_array[i] = diff * diff
        ssr += ssr_array[i]
    return ssr, ssr_array

@njit(parallel=True)
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
    row_idx, row, ignore, range_cap, buffer, w = params

    row_trimmed = row[buffer: len(row) - buffer]
    n_trimmed = row_trimmed.shape[0]
    W_trimmed = precompute_kernel(n_trimmed, w)

    # sra, ssr_array = calculate_nwkr_sra(row, W)
    sra, ssr_array = calculate_nwkr_sra(row_trimmed, W_trimmed)
    # print('ssr_array shape:',ssr_array.shape)

    best_score = -np.inf
    best_window = (0, 0)
    
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
        if pos_i < len(valid_trimmed)-1 and valid_trimmed[pos_i+1] - i > 1:
            continue
        for j in valid_trimmed[pos_i+1 : pos_i+1 + range_cap]:
            inside  = np.asarray([k for k in valid_trimmed if i <= k <= j], dtype=np.int64)
            outside = np.asarray([k for k in all_ch_trimmed if k not in inside], dtype=np.int64)

            sc = score_variance_nwkr(
                np.asarray(all_ch_trimmed),
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

    oi, oj = best_window
    best_window_original = (oi + buffer, oj + buffer)

    return row_idx, best_window_original, best_score

def polynomial_scan_ranges_parallel(
    spec_arrays: np.ndarray,
    score_fn,
    atm_interfs: List[List[Tuple[int,int]]],
    range_cap: int = 20,
    buffer: int    = 10,
    w: float       = 5.0,
):
    n_rows, _ = spec_arrays.shape

    params = [
        (i, spec_arrays[i], atm_interfs[i], range_cap, buffer, w)
        for i in range(n_rows)
    ]

    results = []
    with ProcessPoolExecutor() as exe:
        futures = [exe.submit(_scan_row, p) for p in params]
        for f in as_completed(futures):
            results.append(f.result())

    results.sort(key=lambda x: x[0])
    windows = [r[1] for r in results]
    scores  = [r[2] for r in results]
    return windows, scores


def plot_top_k(
    df: pd.DataFrame,
    actual_spec_arrays: np.ndarray,
    windows: list,
    atm_interfs:list,
    scores: list,
    meta: dict,
    sr_w: int = 5,
    sra_w: int = 5,
    k: int = 10,
    per_fig: int = 10,
    buffer: int = 10,
    out_dir: str = "Images",
    data_dir: str = "Data",
):
    os.makedirs(out_dir, exist_ok=True)

    scores_np = np.array(scores)
    finite = np.isfinite(scores_np)
    indexes = np.where(finite)[0]
    top = indexes[np.argsort(scores_np[finite])[:]][::-1]

    top_uids = np.array(meta['uid'])[top]
    top_scores = np.asarray(scores)[top]
    top_windows = np.asarray(windows)[top]
    
    sub_df = df.loc[df["uid"].isin(top_uids)].copy()
    score_map  = dict(zip(top_uids, top_scores))
    window_map = dict(zip(top_uids, top_windows))

    sub_df["score"]  = sub_df["uid"].map(score_map)
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

            if buffer > 0:
                ax.axvspan(0, buffer-1,     color='gray', alpha=0.2)
                ax.axvspan(len(spec)-buffer, len(spec)-1, color='gray', alpha=0.2)

            ax.axvspan(a, b, color='C1', alpha=0.3)

            ax.set_title(
                f"UID={row.orig_idx}  Score={row.score:.2f}  "
                f"Range=[{a},{b}]"
            )
            ax.set_xlabel("Channel")
            ax.set_ylabel("Amplitude")
            ax.legend()

        plt.tight_layout(rect=[0,0,1,0.92])
        plt.suptitle(f"Items {fig_i*per_fig+1}–{fig_i*per_fig+fig_k} of Top {k}, sr_w {sr_w}, sra_w {sra_w}.", y=0.98)
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
        new.append(adjusted)
    return new

def superresolve(specs: np.ndarray, factor: int = 4) -> np.ndarray:
    n_rows, n_ch = specs.shape
    n_blk = n_ch // factor
    trimmed = specs[:, :n_blk * factor]
    return trimmed.reshape(n_rows, n_blk, factor).mean(axis=2)


def main():
    DATA_PATH = "Data/bandpass_qa0_no_partitions_labelled_filt.parquet"
    INTERFERENCE_PATH = "Data/full_spectrum.gzip"
    W         = 3
    RANGE_CAP = 3 * W
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
        windows_sr, scores = polynomial_scan_ranges_parallel(
            actual_specs_sr,
            score_variance_nwkr,
            atm_interfs=atm_interfs_sr,
            range_cap=RANGE_CAP,
            buffer=BUFFER // SR_FACTOR,
            w=W
        )
        t3 = time.perf_counter()
        print(f"  Scan time: {t3-t2:.3f}s")

        out_dir = os.path.join("Images", f"length_{length}")
        os.makedirs(out_dir, exist_ok=True)

        data_dir = os.path.join("Data", f"length_{length}")
        os.makedirs(data_dir, exist_ok=True)

        windows = [(x * SR_FACTOR, y * SR_FACTOR) for x, y in windows_sr]

        plot_top_k(
            df=df,
            actual_spec_arrays=actual_specs,
            windows=windows,
            atm_interfs=atm_interfs,
            scores=scores,
            meta=meta,
            sr_w=W,
            sra_w=W,
            k=min(TOP_K, n_rows),
            per_fig=PER_FIG,
            buffer=BUFFER // SR_FACTOR,
            out_dir=out_dir,
            data_dir=data_dir,
        )


if __name__ == "__main__":
    main()
