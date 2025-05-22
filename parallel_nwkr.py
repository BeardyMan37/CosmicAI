import ast
import time
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


import ast
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

def load_data_by_length(path: str
) -> Tuple[pd.DataFrame, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """Load spectrograms + metadata, group by the length of each spectrum."""
    if path.endswith('.csv'):
        df = pd.read_csv(path, sep='|', dtype=str, header=0)
        specs = [np.array(ast.literal_eval(s), dtype=float)
                 for s in df['spec_arrays']]
        freqs = [np.array(ast.literal_eval(s), dtype=float)
                 for s in df['frequency_array']]
    elif path.endswith('.parquet'):
        df = pd.read_parquet(path)
        df['uid'] = df.index
        df = df.reset_index(drop=True)
        specs = [np.array(x, dtype=float)
                 for x in df['amplitude_corr_tsys'].tolist()]
        freqs = [np.array(x, dtype=float)
                 for x in df['frequency_array'].tolist()]
    else:
        raise ValueError(f"Unsupported extension: {path!r}")

    keep = [not np.all(s == 0.0) for s in specs]
    specs = [s for s,k in zip(specs, keep) if k]
    freqs = [f for f,k in zip(freqs, keep) if k]

    uid = df['uid'].values[keep]
    ref = df['ref_antenna_name'].values[keep]
    ant = df['antenna'].values[keep]
    pol = df['polarization'].values[keep]

    length_groups: Dict[int, List[int]] = {}
    for i, s in enumerate(specs):
        L = s.shape[0]
        length_groups.setdefault(L, []).append(i)

    result: Dict[int, Tuple[np.ndarray, ...]] = {}
    for L, idxs in length_groups.items():
        specs_L = np.vstack([specs[i] for i in idxs])
        freqs_L = np.vstack([freqs[i] for i in idxs])
        uid_L   = uid[idxs]
        ref_L   = ref[idxs]
        ant_L   = ant[idxs]
        pol_L   = pol[idxs]
        result[L] = (specs_L, uid_L, ref_L, ant_L, pol_L, freqs_L)

    return df, result



def calculate_nwkr_sra(array: np.ndarray, w: float = 5.0) -> float:
    """Compute the leave-one-out NWKR sum of squared residuals (Gaussian kernel)."""
    indices = np.arange(len(array))
    residuals = []
    for idx in range(len(array)):
        weights = np.exp(-((idx - indices) ** 2) / (w ** 2))
        pred = np.sum(weights * array[indices]) / np.sum(weights)
        residuals.append(array[idx] - pred)
    return np.sum(np.array(residuals) ** 2)


def score_variance_nwkr(array: np.ndarray, a: int, b: int, w: float = 5.0) -> float:
    """Compute -SSR for inside [a,b] and outside regions using Gaussian NWKR."""
    n = len(array)
    idx_all = np.arange(n)
    inside = idx_all[a:b+1]
    outside = np.concatenate([idx_all[:a], idx_all[b+1:]])
    
    def ssr_region(idxs):
        if idxs.size == 0:
            return 0.0
        D = np.abs(np.subtract.outer(idxs, idxs))
        W = np.exp(-(D * D) / (w * w))
        sums = W.sum(axis=1)
        preds = np.einsum('ij,j->i', W, array[idxs]) / sums
        diff = array[idxs] - preds
        return np.sum(diff * diff)
    
    sri = ssr_region(inside)
    sro = ssr_region(outside)
    return -(sri + sro)


def _scan_row(params):
    """Worker function: scan one row, find best (i,j), return its score and window."""
    idx, row, score_fn, range_cap, buffer, w = params
    n = len(row)
    best_score = -np.inf
    best_window = (0, 0)
    
    sra = calculate_nwkr_sra(row, w) if score_fn is score_variance_nwkr else None

    start, end = buffer, n - buffer - 1
    for i in range(start, end):
        max_j = min(i + range_cap + 1, n - buffer)
        for j in range(i + 1, max_j):
            sc = score_fn(row, i, j, w)
            if sra is not None:
                sc = sc / sra + 1
            if sc > best_score:
                best_score = sc
                best_window = (i, j)
    return idx, best_window, best_score


def polynomial_scan_ranges_parallel(
    spec_arrays: np.ndarray,
    score_fn,
    range_cap: int = 20,
    buffer: int = 10,
    w: float = 5.0
):
    """
    Parallel scan over each row of spec_arrays using score_fn.
    Returns lists of best windows and scores.
    """
    n_rows = spec_arrays.shape[0]
    params = [
        (i, spec_arrays[i], score_fn, range_cap, buffer, w)
        for i in range(n_rows)
    ]
    
    start = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as exe:
        futures = [exe.submit(_scan_row, p) for p in params]
        for f in tqdm(as_completed(futures), total=n_rows, desc="Rows"):
            results.append(f.result())
    duration = time.perf_counter() - start
    
    results.sort(key=lambda x: x[0])
    windows = [res[1] for res in results]
    scores = [res[2] for res in results]
    
    print(f"Parallel scan completed in {duration:.2f}s")
    return windows, scores


def plot_top_k(
    df: pd.DataFrame,
    spec_arrays: np.ndarray,
    windows: list,
    scores: list,
    meta: dict,
    sr_w: int = 5,
    sra_w: int = 5,
    k: int = 10,
    per_fig: int = 10,
    out_dir: str = "Images",
    data_dir: str = "Data",
):
    os.makedirs(out_dir, exist_ok=True)

    scores_np = np.array(scores)
    finite = np.isfinite(scores_np)
    idxs = np.where(finite)[0]
    top = idxs[np.argsort(scores_np[finite])[-k:]][::-1]
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

    sub_df.to_csv(f"{data_dir}/bandpass_filtered.csv", index=False)

    n_figs = math.ceil(k / per_fig)
    for fig_i in range(n_figs):
        start = fig_i * per_fig
        end = min(start + per_fig, k)
        sub_top = top[start:end]
        sub_k = len(sub_top)

        fig, axes = plt.subplots(sub_k, 1, figsize=(10, 3 * sub_k))
        fig.suptitle(f"Top {k} (items {start+1}–{end}), SR_w={sr_w}, SRA_w={sra_w}",
                     fontsize=16)
        if sub_k == 1:
            axes = [axes]

        for ax, idx in zip(axes, sub_top):
            row = spec_arrays[idx]
            a, b = windows[idx]
            x = np.arange(len(row))

            ax.plot(x, row, color='C0')
            ax.axvspan(a, b, color='C1', alpha=0.3)
            ax.set_title(
                f"Idx={idx}, Score={scores[idx]:.2f}, "
                f"Ref={meta['ref'][idx]}, Ant={meta['ant'][idx]}, "
                f"Pol={meta['pol'][idx]}, "
                f"Freq={meta['freq'][idx][a]:.0f}-{meta['freq'][idx][b]:.0f}"
            )
            ax.set_xlabel("Channel")
            ax.set_ylabel("Amplitude")

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        fname = os.path.join(out_dir, f"top_{k}_NWKR_ss_fig{fig_i+1}.png")
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)


def main():
    CSV_PATH = "Data/bandpass_qa0_no_partitions.parquet"
    RANGE_CAP = 20
    BUFFER    = 10
    W         = 3
    TOP_K     = 100
    PER_FIG   = 10

    t0 = time.perf_counter()
    df, groups = load_data_by_length(CSV_PATH)
    t1 = time.perf_counter()
    print(f"Loaded & grouped data in {t1-t0:.3f}s")
    print(f"Found lengths: {sorted(groups.keys())}")

    for length in sorted(groups):
        if length > 256:
            break

        specs, uid, ref, ant, pol, freqs = groups[length]
        n_rows, row_len = specs.shape
        print(f"\nLength={length}: {n_rows} rows, {row_len} channels")

        meta = {'uid': uid, 'ref': ref, 'ant': ant, 'pol': pol, 'freq': freqs}

        t2 = time.perf_counter()
        windows, scores = polynomial_scan_ranges_parallel(
            specs,
            score_variance_nwkr,
            range_cap=RANGE_CAP,
            buffer=BUFFER,
            w=W
        )
        t3 = time.perf_counter()
        print(f"  Scan time: {t3-t2:.3f}s")

        out_dir = os.path.join("Images", f"length_{length}")
        os.makedirs(out_dir, exist_ok=True)

        data_dir = os.path.join("Data", f"length_{length}")
        os.makedirs(data_dir, exist_ok=True)

        plot_top_k(
            df=df,
            spec_arrays=specs,
            windows=windows,
            scores=scores,
            meta=meta,
            sr_w=W,
            sra_w=W,
            k=min(TOP_K, n_rows),
            per_fig=PER_FIG,
            out_dir=out_dir,
            data_dir=data_dir,
        )


if __name__ == "__main__":
    main()
