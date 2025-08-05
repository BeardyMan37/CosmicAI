#!/usr/bin/env python3
"""
make_score_plot.py  –  make scatter plot of scan stats score vs segment width (MHz)
  color coded by QA2 flag reason. rows with a flag in PRIORITY list are RED (anomalous).
  rows with no flags are green. rows with one or more other flags are blue.

  A CSV file is produced with a subset of the input columns, and the "mapped" QA2 flag reasons
    for easy exploration in other tools (eg topcat). 0=good, 1=priority flags, 2=other flags

  A second scatter plot is also made with segment width in channels on the x axis.
   (it will have the same name with _chan before .png)

usage
-----
python make_score_plot.py \
       --parquet bandpass_qa0_no_partitions_labelled_filt_scored.parquet \
       --csv     bandpass_segment_width.csv \
       --png     segment_width_vs_score.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import ast

KEYS  = ["eb_uid", "antenna", "polarization", "spw_name_ms","win_start","win_end"]
KEEP  = KEYS + ["score", "segment_width", "qa2flag","win_chans"]

# ------------------------------------------------------------------
# Priority QA2 flags → qa2flag = 1
# ------------------------------------------------------------------
PRIORITY = {
    "bandpass_amplitude_frequency",
    "bandpassflag_amplitude_frequency"        # ← add more as needed
}

# ----------------------------------------------------------------------
def channel_spacing(freq_array):
    """Median |Δν| in GHz, robust to string-encoded arrays."""
    if freq_array is None or (isinstance(freq_array, float) and pd.isna(freq_array)):
        return np.nan

    # decode if it looks like a stringified list/array
    if isinstance(freq_array, str):
        try:
            freq_array = ast.literal_eval(freq_array)
        except Exception:
            return np.nan

    if not isinstance(freq_array, (list, tuple, np.ndarray)) or len(freq_array) < 2:
        return np.nan

    diffs = np.diff(np.asarray(freq_array, dtype=float))
    return float(np.median(np.abs(diffs)))

def qa2_flag_code(flags):
    """
    Return 0 / 1 / 2  (no flags / priority flag / other flag).

    * priority = any list element whose lower-case *contains* a PRIORITY token
    * robust to lists, tuples, numpy arrays, or stringified lists
    """
    import ast, numpy as np, pandas as pd

    # ---- normalise to Python list ---------------------------------
    if flags is None or (isinstance(flags, float) and pd.isna(flags)):
        lst = []
    elif isinstance(flags, (list, tuple)):
        lst = list(flags)
    elif isinstance(flags, np.ndarray):
        lst = flags.tolist()
    else:
        s = str(flags).strip()
        if s in ("", "[]", "None"):
            lst = []
        else:                                # try to parse "['flag', ...]"
            try:
                parsed = ast.literal_eval(s)
                lst = list(parsed) if isinstance(parsed, (list, tuple, np.ndarray)) else [s]
            except Exception:
                lst = [s]

    # clean + lower-case for comparison
    clean = [str(f).strip().lower() for f in lst]

    # ---- classify -------------------------------------------------
    if not clean:
        return 0

    if any(any(token in f for token in PRIORITY) for f in clean):
        return 1
    return 2

def scatter(ax, x, y, qa2, title, xlabel):
    """Colour-blind-safe scatter on *ax*."""
    colours = {0: "#228833", 1: "#CC3311", 2: "#4477AA"}   # CB-safe
    markers = {0: "o",       1: "x",       2: "^"}
    order   = [2, 0, 1]                                    # draw X’s last

    for code in order:
        m = qa2 == code
        face = "none" if code in (0, 2) else colours[code]
        ax.scatter(
            x[m], y[m],
            s=50 if code == 1 else 36,
            marker=markers[code],
            edgecolor=colours[code],
            facecolor=face,
            linewidths=1.8 if code == 1 else 1.0,
            alpha=0.85,
            label=f"qa2flag={code}",
            zorder=10 if code == 1 else 5,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--parquet", required=True, help="input scored Parquet")
    ap.add_argument("--csv",     required=True, help="output CSV file")
    ap.add_argument("--png",     default="segment_width_vs_score.png", help="output PNG")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.parquet)

    # 1 – segment_width in MHz. freq axis is actually in Hz...
    df["delta_GHz"]      = df["frequency_array"].apply(channel_spacing)
    df["win_chans"] = (df["win_end"]-df["win_start"]).abs().astype(int)
    df["segment_width"]  = (df["win_end"] - df["win_start"]).abs() * df["delta_GHz"] /1e6

    # 2 – qa2flag code
    df["qa2flag"] = df["QA2 Flag(s)"].apply(qa2_flag_code)

    # 3 – write CSV
    out = df[KEEP].copy()
    out.to_csv(args.csv, index=False)
    print(f"✅  wrote {len(out):,} rows → {args.csv}")

    # 4 – sanity prints
    print("\n--- sanity checks ------------------------------")
    print(f"segment_width MHz : min={out['segment_width'].min():.2f} "
          f"median={out['segment_width'].median():.2f} "
          f"max={out['segment_width'].max():.2f}")
    print("\nqa2flag counts:")
    print(out["qa2flag"].value_counts().sort_index().to_string())
    if out["segment_width"].isna().any():
        print(f"\n⚠️  rows with NaN segment_width : {out['segment_width'].isna().sum():,}")

    # 5 – scatter plot
    # ---- create 1) segment-width and 2) win_chans scatter plots ---------
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    scatter(ax1,
        out["segment_width"], out["score"], out["qa2flag"],
        "Segment-width vs score", "segment_width  [MHz]")
    fig1.tight_layout()
    fig1.savefig(args.png, dpi=180)
    plt.close(fig1)

    png2 = Path(args.png).with_stem(Path(args.png).stem + "_chans")
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    scatter(ax2,
        out["win_chans"], out["score"], out["qa2flag"],
        "Win-channels vs score", "win_chans  [channels]")
    fig2.tight_layout()
    fig2.savefig(png2, dpi=180)
    plt.close(fig2)

    print(f"✅  plot saved → {Path(args.png).resolve()}")
    print(f"✅  plot saved → {png2.resolve()}")


if __name__ == "__main__":
    main()

