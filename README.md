# RegressionScanStats

Automated spectral line detection in ALMA bandpass calibration data using Nadaraya-Watson Kernel Regression (NWKR) scan statistics.

---

## Installation

```bash
git clone https://github.com/BeardyMan37/RegressionScanStats.git
cd RegressionScanStats
conda env create -f environment.yml
conda activate RegressionScanStats
```

---

## Repository Structure

```
RegressionScanStats/
├── helpers/                        # Core library
│   ├── config.py                    # Global config (kernel kind, ref_freq)
│   ├── scan.py                      # NWKR scan row (optimized + naive)
│   ├── kernels.py                   # Gaussian and Laplace kernel vectors
│   ├── kernel_optimized_state.py    # Incremental O(r) state updates
│   ├── scoring.py                   # SRA/SRI computation, epidemic score
│   ├── superres.py                  # Superresolution + window refinement
│   ├── regressors.py                # Mean/poly/KRR regression baselines
│   ├── predictors.py                # Truncated kernel prediction
│   ├── parallel_exec.py             # Parallelised scan over spectrum groups
│   ├── io_preprocess.py             # Data loading and grouping by length
│   ├── warmup.py                    # Numba/cache warmup
│   └── viz.py                       # Plotting utilities
│
├── scan_tool.py                     # General-purpose scan statistics CLI (mean / poly / NWKR)
├── benchmark_regressor_on_syth.py   # Benchmarking harness (runtime + parameter)
├── score_experiment.py              # Parallelised real-data scoring experiment
├── gen_synth_data.py                # Synthetic dataset generation (i.i.d.)
├── gen_ar2_data.py                  # AR(2) correlated background generation
├── create_dataset.py                # Labelled real dataset construction
├── scan_statistics.py               # ALMA-specific CLI entry point
├── calculate_stats.py               # Score/window statistics computation
├── merge_stats.py                   # Merge results across spectrum groups
├── infer_perpol_labels.py           # Per-polarisation label inference
├── make_score_plot.py               # Score distribution plotting
│
├── explore_data.ipynb               # Data exploration
├── explore_scan_stat.ipynb          # Scan statistic exploration
├── filter_data.ipynb                # Data filtering
├── gen_synth_data.ipynb             # Synthetic data exploration
├── calc_window_score.ipynb          # Window score calculation
│
├── utils/                           # Utility scripts
├── Kats/                            # Kats detector integration
└── environment.yml                  # Conda environment
```

---

## General-Purpose Scanner (`scan_tool.py`)

`scan_tool.py` is a self-contained scan statistics tool that works on **any 1-D signal** — not just ALMA spectra. It supports three regression families:

| Method | Description | Complexity |
|--------|-------------|------------|
| `mean` | Constant-mean model via prefix sums | O(nw) |
| `poly` | Polynomial of degree *d* via Vandermonde prefix sums | O(nwd³) |
| `nwkr` | Nadaraya-Watson kernel regression via incremental state | O(nwr) |

The score for a candidate interval I_{a,b} is:

```
S = 1 - (SSE_in + SSE_out) / SSE_all
```

The detector returns the interval that maximises S.

### Accepted input formats

`.npy`, `.npz`, `.csv`, `.tsv`, `.parquet`, plain text (one value per line).

### Basic usage

```bash
# NWKR (Gaussian) on a CSV column
python scan_tool.py --input data.csv --column amplitude --method nwkr

# NWKR (Laplace) — write result to JSON
python scan_tool.py --input data.npy --method nwkr --kernel laplace --output result.json

# Polynomial degree-2, skip edge channels and a known interference range
python scan_tool.py --input data.csv --method poly --degree 2 \
    --buffer 20 --ignore 50:60 100:110

# Mean model with explicit max window width
python scan_tool.py --input values.txt --method mean --max-window 80

# Per-row mode: each row in the parquet column is a separate spectrum
python scan_tool.py --input spectra.parquet --column amplitude \
    --method nwkr --per-row --output results.json
```

### CLI reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--input`, `-i` | *(required)* | Input file path |
| `--column`, `-c` | first column | Column name (CSV / parquet / npz) |
| `--method`, `-m` | `nwkr` | Regression family: `mean`, `poly`, `nwkr` |
| `--kernel`, `-k` | `gaussian` | NWKR kernel: `gaussian` or `laplace` |
| `--degree`, `-d` | `2` | Polynomial degree (only for `poly`) |
| `--max-window`, `-w` | `n // 5` | Maximum window width in samples |
| `--kernel-width` | auto | NWKR bandwidth *w* |
| `--buffer`, `-b` | `0` | Edge samples to ignore on each side |
| `--ignore` | — | Ranges to ignore, e.g. `--ignore 50:60 100:110` |
| `--output`, `-o` | stdout | Write JSON result to file |
| `--per-row` | off | Treat each cell in the column as a separate signal |
| `--sep` | `,` | CSV delimiter (`tab` for TSV) |

### Python API

```python
from scan_tool import scan

result = scan(
    signal,
    method="nwkr",          # "mean" | "poly" | "nwkr"
    kernel="gaussian",       # "gaussian" | "laplace"  (nwkr only)
    degree=2,                # poly only
    max_window=100,
    kernel_width=None,       # auto-derived if None
    buffer=0,
    ignore=[(50, 60)],       # optional exclusion ranges
)

print(result["score"])      # float in [0, 1]
print(result["win_start"])  # start index in original signal
print(result["win_end"])    # end index in original signal
```

### Output fields

| Field | Description |
|-------|-------------|
| `score` | Scan statistic score Φ(x) ∈ [0, 1] |
| `win_start` | Best window start in original signal coordinates |
| `win_end` | Best window end in original signal coordinates |
| `win_start_filtered` | Best window start in filtered (buffer/ignore-removed) coordinates |
| `win_end_filtered` | Best window end in filtered coordinates |
| `method` | Method used |
| `filtered_len` | Number of samples after buffer/ignore removal |
| `row` | Row index (per-row mode only) |

---

## ALMA-Specific Scanner (`scan_statistics.py`)

For ALMA QA2 bandpass calibration spectra, use `scan_statistics.py`, which adds atmospheric interference filtering, per-polarisation grouping, and ALMA-specific hyperparameter derivation.

```bash
python scan_statistics.py \
    --data-path data/my_spectra.parquet \
    --interference-path data/atmospheric_transmission.parquet \
    --kernel-kind gaussian \
    --out-root images/latest_run \
    --data-root data/latest_run
```

| Argument | Description |
|----------|-------------|
| `--data-path` | Path to raw spectra parquet |
| `--interference-path` | Path to atmospheric transmission parquet |
| `--kernel-kind` | `gaussian` or `laplace` (default: `gaussian`) |
| `--out-root` | Output directory for plots |
| `--data-root` | Output directory for result parquets |
| `--workers` | Number of parallel workers (default: all cores) |
| `--buffer-coeff` | Buffer coefficient for edge trimming (default: 20) |

---

## Building the Labelled Dataset

```bash
python create_dataset.py \
    --parquet data/qa2_raw.parquet \
    --out data/qa2_labelled_dataset.parquet
```

---

## Benchmarking

### Runtime scaling

```bash
# vs spectrum length n
python benchmark_regressor_on_syth.py runtime --variable n

# vs window cap w
python benchmark_regressor_on_syth.py runtime --variable w --n 1000

# vs truncation radius r
python benchmark_regressor_on_syth.py runtime --variable r --n 1000
```

### Detection quality (AR(2) synthetic data)

```bash
# SNR sweep
python benchmark_regressor_on_syth.py parameter \
    --variable snr \
    --w 25 \
    --snr 1.0 2.0 3.0 4.0 5.0 \
    --noise 0.05 \
    --methods mean,poly_deg1,poly_deg2,nwkr_gaussian,nwkr_laplace \
    --out-dir data/parameter_benchmark_ar2

# Noise sweep
python benchmark_regressor_on_syth.py parameter \
    --variable noise \
    --w 25 \
    --snr 2.5 \
    --noise 0.01 0.02 0.05 0.10 0.20 \
    --methods mean,poly_deg1,poly_deg2,nwkr_gaussian,nwkr_laplace \
    --out-dir data/parameter_benchmark_ar2
```

Available method keys: `mean`, `poly_deg1`, `poly_deg2`, `nwkr_gaussian`, `nwkr_laplace`, `nwkr_gaussian_naive`, `nwkr_laplace_naive`, `lrt`, `bocpd`, `capa`, `stumpy`, `ruptures_kernelcpd`

---

## Real Data Evaluation

### Parallelised scoring experiment

```bash
python score_experiment.py \
    --parquet data/qa2_labelled_dataset.parquet \
    --methods mean,poly_deg1,poly_deg2,nwkr_gaussian,nwkr_laplace,lrt,capa \
    --max-rows 1000 \
    --workers 16 \
    --out-dir data/score_experiment
```

> **Note:** `stumpy` cannot run in subprocesses due to Numba/OpenMP conflicts. Exclude it from `--methods` and run it separately or use the serial fallback in the script.

### IoU gridsearch

Run from a notebook after the scoring experiment:

```python
from benchmark_regressor_on_syth import run_iou_gridsearch, _resolve_methods

families = _resolve_methods("mean,poly_deg1,nwkr_gaussian,nwkr_laplace", [])
run_iou_gridsearch(
    parquet_path = "data/qa2_labelled_dataset.parquet",
    families     = families,
    iou_values   = [round(v, 2) for v in np.arange(0.50, 1.01, 0.05)],
    max_rows     = 1000,
    out_dir      = "data/real_data_benchmark",
)
```

### Score threshold gridsearch

```python
from benchmark_regressor_on_syth import run_score_threshold_gridsearch

run_score_threshold_gridsearch(
    results_path = "data/score_experiment/score_experiment_results.csv",
    score_values = [round(v, 2) for v in np.arange(0.30, 1.01, 0.05)],
    iou_thresh   = 0.75,
    out_dir      = "data/score_experiment",
)
```